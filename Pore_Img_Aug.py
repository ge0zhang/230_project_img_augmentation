import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage, signal
from skimage.color import rgb2gray
from scipy.ndimage import binary_dilation, median_filter, grey_dilation

# The following comment assumes an input image size of 256x256, a sliding window size of 32x32, a stride of 4,
# and an overlap of 3 between neighboring windows
def Augmentation(Inp, Lab=None, GridRatio=8):
    # Pre-processing input data
    Inp = Inp.astype(np.double)
    Inp_max = np.max(Inp.flatten())
    Inp_min = np.min(Inp.flatten())
    if Inp_min == Inp_max:
        Inp = Inp*254
    else:
        Inp = ((Inp-Inp_min)/(Inp_max-Inp_min))*254
    Inp = np.uint8(Inp+1)

    if len(Inp.shape) == 2:
        Inp = np.stack((Inp, Inp, Inp), axis=2)

    Inp0 = Inp.copy()
    Inp = rgb2gray(Inp)
    Lab = rgb2gray(Lab)
    Shp = Inp.shape                    # 256x256

    Lab = np.uint8(Lab > 0)

    Inp2, Lab2 = Img_Aug(Inp0, Shp, Lab, GridRatio)
    Inp2 = Inp2.astype(np.uint8)

    # Reverse the pre-processing
    Lab2 = (Lab2 - 1).astype(np.uint8)*255

    return Inp2, Lab2

def Img_Aug(Inp, Shp, Lab, GridRatio=8):
    # Declaring initial hyperparameters
    s = np.ceil(np.array(Inp.shape[:2])/GridRatio).astype(int)      # size of each window (32x32)
    ol = 8                                                          # overlap size
    nol = s-2*ol                                                    # non-overlap portion
    stride = np.ceil(np.array(Shp)/160).astype(int)                 # stride
    ph = 0                                                          # placeholder index
    n = np.ceil((Inp.shape[0]-(1+s[0]))/stride[0])*np.ceil(
        (Inp.shape[1]-(1+s[1]))/stride[1])                          # number of samples (56*56)
    n = int(n*2)                                                    # double the number of samples consider the flipped images
    p_ini = np.array([0, 0])
    p = p_ini.copy()

    # Preprocessing
    if len(np.unique(Lab)) == 2:
        Lab = Lab + 1

    Inp0 = Inp.copy()
    Inp = rgb2gray(Inp)

    # Crop the images through sliding window
    I1 = np.zeros((s[0], s[1], n), dtype=np.uint8)
    I2 = np.zeros((s[0], s[1], n), dtype=np.uint8)
    I3 = np.zeros((s[0], s[1], 3, n), dtype=np.uint8)               # 32x32x3x6272

    while (p[0] + s[0]) < Inp.shape[0]:
        p[1] = p_ini[1]
        while (p[1] + s[1]) < Inp.shape[1]:
            I1[:, :, ph] = Inp[p[0]:p[0]+s[0], p[1]:p[1]+s[1]]          # storing the input image windows
            I2[:, :, ph] = Lab[p[0]:p[0]+s[0], p[1]:p[1]+s[1]]          # storing the label image windows
            I3[:, :, :, ph] = Inp0[p[0]:p[0]+s[0], p[1]:p[1]+s[1], :]   # storing the input image windows in RBG
            ph += 1
            # Sliding the window
            p[1] += stride[1]
        p[0] += stride[0]

    # For the flipped images
    Inp_F = np.flip(np.flip(Inp, axis=1), axis=0)
    Lab_F = np.flip(np.flip(Lab, axis=1), axis=0)
    Inp0_F = np.flip(np.flip(Inp0, axis=1), axis=0)

    p[0] = p_ini[0]
    while (p[0] + s[0]) < Inp.shape[0]:
        p[1] = p_ini[1]
        while (p[1] + s[1]) < Inp.shape[1]:
            I1[:, :, ph] = Inp_F[p[0]:p[0] + s[0], p[1]:p[1] + s[1]]
            I2[:, :, ph] = Lab_F[p[0]:p[0] + s[0], p[1]:p[1] + s[1]]
            I3[:, :, :, ph] = Inp0_F[p[0]:p[0] + s[0], p[1]:p[1] + s[1], :]
            ph += 1
            # Sliding the window
            p[1] += stride[1]
        p[0] += stride[0]

    # Augumented images synthesis
    Win2 = np.zeros(s)
    Win2[ol:-ol, ol:-ol] = 1                                              # non-overlapping region
    Win2 = 1 - Win2                                                       # overlapping edges
    Win2 = Win2.flatten()
    I1_o = np.zeros((I1.shape[2], np.sum(Win2 == 1)), dtype=np.uint8)     # container for the overlapping portion

    for j in range(I1.shape[2]):
        t = I1[:, :, j].flatten()
        t = t[Win2 == 1]
        t = np.reshape(t, (1,t.shape[0]))                                 # 348x1
        I1_o[j, :] = t

    wrk_map = np.zeros(Shp+s)                                                   # a workin map for relocating windows (256+32)x(256+32)
    wrk_map0 = np.repeat(wrk_map[:, :, np.newaxis], 3, axis=2)
    L0 = np.zeros_like(wrk_map)

    ps1 = np.arange(ol, wrk_map.shape[0]-nol[0]-ol, nol[0])                     # index of last window position
    ps2 = np.arange(ol, wrk_map.shape[1]-nol[1]-ol, nol[1])

    Loc = np.zeros((len(ps1) * len(ps2), 2))                                    # location of each window

    # Indicies for shuffling window locations
    idx = np.arange(0, len(Loc))
    idx_s = np.random.permutation(np.array(idx).flatten())
    idx = idx_s.reshape(idx.shape)

    ph = 0
    # Load coordinates for locations
    for m in range(len(ps1)):
        for n in range(len(ps2)):
            Loc[ph, :] = [ps1[m], ps2[n]]
            ph += 1

    for k in range(Loc.shape[0]):
        # Shuffle locations
        p = Loc[idx[k], :].astype(np.int)
        # Choose the window location
        t = wrk_map[p[0]-ol:p[0]+nol[0]+ol, p[1]-ol:p[1]+nol[1]+ol]             # 32x32

        Win = t > 0
        Win3 = Win.flatten()
        Win3 = Win3[Win2 != 0]                                                  # choose the overlapping edges (348 pixels)
        Win3 = np.tile(Win3[:, np.newaxis], (1,I1.shape[2])).T                  # 6272x348

        # Choosing the best window
        # If no neighboring and overlapping windows
        if np.sum(Win) == 0:
            # Randomly choose one of the windows
            ID = np.random.randint(0, I1.shape[2], size=(1, 1))
            ID = int(ID[0, 0])
            E = 0                      # error to guide the edge interpolation
        else:
            # Look at the overlapping edges (348 pixels)
            t2 = np.squeeze(t.flatten())
            t2 = t2[Win2 != 0]
            t2 = np.reshape(t2, (t2.shape[0], 1))
            t3 = np.tile(t2, (1, I1.shape[2])).T
            Err = np.abs(t3 - I1_o) * Win3
            Win4 = Win.flatten()
            Win4 = Win4[Win2 != 0]
            Win4 = np.repeat(Win4[:, np.newaxis], I1.shape[2], axis=1).T
            Err = np.sum(Err * Win4, axis=1) / np.sum(Win)

            # Choosing the window with edge that has the smallest error with the existing neighboring edges
            indices = np.argsort(Err)
            ID = indices[0]
            E = Err[ID]

            if np.max(Win) == 0:
                ID = np.random.randint(0, I1.shape[2], size=(1, 1))
                ID = int(ID[0, 0])
                E = 0

        w = ndimage.morphology.distance_transform_edt(np.logical_not(1-Win))
        w = w / np.max(w.flatten())
        w[np.isnan(w)] = 0

        # For RGB input
        IMG = I3[:, :, :, ID]
        w0 = np.repeat(w[:, :, np.newaxis], 3, axis=2)
        t0 = wrk_map0[p[0]-ol:p[0]+nol[0]+ol, p[1]-ol:p[1]+nol[1]+ol, :].astype(np.float64)
        w0 = w0.astype(np.float64)
        IMG = IMG.astype(np.float64)
        wrk_map0[p[0]-ol:p[0]+nol[0]+ol, p[1]-ol:p[1]+nol[1]+ol, :] = (t0*w0)+(IMG*(1-w0))

        # # Smoothing the image
        # for ii in range(1):
        #     wrk_map0[p[0] - ol:p[0] + nol[0] + ol, p[1] - ol:p[1] + nol[1] + ol] = median_filter(
        #         wrk_map0[p[0] - ol:p[0] + nol[0] + ol, p[1] - ol:p[1] + nol[1] + ol], size=(3, 3, 3), mode='reflect')

        # For label
        IMG = I2[:, :, ID]
        t0 = L0[p[0]-ol:p[0]+nol[0]+ol, p[1]-ol:p[1]+nol[1]+ol].astype(np.float64)
        w0 = w0.astype(np.float64)
        IMG = IMG.astype(np.float64)
        BD = binary_dilation(Win, structure=np.ones((ol, ol)))
        lb = (t0*w) + (IMG*(1-w))
        lb = lb.astype(np.uint8)
        lb0 = lb.copy()

        # Smoothing the edge
        for ii in range(3):
            lb = median_filter(lb, size=(3, 3), mode='reflect')

        lb[BD == 0] = lb0[BD == 0]
        L0[p[0]-ol:p[0]+nol[0]+ol, p[1]-ol:p[1]+nol[1]+ol] = lb

    L0 = ndimage.median_filter(L0, size=(3, 3), mode='reflect')
    L0 = L0[:Shp[0], :Shp[1]]

    wrk_map0 = wrk_map0[:Shp[0], :Shp[1]]

    return wrk_map0, L0