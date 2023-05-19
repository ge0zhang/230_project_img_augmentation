import cv2
import numpy as np
import Pore_Img_Aug
import matplotlib.pyplot as plt

from Pore_Img_Aug import Augmentation

# Loading the original images pair
inp = cv2.imread('./input1.tif')
lab = cv2.imread('./label1.tif')

# Number of synthetic (augumented) images pairs wanted
n = 3

# Plotting the results
plt.figure()
plt.subplot(2, 4, 1)
plt.imshow(inp)
plt.title('Original Input')

plt.subplot(2, 4, 2 + n)
plt.imshow(lab)
plt.title('Original Label')

for i in range(n):
    inp2, lab2 = Augmentation(inp, lab, GridRatio=2)

    plt.subplot(2, 4, i+2)
    plt.imshow(cv2.cvtColor(inp2, cv2.COLOR_BGR2RGB))
    plt.title('Augumented Input #' + str(i+1))

    plt.subplot(2, 4, i+n+3)
    plt.imshow(cv2.cvtColor(lab2, cv2.COLOR_BGR2RGB))
    plt.title('Augumented Label #' + str(i+1))

    plt.draw()

plt.show()
