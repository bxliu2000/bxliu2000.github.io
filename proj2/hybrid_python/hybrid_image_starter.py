import matplotlib.pyplot as plt
from align_image_code import align_images
import skimage.io as skio
from skimage.color import rgb2gray
from scipy import signal
import numpy as np
import cv2

# First load images
# low sf
im1 = skio.imread('./nutmeg.jpg')/255.

# high sf
im2 = skio.imread('./DerekPicture.jpg')/255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
im1_aligned = rgb2gray(im1_aligned)
im2_aligned = rgb2gray(im2_aligned)


def convolve_image(im, fil, color):
    if color:
        r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]
        r = signal.convolve2d(r, fil, mode="same")
        g = signal.convolve2d(g, fil, mode="same")
        b = signal.convolve2d(b, fil, mode="same")
        return np.dstack((r, g, b))
    return signal.convolve2d(im, fil, mode="same")



def hybrid_image(im1, im2, sigma1, sigma2):
    gaussian1 = cv2.getGaussianKernel(50, sigma1)
    gaussian1 = gaussian1 * np.transpose(gaussian1)
    
    gaussian2 = cv2.getGaussianKernel(50, sigma2)
    gaussian2 = gaussian2 * np.transpose(gaussian2)
    
    im1 = im1 - convolve_image(im1, gaussian1, False)
    im2 = convolve_image(im2, gaussian2, False)

    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1)))))
    plt.show()
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2)))))
    plt.show()

    return np.clip(im1 + im2, 0, 1)

sigma1 = 0
sigma2 = 0

plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned)))))
plt.show()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_aligned)))))
plt.show()

hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)


plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid)))))
plt.show()

skio.imshow(hybrid)
skio.show()


## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
#pyramids(hybrid, N)