import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def smoothImage(image):
    originalNp = np.array(image)
    kernelSize = 7  # Kernel size (odd number)
    sigma = 1.0  # Standard deviation
    gaussianKernel = createGaussianKernel(kernelSize, sigma)

    smoothedNp = convolve(originalNp, gaussianKernel)
    return Image.fromarray(smoothedNp)

def createGaussianKernel(size, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)