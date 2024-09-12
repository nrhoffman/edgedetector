import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def sobelEdgeDetection(image):
    sobelX = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

    sobelY = np.array([[ 1,  2,  1],
                    [ 0,  0,  0],
                    [-1, -2, -1]])
    
    imageNp = np.array(image)

    gradientX = convolve(imageNp, sobelX)
    gradientY = convolve(imageNp, sobelY)

    gradientMagnitude = np.sqrt(gradientX**2 + gradientY**2)

    gradientMagnitude = (gradientMagnitude / gradientMagnitude.max()) * 255

    return Image.fromarray(gradientMagnitude.astype(np.uint8))