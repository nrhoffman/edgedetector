from numpy import asarray
from PIL import Image

def toGrayscale(image):
    originalNp = asarray(image)

    grayscaleNp = (0.2989 * originalNp[:, :, 0] + 0.5870 * originalNp[:, :, 1] + 0.1140 * originalNp[:, :, 2])

    return Image.fromarray(grayscaleNp)