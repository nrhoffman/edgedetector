import cv2
import smoothImage
import sobelEdgeDetection
import toGrayscale
import findcontours
import tensorflow as tf
from numpy import asarray
from PIL import Image

# def draw_bounding_boxes(image, predictions):
#     for pred in predictions:
#         x, y, w, h = pred['bbox']
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw bounding box
#     return image

image = Image.open('../images/planes_example_2.png')

grayscaleImage = toGrayscale.toGrayscale(image)

smoothedImage = smoothImage.smoothImage(grayscaleImage)

edgeDetectedImage = sobelEdgeDetection.sobelEdgeDetection(smoothedImage)

predictions = findcontours.FindContours(edgeDetectedImage)

# image_with_boxes = draw_bounding_boxes(edgeDetectedImage, predictions)

cv2.imshow('Detection Result', predictions)

cv2.waitKey(0)
cv2.destroyAllWindows()