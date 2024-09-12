import cv2
import numpy as np
from PIL import Image

def FindContours(image):
    image_np = np.array(image)

    # Convert image to binary using a threshold
    _, binary_image = cv2.threshold(image_np, 50, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)  # Convert to color image for visualization

    # Loop through contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box
        cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to Pillow Image and save/show
    return image_with_contours