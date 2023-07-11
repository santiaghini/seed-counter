import cv2
import numpy as np
import skimage.morphology

# Load image
image = cv2.imread('path_to_your_image.jpg', 0)

# Threshold the image: this will require tuning
_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Skeletonize the image
skeleton = skimage.morphology.skeletonize(image / 255)

# Count the number of white pixels
length_in_pixels = np.count_nonzero(skeleton)

# Convert to centimeters
pixels_per_centimeter = 50  # this is something you would need to determine for your setup
length_in_centimeters = length_in_pixels / pixels_per_centimeter

print(f"The length of the root is approximately {length_in_centimeters:.2f} centimeters")
