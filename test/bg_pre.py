import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def count_seeds(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Adaptive thresholding to segment objects
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    # Morphological operations to remove noise
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    # Sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(image,markers)
    image[markers == -1] = [255,0,0]

    # Label the image and count the regions
    labeled_img = label(image)
    regions = regionprops(labeled_img)
    seed_count = len(regions)

    return seed_count


def plot_full(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":    
    image_path = 'images/VZ254_Brightfield.tif'
    print(f'The total count of seeds is: {count_seeds(image_path)}')
