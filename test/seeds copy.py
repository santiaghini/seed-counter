import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_full

DISTANCE_THRESHOLD = 14 # change according to scale of image
SMALL_AREA = 5 # change according to scale of image

# Eliminate the scale component
# img[-50:, :500] = [0,0,0]

# Function to read and process the image
def process_rfp(image_path, prefix, output_dir=None):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for initial filtering
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the image and threshold to make white-ish areas black
    _, inverted_thresholded = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

    # Use the inverted thresholded image as a mask on the original image
    masked_image = cv2.bitwise_and(image, image, mask=inverted_thresholded)

    # Split the masked image into B, G, R channels
    _, _, red_channel = cv2.split(masked_image)

    # Threshold the red channel image to isolate bright red regions
    _, thresholded = cv2.threshold(red_channel, 240, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    eroded = cv2.erode(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

    # TODO: PARAMETERIZE THIS
    # ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
    ret, sure_fg = cv2.threshold(dist_transform, DISTANCE_THRESHOLD, 255, 0)

    # Finding unknown region
    
    # Label connected components
    # Ensure sure_fg is 8-bit before applying connectedComponentsWithStats
    sure_fg = np.uint8(sure_fg)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
    # labels is markers

    # Get the areas of all components
    areas = stats[1:, cv2.CC_STAT_AREA]
    avg_area = np.mean(areas)

    # Get indices of all areas greater than 1.5 * avg_area
    idxs_large_area = np.where(areas > 1.5 * avg_area)[0] + 1

    component_mask = np.zeros(sure_fg.shape, dtype="uint8")
    for i in idxs_large_area:
        component_mask[labels == i] = 255

    # Perform a distance transform on the component
    dist_transform = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)

    # _, thresh = cv2.threshold(dist_transform, 0.6*dist_transform.max(), 255, 0)
    _, thresh = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

    # Convert sure_fg back to 8-bit
    thresh = np.uint8(thresh)

    # perform connected components analysis on the new markers
    num_labels_new, labels_new = cv2.connectedComponents(thresh, 8, cv2.CV_32S)

    # increment labels_new by num_labels to ensure new unique labels
    labels_new = labels_new + num_labels

    # replace the old components in the labels array with 0s
    for i in idxs_large_area:
        labels[labels == i] = 0

    # replace the old component in the labels array with the new components
    labels[labels_new > num_labels] = labels_new[labels_new > num_labels]

    components = np.copy(labels)
    components[components > 0] = 255
    components = np.uint8(components)

    num_labels_final, labels_final, stats_final, centroids_final = cv2.connectedComponentsWithStats(components, 8, cv2.CV_32S)

    areas_final = stats_final[1:, cv2.CC_STAT_AREA]
    idxs_small_area = np.where(areas_final < SMALL_AREA)[0] + 1
    mask_small = np.isin(labels_final, idxs_small_area)
    labels_final[mask_small] = 0

    # Subtract 1 to exclude the background
    num_seeds = len(np.unique(labels_final)) - 1

    # update labels_final to be labels with every pixel > 0 set to 255
    final_sure_fg = np.copy(labels_final)
    final_sure_fg[final_sure_fg > 0] = 255
    final_sure_fg = np.uint8(final_sure_fg)

    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(final_sure_fg, cv2.COLOR_BGR2RGB), cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    unknown = cv2.subtract(sure_bg, final_sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = labels_final+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(image,markers)

    # plot markers and fit to screen
    plt.figure(figsize=(10,10))
    plt.imshow(markers, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # plt.imshow(markers)
    # plt.show()
    
    # Increase margin of markers to 3 pixels
    kernel = np.ones((7,7),np.uint8)
    mask = markers == -1
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    image[dilated_mask == 1] = [255,255,0]
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Return the number of seeds
    return num_seeds


def process_bf(image_path):
    pass

def count_seeds(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (25, 25), 0)
    # blurred = cv2.medianBlur(image, 15)

    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    plot_full(thresh)

    num_seeds = num_labels - 1

    return num_seeds