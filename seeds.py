import cv2
import numpy as np

from config import INITIAL_BRIGHTNESS_THRESHOLDS, SMALL_AREA_PRE_PASS, DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX
from utils import plot_full


REF_RADIAL_THRESH = 8
REF_MEDIAN_AREA = 1250


def get_radial_thresh(median_area):
    radial_thresh = median_area * REF_RADIAL_THRESH / REF_MEDIAN_AREA
    return radial_thresh


def process_seed_image(image_path, img_type, sample_name, initial_brightness_thresh, radial_threshold, output_dir=None, plot=False):
    if img_type not in [DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX]:
        raise Exception(f'Image type must be either {DEFAULT_BRIGHTFIELD_SUFFIX} (brightfield) or {DEFAULT_FLUORESCENT_SUFFIX} (fluorescent)')
    
    if not initial_brightness_thresh:
        initial_brightness_thresh = INITIAL_BRIGHTNESS_THRESHOLDS[img_type]
    
    # Load the image
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if img_type == DEFAULT_BRIGHTFIELD_SUFFIX:
        gray = cv2.bitwise_not(gray)

    # Eliminate scale bar
    gray[-50:, :500] = 0

    if plot:
        plot_full(gray, 'Grayscale image', cmap='gray')

    # Threshold the unique channel image to isolate bright regions
    _, thresholded = cv2.threshold(gray, initial_brightness_thresh, 255, cv2.THRESH_BINARY)

    # Segment by component areas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded, connectivity=8)

    # Get the areas of all components
    areas = stats[1:, cv2.CC_STAT_AREA]

    if plot:
        plot_full(thresholded, 'Thresholded image')

    ####### START Filter small areas
    # Get indices of all areas smaller than SMALL_AREA
    idxs_small_area = np.where(areas < SMALL_AREA_PRE_PASS)[0] + 1

    # Create a mask for small areas
    mask_small = np.isin(labels, idxs_small_area)

    # Remove areas smaller than SMALL_AREA
    thresholded[mask_small] = 0
    ####### END Filter small areas

    # get updated areas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded, connectivity=8)

    # Get the areas of all components
    areas = stats[1:, cv2.CC_STAT_AREA]

    if plot:
        plot_full(thresholded, 'Thresholded image (filtered small areas)')

    ####### START Obtain and verify median area
    median_area = np.median(areas)

    temp_areas = np.copy(areas)
    # if areas has an even length, add a zero to the start of the end of the array
    if len(temp_areas) % 2 == 0:
        temp_areas = np.append(temp_areas, 0)
        median_area = np.median(temp_areas)

    median_area_label = np.where(areas == median_area)[0][0] + 1
    median_area_mask = np.zeros(thresholded.shape, dtype="uint8")
    median_area_mask[labels == median_area_label] = 255
    if plot:
        plot_full(median_area_mask, 'Median area mask')
    ####### END Get median area and verify

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # eroded = cv2.erode(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    if not radial_threshold:
        radial_threshold = get_radial_thresh(median_area)
    # Get centers of components from threshold
    ret, sure_fg = cv2.threshold(dist_transform, radial_threshold, 255, 0)
    
    ####### START Second pass on large areas
    # Ensure sure_fg is 8-bit before applying connectedComponentsWithStats
    sure_fg = np.uint8(sure_fg)
    # Label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg, connectivity=8) # labels is markers

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
    ####### END Second pass on large areas

    # Perform connected components for final count
    num_labels_final, labels_final, stats_final, centroids_final = cv2.connectedComponentsWithStats(components, 8, cv2.CV_32S)

    ####### Filter small areas (second pass)
    # areas_final = stats_final[1:, cv2.CC_STAT_AREA]
    # idxs_small_area = np.where(areas_final < SMALL_AREA_POST_PASS)[0] + 1
    # mask_small = np.isin(labels_final, idxs_small_area)
    # labels_final[mask_small] = 0
    #######

    # Subtract 1 to exclude the background
    num_seeds = len(np.unique(labels_final)) - 1
    # update labels_final to be labels with every pixel > 0 set to 255
    final_sure_fg = np.copy(labels_final)
    final_sure_fg[final_sure_fg > 0] = 255
    final_sure_fg = np.uint8(final_sure_fg)

    # Plot the final sure foreground (count of seeds is computed from these)
    image_final_components = cv2.cvtColor(final_sure_fg, cv2.COLOR_BGR2RGB)
    if plot:
        plot_full(image_final_components, 'Final sure foreground (count of seeds is computed from these)')

    ####### START Watershed (for visualization only)
    unknown = cv2.subtract(sure_bg, final_sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = labels_final+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    # Perform watershed
    markers = cv2.watershed(image,markers)
    ####### END Watershed

    # Plot markers
    if plot:
        plot_full(markers, 'Markers')
    
    # Increase margin of markers to 3 pixels
    kernel = np.ones((7,7),np.uint8)
    mask = markers == -1
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    image[dilated_mask == 1] = [255,255,0]

    # Plot the original image with contours
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if plot:
        plot_full(image_with_contours, 'Image with contours')

    if output_dir is not None:
        cv2.imwrite(f'{output_dir}/{sample_name}_{img_type}_contours.png', image)
    
    # Return the number of seeds
    return num_seeds


# Method for counting seeds based on brightness at the center of the seed (useful when brightness changes outwards from center)
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