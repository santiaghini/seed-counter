from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import (
    DEFAULT_BRIGHTFIELD_SUFFIX,
    DEFAULT_FLUORESCENT_SUFFIX,
)
from utils import plot_all, plot_full


def mask_red_marker(
    image: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    thr_L, seeds_mask = cv2.threshold(
        L, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seeds_mask = cv2.morphologyEx(seeds_mask, cv2.MORPH_OPEN, kernel, 1)

    # ────────────────────────────────────────────────────────────────
    # 2.  NON-marker seeds  = yellow / tan  → high b*
    #     • analyse b* *only* inside the seed mask
    #     • let Otsu find the valley between “tan” and “not-tan”
    # ────────────────────────────────────────────────────────────────
    thr_marker, non_marker = cv2.threshold(b, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # At this point:  white = high-b* seeds  (yellows/tans)  
    #                 black = lower-b* seeds  (reds / purples / anything else)

    non_marker = cv2.bitwise_and(non_marker, seeds_mask)      # remove stray BG hits

    # ────────────────────────────────────────────────────────────────
    # 3.  MARKER-positive seeds  = all_seeds ⊖ non_marker
    # ────────────────────────────────────────────────────────────────
    marker = cv2.bitwise_xor(seeds_mask, non_marker)

    # optional clean-up (little specks, gaps)
    marker  = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel, 1)

    # ---------------------------------------------------------------
    #  A. distance-transform: keep only pixels ≥ 2 px from any hole
    # ---------------------------------------------------------------
    dist   = cv2.distanceTransform(marker, cv2.DIST_L2, 5)
    core   = (dist > 2).astype(np.uint8) * 255   # adjust 2 → 3 if rims persist

    # ---------------------------------------------------------------
    # optional: restore original seed size if you shrank it too much
    # ---------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    core   = cv2.dilate(core, kernel, iterations=1)   # puts back ~1 px


    return marker


def enhance_contrast(gray_img: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (adaptive histogram equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def normalize_seeds_bright(image):
    """
    Ensures seeds are always the high (bright) values in the image.
    Works for both grayscale and color images.
    """
    # Convert to grayscale if image is color
    L_img = image.copy()

    # Flatten and find the mode (most common pixel value)
    pixels = L_img.flatten()
    mode_val = np.bincount(pixels).argmax()

    # If background is bright (mode > 127), seeds are dark: invert
    if mode_val > 127:
        normalized = cv2.bitwise_not(L_img)
    else:
        normalized = L_img

    return normalized


def dt_threshold_from_median(num_areas, area_labels, areas_stats, median_area, median_mask, seed_mask, dist_transform, frac=0.40, area_tol=0.25):
    lo, hi   = median_area * (1 - area_tol), median_area * (1 + area_tol)

    # 3. build mask of seeds whose area ≈ median
    median_mask = np.zeros_like(seed_mask, np.uint8)
    for cid in range(1, num_areas):
        if lo <= areas_stats[cid, cv2.CC_STAT_AREA] <= hi:
            median_mask[area_labels == cid] = 255

    # 4. “median radius” = max DT value among those seeds
    median_radius = dist_transform[median_mask > 0].max()
    dt_thresh     = median_radius * frac
    return dt_thresh


def dt_threshold_from_reference(dist_transform, ref_radius_px: int = 10, ref_dt_thresh: int = 10):
    from scipy import ndimage

    # --- estimate current seed radius ---------------------------
    #     • find local maxima of dt  (centre of each seed)
    peaks = (dist_transform == ndimage.maximum_filter(dist_transform, size=7)) & (dist_transform > 0)
    peak_vals = dist_transform[peaks]                       # one value ≈ radius for each seed
    curr_radius_px = np.median(peak_vals)       # robust against odd seeds

    # --- scale the reference threshold --------------------------
    scale      = curr_radius_px / ref_radius_px
    dt_thresh  = ref_dt_thresh * scale
    print(f"median seed radius = {curr_radius_px:.2f}px  |  "
          f"scale = {scale:.2f}  |  DT threshold = {dt_thresh:.2f}")
    return dt_thresh


def process_seed_image(
    image: np.ndarray,
    img_type: str,
    sample_name: str,
    initial_brightness_thresh: int | None,
    radial_threshold: float | None,
    image_L: np.ndarray | None,
    output_dir: str | None = None,
    plot: bool = False,
    remove_scale_bar: bool = True,
    radial_threshold_ratio = 0.4,
    radial_threshold_mode: str = "auto_infer",
) -> int:

    plots: list[tuple[np.ndarray, str, str | None]] = []

    if img_type not in [DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX]:
        raise Exception(f'Image type must be either {DEFAULT_BRIGHTFIELD_SUFFIX} (brightfield) or {DEFAULT_FLUORESCENT_SUFFIX} (fluorescent)')
    
    # Only convert to LAB and split if image has 3 channels (i.e., is color)
    if image_L is None and len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)  # keep only L as the lightness channel
    else:
        # If already single channel, assume this is L
        L = image_L

    # We expect seeds to be the bright regions in the image, so we normalize the L channel
    # to ensure that the seeds are always the high (bright) values in the image.
    L_norm = normalize_seeds_bright(L)

    # FIXME: normalize size of the image

    # Eliminate scale bar
    if remove_scale_bar:
        # gray[-50:, :500] = 0
        L_norm[-50:, :500] = 0

    if plot:
        plots.append((L_norm, 'Grayscale image', 'gray'))

    # Threshold the unique channel image to isolate bright regions
    if not initial_brightness_thresh:
        threshold_value, thresholded_img  = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
    else:
        threshold_value, thresholded_img = cv2.threshold(L_norm, initial_brightness_thresh, 255, cv2.THRESH_BINARY)

    print(f"Threshold value: {threshold_value}")

    if plot:
        plots.append((thresholded_img, 'Thresholded image', None))

    # Segment by component areas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)

    # Get the areas of all components
    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the background, which is the first label

    ####### START Filter small areas
    SMALL_AREA_FACTOR = 0.1
    area_95th = np.percentile(areas, 95)
    idxs_small_area = np.where(areas < area_95th * SMALL_AREA_FACTOR)[0] + 1

    # Create a mask for small areas
    mask_small = np.isin(labels, idxs_small_area)

    # Remove areas smaller than median_area * SMALL_AREA_FACTOR
    thresholded_img[mask_small] = 0
    ####### END Filter small areas

    # filter again since we removed small areas
    # Segment by component areas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)

    # Get the areas of all components
    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude the background, which is the first label

    # --- Filter very large areas ---
    # Remove the top five largest areas if they are much larger than the median area
    median_area = np.median(areas)
    LARGE_AREA_FACTOR = 20

    # stats already computed with cv2.connectedComponentsWithStats
    if median_area == 0:
        raise RuntimeError("No valid seed area found")

    # labels to drop (indices are 0-based w.r.t. 'areas', so +1 later)
    too_big = np.where(areas > LARGE_AREA_FACTOR * median_area)[0]

    # keep at most the 5 largest, if you really need the limit
    if too_big.size > 5:
        top5 = too_big[np.argsort(areas[too_big])[-5:]]
        too_big = top5

    thresholded_img_pre_remove_big_areas = np.copy(thresholded_img)

    # zero them in one shot
    for lbl in too_big + 1:          # +1 because label 0 = background
        thresholded_img[labels == lbl] = 0

    # get updated areas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)

    # Get the areas of all components
    areas = stats[1:, cv2.CC_STAT_AREA]

    if plot:
        plots.append((thresholded_img, 'Thresholded image (filtered small areas)', None))

    ####### START Obtain and verify median area
    median_area = np.median(areas)

    temp_areas = np.copy(areas)
    # if areas has an even length, add a zero to the start of the end of the array
    if len(temp_areas) % 2 == 0:
        temp_areas = np.append(temp_areas, 0)
        median_area = np.median(temp_areas)

    median_area_label = np.where(areas == median_area)[0][0] + 1
    median_area_mask = np.zeros(thresholded_img.shape, dtype="uint8")
    median_area_mask[labels == median_area_label] = 255
    if plot:
        plots.append((median_area_mask, 'Median area mask', 'gray'))

    ####### END Get median area and verify

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # eroded = cv2.erode(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    if not radial_threshold:
        # radial_threshold = 50
        if radial_threshold_mode == "from_ref":
            radial_threshold = dt_threshold_from_reference(dist_transform)
        elif radial_threshold_mode == "auto_infer":
            radial_threshold = dt_threshold_from_median(
                num_areas=num_labels, area_labels=labels, areas_stats=stats, median_area=median_area, median_mask=median_area_mask, seed_mask=opening, dist_transform=dist_transform, frac=radial_threshold_ratio
            )
        else:
            raise ValueError(f"Unknown mode: {radial_threshold_mode}")
        # radial_threshold = get_radial_thresh(median_area)
    # radial_threshold = max(radial_threshold, 0.3 * dist_transform.max())
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
        plots.append((image_final_components, 'Final sure foreground (count of seeds is computed from these)', None))

    ####### START Watershed (for visualization only)
    unknown = cv2.subtract(sure_bg, final_sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = labels_final+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    # Perform watershed
    markers = cv2.watershed(image, markers)
    ####### END Watershed

    if plot:
        plots.append((markers, 'Markers', None))
    
    # Increase margin of markers to 3 pixels
    kernel = np.ones((7,7),np.uint8)
    mask = markers == -1
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    image_with_contours_dil = image.copy()
    image_with_contours_dil[dilated_mask == 1] = [255,255,0]

    # Plot the original image with contours
    image_with_contours_final = cv2.cvtColor(image_with_contours_dil, cv2.COLOR_BGR2RGB)
    if plot:
        plots.append((image_with_contours_final, 'Image with contours', None))

    if output_dir is not None:
        cv2.imwrite(f'{output_dir}/{sample_name}_{img_type}_brightness{initial_brightness_thresh}_radial{radial_threshold}_contours.png', image)
    
    # Plot all images at once
    if plot:
        plot_all(plots)

    # Return the number of seeds
    return num_seeds


def process_color_image(
    image_path: str,
    sample_name: str,
    rgb_color: tuple[int, int, int],
    bf_thresh: int,
    fl_thresh: int,
    radial_thresh: float | None,
    output_dir: str | None = None,
    plot: bool = False,
) -> tuple[int, int]:
    """Process a single RGB image for total and colored seeds."""
    original = cv2.imread(image_path)
    total = process_seed_image(
        image=original,
        img_type=DEFAULT_BRIGHTFIELD_SUFFIX,
        sample_name=sample_name,
        initial_brightness_thresh=None,
        radial_threshold=radial_thresh,
        output_dir=output_dir,
        plot=plot,
        image_L=None,
    )

    red_masked = mask_red_marker(original)
    colored = process_seed_image(
        image=original,
        image_L=red_masked,
        img_type=DEFAULT_FLUORESCENT_SUFFIX,
        sample_name=sample_name,
        initial_brightness_thresh=fl_thresh,
        radial_threshold=radial_thresh,
        output_dir=output_dir,
        plot=plot,
    )
    return total, colored
