from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import (
    DEFAULT_BRIGHTFIELD_SUFFIX,
    DEFAULT_FLUORESCENT_SUFFIX,
)
from utils import plot_all


def mask_red_marker(
    image: np.ndarray,
) -> np.ndarray:
    # Use LAB color space to separate the red marker
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # Threshold the L channel to get all the seeds
    thr_L, seeds_mask = cv2.threshold(
        L, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seeds_mask = cv2.morphologyEx(seeds_mask, cv2.MORPH_OPEN, kernel, 1)

    # threshold to obtain the non-marker seeds
    thr_marker, non_marker = cv2.threshold(
        b, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    non_marker = cv2.bitwise_and(non_marker, seeds_mask)  # remove stray BG hits

    # xor operation to get the marker seeds (red)
    marker = cv2.bitwise_xor(seeds_mask, non_marker)

    # clean-up (little specks, gaps)
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel, 1)

    # distance-transform: keep only pixels ≥ 2 px from any hole
    dist = cv2.distanceTransform(marker, cv2.DIST_L2, 5)
    core = (dist > 3).astype(np.uint8) * 255

    # restore original seed size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.dilate(core, kernel, iterations=1)  # puts back ~1 px

    return core


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


def dt_threshold_from_median(
    num_areas: int,
    area_labels: np.ndarray,
    areas_stats: np.ndarray,
    median_area: float,
    median_mask: np.ndarray,
    seed_mask: np.ndarray,
    dist_transform: np.ndarray,
    frac: float = 0.40,
    area_tol: float = 0.25,
) -> float:
    lo, hi = median_area * (1 - area_tol), median_area * (1 + area_tol)

    # Build a mask of seeds whose area is within area_tol of the median area
    median_mask = np.zeros_like(seed_mask, np.uint8)
    for cid in range(1, num_areas):
        if lo <= areas_stats[cid, cv2.CC_STAT_AREA] <= hi:
            median_mask[area_labels == cid] = 255

    # Find the maximum distance transform value among those seeds (median radius)
    median_radius = dist_transform[median_mask > 0].max()
    # Return a threshold as a fraction of the median radius
    dt_thresh = median_radius * frac
    return dt_thresh


def process_seed_image(
    image: np.ndarray,
    img_type: str,
    sample_name: str,
    initial_brightness_thresh: int | None = None,
    radial_threshold: float | None = None,
    image_L: np.ndarray | None = None,
    output_dir: str | None = None,
    plot: bool = False,
    remove_scale_bar: bool = True,
    radial_threshold_ratio: float | None = None,
    large_area_factor: float | None = None,
) -> int:
    """
    Process a seed image to count the number of seeds.
    Applies normalization, thresholding, area filtering, and segmentation.

    Parameters:
        image (np.ndarray): The input image (BGR or grayscale).
        img_type (str): The type of image, e.g., 'bf' (brightfield) or 'fl' (fluorescent).
        sample_name (str): The sample name, used for output file naming.
        initial_brightness_thresh (int | None): If provided, use this fixed threshold for binarization;
            if None, use automatic thresholding (Triangle method).
        radial_threshold (float | None): If provided, use this value for distance transform thresholding;
            if None, compute automatically.
        image_L (np.ndarray | None): Optional precomputed L (lightness) channel; if None, extract from image.
        output_dir (str | None): If provided, save output images to this directory.
        plot (bool): If True, display intermediate and final processing steps as plots.
        remove_scale_bar (bool): If True, mask out the lower left region (assumed to be a scale bar).
        radial_threshold_ratio (float | None): Fraction of the median distance transform value to use for
            thresholding; if None, use default (0.4).
        large_area_factor (float): Factor to determine the maximum allowed area for a seed (relative to median area).
            Used to filter out very large objects. Default is None, which means that the operation will not be performed.

    Returns:
        int: The number of seeds detected in the image.
    """
    # Print the values or shapes of arguments for debugging
    print(f"process_seed_image called with:")
    print(f"  image shape: {image.shape if hasattr(image, 'shape') else type(image)}")
    print(f"  img_type: {img_type}")
    print(f"  sample_name: {sample_name}")
    print(f"  initial_brightness_thresh: {initial_brightness_thresh}")
    print(f"  radial_threshold: {radial_threshold}")
    print(
        f"  image_L shape: {image_L.shape if image_L is not None and hasattr(image_L, 'shape') else image_L}"
    )
    print(f"  output_dir: {output_dir}")
    print(f"  plot: {plot}")
    print(f"  remove_scale_bar: {remove_scale_bar}")
    print(f"  radial_threshold_ratio: {radial_threshold_ratio}")
    print(f"  large_area_factor: {large_area_factor}")

    if radial_threshold_ratio is None:
        radial_threshold_ratio = 0.4

    plots: list[tuple[np.ndarray, str, str | None]] = []

    if img_type not in [DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX]:
        raise Exception(
            f"Image type must be either {DEFAULT_BRIGHTFIELD_SUFFIX} (brightfield) or {DEFAULT_FLUORESCENT_SUFFIX} (fluorescent)"
        )

    # --- Extract L channel (lightness) ---
    if image_L is None and len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L_channel, a, b = cv2.split(lab)
    else:
        L_channel = image_L

    # --- Normalize so seeds are always bright ---
    L_norm = normalize_seeds_bright(L_channel)

    if remove_scale_bar:
        L_norm[-50:, :500] = 0

    if plot:
        plots.append((L_norm, "Luminosity image", "jet"))

    # --- Threshold to isolate bright regions (seeds) ---
    if not initial_brightness_thresh:
        threshold_value, thresholded_img = cv2.threshold(
            L_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE
        )
        initial_brightness_thresh = threshold_value
    else:
        threshold_value, thresholded_img = cv2.threshold(
            L_norm, initial_brightness_thresh, 255, cv2.THRESH_BINARY
        )

    print(f"Intensity threshold value: {threshold_value}")

    if plot:
        thresholded_img_copy = np.copy(thresholded_img)
        plots.append((thresholded_img_copy, "Thresholded image", None))

    # --- Initial connected components: segment all regions ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresholded_img, connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background

    # --- Filter out small areas ---
    SMALL_AREA_FACTOR = 0.1
    # 95th percentile of areas works well for most images, as the initial pass has a lot of noise
    area_95th = np.percentile(areas, 95)
    idxs_small_area = np.where(areas < area_95th * SMALL_AREA_FACTOR)[0] + 1
    mask_small = np.isin(labels, idxs_small_area)
    thresholded_img[mask_small] = 0

    # Fallback: filter everything less than 1/1000th of the total image area
    total_area = image.shape[0] * image.shape[1]
    min_area = total_area / 100_000
    idxs_tiny_area = np.where(areas < min_area)[0] + 1
    mask_tiny = np.isin(labels, idxs_tiny_area)
    thresholded_img[mask_tiny] = 0

    # --- Re-segment after removing small areas ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresholded_img, connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]

    # --- Filter out very large areas ---
    if large_area_factor is not None:
        # Remove regions that are much larger than typical seeds (likely merged seeds or artifacts).
        median_area = np.median(areas)

        if median_area == 0:
            raise RuntimeError("No valid seed area found")

        too_big = np.where(areas > large_area_factor * median_area)[0]
        if too_big.size > 5:
            top5 = too_big[np.argsort(areas[too_big])[-5:]]
            too_big = top5

        for lbl in too_big + 1:  # +1 for label offset
            thresholded_img[labels == lbl] = 0

        # --- Re-segment after removing large areas ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresholded_img, connectivity=8
        )
        areas = stats[1:, cv2.CC_STAT_AREA]

    if plot:
        plots.append(
            (thresholded_img, "Thresholded image (filtered small areas)", None)
        )

    # --- Obtain and verify median area ---
    # Find the median area and create a mask for a typical seed.
    median_area = np.median(areas)
    temp_areas = np.copy(areas)
    if len(temp_areas) % 2 == 0:
        temp_areas = np.append(temp_areas, 0)
        median_area = np.median(temp_areas)

    median_area_label = np.where(areas == median_area)[0][0] + 1
    median_area_mask = np.zeros(thresholded_img.shape, dtype="uint8")
    median_area_mask[labels == median_area_label] = 255

    if plot:
        plots.append((median_area_mask, "Median area mask", "gray"))

    # --- Morphological noise removal ---
    # Use morphological opening to remove small noise and smooth the object edges.
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # --- Distance transform for sure foreground ---
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # --- Determine radial threshold ---
    if not radial_threshold:
        # Compute radial threshold by finding the distance transform value at the median area
        radial_threshold = dt_threshold_from_median(
            num_areas=num_labels,
            area_labels=labels,
            areas_stats=stats,
            median_area=median_area,
            median_mask=median_area_mask,
            seed_mask=opening,
            dist_transform=dist_transform,
            frac=radial_threshold_ratio,
        )

    # --- Threshold distance transform to get sure foreground ---
    # This isolates the core of each seed.
    ret, sure_fg = cv2.threshold(dist_transform, radial_threshold, 255, 0)

    # --- Second pass: split large areas ---
    # Convert sure foreground to uint8 for connected components analysis
    sure_fg = np.uint8(sure_fg)
    # Find connected components in the sure foreground mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        sure_fg, connectivity=8
    )
    # Calculate the area of each component (excluding background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    avg_area = np.mean(areas)

    # Identify indices of components that are significantly larger than average (likely merged seeds)
    idxs_large_area = (
        np.where(areas > 1.5 * avg_area)[0] + 1
    )  # +1 to account for background label

    # Create a mask for the large components
    component_mask = np.zeros(sure_fg.shape, dtype="uint8")
    for i in idxs_large_area:
        component_mask[labels == i] = 255

    # Apply distance transform to the mask of large components
    dist_transform = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)

    # Threshold the distance transform to find potential seed centers within large components
    _, thresh = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    thresh = np.uint8(thresh)

    num_labels_new, labels_new = cv2.connectedComponents(thresh, 8, cv2.CV_32S)

    # Offset new labels so they don't overlap with existing ones
    labels_new = labels_new + num_labels

    # Remove the original large components from the label image
    for i in idxs_large_area:
        labels[labels == i] = 0

    # Add the new split components into the label image
    labels[labels_new > num_labels] = labels_new[labels_new > num_labels]

    # Create a binary mask of all components (for final counting)
    components = np.copy(labels)
    components[components > 0] = 255
    components = np.uint8(components)

    # --- Final connected components for seed count ---
    num_labels_final, labels_final, stats_final, centroids_final = (
        cv2.connectedComponentsWithStats(components, 8, cv2.CV_32S)
    )

    # --- Count seeds (exclude background) ---
    num_seeds = len(np.unique(labels_final)) - 1
    final_sure_fg = np.copy(labels_final)
    final_sure_fg[final_sure_fg > 0] = 255
    final_sure_fg = np.uint8(final_sure_fg)

    # --- Plot final sure foreground ---
    image_final_components = cv2.cvtColor(final_sure_fg, cv2.COLOR_BGR2RGB)
    if plot:
        plots.append(
            (
                image_final_components,
                "Final sure foreground (count of seeds is computed from these)",
                None,
            )
        )

    # --- Watershed for visualization only ---
    # Use watershed to visualize boundaries between seeds.
    unknown = cv2.subtract(sure_bg, final_sure_fg)
    markers = labels_final + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    if plot:
        plots.append((markers, "Markers", None))

    # --- Draw contours for visualization ---
    # Highlight the boundaries found by watershed.
    kernel = np.ones((7, 7), np.uint8)
    mask = markers == -1
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    image_with_contours_final = image.copy()
    image_with_contours_final[dilated_mask == 1] = [255, 255, 0]

    # Plot the original image with contours
    if plot:
        plots.append((image_with_contours_final, "Image with contours", None))

    # --- Save output image if requested ---
    if output_dir is not None:
        cv2.imwrite(
            f"{output_dir}/{sample_name}_{img_type}_brightness={initial_brightness_thresh}_radial={radial_threshold}_contours.png",
            image_with_contours_final,
        )

    # --- Show all plots if requested ---
    if plot:
        plot_all(plots)

    return num_seeds


def process_color_image(
    image_path: str,
    sample_name: str,
    bf_thresh: int | None = None,
    fl_thresh: int | None = None,
    radial_threshold: float | None = None,
    radial_threshold_ratio: float | None = None,
    output_dir: str | None = None,
    large_area_factor: float | None = None,
    plot: bool = False,
) -> tuple[int, int]:
    """Process a single RGB image for total and colored seeds."""
    original_image = cv2.imread(image_path)
    total = process_seed_image(
        image=original_image,
        img_type=DEFAULT_BRIGHTFIELD_SUFFIX,
        sample_name=sample_name,
        initial_brightness_thresh=bf_thresh,
        radial_threshold=radial_threshold,
        radial_threshold_ratio=radial_threshold_ratio,
        large_area_factor=large_area_factor,
        output_dir=output_dir,
        plot=plot,
        image_L=None,
    )

    red_masked = mask_red_marker(original_image)
    colored = process_seed_image(
        image=original_image,
        image_L=red_masked,
        img_type=DEFAULT_FLUORESCENT_SUFFIX,
        sample_name=sample_name,
        initial_brightness_thresh=fl_thresh,
        radial_threshold=radial_threshold,
        radial_threshold_ratio=radial_threshold_ratio,
        large_area_factor=large_area_factor,
        output_dir=output_dir,
        plot=plot,
    )
    return total, colored
