import cv2

from utils import plot_full

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


if __name__ == "__main__":    
    image_path = 'images/VZ254_Brightfield.tif'
    print(f'The total count of seeds is: {count_seeds(image_path)}')