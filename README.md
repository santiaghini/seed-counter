# Seed Counter

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://seed-counter-brophy.streamlit.app)

Seed Counter is a tool that counts the number of seeds in an image. This tool was developed for segregation analysis of transgenic Arabidopsis lines to identify single insertion lines, and as such is optimized for counting the number of fluorescent seeds in a mixed pool of fluorescent and non-fluorescent seeds.

Seed Counter takes a batch of images as input and outputs in `.csv` format the count of seeds for each input image given.

**Raw Image**
<div style="text-align:center">
    <img src="readme_imgs/raw_fl.png" alt="raw_fl" height="300">
    <img src="readme_imgs/raw_bf.png" alt="raw_bf" height="300">
</div>

<br>

**Segmented Image**
<div style="text-align:center">
    <img src="readme_imgs/segmented_fl.png" alt="segmented_fl" height="300">
    <img src="readme_imgs/segmented_bf.png" alt="segmented_bf" height="300">
</div>

## Getting started
Seed Counter requires the following for each input image:
- A fluorescence intensity image (grayscale or false color), to count seeds expressing a fluorescent marker
- A brightfield image, to count total number of seeds

Images should be in a directory with a specific naming convention:

`<sample>_<image_type>.<extension>`
- `sample`: name/ID for the sample being analyzed, each image pair (fluorescent + brightfield) in the sample should have the same prefix.
- `image_type`: either `FL` (fluorescent image) or `BF` (brightfield image)
- `extension`: extension of the image (usually `.tif`)

Example `VZ254_BF.tif` and `VZ254_FL.tif`

## Usage

First, install the required dependencies. Make sure you have at least `Python 3.7`. It is recommended to create a virtual environment first.
```bash
pip install -r requirements.txt
```

After installing requirements, 1) `cd` into the folder containing this repo and 2) make sure that the images are in a directory and in the format and specification specified above. Then, *Seed Segmenter* can be run with:
```bash
python run.py --dir ./images --output ./output_directory --intensity_thresh 30,30
```
- `-d, --dir`: directory with input images with the specified format (required).
- `-o, --output`: output directory to store results (required).
- `-n, --nostore`: flag, if present, does not store the processed images with contours for counting.
- `-p, --plot`: flag, if present, plots intermediate steps for each image. Default is `False`.
- `-t, --intensity_thresh`: intensity threshold to capture seeds. Format is <brightfield_thresh>,<fluorescent_thresh>. Default is `30,30`.
- `-r, --radial_thresh`: radial threshold to capture seeds (float). This value balances how many smalls seeds are capture versus how much seeds can be separated if together. Usually, range for this value should be around `8.0` and `16.0`. Read [Debugging]() bellow to tune this value. By default this value is set automatically using the median seed area in the image.

You can get details of all arguments by running:
```bash
python run.py --help
```

### Debugging
In some cases, seeds might not be separated properly or some seeds might be left out from the final result. There are two parameters to make adjustments and fix this.

`--intensity_thresh`
- If the seeds are too dim and are not being segmented, try decreasing the threshold (default is `30`).
- If the seeds are bright and there are other shapes in the image being captured, try increasing the threshold.

`--radial_thresh`
- Note: this value usually ranges from `8.0` to `18.0`.
- This parameter has a direct tradeoff between capturing small seeds and separating those that are together. A low value captures small seeds (but doesn't separate very well) and a high value separates well (but leaves out small seeds).
- If there are seeds that are smaller and are not captured, try setting a low value (e.g. `10.0`).
- If there most seeds have the same size and there are many seeds that weren't separated properly, try setting a high value (e.g. `16.0`)

## Image Acquisition

You may need to play around with the settings on your microscope to acquire images with the right contrast/brightness (or adjust the intensity_thresh parameter to work with your images). These are the settings used for our lab's Leica Widefield scope to generate the example images, for seeds carrying FAST markers for either RFP or GFP (https://pubmed.ncbi.nlm.nih.gov/19891705/)

- Epifluorescence with mCherry filter: 100 ms exposure (FastRed marker)
- Epifluorescence with GFP filter: 500 ms exposure (FastGreen marker)
- Brightfield with transillumination light: 100 ms exposure, 60% intensity, 50% aperture

With lower exposure (50 ms and 250 ms for FastRed and FastGreen respectively), non-fluorescent seeds only have background levels of signal, and we found that a lower intensity threshold of 20 was both sufficient and necessary.
