# Seed Segmenter

Seed Segementer is a tool that allows you to take an image of fluorescent seeds and output the count of gene-carrier seeds (red) and "normal" seeds.

It takes a batch of images and outputs in `.csv` format the count of seeds for each input image given.

**Before**
<div style="text-align:center">
    <img src="readme_imgs/before.jpg" alt="Before" height="300">
</div>

<br>

**After**
<div style="text-align:center">
    <img src="readme_imgs/segmented.png" alt="After" height="300">
</div>

## Getting started
Seed segmenter requires the following for each input image:
- A fluorescent image with marker seeds
- A brightfield image (to count total number of seeds)

Images should be in a directory with a specific naming convention:

`<prefix>_<image_type>.<extension>`
- `prefix`: name of image, each image pair (fluorescent + brightfield) should have the same prefix.
- `image_type`: either `FL` (fluorescent image) or `BF` (brightfield image)
- `extension`: extension of the image (usually `.tif`)

Example `VZ254_BF.tif` and `VZ254_FL.tif`

## Usage

First, install the required dependencies. Make sure you have at least `Python 3.7`. It is recommended to create a virtual environment first.
```bash
pip install -r requirements.txt
```

After installing requirements, and making sure that the images are in a directory and in the format and specification specified above, Seed Segmenter can be run with:
```bash
python run.py --dir ./images --output ./output_directory --store
```
- `--dir`: directory with input images with the specified format (required).
- `--output`: output directory to store results (required)
- `--nostore`: flag, if present, does not store the processed images with contours for counting.
- `--plot`: flag, if present, plots intermediate steps for each image.

You can get details of all arguments by running:
```bash
python run.py --help
```

### Debugging
In some cases, seeds might not be separated properly or some seeds might be left out from the final result. If this is the case, you can change parameters in `config.py`, particularly the following constants, which are set by type of image.

```python
# NOTE: This is a number between 0 and 255 (max value) that will be used to threshold image to capture seeds according to brightness
# Increase if seeds are brighter, decrease if seeds are darker
# Balance: the smaller this is set, the more likely seeds will be captured BUT the more likely noise will be captured
INITIAL_BRIGHTNESS_THRESHOLDS = {
    BRIGHTFIELD: 60,
    FLUORESCENT: 60
}
```
