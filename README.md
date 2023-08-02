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
- An mCherry image with fluorescent seeds (to count gene-carrier seeds) 
- A brightfield TL image (to count total number of seeds)

<details>
  <summary>Note on scale</summary>
Scale bar should be around 3.4 mm. If scale is different and results are bad, then scale can be adjusted in `config.py` with the `DISTANCE_THRESHOLDS` constant.
</details>
<br>

Images should be in a directory with a specific naming convention:

`<prefix>_<image_type>.<extension>`
- `prefix`: name of image, each image pair (mCherry + BF TL) should have the same prefix.
- `image_type`: either `RFP` (mCherry image) or `BFTL` (brightfield TL)
- `extension`: extension of the image (usually `.tif`)

Example `VZ254_BFTL.tif` and `VZ254_RFP.tif`

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
- `--store`: flag, if present, stores the processed images with contours for counting.

You can get details of all arguments by running:
```bash
python run.py --help
```

### Debugging
In some cases, seeds might not be separated properly or some seeds might be left out from the final result. If this is the case, you can change parameters in `config.py`, particularly the following constants, which are set by type of image.

```python
# NOTE: This will be the distance threshold used to separate seeds that are too close together
# Increase if seeds are closer together, decrease if seeds are farther apart
# Balance: the smaller this is set, the more likely seeds will be separated BUT the more likely smaller (or dimmer) seeds will be left out
DISTANCE_THRESHOLDS = {
    'BFTL': 10,
    'RFP': 14
}

# NOTE: This will be the ratio of 255 (max value) that will be used to threshold image to capture seeds
# Increase if seeds are brighter, decrease if seeds are darker
# Balance: the smaller this is set, the more likely seeds will be captured BUT the more likely noise will be captured
INITIAL_BRIGHTNESS_THRESHOLDS = {
    'BFTL': 0.93,
    'RFP': 0.90
}
```
