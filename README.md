# Seed Segmenter

Seed Segementer is a tool that allows you to take an image of fluorescent seeds and output the count of marker seeds (and its ratio) and the total seeds.

Seed Counter takes a batch of images as input and outputs in `.csv` format the count of seeds for each input image given.

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
python run.py --dir ./images --output ./output_directory --thresh 60,60
```
- `--dir`: directory with input images with the specified format (required).
- `--output`: output directory to store results (required)
- `--nostore`: flag, if present, does not store the processed images with contours for counting.
- `--plot`: flag, if present, plots intermediate steps for each image.
- `--thresh`: intensity threshold to capture seeds. Format is <brightfield_thresh>,<fluorescent_thresh>. Default is `60,60`.

You can get details of all arguments by running:
```bash
python run.py --help
```

### Debugging
In some cases, seeds might not be separated properly or some seeds might be left out from the final result. If this is the case, you can use the `--thresh` argument for `run.py`. As a general guide:
- If the seeds are too dim and are not being segmented, try decreasing the threshold (default is `60`).
- If the seeds are bright and there are other shapes in the image being captured, try incresing the threshold.