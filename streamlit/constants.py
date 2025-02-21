INSTRUCTIONS_TEXT = """
Seed Counter requires the following for each input image:
- A fluorescence intensity image (grayscale or false color), to count seeds expressing a fluorescent marker
- A brightfield image, to count total number of seeds

You should upload these pairs of images following the following convention:

`<sample>_<image_type>.<extension>`
- `sample`: name/ID for the sample being analyzed, each image pair (fluorescent + brightfield) in the sample should have the same prefix.
- `image_type`: either `FL` (fluorescent image) or `BF` (brightfield image) - suffix can be changed in "Parameters for manual setup"
- `extension`: extension of the image (usually `.tif`)

Example `VZ254_BF.tif` and `VZ254_FL.tif`

Upload your images using the file uploader below. Once you have uploaded all your images, click the "Run Seed Counter" button to run the analysis. You will be able to download the results as a CSV file.
"""

PARAM_HINTS = """
**Intensity**
- If the seeds are too dim and are not being segmented, try decreasing the threshold.
- If the seeds are bright and there are other shapes in the image being captured, try increasing the threshold
- /

**Radial Threshold**
- This parameter has a direct tradeoff between capturing small seeds and separating those that are together. A low value captures small seeds (but doesn't separate very well) and a high value separates well (but leaves out small seeds).
- If there are seeds that are smaller and are not captured, try setting a low value (e.g. `10.0`).
- If there most seeds have the same size and there are many seeds that weren't separated properly, try setting a high value (e.g. `16.0`)
- /
"""
