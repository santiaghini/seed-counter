INSTRUCTIONS_TEXT = """
Seed Counter can operate in two modes:
1. **Fluorescence mode** expects image pairs:
   - A fluorescence intensity image to count seeds expressing a fluorescent marker
   - A brightfield image to count total seeds
   Images should follow the convention `<sample>_<image_type>.<extension>` where:
   - `sample` is an ID for the sample being analyzed
   - `image_type` is either `FL` (fluorescent) or `BF` (brightfield)
   - `extension` is usually `.tif`
   Example: `VZ254_BF.tif` and `VZ254_FL.tif`

2. **Color mode** expects a single RGB image per sample where marked seeds have a distinct color.

Upload your images using the file uploader below. After uploading, click "Run Seed Counter" to run the analysis and download a CSV with the results.
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
