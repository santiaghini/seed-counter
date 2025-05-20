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

2. **Color mode** expects a single RGB image per sample in which the marked seeds appear in a distinct color. The filename of each image is used as the sample name.

Choose the appropriate mode below, upload your images, then click **Run Seed Counter** to process the batch and download a CSV with the results.
"""

PARAM_HINTS = """
**Intensity Thresholds**
- Lower values capture dim seeds but may include noise.
- Higher values reduce noise but can miss faint seeds.
- /

**Radial Threshold Ratio**
- Controls how aggressively touching seeds are split based on their median size.
- Decrease if seeds are small and merging together; increase if large seeds remain unsplit.
- /

**Large Area Factor**
- Objects larger than `factor × median area` are ignored as likely clumps.
- Raise this if valid large seeds are removed; lower it to discard big artifacts.
- /
"""
