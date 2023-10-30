INSTRUCTIONS_TEXT = """
Seed Counter requires the following for each input image:
- A fluorescence intensity image (grayscale or false color), to count seeds expressing a fluorescent marker
- A brightfield image, to count total number of seeds

You should upload these pairs of images following the following convention:

`<prefix>_<image_type>.<extension>`
- `prefix`: name/ID for seed batch being analyzed, each image pair (fluorescent + brightfield) should have the same prefix.
- `image_type`: either `FL` (fluorescent image) or `BF` (brightfield image)
- `extension`: extension of the image (usually `.tif`)

Example `VZ254_BF.tif` and `VZ254_FL.tif`

Upload your images using the file uploader below. Once you have uploaded all your images, click the "Run Seed Counter" button to run the analysis. You will be able to download the results as a CSV file.
"""