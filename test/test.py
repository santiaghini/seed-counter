from seeds import process_bf, process_seed_image

images = [
    "images/VZ075_RUBY mCherry 800ms exposure.tif",
    "images/VZ254_FastRed BF TL 100% intensity 50ms exposure.tif",
    "images/VZ254_RFP100ms.tif"
]

image_path = images[2]

process_seed_image(image_path, "RFP", "VZ075", "output")
