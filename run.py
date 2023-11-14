import argparse
from datetime import datetime
import os

from config import BRIGHTFIELD, FLUORESCENT, TARGET_RATIO
from utils import build_results_csv, store_results, VALID_EXTENSIONS, Result, validate_filenames, apply_chi_squared, get_results_rounded
from seeds import process_seed_image

def process_batch(prefix_to_filenames, bf_thresh, fl_thresh, radial_thresh, batch_output_dir):
    results = []
    for i, prefix in enumerate(sorted(prefix_to_filenames.keys())):
        yield f'Processing sample {prefix} ({i+1} of {len(prefix_to_filenames.keys())}):'
        result = Result(prefix)
        for file in sorted(prefix_to_filenames[prefix]):
            filename = os.path.basename(file).split('.')[0]
            postfix = filename.split('_')[1]
            if postfix == BRIGHTFIELD:
                yield f'\t{BRIGHTFIELD} (brightfield) image: {filename}'
                total_seeds = process_seed_image(file, postfix, prefix, bf_thresh, radial_thresh, batch_output_dir)
                result.total_seeds = total_seeds
            elif postfix == FLUORESCENT:
                yield f'\t{FLUORESCENT} (fluorescent) image: {filename}'
                fl_seeds = process_seed_image(file, postfix, prefix, fl_thresh, radial_thresh, batch_output_dir)
                result.fl_seeds = fl_seeds
            else:
                yield f'\tUnknown image type for {filename}'

        if result.total_seeds == None:
            yield f"\tCouldn't find {BRIGHTFIELD} (brightfield) image for {prefix}. Remember that image should be named <prefix_id>_{BRIGHTFIELD}.<img_extension>. Example: img1_{BRIGHTFIELD}.tif"
        if result.fl_seeds == None:    
            yield f"\tCouldn't find {FLUORESCENT} (fluorescent) image for {prefix}. Remember that image should be named <prefix_id>_{FLUORESCENT}.<img_extension>. Example: img1_{FLUORESCENT}.tif"

        if result.total_seeds != None and result.fl_seeds != None:
            result.non_fl_seeds = result.total_seeds - result.fl_seeds

        if result.non_fl_seeds:
            result.ratio_fl_total = result.fl_seeds / result.total_seeds

        apply_chi_squared(result, TARGET_RATIO)

        results.append(result)

    yield results


def parse_args():
    help_message = (
        "This script takes an image or directory of images and returns the number of seeds in the image(s)."
    )
    parser = argparse.ArgumentParser(description=help_message)
    parser.add_argument('-d', '--dir', type=str, help='Path to the image directory', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to the output directory', required=True)
    parser.add_argument('-n', '--nostore', action='store_true', help='Do not store contour images')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot images')
    parser.add_argument('-t', '--intensity_thresh', type=str, help='Intensity threshold to capture seeds. Format is <brightfield_thresh>,<fluorescent_thresh>. Example: 30,30')
    parser.add_argument('-r', '--radial_thresh', type=float, help='Radial threshold to capture seeds')

    args = parser.parse_args()

    # parse intensity thresholds
    bf_thresh, fl_thresh = None, None
    if args.intensity_thresh:
        try:
            bf_thresh, fl_thresh = [int(x) for x in args.intensity_thresh.split(',')]
        except:
            raise Exception('Invalid intensity threshold format. Format is <brightfield_thresh>,<fluorescent_thresh>. Example: 60,60')

    return args, bf_thresh, fl_thresh

def print_welcome_msg():
    print(f'Welcome to Seed Counter!')
    print(f'Make sure that your images are in the input directory in pairs: a {BRIGHTFIELD} (brightfield) image and a {FLUORESCENT} (fluorescent) image. For example, for "img1" you need to have two images: img1_{BRIGHTFIELD}.tif and img1_{FLUORESCENT}.tif')
    print()


def collect_img_files(input_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    # filter out non-image files
    files = [f for f in files if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
    files.sort()

    validate_filenames(files)

    image_files = {}
    for file in files:
        prefix = os.path.basename(file).split('_')[0]
        if prefix not in image_files:
            image_files[prefix] = [file]
        else:
            image_files[prefix].append(file)

    return prefix_to_filenames, files


if __name__ == "__main__":
    args, bf_thresh, fl_thresh = parse_args()
    
    print_welcome_msg()

    prefix_to_filenames, files = collect_img_files(args.dir)
    print(f'Found {len(prefix_to_filenames.keys())} unique prefixes in {len(files)} files')

    # Call the process_image function with the specified image path
    results = []
    for message in process_batch(prefix_to_filenames, bf_thresh, fl_thresh, args.radial_thresh, args.output):
        if type(message) == str:
            print(message)
        else:
            results = message

    results_rounded = get_results_rounded(results, 2)
    results_csv = build_results_csv(results_rounded)
    store_results(results_rounded, args.output)

    print("Thanks for your visit!")

