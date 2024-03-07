import argparse
from datetime import datetime
import os

from config import DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX, TARGET_RATIO
from utils import build_results_csv, store_results, VALID_EXTENSIONS, Result, parse_filename, get_results_rounded
from seeds import process_seed_image

def process_batch(sample_to_files, bf_thresh, fl_thresh, radial_thresh, batch_output_dir, bf_suffix=None, fl_suffix=None):
    if not bf_suffix:
        bf_suffix = DEFAULT_BRIGHTFIELD_SUFFIX
    if not fl_suffix:
        fl_suffix = DEFAULT_FLUORESCENT_SUFFIX

    results = []
    for i, sample_name in enumerate(sorted(sample_to_files.keys())):
        yield f'Processing sample {sample_name} ({i+1} of {len(sample_to_files.keys())}):'
        result = Result(sample_name)
        for file_obj in sample_to_files[sample_name]:
            file_path = file_obj['file_path']
            filename = file_obj['file_name']
            img_type = file_obj['img_type']
            
            if img_type == bf_suffix:
                yield f'\t{bf_suffix} (brightfield) image: {filename}'
                img_type_name = DEFAULT_BRIGHTFIELD_SUFFIX
                total_seeds = process_seed_image(file_path, img_type_name, sample_name, bf_thresh, radial_thresh, batch_output_dir)
                result.total_seeds = total_seeds
            elif img_type == fl_suffix:
                yield f'\t{fl_suffix} (fluorescent) image: {filename}'
                img_type_name = DEFAULT_FLUORESCENT_SUFFIX
                fl_seeds = process_seed_image(file_path, img_type_name, sample_name, fl_thresh, radial_thresh, batch_output_dir)
                result.fl_seeds = fl_seeds
            else:
                yield f'\tUnknown image type for {filename}'

        if result.total_seeds == None:
            yield f"\tCouldn't find {bf_suffix} (brightfield) image for {sample_name}. Remember that image should be named <prefix_id>_{bf_suffix}.<img_extension>. Example: img1_{bf_suffix}.tif"
        if result.fl_seeds == None:
            yield f"\tCouldn't find {fl_suffix} (fluorescent) image for {sample_name}. Remember that image should be named <prefix_id>_{fl_suffix}.<img_extension>. Example: img1_{fl_suffix}.tif"

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
    parser.add_argument('-s', '--img_type_suffix', type=str, help='Image type suffix. Format is <brightfield_suffix>,<fluorescent_suffix>. Example: BF,FL')

    args = parser.parse_args()

    # parse intensity thresholds
    bf_thresh, fl_thresh = None, None
    if args.intensity_thresh:
        try:
            bf_thresh, fl_thresh = [int(x) for x in args.intensity_thresh.split(',')]
        except:
            raise Exception('Invalid intensity threshold format. Format is <brightfield_thresh>,<fluorescent_thresh>. Example: 60,60')
        
    if args.img_type_suffix:
        try:
            bf_suffix, fl_suffix = args.img_type_suffix.split(',')
        except:
            raise Exception('Invalid image type suffix format. Format is <brightfield_suffix>,<fluorescent_suffix>. Example: BF,FL')

    return args, bf_thresh, fl_thresh, bf_suffix, fl_suffix

def print_welcome_msg():
    print(f'Welcome to Seed Counter!')
    print(f'Make sure that your images are in the input directory in pairs: a {DEFAULT_BRIGHTFIELD_SUFFIX} (brightfield) image and a {DEFAULT_FLUORESCENT_SUFFIX} (fluorescent) image. For example, for "img1" you need to have two images: img1_{DEFAULT_BRIGHTFIELD_SUFFIX}.tif and img1_{DEFAULT_FLUORESCENT_SUFFIX}.tif')
    print()


def collect_img_files(input_dir):
    file_names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and os.path.splitext(f)[-1].lower() in VALID_EXTENSIONS]

    sample_to_files = []
    for filename in file_names:
        sample_name, img_type = parse_filename(filename, bf_suffix, fl_suffix)
        file_obj = {
            'file_path': os.path.join(input_dir, filename),
            'file_name': filename,
            'img_type': img_type
        }
        if sample_name not in sample_to_files:
            sample_to_files[sample_name] = [file_obj]
        else:
            sample_to_files[sample_name] = [file_obj]

    return sample_to_files, file_names


if __name__ == "__main__":
    args, bf_thresh, fl_thresh, bf_suffix, fl_suffix = parse_args()

    bf_suffix = DEFAULT_BRIGHTFIELD_SUFFIX if not bf_suffix else bf_suffix
    fl_suffix = DEFAULT_FLUORESCENT_SUFFIX if not fl_suffix else fl_suffix

    print_welcome_msg()

    sample_to_files, file_names = collect_img_files(args.dir, bf_suffix, fl_suffix)
    print(f'Found {len(sample_to_files.keys())} unique prefixes in {len(file_names)} files')

    # Call the process_image function with the specified image path
    results = []
    for message in process_batch(sample_to_files, bf_thresh, fl_thresh, args.radial_thresh, args.output, bf_suffix=bf_suffix, fl_suffix=fl_suffix):
        if type(message) == str:
            print(message)
        else:
            results = message

    results_rounded = get_results_rounded(results, 2)
    results_csv = build_results_csv(results_rounded)
    store_results(results_csv, args.output) ###

    print("Thanks for your visit!")
