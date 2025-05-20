from __future__ import annotations

import argparse
import cv2
import os

from config import (
    DEFAULT_BRIGHTFIELD_SUFFIX,
    DEFAULT_FLUORESCENT_SUFFIX,
    INITIAL_BRIGHTNESS_THRESHOLDS,
)
from utils import (
    build_results_csv,
    store_results,
    VALID_EXTENSIONS,
    Result,
    parse_filename,
    get_results_rounded,
)
from seeds import process_seed_image, process_color_image

DEFAULT_BRIGHTFIELD_THESHOLD = INITIAL_BRIGHTNESS_THRESHOLDS[DEFAULT_BRIGHTFIELD_SUFFIX]
DEFAULT_FLUORESCENT_THRESHOLD = INITIAL_BRIGHTNESS_THRESHOLDS[DEFAULT_FLUORESCENT_SUFFIX]

from typing import Dict, Iterable, Iterator, List, Tuple
from PIL import ImageColor


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""
    return ImageColor.getrgb(color)


def process_batch(
    sample_to_files: Dict[str, List[Dict[str, str]]],
    bf_thresh: int,
    fl_thresh: int,
    radial_thresh: float | None,
    batch_output_dir: str | None,
    bf_suffix: str | None = None,
    fl_suffix: str | None = None,
    plot: bool = False,
) -> Iterator[str | List[Result]]:
    """Process a batch of images and optionally store intermediate outputs.

    Parameters
    ----------
    sample_to_files : dict
        Mapping of sample names to image file info.
    bf_thresh, fl_thresh : int
        Brightfield and fluorescent intensity thresholds.
    radial_thresh : float
        Radial threshold for seed segmentation.
    batch_output_dir : str or None
        Directory to save intermediate images. If ``None`` intermediate
        images are not saved.
    bf_suffix, fl_suffix : str, optional
        Suffixes identifying brightfield and fluorescent images.
    plot : bool, optional
        Display plots of intermediate steps.
    """

    bf_suffix = bf_suffix or DEFAULT_BRIGHTFIELD_SUFFIX
    fl_suffix = fl_suffix or DEFAULT_FLUORESCENT_SUFFIX
    
    results = []
    for i, sample_name in enumerate(sorted(sample_to_files.keys())):
        yield f'Processing sample {sample_name} ({i+1} of {len(sample_to_files.keys())}):'
        result = Result(sample_name)
        for file_obj in sample_to_files[sample_name]:
            file_path = file_obj['file_path']
            filename = file_obj['file_name']
            img_type = file_obj['img_type']

            image = cv2.imread(file_path)
            if img_type == bf_suffix:
                yield f'\t{bf_suffix} (brightfield) image: {filename}'
                img_type_name = DEFAULT_BRIGHTFIELD_SUFFIX
                total_seeds = process_seed_image(image, img_type_name, sample_name, bf_thresh, radial_thresh, batch_output_dir, plot=plot)
                result.total_seeds = total_seeds
            elif img_type == fl_suffix:
                yield f'\t{fl_suffix} (fluorescent) image: {filename}'
                img_type_name = DEFAULT_FLUORESCENT_SUFFIX
                fl_seeds = process_seed_image(image, img_type_name, sample_name, fl_thresh, radial_thresh, batch_output_dir, plot=plot)
                result.fl_seeds = fl_seeds
            else:
                yield f'\tUnknown image type for {filename}'

        if result.total_seeds == None:
            yield f"\tCouldn't find {bf_suffix} (brightfield) image for {sample_name}. Remember that image should be named <prefix_id>_{bf_suffix}.<img_extension>. Example: img1_{bf_suffix}.tif"
        if result.fl_seeds == None:
            yield f"\tCouldn't find {fl_suffix} (fluorescent) image for {sample_name}. Remember that image should be named <prefix_id>_{fl_suffix}.<img_extension>. Example: img1_{fl_suffix}.tif"

        results.append(result)

    yield results


def process_color_batch(
    sample_to_file: Dict[str, Dict[str, str]],
    bf_thresh: int,
    fl_thresh: int,
    radial_thresh: float | None,
    rgb_color: Tuple[int, int, int],
    batch_output_dir: str | None,
    plot: bool = False,
) -> Iterator[str | List[Result]]:
    """Process a batch of single RGB images."""

    print(f"{sample_to_file=}")
    results = []
    for i, sample_name in enumerate(sorted(sample_to_file.keys())):
        yield f'Processing sample {sample_name} ({i+1} of {len(sample_to_file)}):'
        print(f"{sample_to_file[sample_name]=}")
        assert len(sample_to_file[sample_name]), "Sample should have only one image"
        file_path = sample_to_file[sample_name][0]['file_path']
        total, colored = process_color_image(
            file_path,
            sample_name,
            rgb_color,
            bf_thresh,
            fl_thresh,
            radial_thresh,
            batch_output_dir,
            plot=plot,
        )
        result = Result(sample_name)
        result.total_seeds = total
        result.fl_seeds = colored
        results.append(result)

    yield results


def parse_args() -> tuple[argparse.Namespace, int, int, str, str, Tuple[int, int, int]]:
    help_message = (
        "This script takes an image or directory of images and returns the number of seeds in the image(s)."
    )
    parser = argparse.ArgumentParser(description=help_message, argument_default=argparse.SUPPRESS)
    parser.add_argument('-d', '--dir', type=str, help='Path to the image directory. Required', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to the output directory. Required', required=True)
    parser.add_argument('-n', '--nostore', action='store_true', help='Do not store output images in output directory (but still store results .csv file)', default=False)
    parser.add_argument('-p', '--plot', action='store_true', help='Plot images', default=False)
    parser.add_argument('-t', '--intensity_thresh', type=str, help='Intensity threshold to capture seeds. Format is <brightfield_thresh>,<fluorescent_thresh>. Example: "30,30". Default: "%(default)s"', default=f"{DEFAULT_BRIGHTFIELD_THESHOLD},{DEFAULT_FLUORESCENT_THRESHOLD}")
    parser.add_argument('-r', '--radial_thresh', type=float, help='Radial threshold to capture seeds. If not given, this value is set by taking the median area as a reference.', default=None)
    parser.add_argument('-s', '--img_type_suffix', type=str, help='Image type suffix. Format is <brightfield_suffix>,<fluorescent_suffix>. Example: BF,FL. Default: "%(default)s"', default=f"{DEFAULT_BRIGHTFIELD_SUFFIX},{DEFAULT_FLUORESCENT_SUFFIX}")
    parser.add_argument('--mode', type=str, choices=['fluorescence', 'color'], default='fluorescence', help='Counting mode to use.')
    parser.add_argument('--marker_color', type=str, default='#ff0000', help='Hex color for marker seeds when using color mode.')

    args = parser.parse_args()

    try:
        bf_thresh, fl_thresh = [int(x) for x in args.intensity_thresh.split(',')]
    except:
        raise Exception('Invalid intensity threshold format. Format is <brightfield_thresh>,<fluorescent_thresh>. Example: 60,60')
        
    try:
        bf_suffix, fl_suffix = args.img_type_suffix.split(',')
    except Exception:
        raise Exception('Invalid image type suffix format. Format is <brightfield_suffix>,<fluorescent_suffix>. Example: BF,FL')

    marker_rgb = hex_to_rgb(args.marker_color)

    return args, bf_thresh, fl_thresh, bf_suffix, fl_suffix, marker_rgb

def print_welcome_msg() -> None:
    print('Welcome to Seed Counter!')
    print(
        'For fluorescence mode, provide pairs of images: '
        f'{DEFAULT_BRIGHTFIELD_SUFFIX} (brightfield) and {DEFAULT_FLUORESCENT_SUFFIX} (fluorescent).'
    )
    print('For color mode, provide a single RGB image per sample.')
    print()


def collect_img_files(
    input_dir: str,
    bf_suffix: str,
    fl_suffix: str,
) -> tuple[Dict[str, List[Dict[str, str]]], List[str]]:
    file_names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and os.path.splitext(f)[-1].lower() in VALID_EXTENSIONS]

    sample_to_files = {}
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
            sample_to_files[sample_name].append(file_obj)

    return sample_to_files, file_names


def collect_single_img_files(
    input_dir: str,
) -> tuple[Dict[str, List[Dict[str, str]]], List[str]]:
    file_names = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[-1].lower() in VALID_EXTENSIONS
    ]

    sample_to_file = {}
    for filename in file_names:
        sample_name = os.path.splitext(filename)[0]
        file_obj = {
            'file_path': os.path.join(input_dir, filename),
            'file_name': filename,
        }
        sample_to_file[sample_name] = [file_obj]

    return sample_to_file, file_names


if __name__ == "__main__":
    args, bf_thresh, fl_thresh, bf_suffix, fl_suffix, marker_rgb = parse_args()

    print_welcome_msg()

    if args.mode == 'fluorescence':
        sample_to_files, file_names = collect_img_files(args.dir, bf_suffix, fl_suffix)
    else:
        sample_to_files, file_names = collect_single_img_files(args.dir)
    print(f'Found {len(sample_to_files.keys())} unique samples in {len(file_names)} files')

    # Determine whether to store intermediate images
    img_output_dir = None if args.nostore else args.output

    # Process images
    results = []
    if args.mode == 'fluorescence':
        iterator = process_batch(
            sample_to_files,
            bf_thresh,
            fl_thresh,
            args.radial_thresh,
            img_output_dir,
            bf_suffix=bf_suffix,
            fl_suffix=fl_suffix,
            plot=args.plot,
        )
    else:
        iterator = process_color_batch(
            sample_to_files,
            bf_thresh,
            fl_thresh,
            args.radial_thresh,
            marker_rgb,
            img_output_dir,
            plot=args.plot,
        )

    for message in iterator:
        if isinstance(message, str):
            print(message)
        else:
            results = message

    results_rounded = get_results_rounded(results, 2)
    results_csv = build_results_csv(results_rounded)

    # Results CSV is always stored in the output directory
    store_results(results_csv, args.output)

    print("Thanks for your visit!")
