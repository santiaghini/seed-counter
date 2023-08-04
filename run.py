import argparse
from datetime import datetime
import os

from config import BRIGHTFIELD, FLUORESCENT
from seeds import process_seed_image


def parse_args():
    help_message = (
        "This script takes an image or directory of images and returns the number of seeds in the image(s)."
    )
    parser = argparse.ArgumentParser(description=help_message)
    parser.add_argument('-d', '--dir', type=str, help='Path to the image directory', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to the output directory', required=True)
    parser.add_argument('-n', '--nostore', action='store_true', help='Do not store contour images')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot images')
    parser.add_argument('-t', '--thresh', type=str, help='Intensity threshold to capture seeds. Format is <brightfield_thresh>,<fluorescent_thresh>. Example: 60,60')

    args = parser.parse_args()

    # parse intensity thresholds
    if args.thresh:
        try:
            bf_thresh, fl_thresh = args.thresh.split(',')
            args.bf_thresh = int(bf_thresh)
            args.fl_thresh = int(fl_thresh)
        except:
            raise Exception('Invalid intensity threshold format. Format is <brightfield_thresh>,<fluorescent_thresh>. Example: 60,60')

    return args

def print_welcome_msg():
    print(f'Welcome to Seed Counter!')
    print(f'Make sure that your images are in the input directory in pairs: a {BRIGHTFIELD} (brightfield) image and a {FLUORESCENT} (fluorescent) image. For example, for "img1" you need to have two images: img1_{BRIGHTFIELD}.tif and img1_{FLUORESCENT}.tif')


def collect_img_files(args):
    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, f))]
    # filter out non-image files
    valid_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
    files.sort()

    image_files = {}
    for file in files:
        prefix = os.path.basename(file).split('_')[0]
        if prefix not in image_files:
            image_files[prefix] = [file]
        else:
            image_files[prefix].append(file)

    return image_files, files


def store_results(results, output_folder):
    # save file with current timestamp
    output_file = os.path.join(output_folder, f'results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
    with open(output_file, 'w') as f:
        f.write('prefix,fl_seeds,dark_seeds,total_seeds,ratio fl/total\n')
        for i, result in enumerate(results):
            f.write(f'{result["prefix"]},{result.get("fl_seeds")},{result.get("dark_seeds")},{result.get("total_seeds")},{result.get("ratio fl/total")}\n')

    print(f"Finished processing all files and stored results in {output_file}")


if __name__ == "__main__":
    args = parse_args()

    print_welcome_msg()

    image_files, files = collect_img_files(args)
    print(f'Found {len(image_files.keys())} unique prefixes in {len(files)} files')

    # Call the process_image function with the specified image path
    results = []
    for i, prefix in enumerate(sorted(image_files.keys())):
        print(f'Processing image {i+1} of {len(image_files.keys())}')
        result = {'prefix': prefix}
        for file in sorted(image_files[prefix]):
            filename = os.path.basename(file).split('.')[0]
            postfix = filename.split('_')[1]
            if postfix == BRIGHTFIELD:
                print(f'\t{BRIGHTFIELD} (brightfield) image: {filename}')
                total_seeds = process_seed_image(file, postfix, prefix, args.bf_thresh, args.output if not args.nostore else None, args.plot)
                result['total_seeds'] = total_seeds
            elif postfix == FLUORESCENT:
                print(f'\t{FLUORESCENT} (fluorescent) image: {filename}')
                fl_seeds = process_seed_image(file, postfix, prefix, args.fl_thresh, args.output if not args.nostore else None, args.plot)
                result['fl_seeds'] = fl_seeds
            else:
                print(f'\tUnknown image type for {filename}')

        if 'total_seeds' not in result:
            print(f"\tCouldn't find {BRIGHTFIELD} (brightfield) image for {prefix}. Remember that image should be named <prefix_id>_{BRIGHTFIELD}.<img_extension>. Example: img1_{BRIGHTFIELD}.tif")
        if 'fl_seeds' not in result:    
            print(f"\tCouldn't find {FLUORESCENT} (fluorescent) image for {prefix}. Remember that image should be named <prefix_id>_{FLUORESCENT}.<img_extension>. Example: img1_{FLUORESCENT}.tif")

        result['dark_seeds'] = result['total_seeds'] - result['fl_seeds'] if 'total_seeds' in result and 'fl_seeds' in result else None

        if result['dark_seeds']:
            result['ratio fl/total'] = round(result['fl_seeds'] / result['total_seeds'], 2)

        results.append(result)
    
    store_results(results, args.output)

    print("Thanks for your visit!")

