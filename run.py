import argparse
from datetime import datetime
import os

from seeds import process_seed_image


def parse_args():
    help_message = (
        "This script takes an image or directory of images and returns the number of seeds in the image(s)."
    )
    parser = argparse.ArgumentParser(description=help_message)
    parser.add_argument('-d', '--dir', type=str, help='Path to the image directory', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to the output directory', required=True)
    parser.add_argument('-s', '--store', action='store_true', default=True, help='Store output images')

    return parser.parse_args()


def store_results(results, output_folder):
    # save file with current timestamp
    output_file = os.path.join(output_folder, f'results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
    with open(output_file, 'w') as f:
        f.write('prefix,red_seeds,dark_seeds,total_seeds\n')
        for i, result in enumerate(results):
            f.write(f'{result["prefix"]},{result.get("red_seeds")},{result.get("dark_seeds")},{result.get("total_seeds")}\n')

    print(f"Finished processing all files and stored results in {output_file}")


if __name__ == "__main__":
    args = parse_args()
    
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

    print(f'Found {len(image_files.keys())} unique prefixes in {len(files)} files')

    # Call the process_image function with the specified image path
    results = []
    for i, prefix in enumerate(sorted(image_files.keys())):
        print(f'Processing image {i+1} of {len(image_files.keys())}')
        result = {'prefix': prefix}
        for file in image_files[prefix]:
            filename = os.path.basename(file)
            postfix = filename.split('_')[1]
            if postfix == 'BFTL':
                print(f'\tBFTL image: {filename}')
                total_seeds = process_seed_image(file, postfix, prefix, args.output if args.store else None)
                result['total_seeds'] = total_seeds
            elif postfix == 'RFP':
                print(f'\tRFP image: {filename}')
                red_seeds = process_seed_image(file, postfix, prefix, args.output if args.store else None)
                result['red_seeds'] = red_seeds
            else:
                print(f'\tUnknown image type for {filename}')

        if 'total_seeds' not in result:
            print(f"\tCouldn't find BFTL image for {prefix}")
        if 'red_seeds' not in result:    
            print(f"\tCouldn't find RFP image for {prefix}")

        result['dark_seeds'] = result.get('total_seeds',0) - result.get('red_seeds',0)

        results.append(result)
    
    store_results(results, args.output)

    print("Thanks for your visit!")

