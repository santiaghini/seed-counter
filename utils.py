from datetime import datetime
import os

import matplotlib.pyplot as plt

from config import BRIGHTFIELD, FLUORESCENT
from seeds import process_seed_image

VALID_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']

def plot_full(img, title='', cmap='jet'):
    # add title
    # plt.text(0, 0, title, color='white', fontsize=8, ha='left', va='top')
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()


class Result:
    def __init__(self, prefix):
        self.prefix = prefix
        self.fl_seeds = None
        self.non_fl_seeds = None
        self.total_seeds = None
        self.ratio_fl_total = None

def process_batch(prefix_to_filenames, bf_thresh, fl_thresh, radial_thresh, batch_output_dir):
    results = []
    for i, prefix in enumerate(sorted(prefix_to_filenames.keys())):
        yield (f'Processing image {i+1} of {len(prefix_to_filenames.keys())}')
        result = Result(prefix)
        for file in sorted(prefix_to_filenames[prefix]):
            filename = os.path.basename(file).split('.')[0]
            postfix = filename.split('_')[1]
            if postfix == BRIGHTFIELD:
                yield (f'\t{BRIGHTFIELD} (brightfield) image: {filename}')
                total_seeds = process_seed_image(file, postfix, prefix, bf_thresh, radial_thresh, batch_output_dir)
                result.total_seeds = total_seeds
            elif postfix == FLUORESCENT:
                yield (f'\t{FLUORESCENT} (fluorescent) image: {filename}')
                fl_seeds = process_seed_image(file, postfix, prefix, fl_thresh, radial_thresh, batch_output_dir)
                result.fl_seeds = fl_seeds
            else:
                yield (f'\tUnknown image type for {filename}')

        if 'total_seeds' not in result:
            print(f"\tCouldn't find {BRIGHTFIELD} (brightfield) image for {prefix}. Remember that image should be named <prefix_id>_{BRIGHTFIELD}.<img_extension>. Example: img1_{BRIGHTFIELD}.tif")
        if 'fl_seeds' not in result:    
            print(f"\tCouldn't find {FLUORESCENT} (fluorescent) image for {prefix}. Remember that image should be named <prefix_id>_{FLUORESCENT}.<img_extension>. Example: img1_{FLUORESCENT}.tif")

        if result.total_seeds != None and result.fl_seeds != None:
            result.non_fl_seeds = result.total_seeds - result.fl_seeds

        if result.non_fl_seeds:
            result.ratio_fl_total = round(result.fl_seeds / result.total_seeds, 2)

        results.append(result)

    return results


def build_results_csv(results):
    col_names = ['prefix', 'fl_seeds', 'non_fl_seeds', 'total_seeds', 'ratio_fl_total']
    rows = [col_names]
    for result in results:
        row = [result.prefix, result.fl_seeds, result.non_fl_seeds, result.total_seeds, result.ratio_fl_total]
        rows.append(row)
    return rows


def store_results(results_csv, batch_output_dir, batch_id=None):
    # save file with current timestamp
    if not batch_id:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(batch_output_dir, f'results_{batch_id}.csv')

    with open(output_file, 'w') as f:
        for row in results_csv:
            f.write(','.join([str(r) for r in row]) + '\n')