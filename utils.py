from datetime import datetime
import os
from scipy.stats import chisquare

import matplotlib.pyplot as plt

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
        self.prefix: str = prefix
        self.fl_seeds: int = None
        self.non_fl_seeds: int = None
        self.total_seeds: int = None
        self.ratio_fl_total: float = None
        self.chisquare: float = None
        self.pvalue: float = None


def apply_chi_squared(result, expected_ratio):
    if result.fl_seeds == None or result.non_fl_seeds == None:
        return
    observed = [result.fl_seeds, result.non_fl_seeds]
    total = result.total_seeds
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    chi2, p = chisquare(observed, f_exp=expected)
    result.chisquare = chi2
    result.pvalue = p


def build_results_csv(results):
    col_names = ['sample', 'fl_seeds', 'non_fl_seeds', 'total_seeds', 'ratio_fl_total', 'chisquare', 'pvalue']
    rows = [col_names]
    for result in results:
        row = [result.prefix, result.fl_seeds, result.non_fl_seeds, result.total_seeds, result.ratio_fl_total, result.chisquare, result.pvalue]
        rows.append(row)
    return rows


def get_results_rounded(results, decimals=2):
    new_results = results.copy()
    for result in results:
        result.fl_seeds = round(result.fl_seeds)
        result.non_fl_seeds = round(result.non_fl_seeds)
        result.total_seeds = round(result.total_seeds)
        result.ratio_fl_total = round(result.ratio_fl_total, decimals)
        result.chisquare = round(result.chisquare, decimals)
        result.pvalue = round(result.pvalue, decimals)

    return new_results


def store_results(results_csv, batch_output_dir, batch_id=None, filename=None):
    # save file with current timestamp
    if not batch_id:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not filename:
        filename = f'results_{batch_id}.csv'

    output_path = os.path.join(batch_output_dir, filename)

    os.makedirs(batch_output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        for row in results_csv:
            f.write(','.join([str(r) for r in row]) + '\n')

    return output_path


def validate_filenames(filenames):
    for filename in filenames:
        try:
            pieces = filename.split('_')
            extension = pieces[-1].split('.')[1]
        except:
            raise Exception(f'Invalid filename: {filename}. Filenames must be in the format <sample>_<image_type>.<extension>. Example: VZ254_BF.tif')

        if "." + extension not in VALID_EXTENSIONS:
            raise Exception(f'Invalid extension: {extension}. Valid extensions are: {VALID_EXTENSIONS}')