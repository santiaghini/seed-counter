from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

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
    
    def __repr__(self):
        return f"Result(prefix={self.prefix}, fl_seeds={self.fl_seeds}, non_fl_seeds={self.non_fl_seeds}, total_seeds={self.total_seeds}, ratio_fl_total={self.ratio_fl_total}, chisquare={self.chisquare}, pvalue={self.pvalue})"


def apply_chi_squared(result, expected_ratio):
    if result.fl_seeds == None or result.non_fl_seeds == None:
        return
    observed = np.array([result.fl_seeds, result.non_fl_seeds])
    total = result.total_seeds
    expected = np.array([expected_ratio, (1 - expected_ratio)]) * total
    chi2_result = chisquare(f_obs=observed, f_exp=expected)
    print(f"chi2_result: {chi2_result}")
    chi2, p = chi2_result
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
        result.fl_seeds = round_if_not_none(result.fl_seeds)
        result.non_fl_seeds = round_if_not_none(result.non_fl_seeds)
        result.total_seeds = round_if_not_none(result.total_seeds)
        result.ratio_fl_total = round_if_not_none(result.ratio_fl_total, decimals)
        result.chisquare = round_if_not_none(result.chisquare, 4)
        result.pvalue = round_if_not_none(result.pvalue, 4)

    return new_results

def round_if_not_none(num, decimals=2):
    return round(num, decimals) if num else None


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


def parse_filename(filename, bf_suffix, fl_suffix):
    reminder = f"Filenames must be in the format <sample_name>_<image_type_suffix>.<extension>. Example: VZ254_{bf_suffix}.tif"
    try:
        pieces = filename.split('.')
        name = pieces[0]
        extension = pieces[-1]

        name_pieces = name.split('_')
        sample_name = name_pieces[0]
        img_type = name_pieces[-1]

    except Exception as e:
        print(e)
        raise Exception(f'Invalid filename: {filename}. {reminder}')
    

    if "." + extension not in VALID_EXTENSIONS:
        raise Exception(f'Invalid extension: {extension}. Valid extensions are: {VALID_EXTENSIONS}. {reminder}')
    

    if img_type not in [bf_suffix, fl_suffix]:
        raise Exception(f'Invalid suffix for image type: {img_type}. Valid suffixes are: {bf_suffix} (brightfield) and {fl_suffix} (fluorescent). {reminder}')
    
    return sample_name, img_type
