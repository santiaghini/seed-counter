from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

from config import TARGET_RATIO

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

def plot_all(plots):
    assert len(plots) == 7, "This function is designed for 7 plots. If you need more or less, please modify the function."
    default_cmap = 'jet'
    num_plots = len(plots)
    num_cols = 3
    num_rows = 2

    # Create a figure with a larger size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for ax, (image, title, cmap) in zip(axes, plots):
        cmap = cmap if cmap else default_cmap
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Last plot is full screen (final result)
    plot_full(*plots[-1])


class Result:
    def __init__(self, prefix):
        self.prefix: str = prefix
        self.__fl_seeds: int = None
        self.__non_fl_seeds: int = None
        self.__total_seeds: int = None
        self.__ratio_fl_total: float = None
        self.__chisquare: float = None
        self.__pvalue: float = None

        self.target_ratio = TARGET_RATIO
    
    def __repr__(self):
        return f"Result(prefix={self.prefix}, fl_seeds={self.__fl_seeds}, non_fl_seeds={self.__non_fl_seeds}, total_seeds={self.__total_seeds}, ratio_fl_total={self.__ratio_fl_total}, chisquare={self.__chisquare}, pvalue={self.__pvalue})"
    
    def to_dict(self):
        return {
            'prefix': self.prefix,
            'fl_seeds': self.__fl_seeds,
            'non_fl_seeds': self.__non_fl_seeds,
            'total_seeds': self.__total_seeds,
            'ratio_fl_total': self.__ratio_fl_total,
            'chisquare': self.__chisquare,
            'pvalue': self.__pvalue
        }
    
    @property
    def fl_seeds(self):
        return self.__fl_seeds
    
    @fl_seeds.setter
    def fl_seeds(self, value):
        self.__fl_seeds = value
        self.update_values()

    @property
    def total_seeds(self):
        return self.__total_seeds
    
    @total_seeds.setter
    def total_seeds(self, value):
        self.__total_seeds = value
        self.update_values()

    @property
    def non_fl_seeds(self):
        return self.__non_fl_seeds
    
    @property
    def ratio_fl_total(self):
        return self.__ratio_fl_total
    
    @property
    def chisquare(self):
        return self.__chisquare
    
    @property
    def pvalue(self):
        return self.__pvalue
    
    def update_values(self):
        if self.__total_seeds != None and self.__fl_seeds != None:
            self.__non_fl_seeds = self.__total_seeds - self.__fl_seeds
            self.__ratio_fl_total = self.__fl_seeds / self.__total_seeds
            self.__chisquare, self.__pvalue = compute_chi2(self, self.target_ratio)

        else:
            self.__non_fl_seeds = None
            self.__ratio_fl_total = None
            self.__chisquare = None
            self.__pvalue = None

    def round_all(self, decimals=2):
        self.__fl_seeds = round_if_not_none(self.__fl_seeds)
        self.__non_fl_seeds = round_if_not_none(self.__non_fl_seeds)
        self.__total_seeds = round_if_not_none(self.__total_seeds)
        self.__ratio_fl_total = round_if_not_none(self.__ratio_fl_total, decimals)
        self.__chisquare = round_if_not_none(self.__chisquare, 4)
        self.__pvalue = round_if_not_none(self.__pvalue, 4)


def compute_chi2(result, expected_ratio):
    observed = np.array([result.fl_seeds, result.non_fl_seeds])
    total = result.total_seeds
    expected = np.array([expected_ratio, (1 - expected_ratio)]) * total
    chi2_result = chisquare(f_obs=observed, f_exp=expected)
    chi2, p = chi2_result
    return chi2, p


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
        result.round_all(decimals)

    return new_results


def round_if_not_none(num, decimals=2):
    return round(num, decimals) if num is not None else None


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
