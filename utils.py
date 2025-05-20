from __future__ import annotations

import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from scipy.stats import chisquare

from config import TARGET_RATIO

VALID_EXTENSIONS: list[str] = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']

def plot_full(img: np.ndarray, title: str = '', cmap: str = 'jet') -> None:
    # add title
    # plt.text(0, 0, title, color='white', fontsize=8, ha='left', va='top')
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_all(plots: list[tuple[np.ndarray, str, str | None]]) -> None:
    num_plots = len(plots)
    if num_plots == 0:
        return

    # Choose up to 3 columns for aesthetics, but you can change this
    max_cols = 3
    num_cols = min(num_plots, max_cols)
    num_rows = math.ceil(num_plots / num_cols)

    # Set figure size: 5x5 inches per subplot is a good default
    fig_width = 5 * num_cols
    fig_height = 5 * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    # If only one plot, axes is not an array
    if num_plots == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    default_cmap = 'jet'
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


@dataclass
class Result:
    prefix: str
    __fl_seeds: int | None = field(default=None, init=False, repr=False)
    __non_fl_seeds: int | None = field(default=None, init=False, repr=False)
    __total_seeds: int | None = field(default=None, init=False, repr=False)
    __ratio_fl_total: float | None = field(default=None, init=False, repr=False)
    __chisquare: float | None = field(default=None, init=False, repr=False)
    __pvalue: float | None = field(default=None, init=False, repr=False)
    target_ratio: float = field(default=TARGET_RATIO, init=False, repr=False)
    
    def __repr__(self) -> str:
        return f"Result(prefix={self.prefix}, fl_seeds={self.__fl_seeds}, non_fl_seeds={self.__non_fl_seeds}, total_seeds={self.__total_seeds}, ratio_fl_total={self.__ratio_fl_total}, chisquare={self.__chisquare}, pvalue={self.__pvalue})"
    
    def to_dict(self) -> dict[str, float | int | str | None]:
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
    def fl_seeds(self) -> int | None:
        return self.__fl_seeds
    
    @fl_seeds.setter
    def fl_seeds(self, value: int | None) -> None:
        self.__fl_seeds = value
        self.update_values()

    @property
    def total_seeds(self) -> int | None:
        return self.__total_seeds
    
    @total_seeds.setter
    def total_seeds(self, value: int | None) -> None:
        self.__total_seeds = value
        self.update_values()

    @property
    def non_fl_seeds(self) -> int | None:
        return self.__non_fl_seeds
    
    @property
    def ratio_fl_total(self) -> float | None:
        return self.__ratio_fl_total
    
    @property
    def chisquare(self) -> float | None:
        return self.__chisquare
    
    @property
    def pvalue(self) -> float | None:
        return self.__pvalue
    
    def update_values(self) -> None:
        if self.__total_seeds != None and self.__fl_seeds != None:
            self.__non_fl_seeds = self.__total_seeds - self.__fl_seeds
            self.__ratio_fl_total = self.__fl_seeds / self.__total_seeds
            self.__chisquare, self.__pvalue = compute_chi2(self, self.target_ratio)

        else:
            self.__non_fl_seeds = None
            self.__ratio_fl_total = None
            self.__chisquare = None
            self.__pvalue = None

    def round_all(self, decimals: int = 2) -> None:
        self.__fl_seeds = round_if_not_none(self.__fl_seeds)
        self.__non_fl_seeds = round_if_not_none(self.__non_fl_seeds)
        self.__total_seeds = round_if_not_none(self.__total_seeds)
        self.__ratio_fl_total = round_if_not_none(self.__ratio_fl_total, decimals)
        self.__chisquare = round_if_not_none(self.__chisquare, 4)
        self.__pvalue = round_if_not_none(self.__pvalue, 4)


def compute_chi2(result: Result, expected_ratio: float) -> tuple[float, float]:
    observed = np.array([result.fl_seeds, result.non_fl_seeds])
    total = result.total_seeds
    expected = np.array([expected_ratio, (1 - expected_ratio)]) * total
    chi2_result = chisquare(f_obs=observed, f_exp=expected)
    chi2, p = chi2_result
    return chi2, p


def build_results_csv(results: list[Result]) -> list[list[str | float | int | None]]:
    col_names = ['sample', 'fl_seeds', 'non_fl_seeds', 'total_seeds', 'ratio_fl_total', 'chisquare', 'pvalue']
    rows = [col_names]
    for result in results:
        row = [result.prefix, result.fl_seeds, result.non_fl_seeds, result.total_seeds, result.ratio_fl_total, result.chisquare, result.pvalue]
        rows.append(row)
    return rows


def get_results_rounded(results: list[Result], decimals: int = 2) -> list[Result]:
    new_results = results.copy()
    for result in results:
        result.round_all(decimals)

    return new_results


def round_if_not_none(num: float | int | None, decimals: int = 2) -> float | int | None:
    return round(num, decimals) if num is not None else None


def store_results(
    results_csv: list[list[str | float | int | None]],
    batch_output_dir: str,
    batch_id: str | None = None,
    filename: str | None = None,
) -> str:
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


def parse_filename(filename: str, bf_suffix: str, fl_suffix: str) -> tuple[str, str]:
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
