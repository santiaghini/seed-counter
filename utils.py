from datetime import datetime
import os

import matplotlib.pyplot as plt

from config import BRIGHTFIELD, FLUORESCENT

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


def build_results_csv(results):
    col_names = ['prefix', 'fl_seeds', 'non_fl_seeds', 'total_seeds', 'ratio_fl_total']
    rows = [col_names]
    for result in results:
        row = [result.prefix, result.fl_seeds, result.non_fl_seeds, result.total_seeds, result.ratio_fl_total]
        rows.append(row)
    return rows


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