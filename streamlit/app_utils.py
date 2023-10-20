import os
from datetime import datetime
import sys

# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import BRIGHTFIELD, FLUORESCENT
from seeds import process_seed_image
from utils import VALID_EXTENSIONS, build_results_csv, process_batch, store_results

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

def get_batch_id():
    # generate a batch run name based on the current date and time
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_files(uploaded_files, batch_id):
    # filter out non-image files
    files = [f for f in uploaded_files if f.name.split('.')[1].lower() in VALID_EXTENSIONS]
    files.sort()

    prefix_to_filenames = {}
    for file in files:
        # save the uploaded file in INPUT_DIR/batch_run/file.name
        file_path = os.path.join(INPUT_DIR, batch_id, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

        prefix = os.path.basename(file_path).split('_')[0]
        if prefix not in prefix_to_filenames:
            prefix_to_filenames[prefix] = [file_path]
        else:
            prefix_to_filenames[prefix].append(file_path)

    return prefix_to_filenames, len(files)


def run_batch(batch_id, run_params, prefix_to_filenames):
    bf_thresh = run_params['bf_intensity_thresh']
    fl_thresh = run_params['fl_intensity_thresh']
    radial_thresh = run_params['radial_thresh']

    batch_output_dir = os.path.join(OUTPUT_DIR, batch_id)

    results = None
    for m in process_batch(prefix_to_filenames, bf_thresh, fl_thresh, radial_thresh, batch_output_dir):
        if type(m) == str:
            yield m
        else:
            results = m

    return results, batch_output_dir
