import os
from datetime import datetime
import sys

# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import VALID_EXTENSIONS
from run import process_batch

BATCHES_DIR = 'batches'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

def get_batch_id():
    # generate a batch run name based on the current date and time
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_folders(batch_id):
    batch_dir = os.path.join(BATCHES_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    input_dir = os.path.join(batch_dir, INPUT_DIR)
    os.makedirs(input_dir, exist_ok=True)

    output_dir = os.path.join(batch_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"batch_dir: {batch_dir}")
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")

    return batch_dir, input_dir, output_dir


def load_files(uploaded_files, input_dir):
    # filter out non-image files
    files = [f for f in uploaded_files if '.' + f.name.split('.')[1].lower() in VALID_EXTENSIONS]
    files.sort(key=lambda f: f.name)

    prefix_to_filenames = {}
    # create directory for batch run
    for file in files:
        # save the uploaded file in INPUT_DIR/batch_run/file.name
        file_path = os.path.join(input_dir, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

        prefix = os.path.basename(file_path).split('_')[0]
        if prefix not in prefix_to_filenames:
            prefix_to_filenames[prefix] = [file_path]
        else:
            prefix_to_filenames[prefix].append(file_path)

    return prefix_to_filenames, len(files)


def run_batch(batch_id, run_params, prefix_to_filenames, output_dir):
    bf_thresh = run_params['bf_intensity_thresh']
    fl_thresh = run_params['fl_intensity_thresh']
    radial_thresh = run_params['radial_thresh']

    yield f'Running batch {batch_id} with params: {run_params}'
    results = None
    for m in process_batch(prefix_to_filenames, bf_thresh, fl_thresh, radial_thresh, output_dir):
        if type(m) == str:
            yield m
        else:
            results = m

    yield results
