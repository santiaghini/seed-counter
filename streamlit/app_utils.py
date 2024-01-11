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


def load_files(parsed_filenames, input_dir):
    # filter out non-image files
    parsed_filenames.sort(key=lambda obj: obj['file_name'])

    sample_to_files = {}
    # create directory for batch run
    for obj in parsed_filenames:
        # save the uploaded file in INPUT_DIR/batch_run/file.name
        file = obj['file']
        file_path = os.path.join(input_dir, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

        sample_name = obj['sample_name']
        file_obj = {
                'file_path': file_path,
                'file_name': file.name,
                'img_type': obj['img_type']
            }
        
        if sample_name not in sample_to_files:
            sample_to_files[sample_name] = [file_obj]
        else:
            sample_to_files[sample_name].append(file_obj)

    return sample_to_files


def run_batch(batch_id, run_params, sample_to_files, output_dir):
    bf_suffix = run_params['bf_suffix']
    fl_suffix = run_params['fl_suffix']
    bf_thresh = run_params['bf_intensity_thresh']
    fl_thresh = run_params['fl_intensity_thresh']
    radial_thresh = run_params['radial_thresh']

    yield f'Running batch {batch_id} with params: {run_params}'
    results = None
    for m in process_batch(sample_to_files, bf_thresh, fl_thresh, radial_thresh, output_dir, bf_suffix=bf_suffix, fl_suffix=fl_suffix):
        if type(m) == str:
            yield m
        else:
            results = m

    yield results
