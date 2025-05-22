from __future__ import annotations

import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from run import process_fluorescent_batch, process_colorimetric_batch
from utils import CountMethod, Result

BATCHES_DIR = "batches"
INPUT_DIR = "input"
OUTPUT_DIR = "output"


@dataclass
class AppRunParams:
    mode: CountMethod
    bf_suffix: str | None
    fl_suffix: str | None
    bf_intensity_thresh: int | None
    fl_intensity_thresh: int | None
    radial_threshold_ratio: float | None
    large_area_factor: float | None


def get_batch_id() -> str:
    """Return a unique identifier for the current batch run."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_folders(batch_id: str) -> tuple[str, str, str]:
    """Create directories for a batch and return their paths."""
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


def results_list_to_dict(results: List[Result]) -> Dict[str, Result]:
    """Convert a list of :class:`Result` objects into a dictionary keyed by prefix."""
    results_dict = {}
    for result in results:
        results_dict[result.prefix] = result
    return results_dict


def dict_to_results_list(results_dict: Dict[str, Result]) -> List[Result]:
    """Convert a dictionary of results back into a list."""
    results_list = []
    for prefix, result in results_dict.items():
        results_list.append(result)
    return results_list


def load_files(
    parsed_filenames: List[Dict[str, Any]], input_dir: str
) -> Dict[str, List[Dict[str, str]]]:
    """Save uploaded files to ``input_dir`` and build a mapping for processing."""
    # filter out non-image files
    parsed_filenames.sort(key=lambda obj: obj["file_name"])

    sample_to_files = {}
    # create directory for batch run
    for obj in parsed_filenames:
        # save the uploaded file in INPUT_DIR/batch_run/file.name
        file = obj["file"]
        file_path = os.path.join(input_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        sample_name = obj["sample_name"]
        file_obj = {
            "file_path": file_path,
            "file_name": file.name,
            "img_type": obj["img_type"],
        }

        if sample_name not in sample_to_files:
            sample_to_files[sample_name] = []
        sample_to_files[sample_name].append(file_obj)

    return sample_to_files


def run_batch(
    batch_id: str,
    run_params: AppRunParams,
    sample_to_files: Dict[str, List[Dict[str, str]]],
    output_dir: str,
) -> Iterable[str | List[Result]]:
    """Run SeedCounter for a prepared batch of images.

    Parameters
    ----------
    batch_id:
        Identifier for this batch run used when saving outputs.
    run_params:
        Configuration parameters collected from the UI.
    sample_to_files:
        Mapping of sample names to lists of file descriptor dictionaries.
    output_dir:
        Directory where result images and CSV files will be stored.

    Yields
    ------
    str or List[Result]
        Progress messages followed by the final list of :class:`Result` objects.
    """
    bf_suffix = run_params.bf_suffix
    fl_suffix = run_params.fl_suffix
    bf_thresh = run_params.bf_intensity_thresh
    fl_thresh = run_params.fl_intensity_thresh
    radial_threshold_ratio = run_params.radial_threshold_ratio
    large_area_factor = run_params.large_area_factor

    print(f"{run_params=}")

    yield f"Running batch {batch_id} with params: {run_params}"
    results = None
    if run_params.mode == CountMethod.FLUORESCENCE:
        iterator = process_fluorescent_batch(
            sample_to_files=sample_to_files,
            bf_thresh=bf_thresh,
            fl_thresh=fl_thresh,
            radial_thresh=None,
            batch_output_dir=output_dir,
            bf_suffix=bf_suffix,
            fl_suffix=fl_suffix,
            radial_threshold_ratio=radial_threshold_ratio,
            large_area_factor=large_area_factor,
        )
    else:
        iterator = process_colorimetric_batch(
            sample_to_file=sample_to_files,
            bf_thresh=bf_thresh,
            fl_thresh=fl_thresh,
            radial_thresh=None,
            batch_output_dir=output_dir,
            radial_threshold_ratio=radial_threshold_ratio,
            large_area_factor=large_area_factor,
        )

    for m in iterator:
        if isinstance(m, str):
            yield m
        else:
            results = m

    yield results
