from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
from PIL import Image

# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app_utils import (
    AppRunParams,
    create_folders,
    dict_to_results_list,
    get_batch_id,
    load_files,
    results_list_to_dict,
    run_batch,
)
from config import (
    DEFAULT_BRIGHTFIELD_SUFFIX,
    DEFAULT_FLUORESCENT_SUFFIX,
    INITIAL_BRIGHTNESS_THRESHOLDS,
)
from constants import INSTRUCTIONS_TEXT, PARAM_HINTS
from utils import (
    CountMethod,
    Result,
    store_results,
    parse_filename,
    round_if_not_none,
)

st.set_page_config(
    page_title="SeedSeg by the Brophy Lab",
    menu_items={
        "About": "Run by the Brophy Lab",
        "Report a Bug": "mailto:jbrophy@stanford.edu",
        "Get help": None,
    },
    page_icon=":seedling:",
    # layout='wide'
)


##########################           GLOBALS           ##########################
RADIAL_THRESH_DEFAULT = 0.4
RADIAL_THRESH_RATIO = (0, 1)
LARGE_AREA_FACTOR_DEFAULT = 20.0

BATCH_ID = None
RUN_PARAMS = AppRunParams(
    mode=CountMethod.FLUORESCENCE,
    bf_suffix=None,
    fl_suffix=None,
    bf_intensity_thresh=None,
    fl_intensity_thresh=None,
    radial_threshold_ratio=RADIAL_THRESH_DEFAULT,
    large_area_factor=None,
)
PREFIX_TO_FILENAMES = None

##########################           STATE           ##########################
LOBBY = "lobby"
LOADING = "loading"
RESULTS = "results"
STEPS = [LOBBY, LOADING, RESULTS]
if "curr_step" not in st.session_state:
    st.session_state.curr_step = LOBBY

if "curr_batch" not in st.session_state:
    st.session_state.curr_batch = None
    # Expect dict with keys: batch_id, batch_output_dir

if "run_results" not in st.session_state:
    st.session_state.run_results = {
        "results": None,
        "output_dir": None,
    }

if "results" not in st.session_state:
    st.session_state.results = None

if "results_csv" not in st.session_state:
    st.session_state.results_csv = True

if "logging" not in st.session_state:
    st.session_state.logging = True

if "clicked_run" not in st.session_state:
    st.session_state.clicked_run = False

if "logs_content" not in st.session_state:
    st.session_state.logs_content = ""

if "expanded_params" not in st.session_state:
    st.session_state.expanded_params = False

if "has_clicked_once" not in st.session_state:
    st.session_state.has_clicked_once = False


##########################           LOGIC           ##########################


def clear() -> None:
    st.session_state.logs_content = ""
    st.session_state.run_results = {
        "results": None,
        "results_csv": None,
        "results_csv_path": None,
        "output_dir": None,
    }


def click_reset_button() -> None:
    st.session_state.clicked_run = False
    clear()


def click_run_button() -> None:
    success = run_for_batch(RUN_PARAMS, uploaded_files)
    if success and not st.session_state.clicked_run:
        st.session_state.clicked_run = True

    if success and not st.session_state.has_clicked_once:
        st.session_state.has_clicked_once = True


@st.cache_data
def run_for_batch(run_params: AppRunParams, files_uploaded: List[UploadedFile]) -> int:
    """Execute a batch run with the given parameters and uploaded files.

    Returns ``1`` on success and ``0`` if any error occurred while parsing
    filenames.
    """
    parsed_filenames = []
    for f in files_uploaded:
        try:
            if run_params.mode == CountMethod.FLUORESCENCE:
                sample_name, img_type = parse_filename(
                    f.name, run_params.bf_suffix, run_params.fl_suffix
                )
                parsed_filenames.append(
                    {
                        "file_name": f.name,
                        "sample_name": sample_name,
                        "img_type": img_type,
                        "file": f,
                    }
                )
            else:
                sample_name = f.name.split(".")[0]
                parsed_filenames.append(
                    {
                        "file_name": f.name,
                        "sample_name": sample_name,
                        "img_type": None,
                        "file": f,
                    }
                )

        except Exception as e:
            st.error(e)
            return 0

    clear()

    BATCH_ID = get_batch_id()
    batch_dir, input_dir, output_dir = create_folders(BATCH_ID)

    sample_to_files = load_files(parsed_filenames, input_dir)
    results: List[Result] = None

    print(f"sample_to_filenames: {sample_to_files}")

    print("running batch...")
    for m in run_batch(
        batch_id=BATCH_ID,
        run_params=run_params,
        sample_to_files=sample_to_files,
        output_dir=output_dir,
    ):
        if type(m) == str:
            print(m)
            st.session_state.logs_content += m + "\n"
        else:
            results = m

    st.session_state.logs_content += "Done!\n"

    print("Results", results)

    st.session_state.run_results = {
        "results": results,
        "output_dir": output_dir,
    }

    return 1


@st.cache_data
def get_output_imgs(output_dir: str) -> List[str]:
    return [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f)) and f.endswith(".png")
    ]


@st.cache_data
def build_prefix_to_output_imgs(
    output_imgs: List[str],
    mode: CountMethod,
) -> Dict[str, Dict[str, str | None]]:
    prefix_to_output_imgs = {}
    for img in output_imgs:
        if mode == CountMethod.FLUORESCENCE:
            prefix = os.path.basename(img).split("_")[0]
        else:
            prefix = None
            for suffix in (DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX):
                target = f"_{suffix}_"
                basename = os.path.basename(img)
                if target in basename:
                    idx = basename.find(target)
                    prefix = basename[:idx]
                    print(f"prefix: {prefix}")
                    break
            # If no suffix is found, use the base name of the image
            if not prefix:
                prefix = os.path.basename(img)

        if prefix not in prefix_to_output_imgs:
            prefix_to_output_imgs[prefix] = {
                DEFAULT_BRIGHTFIELD_SUFFIX: None,
                DEFAULT_FLUORESCENT_SUFFIX: None,
            }
        if DEFAULT_BRIGHTFIELD_SUFFIX in img:
            prefix_to_output_imgs[prefix][DEFAULT_BRIGHTFIELD_SUFFIX] = img
        else:
            prefix_to_output_imgs[prefix][DEFAULT_FLUORESCENT_SUFFIX] = img

    print(f"prefix_to_output_imgs: {prefix_to_output_imgs}")

    return prefix_to_output_imgs


def build_caption(result: Result, img_type: str, mode: CountMethod) -> str:
    """Return a multi-line caption describing the output image."""

    if mode == CountMethod.FLUORESCENCE:
        if img_type == DEFAULT_BRIGHTFIELD_SUFFIX:
            first_line = "Brightfield Seeds"
            brightness = result.bf_thresh
        else:
            first_line = "Fluorescent Seeds"
            brightness = result.marker_thresh
    else:  # COLORIMETRIC
        if img_type == DEFAULT_BRIGHTFIELD_SUFFIX:
            first_line = "All Seeds"
            brightness = result.bf_thresh
        else:
            first_line = "Marker Seeds"
            brightness = result.marker_thresh

    radial_thresh = round_if_not_none(result.radial_threshold, 2)
    radial_ratio = round_if_not_none(result.radial_threshold_ratio, 2)
    brightness = round_if_not_none(brightness, 2)

    caption_lines = [
        first_line,
        f"Sample: {result.prefix}",
        f"Brightness Threshold: {brightness}",
        f"Radial Threshold: {radial_thresh}",
        f"Radial Threshold Ratio: {radial_ratio}",
    ]
    return " | ".join(caption_lines)


##########################           UI           ##########################

st.title(":seedling: SeedSeg")
st.subheader("The Brophy Lab's automated seed counting app")

st.markdown("[Link to GitHub Repo](https://github.com/santiaghini/seedseg)")

on = st.toggle
if on:
    st.session_state.logging = True
else:
    st.session_state.logging = False

############# STEP: LOBBY #############

with st.expander("**Instructions** (click to expand)"):
    st.subheader("Instructions")
    st.markdown(INSTRUCTIONS_TEXT)

st.subheader("Upload your images")

mode_option = st.radio(
    "Select counting mode",
    ["Fluorescence", "Colorimetric"],
    horizontal=True,
    help="Fluorescence: paired fluorescent/brightfield images. Colorimetric: single RGB images",
)
RUN_PARAMS.mode = (
    CountMethod.FLUORESCENCE
    if mode_option == "Fluorescence"
    else CountMethod.COLORIMETRIC
)

if RUN_PARAMS.mode == CountMethod.FLUORESCENCE:
    uploader_text = "Upload files with the format <sample_name>_<img_type>.tif"
else:
    uploader_text = "Upload RGB images. File name is used as sample name"

if st.session_state.clicked_run:
    st.button("Reset", on_click=click_reset_button)

uploaded_files = st.file_uploader(
    uploader_text,
    accept_multiple_files=True,
    help="Load the images for this run. You can drag multiple files at once.",
)

st.markdown(":gray[*Pro Tip: To clear all uploaded files, reload the page.*]")

### Parameter box

intensity_naming = {
    CountMethod.FLUORESCENCE: {
        "bf": "Brightfield Intensity Threshold",
        "fl": "Fluorescence Intensity Threshold",
    },
    CountMethod.COLORIMETRIC: {
        "bf": "All Seeds Intensity Threshold",
        "fl": "Marker Intensity Threshold",
    },
}

with st.expander("**Parameters for manual setup**"):
    st.subheader("Parameters")

    th_min, th_max = float(RADIAL_THRESH_RATIO[0]), float(RADIAL_THRESH_RATIO[1])

    suff_col1, suff_col2 = st.columns(2)
    with suff_col1:
        RUN_PARAMS.bf_suffix = None
        if RUN_PARAMS.mode == CountMethod.FLUORESCENCE:
            enable_bf_suffix = st.checkbox(
                "Manually set Brightfield suffix",
                value=True,
                disabled=RUN_PARAMS.mode == CountMethod.COLORIMETRIC,
                help="Check to override the default brightfield filename suffix",
            )
            RUN_PARAMS.bf_suffix = st.text_input(
                "Brightfield suffix",
                value=DEFAULT_BRIGHTFIELD_SUFFIX,
                disabled=(
                    not enable_bf_suffix or RUN_PARAMS.mode != CountMethod.FLUORESCENCE
                ),
                help="Suffix that identifies brightfield images (e.g. 'BF')",
            )
            if not enable_bf_suffix:
                RUN_PARAMS.bf_suffix = None

        enable_bf_thresh = st.checkbox(
            f"Manually set {intensity_naming[RUN_PARAMS.mode]['bf']}",
            value=False,
            help="Check to override the automatic thresholding",
        )
        RUN_PARAMS.bf_intensity_thresh = st.slider(
            intensity_naming[RUN_PARAMS.mode]["bf"],
            0,
            255,
            INITIAL_BRIGHTNESS_THRESHOLDS[DEFAULT_BRIGHTFIELD_SUFFIX],
            disabled=not enable_bf_thresh,
            help="Lower to capture dim seeds, raise to reduce background",
        )
        if not enable_bf_thresh:
            RUN_PARAMS.bf_intensity_thresh = None

        enable_radial_thresh = st.checkbox(
            "Manually set Radial Threshold Ratio",
            value=False,
            help=f"Check to override the default value of {RADIAL_THRESH_DEFAULT}",
        )
        RUN_PARAMS.radial_threshold_ratio = st.slider(
            "Radial Threshold Ratio",
            th_min,
            th_max,
            float(RADIAL_THRESH_DEFAULT),
            step=0.01,
            disabled=not enable_radial_thresh,
            help="Fraction of the median seed radius which is removed for splitting seeds",
        )
        if not enable_radial_thresh:
            RUN_PARAMS.radial_threshold_ratio = RADIAL_THRESH_DEFAULT

    with suff_col2:
        RUN_PARAMS.fl_suffix = None
        RUN_PARAMS.fl_intensity_thresh = None
        if RUN_PARAMS.mode == CountMethod.FLUORESCENCE:
            enable_fl_suffix = st.checkbox(
                "Manually set Fluorescent suffix",
                value=True,
                disabled=RUN_PARAMS.mode == CountMethod.COLORIMETRIC,
                help="Check to override the default fluorescent filename suffix",
            )
            RUN_PARAMS.fl_suffix = st.text_input(
                "Fluorescent suffix",
                value=DEFAULT_FLUORESCENT_SUFFIX,
                disabled=(
                    not enable_fl_suffix or RUN_PARAMS.mode != CountMethod.FLUORESCENCE
                ),
                help="Suffix that identifies fluorescent images (e.g. 'FL')",
            )
            if not enable_fl_suffix:
                RUN_PARAMS.fl_suffix = None

            enable_fl_thresh = st.checkbox(
                f"Manually set {intensity_naming[RUN_PARAMS.mode]['fl']}",
                value=False,
                help="Check to override automatic thresholding for the fluorescent image",
            )
            RUN_PARAMS.fl_intensity_thresh = st.slider(
                intensity_naming[RUN_PARAMS.mode]["fl"],
                0,
                255,
                INITIAL_BRIGHTNESS_THRESHOLDS[DEFAULT_FLUORESCENT_SUFFIX],
                disabled=not enable_fl_thresh,
                help="Lower to capture dim seeds, raise to reduce background",
            )
            if not enable_fl_thresh:
                RUN_PARAMS.fl_intensity_thresh = None

    st.divider()
    if st.checkbox("Show me tips on how to tune these parameters 🔍"):
        st.markdown(PARAM_HINTS)


### RUN BUTTON
st.button("Run SeedSeg", disabled=not uploaded_files, on_click=click_run_button)
if st.session_state.clicked_run:
    print(f"uploaded_files: {uploaded_files}")

    run_results = st.session_state.run_results

    st.subheader("Results")

    with st.expander("__Logs__"):
        for line in st.session_state.logs_content.split("\n"):
            st.write(line)

    if run_results["results"]:
        results: List[Result] = run_results["results"]
        results_csv_path = store_results(results, run_results["output_dir"], BATCH_ID)

        results_dict = [result.to_dict() for result in results]
        # build pandas df from results_dict
        df = pd.DataFrame(results_dict)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download results as CSV",
            data=open(results_csv_path, "rb"),
            file_name=f"seedseg_{BATCH_ID}.csv",
            mime="text/csv",
        )

        st.subheader("Output Images with Seeds Highlighted")

        # results list to dict
        results_dict: Dict[str, Result] = results_list_to_dict(run_results["results"])

        # get all png files from output_dir
        output_imgs = get_output_imgs(run_results["output_dir"])
        # group images by prefix
        prefix_to_output_imgs = build_prefix_to_output_imgs(
            output_imgs, RUN_PARAMS.mode
        )
        print(f"prefix_to_output_imgs: {prefix_to_output_imgs}")

        col1, col2 = st.columns([1, 3])
        prefixes = sorted(list(prefix_to_output_imgs.keys()))
        prefix = prefixes[0]
        with col1:
            prefix = st.radio(
                "Select a sample to display the results",
                list(prefix_to_output_imgs.keys()),
            )

        with col2:
            readable_type_map = {
                DEFAULT_BRIGHTFIELD_SUFFIX: "Brightfield",
                DEFAULT_FLUORESCENT_SUFFIX: "Fluorescent",
            }
            suffix_to_key = {
                DEFAULT_BRIGHTFIELD_SUFFIX: "total_seeds",
                DEFAULT_FLUORESCENT_SUFFIX: "marker_seeds",
            }
            for img_type in [
                DEFAULT_BRIGHTFIELD_SUFFIX,
                DEFAULT_FLUORESCENT_SUFFIX,
            ]:  # one for brightfield, one for fluorescent
                try:
                    image_path = prefix_to_output_imgs[prefix][img_type]
                except KeyError:
                    st.write(
                        f"Missing image for {readable_type_map.get(img_type, img_type)} ({img_type}) for sample '{prefix}'."
                    )
                    image_path = None

                if image_path:
                    image = Image.open(image_path)
                    caption = build_caption(
                        results_dict[prefix], img_type, RUN_PARAMS.mode
                    )
                    st.image(image, caption=caption, width=500)

                value = results_dict[prefix].__getattribute__(suffix_to_key[img_type])

                old_value = suffix_to_key[img_type]

                new_value = st.number_input(
                    f"Manually override {suffix_to_key[img_type]} value:",
                    value=value,
                    placeholder="Type a number",
                    step=1,
                    key=image_path,
                )

                if new_value != value:
                    print(
                        f"Updated value for {prefix} - {suffix_to_key[img_type]}: {new_value}"
                    )
                    results_dict[prefix].__setattr__(suffix_to_key[img_type], new_value)
                    new_results = dict_to_results_list(results_dict)
                    st.session_state.run_results = {
                        "results": new_results,
                        "output_dir": run_results["output_dir"],
                    }
                    st.rerun()
