import os
import sys
from typing import Dict, List

import streamlit as st
import pandas as pd
from PIL import Image


# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app_utils import get_batch_id, load_files, run_batch, create_folders, results_list_to_dict, dict_to_results_list
from config import DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX, INITIAL_BRIGHTNESS_THRESHOLDS
from utils import build_results_csv, store_results, parse_filename, get_results_rounded, Result
from constants import INSTRUCTIONS_TEXT, PARAM_HINTS

st.set_page_config(
    page_title='SeedCounter by the Brophy Lab', menu_items={"About": "Run by the Brophy Lab", "Report a Bug": "mailto:jbrophy@stanford.edu", "Get help": None}
    page_icon=':seedling:',
    # layout='wide'
)

##########################           GLOBALS           ##########################
RADIAL_THRESH_DEFAULT = 12
RADIAL_THRESH_RANGE = (6, 20)

BATCH_ID = None
RUN_PARAMS = {
    'bf_suffix': None,
    'fl_suffix': None,
    'bf_intensity_thresh': None,
    'fl_intensity_thresh': None,
    'radial_thresh': None
}
PREFIX_TO_FILENAMES = None

##########################           STATE           ##########################
LOBBY = 'lobby'
LOADING = 'loading'
RESULTS = 'results'
STEPS = [LOBBY, LOADING, RESULTS]
if 'curr_step' not in st.session_state:
    st.session_state.curr_step = LOBBY

if 'curr_batch' not in st.session_state:
    st.session_state.curr_batch = None
    # Expect dict with keys: batch_id, batch_output_dir

if 'run_results' not in st.session_state:
    st.session_state.run_results = {
        'results': None,
        'output_dir': None,
    }

if 'results' not in st.session_state:
    st.session_state.results = None

if 'results_csv' not in st.session_state:
    st.session_state.results_csv = True

if 'logging' not in st.session_state:
    st.session_state.logging = True

if 'clicked_run' not in st.session_state:
    st.session_state.clicked_run = False

if 'logs_content' not in st.session_state:
    st.session_state.logs_content = ""

if 'expanded_params' not in st.session_state:
    st.session_state.expanded_params = False

if 'has_clicked_once' not in st.session_state:
    st.session_state.has_clicked_once = False


##########################           LOGIC           ##########################

def clear():
    st.session_state.logs_content = ""
    st.session_state.run_results = {
        'results': None,
        'results_csv': None,
        'results_csv_path': None,
        'output_dir': None,
    }

def click_reset_button():
    st.session_state.clicked_run = False
    clear()

def click_run_button():
    success = run_for_batch(RUN_PARAMS, uploaded_files)
    if success and not st.session_state.clicked_run:
        st.session_state.clicked_run = True

    if success and not st.session_state.has_clicked_once:
        st.session_state.has_clicked_once = True

@st.cache_data
def run_for_batch(run_params, files_uploaded):
    parsed_filenames = []
    for f in files_uploaded:
        try:
            sample_name, img_type = parse_filename(f.name, run_params['bf_suffix'], run_params['fl_suffix'])
            parsed_filenames.append({
                    'file_name': f.name,
                    'sample_name': sample_name,
                    'img_type': img_type,
                    'file': f
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
    for m in run_batch(BATCH_ID, run_params, sample_to_files, output_dir):
        if type(m) == str:
            print(m)
            st.session_state.logs_content += m + '\n'
        else:
            results = m

    st.session_state.logs_content += "Done!\n"

    print("Results", results)

    st.session_state.run_results = {
        'results': results,
        'output_dir': output_dir,
    }

    return 1

@st.cache_data
def get_output_imgs(output_dir):
    return [
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f)) and f.endswith('.png')
    ]

@st.cache_data
def build_prefix_to_output_imgs(output_imgs):
    prefix_to_output_imgs = {}
    for img in output_imgs:
        prefix = os.path.basename(img).split('_')[0]
        if prefix not in prefix_to_output_imgs:
            prefix_to_output_imgs[prefix] = {
                DEFAULT_BRIGHTFIELD_SUFFIX: None,
                DEFAULT_FLUORESCENT_SUFFIX: None
            }
        if DEFAULT_BRIGHTFIELD_SUFFIX in img:
            prefix_to_output_imgs[prefix][DEFAULT_BRIGHTFIELD_SUFFIX] = img
        else:
            prefix_to_output_imgs[prefix][DEFAULT_FLUORESCENT_SUFFIX] = img

    return prefix_to_output_imgs


##########################           UI           ##########################

st.title(":seedling: Seed Counter by Brophy Lab")

st.markdown("[Link to GitHub Repo](https://github.com/santiaghini/seed-counter)")

on = st.toggle
if on:
    st.session_state.logging = True
else:
    st.session_state.logging = False

############# STEP: LOBBY #############

with st.expander("**Instructions** (click to expand)"):
    st.subheader("Instructions")
    st.markdown(INSTRUCTIONS_TEXT)

st.header("Upload your images")

if st.session_state.clicked_run:
    st.button("Reset", on_click=click_reset_button)

uploaded_files = st.file_uploader("Upload files with the format <sample_name>_<img_type>.tif", accept_multiple_files=True)

st.markdown(":gray[*Pro Tip: To clear all uploaded files, reload the page.*]")

### Parameter box
with st.expander("**Parameters for manual setup**"):
    # if not st.session_state.expanded_params:
    #     st.session_state.expanded_params = True

    st.subheader("Parameters")

    th_min, th_max = RADIAL_THRESH_RANGE

    suff_col1, suff_col2 = st.columns(2)
    with suff_col1:
        RUN_PARAMS['bf_suffix'] = st.text_input('Brightfield suffix', value=DEFAULT_BRIGHTFIELD_SUFFIX)
        RUN_PARAMS['bf_intensity_thresh'] = st.slider(
            'Brightfield Intensity Threshold', 
            0, 
            255, 
            INITIAL_BRIGHTNESS_THRESHOLDS[DEFAULT_BRIGHTFIELD_SUFFIX]
        )
        RUN_PARAMS['radial_thresh'] = st.slider(
            'Radial Threshold', 
            th_min, 
            th_max, 
            RADIAL_THRESH_DEFAULT
        )
        
    with suff_col2:
        RUN_PARAMS['fl_suffix'] = st.text_input('Fluorescent suffix', value=DEFAULT_FLUORESCENT_SUFFIX)
        RUN_PARAMS['fl_intensity_thresh'] = st.slider(
            'Fluorescent Intensity Threshold', 
            0, 
            255, 
            INITIAL_BRIGHTNESS_THRESHOLDS[DEFAULT_FLUORESCENT_SUFFIX]
        )
    

    if st.checkbox("Show me tips on how to tune these parameters 🔍"):
        st.markdown(PARAM_HINTS)

### RUN BUTTON
st.button("Run Seed Counter", disabled=not uploaded_files, on_click=click_run_button)
if st.session_state.clicked_run:
    print(f"uploaded_files: {uploaded_files}")

    run_results = st.session_state.run_results

    # with st.spinner('Processing images...'):
    #     while run_results['results'] is None:
    #         pass

    st.header("Results")

    with st.expander("__Logs__"):
        for line in st.session_state.logs_content.split('\n'):
            st.write(line)


    if run_results['results']:

        results_rounded = get_results_rounded(run_results['results'], 2)
        results_csv = build_results_csv(results_rounded)
        results_csv_path = store_results(results_csv, run_results['output_dir'], BATCH_ID)

        print(f"results_csv: {results_csv}")

        results_dict = [result.to_dict() for result in results_rounded]
        # build pandas df from results_dict
        df = pd.DataFrame(results_dict)

        st.table(df)

        st.download_button(
            label="Download results as CSV",
            data=open(results_csv_path, 'rb'),
            file_name=f'seed_counter_{BATCH_ID}.csv',
            mime='text/csv',
        )

        st.header("Output Images with Seeds Highlighted")

        # results list to dict
        results_dict: Dict[str, Result] = results_list_to_dict(run_results['results'])

        # get all png files from output_dir
        output_imgs = get_output_imgs(run_results['output_dir'])
        # group images by prefix
        prefix_to_output_imgs = build_prefix_to_output_imgs(output_imgs)

        col1, col2 = st.columns([1, 3])
        prefixes = sorted(list(prefix_to_output_imgs.keys()))
        prefix = prefixes[0]
        with col1:
            prefix = st.radio(
                "Select a sample to display the results",
                list(prefix_to_output_imgs.keys())
            )

        with col2:
            readable_type_map = {
                DEFAULT_BRIGHTFIELD_SUFFIX: "Brightfield",
                DEFAULT_FLUORESCENT_SUFFIX: "Fluorescent"
            }
            suffix_to_key = {
                DEFAULT_BRIGHTFIELD_SUFFIX: "total_seeds",
                DEFAULT_FLUORESCENT_SUFFIX: "fl_seeds"
            }
            for img_type in [DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX]: # one for brightfield, one for fluorescent
                image_path = prefix_to_output_imgs[prefix][img_type]
                if image_path:
                    image = Image.open(image_path)
                    caption = f"{readable_type_map[img_type]} - {image_path}"
                    st.image(image, caption=caption, width=500)

                value = results_dict[prefix].__getattribute__(suffix_to_key[img_type])

                old_value = suffix_to_key[img_type]

                new_value = st.number_input(f"Manually override {suffix_to_key[img_type]} value:", value=value, placeholder="Type a number", step=1, key=image_path)

                if new_value != value:
                    print(f"Updated value for {prefix} - {suffix_to_key[img_type]}: {new_value}")
                    results_dict[prefix].__setattr__(suffix_to_key[img_type], new_value)
                    new_results = dict_to_results_list(results_dict)
                    st.session_state.run_results = {
                        'results': new_results,
                        'output_dir': run_results['output_dir'],
                    }
                    st.rerun()
