'''
- [ ] Validate filenames to be sample name + suffix + extension (e.g. img1_bf.tif)
'''

import streamlit as st
from PIL import Image

import os
import sys

# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app_utils import get_batch_id, load_files, run_batch, create_folders
import seeds
from config import DEFAULT_BRIGHTFIELD_SUFFIX, DEFAULT_FLUORESCENT_SUFFIX, INITIAL_BRIGHTNESS_THRESHOLDS
from utils import build_results_csv, store_results, parse_filename, get_results_rounded
from constants import INSTRUCTIONS_TEXT, PARAM_HINTS

st.set_page_config(
    page_title='Seed Counter by Brophy Lab',
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
        'results_csv': None,
        'results_csv_path': None,
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
                    'filename': f.name,
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

    sample_to_filenames = load_files(parsed_filenames, input_dir)
    results = None

    print(f"sample_to_filenames: {sample_to_filenames}")

    print("running batch...")
    for m in run_batch(BATCH_ID, run_params, sample_to_filenames, output_dir):
        if type(m) == str:
            print(m)
            st.session_state.logs_content += m + '\n'
        else:
            results = m

    st.session_state.logs_content += "Done!\n"

    results_rounded = get_results_rounded(results, 2)
    results_csv = build_results_csv(results_rounded)
    results_csv_path = store_results(results_csv, output_dir, BATCH_ID)

    print(f"results_csv: {results_csv}")

    st.session_state.run_results = {
        'results': results,
        'results_csv': results_csv,
        'results_csv_path': results_csv_path,
        'output_dir': output_dir,
    }
    # if not st.session_state.has_clicked_once:
    #     st.balloons()
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
            prefix_to_output_imgs[prefix] = [img]
        else:
            prefix_to_output_imgs[prefix].append(img)
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

        st.table(run_results['results_csv'])

        st.download_button(
            label="Download results as CSV",
            data=open(run_results['results_csv_path'], 'rb'),
            file_name=f'seed_counter_{BATCH_ID}.csv',
            mime='text/csv',
        )

        st.header("Output Images with Seeds Highlighted")

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
            image_paths = sorted(prefix_to_output_imgs[prefix])

            for image_path in image_paths:
                image = Image.open(image_path)
                caption = f"{'Fluorescent' if 'FL' in image_path else 'Brightfield'} - {image_path}"
                st.image(image, caption=caption, width=500)


############# STEP: LOADING #############



############# STEP: RESULTS #############
