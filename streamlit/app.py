import streamlit as st

import os
import sys

# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app_utils import get_batch_id, load_files, run_batch
import seeds
from config import BRIGHTFIELD, FLUORESCENT, INITIAL_BRIGHTNESS_THRESHOLDS
from utils import build_results_csv, store_results

st.set_page_config(
    page_title='Seed Counter',
    page_icon=':seedling:',
    layout='wide'
)

############# GLOBALS
RADIAL_THRESH_DEFAULT = 12

BATCH_ID = None
RUN_PARAMS = {
    'bf_intensity_thresh': None,
    'fl_intensity_thresh': None,
    'radial_thresh': None
}
PREFIX_TO_FILENAMES = None

############# STATE
LOBBY = 'lobby'
LOADING = 'loading'
RESULTS = 'results'
STEPS = [LOBBY, LOADING, RESULTS]
if 'curr_step' not in st.session_state:
    st.session_state['curr_step'] = LOBBY

if 'curr_batch' not in st.session_state:
    st.session_state['curr_batch'] = None
    # Expect dict with keys: batch_id, batch_output_dir

if 'results' not in st.session_state:
    st.session_state['results'] = None

if 'results_csv' not in st.session_state:
    st.session_state['results_csv'] = True

if 'logging' not in st.session_state:
    st.session_state['logging'] = True

############# UI

st.title(":seedling: Seed Counter")

on = st.toggle
if on:
    st.session_state['logging'] = True
else:
    st.session_state['logging'] = False

############# STEP: LOBBY

if st.session_state['curr_step'] == LOBBY:
    st.header("Instructions")
    st.markdown("This are some instructions on how to use the app.")

    st.header("Upload a file")

    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"filename {i+1}: {uploaded_file.name}")

    ### Parameter box
    manual_setup = st.checkbox("Manual Setup", value=False)
    if manual_setup:
        st.header("Parameters")
        st.markdown("Here you can set the parameters for the seed counting.")
        RUN_PARAMS['bf_intensity_thresh'] = st.slider('Brightfield Intensity Threshold', 0, 255, INITIAL_BRIGHTNESS_THRESHOLDS[BRIGHTFIELD])
        RUN_PARAMS['fl_intensity_thresh'] = st.slider('Fluorescent Intensity Threshold', 0, 255, INITIAL_BRIGHTNESS_THRESHOLDS[FLUORESCENT])
        RUN_PARAMS['radial_thresh'] = st.slider('Radial Threshold', 8, 18, RADIAL_THRESH_DEFAULT)

    ### RUN BUTTON
    if st.button("Run Seed Counter", disabled=not uploaded_files):
        st.session_state['curr_step'] = LOADING

        BATCH_ID = get_batch_id()
        prefix_to_filenames, nfiles = load_files(uploaded_files, BATCH_ID)
        results = None
        batch_output_dir = None
        for m in run_batch(BATCH_ID, RUN_PARAMS, prefix_to_filenames):
            if type(m) == str:
                st.write(m)
            else:
                results, batch_output_dir = m

        st.write("Done!")
        st.session_state['results'] = results

        results_csv = build_results_csv(results)
        store_results(results_csv, batch_output_dir, BATCH_ID)

        st.session_state['curr_step'] = RESULTS
        st.session_state['curr_batch'] = {'batch_id': BATCH_ID, 'batch_output_dir': batch_output_dir}


############# STEP: LOADING

elif st.session_state['curr_step'] == LOADING:
    st.write("Running Seed Counter...")


############# STEP: RESULTS

elif st.session_state['curr_step'] == RESULTS:
    results_csv = st.session_state['results_csv']
    curr_batch = st.session_state['curr_batch']

    st.header("Results")
    st.markdown("Here are the results of the seed counting.")

    st.table(results_csv)

    st.download_button(
        label="Download results as CSV",
        data=open(curr_batch['batch_output_dir'], 'rb'),
        file_name=f'seed_counter_{curr_batch["batch_id"]}.csv',
        mime='text/csv',
    )

    # list of files that are clickable to show the results
    # for the selected file, display images side by side, with the seeds highlighted, with the count of seeds and a button to overlay the original image
