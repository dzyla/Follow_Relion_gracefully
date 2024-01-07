# 
#  ____        ___    ___                           ____            ___                            ____                                    ___          ___    ___                
# /\  __\     /\_ \  /\_ \                         /\  _`\         /\_ \    __                    /\  _`\                                /'___\        /\_ \  /\_ \               
# \ \ \_    __\//\ \ \//\ \     ___   __  __  __   \ \ \L\ \     __\//\ \  /\_\    ___     ___    \ \ \L\_\  _ __    __      ___     __ /\ \__/  __  __\//\ \ \//\ \    __  __    
#  \ \  _\/ __`\\ \ \  \ \ \   / __`\/\ \/\ \/\ \   \ \ ,  /   /'__`\\ \ \ \/\ \  / __`\ /' _ `\   \ \ \L_L /\`'__\/'__`\   /'___\ /'__`\ \ ,__\/\ \/\ \ \ \ \  \ \ \  /\ \/\ \   
#   \ \ \/\ \L\ \\_\ \_ \_\ \_/\ \L\ \ \ \_/ \_/ \   \ \ \\ \ /\  __/ \_\ \_\ \ \/\ \L\ \/\ \/\ \   \ \ \/, \ \ \//\ \L\.\_/\ \__//\  __/\ \ \_/\ \ \_\ \ \_\ \_ \_\ \_\ \ \_\ \  
#    \ \_\ \____//\____\/\____\ \____/\ \___x___/'    \ \_\ \_\ \____\/\____\\ \_\ \____/\ \_\ \_\   \ \____/\ \_\\ \__/.\_\ \____\ \____\\ \_\  \ \____/ /\____\/\____\\/`____ \ 
#     \/_/\/___/ \/____/\/____/\/___/  \/__//__/       \/_/\/ /\/____/\/____/ \/_/\/___/  \/_/\/_/    \/___/  \/_/ \/__/\/_/\/____/\/____/ \/_/   \/___/  \/____/\/____/ `/___/> \
#                                                                                                                                                                           \\___/
# Follow Relion Gracefully (v5)
# Developed by Dawid Zyla, La Jolla Institute for Immunology 
# Non-Profit Open Software License 3.0 

# ## Main Changes
# -> Integration with Streamlit platform (https://streamlit.io/)
# -> Live processing preview incorporated (utilizing https://github.com/cryoEM-CNIO/CNIO_Relion_Tools)
# -> Experimental feature: running Relion in-browser
# 
# ## New Features
# -> Interactive widgets support (volume adjustment, selection, and data plotting)
# -> Enhanced job-specific statistics for deeper insights
# -> Optimized performance through real-time calculations (pre-calculation eliminated)
# -> Elimination of additional software requirements (e.g., Hugo)
# -> Expanded job previews including ModelAngelo, MotionCorr, CtfFind, and Picking
# -> Automated generation of job flow diagrams
# -> Interactive plots for data selection from uploaded STAR files
# -> Enhanced security with optional instance password setting
# -> Note: Potential slower performance for large datasets due to real-time calculations
# -> New feature to preview selected/rejected picks in SPA and STA
# -> Direct download functionality for all final mrc/PDF files via browser
# -> Downloadable selected micrographs and particles star
#
#
# ## To Do
# -> Browser-based operation is in experimental phase and lacks full pipeline support
# -> Plans to extend support to additional jobs (e.g., Dynamight, remaining Tomo jobs, multibody)
# -> Function rewrite targeted at reducing redundancy
# -> Implementation of job support for classic job types (e.g., Class2D, Select)
# -> Pipeline enhancement to facilitate adding jobs to pipeline.star



# Standard library imports
import os
import re
import glob
import math
import time
import textwrap
import subprocess
import argparse

# Data manipulation and computation
import numpy as np
import pandas as pd

# streamlit
import streamlit as st
from stqdm import stqdm
import streamlit.components.v1 as components

# Reading and working with MRC files
import mrcfile

# Image processing and filters
from skimage import measure, exposure
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from skimage.measure import marching_cubes
from scipy.fft import fft2, fftshift, rfft2
from skimage.exposure import rescale_intensity


# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import tempfile
from PIL import Image

# Plotly library for interactive plots
import plotly.subplots as sp
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from pyvis.network import Network
import hmac


# Working with CIF/Star files
from gemmi import cif

# Running Relion
import subprocess
import shlex
import signal
from typing import List


def parse_star(file_path):
    doc = cif.read_file(file_path)

    star_data = {}

    for data in doc:
        try:
            dataframe = pd.DataFrame()
            for item in data:
                for metadata in item.loop.tags:
                    value = data.find_loop(metadata)
                    dataframe[metadata] = np.array(value)
                star_data[data.name] = dataframe

        except AttributeError as e:
            pass
            # print(e)

    return star_data


def get_folders(path):
    """Return a list of folder names in the given directory, sorted case-insensitively."""
    try:
        # List all entries in the directory
        entries = os.listdir(path)
        # Filter out folders
        folders = [
            entry for entry in entries if os.path.isdir(os.path.join(path, entry))
        ]
        # Sort folders case-insensitively
        return sorted(folders, key=str.lower)
    except FileNotFoundError:
        # Return an empty list if the path doesn't exist
        return []


def get_newest_change(folder, include_subfolders=False):
    newest_mod_time = None
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            mod_time = os.path.getmtime(file_path)
            if newest_mod_time is None or mod_time > newest_mod_time:
                newest_mod_time = mod_time
        if not include_subfolders:
            # Break the loop after scanning the top-level directory
            break

    if newest_mod_time is not None:
        return datetime.fromtimestamp(newest_mod_time).strftime("%Y-%m-%d %H-%M")
    else:
        return "No files found in the folder"


def get_relationships_df(df):
    children = {}
    parents = {}

    # Function to extract the job part of the process name
    def extract_job_name(process_name):
        parts = process_name.split("/")
        return "/".join(parts[:-1]) + "/" if len(parts) > 1 else process_name

    # Process each edge
    for index, row in df.iterrows():
        from_node = extract_job_name(row["_rlnPipeLineEdgeFromNode"])
        to_node = extract_job_name(row["_rlnPipeLineEdgeProcess"])

        # Add to_node as a child of from_node
        parents.setdefault(from_node, []).append(to_node)

        # Add from_node as a parent of to_node
        children.setdefault(to_node, []).append(from_node)

    # Create DataFrames for children and parents
    children_df = pd.DataFrame(list(children.items()), columns=["Job", "Children"])
    parents_df = pd.DataFrame(list(parents.items()), columns=["Job", "Parents"])

    # Merge the two DataFrames
    merged_df = pd.merge(children_df, parents_df, on="Job", how="outer")

    return merged_df


def execute_job(selected_job, folder, node_files):
    job_actions = {
        "Import": lambda: plot_import_streamlit(folder, node_files),
        "MotionCorr": lambda: plot_motioncorr_stats_streamlit(os.path.join(folder, node_files[0])),
        "CtfFind": lambda: plot_ctf_stats_streamlit(folder, node_files[0]),
        "Extract": lambda: process_extract_streamlit(node_files, folder),
        "Class2D": lambda: plot_2d_streamlit(folder, selected_job, node_files[0]),
        "AutoPick": lambda: plot_picks_streamlit(folder, selected_job),
        "ManualPick": lambda: plot_picks_streamlit(folder, selected_job),
        "Class3D": lambda: plot_cls3d_stats_streamlit(folder, selected_job, node_files),
        "Initial": lambda: plot_cls3d_stats_streamlit(folder, selected_job, node_files),
        "Refine3D": lambda: plot_cls3d_stats_streamlit(folder, selected_job, node_files),
        "Mask": lambda: plot_mask_streamlit(os.path.join(folder, node_files[0])),
        "LocalRes": lambda: plot_locres_streamlit(node_files, folder, selected_job),
        "CtfRefine": lambda: plot_ctf_refine_streamlit(node_files, folder, selected_job) if "Tomo" not in selected_job else None,
        "Polish": lambda: plot_polish_streamlit(node_files, folder, selected_job),
        "PostProcess": lambda: plot_postprocess_streamlit(node_files, folder),
        "ModelAngelo": lambda: plot_pdb_streamlit(node_files, folder, selected_job),
        "Select": lambda: plot_selection_streamlit(node_files, folder, selected_job),
        "ReconstructParticleTomo": lambda: plot_cls3d_stats_streamlit(folder, selected_job, node_files),
        "PseudoSubtomo": lambda: plot_pseudosubtomo_streamlit(folder, node_files, selected_job)
    }

    for job_type, action in job_actions.items():
        if job_type in selected_job:
            action()
            break
    else:
        st.subheader('Job not supported (yet)')
    return True


# Streamlit app layout
def display_job_info(selected_job, FOLDER, pipeline_processes, pipeline_star):
    process_nodes = pipeline_star["pipeline_nodes"]["_rlnPipeLineNodeName"]
    job_path = os.path.join(FOLDER, selected_job)

    col1, col2 = st.columns(2)
    st.title(selected_job)
    st.divider()

    # Get status and icon
    status = pipeline_processes[
        pipeline_processes["_rlnPipeLineProcessName"] == selected_job
    ]["_rlnPipeLineProcessStatusLabel"].values[0]
    alias = pipeline_processes[
        pipeline_processes["_rlnPipeLineProcessAlias"] == selected_job
    ]["_rlnPipeLineProcessAlias"].values
    icon_status = (
    ":white_check_mark:" if status == "Succeeded" else
    ":heavy_exclamation_mark:" if status == "Failed" else
    ":x:" if status == "Aborted" else
    ":recycle:" if status == "Running" else
    "Unknown"
    )

    # Display newest change
    newest_change = get_newest_change(job_path)

    # Display relationships
    job_relation = get_relationships_df(pipeline_star["pipeline_input_edges"])
    children = job_relation[job_relation["Job"] == selected_job]["Children"].values
    parents = job_relation[job_relation["Job"] == selected_job]["Parents"].values

    # Check if children/parents are not empty and are iterable
    children_text = (
        ", ".join(children[0])
        if len(children) > 0 and isinstance(children[0], list)
        else "None"
    )
    parents_text = (
        ", ".join(parents[0])
        if len(parents) > 0 and isinstance(parents[0], list)
        else "None"
    )

    # Use columns to display time, status, source, and target jobs
    col1, col2 = st.columns(2)

    with col1:
        # st.markdown(f"**Job path:** `{job_path}`")
        st.markdown(f"**Job status:** _{status}_ {icon_status}")
        st.markdown(f"**Job newest change:** _{newest_change}_")
        if alias.size > 0:
            st.markdown(f"**Alias:** _{alias}_")

    with col2:
        st.markdown(f"**Source job(s):** _{children_text}_")
        st.markdown(f"**Target job(s):** _{parents_text}_")

    # List related node files
    node_files = [node for node in process_nodes if selected_job in node]
    
    with col1:
        with st.expander("Related Node Files", expanded=False):
            if node_files:
                st.markdown(f"`{', '.join(node_files)}`")
            else:
                st.markdown("None")
    with col2:
        if st.button("Refresh Data"):
            # Rerun the app to refresh data
            st.cache_data.clear()
            st.rerun()
            

    st.divider()


    job_completed = execute_job(selected_job, FOLDER, node_files)
    print(f'{datetime.now()}: execute_job: {job_completed}')
        

    buttons_done = set()  # Using a set for performance in lookups
    job = selected_job.split("/")[1]

    # Filter the node files to include only those that match the criteria
    valid_node_files = [
        node for node in node_files if "mrc" in node or "cif" in node or "pdf" in node
    ]

    # Calculate the number of columns based on the valid_node_files length
    if len(valid_node_files) > 0:
        columns_count = min(len(valid_node_files), 3)
        cols = st.columns(columns_count)

        for idx, node in enumerate(valid_node_files):
            with cols[idx % columns_count]:
                file_name = f"{job}_{os.path.basename(node)}"
                if file_name not in buttons_done:
                    file_path = os.path.join(FOLDER, node)
                    try:
                        create_download_button(
                            file_path, f"Download {file_name}", file_name
                        )
                        buttons_done.add(file_name)
                    except Exception as e:
                        print(e)

                    if "half" in node:
                        half2_node = node.replace("half1", "half2")
                        half2_file_name = f"{job}_{os.path.basename(half2_node)}"
                        if half2_file_name not in buttons_done:
                            half2_file_path = os.path.join(FOLDER, half2_node)
                            try:
                                create_download_button(
                                    half2_file_path,
                                    f"Download {half2_file_name}",
                                    half2_file_name,
                                )
                                buttons_done.add(half2_file_name)
                            except Exception as e:
                                print(e)

    st.divider()

    note_path = os.path.join(job_path, "note.txt")

    note = get_note(note_path)
    with st.expander("Running parameters:"):
        st.code(note)

def extract_with_comprehension(data):
    return np.array([d['pointIndex'] for d in data])

def interactive_scatter_plot(star_file_path, max_rows=None, own=False, selected_block='', own_name=''):
    if not own:
        columns_to_plot = ["particles", "movies", "micrographs"]

        star_file = parse_star(star_file_path)
        column_present = [
            column for column in columns_to_plot if column in star_file.keys()
        ][0]
        df = star_file[column_present]
    
    else:
        star_file = star_file_path
        df = star_file_path[selected_block]

    df_raw = df.copy()
    
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            continue
    

    # Allow the user to choose X and Y columns
    st.header("Interactive plotting and selection")
    st.write("Select with lasso or box to save selected points to STAR file")
    
    columns = df.columns
    x_axis = st.selectbox("Choose X-axis", columns, index=0)
    y_axis = st.selectbox("Choose Y-axis", columns, index=1 if len(columns) > 1 else 0)
    plot_type = st.checkbox("Show as 2D histogram instead of scatter plot")
    ratio_fix = st.checkbox("Keep data aspect ratio?")
    
    if df.shape[0] > 1:
        max_rows = st.slider('Random size of rows to plot (to speed up) (Max, use all, no shuffle)', 1, df.shape[0], df.shape[0])
    else:
        max_rows=1
        
    if max_rows != df.shape[0]:
        if len(df) > max_rows:
            st.warning(
                f"Dataframe is too large to display at once, displaying a random sample of {max_rows} rows instead."
            )
            df = df.sample(n=max_rows)
    
    # Convert string columns to categorical if necessary
    if df[x_axis].dtype == "object":
        x_is_string = True
        x_axis_conversion = st.radio(
            f"X-axis '{x_axis}' is text. Convert to:", ("Numbers", "Index")
        )
        if x_axis_conversion == "Numbers":
            df[x_axis] = pd.Categorical(df[x_axis]).codes
        else:
            df[x_axis] = range(len(df))
    else:
        x_is_string = False

    if df[y_axis].dtype == "object":
        y_is_string = True
        y_axis_conversion = st.radio(
            f"Y-axis '{y_axis}' is text. Convert to:", ("Numbers", "Index")
        )
        if y_axis_conversion == "Numbers":
            df[y_axis] = pd.Categorical(df[y_axis]).codes
        else:
            df[y_axis] = range(len(df))
    else:
        y_is_string = False

    # Create the plot
    if plot_type:
        log_scale = st.checkbox("Use Log Scale for Color")
        # Calculate histogram
        hist_data, xedges, yedges = np.histogram2d(
            df[x_axis], df[y_axis], bins=[20, 20]
        )
        if log_scale:
            # Apply logarithmic scaling
            hist_data = np.log(hist_data + 1)  # Add 1 to avoid log(0)

        # Create a heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=hist_data,
                x=np.around(xedges, 2),
                y=np.around(yedges, 2),
                colorscale="Viridis" if not log_scale else "Cividis",
            )
        )
        # Set axis labels
        fig.update_layout(
            xaxis_title=x_axis, yaxis_title=y_axis, title="2D Histogram Plot"
        )
    
    
    else:
        # Plot scatter
        fig = px.scatter(df, x=x_axis, y=y_axis, title="Interactive Scatter Plot")

    if ratio_fix:
        fig.update_layout(
            scene=dict(aspectmode="data"),
            height=600,  # Height of the plot
            margin=dict(l=0, r=0, t=0, b=0),
        )
        
    selected_points = plotly_events(fig, select_event=True)
    selected_idx = extract_with_comprehension(selected_points)

    pattern = r"job\d\d\d"

    try:
        # Search for the pattern in the file path
        match = re.search(pattern, star_file_path)

        # Extract the matched string if found
        job_string = match.group(0) if match else ''
        
        basename = os.path.basename(star_file_path).replace('.star', '_mod.star')
        final_star_name = f'{job_string}_{basename}'
    
    except:
        final_star_name = f'subset_selected_{own_name}'

    if len(selected_idx) > 0:
        if not own:
            star_subset = df_raw.iloc[selected_idx]
            key0 = list(star_file.keys())[0]
            modified_star = star_from_df({key0: star_file[key0], column_present: star_subset})

        # get selected data, get all other keys back and then add the selected
        else:
            star_subset = df_raw.iloc[selected_idx]
            keys = star_file.keys()
            star_dict = {}
            for key in keys:
                if key != selected_block:
                    star_dict[key] = star_file[key]
            star_dict[selected_block] = star_subset
            modified_star = star_from_df(star_dict)
        
        
        # Save the modified STAR file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.star') as tmp_file:
            modified_star.write_file(tmp_file.name)

        # Read the file in binary mode
        with open(tmp_file.name, 'rb') as file:
            binary_star_data = file.read()

        # Provide a download button in Streamlit
        st.download_button(
            label=f"**Download selected data as STAR {final_star_name}**",
            data=binary_star_data,
            file_name=final_star_name,
            mime='application/octet-stream'  # MIME type for binary file
        )
        st.divider()
        # Optional: Remove the temporary file if desired
        os.remove(tmp_file.name)
        
    # Show the plot
    #st.plotly_chart(fig, use_container_width=True)


def star_from_df(dicts_of_df):
    out_doc = cif.Document()

    for element in dicts_of_df.keys():
        out_particles = out_doc.add_new_block(element, pos=-1)
        
        # Ensure that the object is a DataFrame
        if isinstance(dicts_of_df[element], pd.DataFrame):
            column_names = dicts_of_df[element].columns
        else:
            raise TypeError(f"The object for key '{element}' is not a DataFrame.")
        
        column_names_to_star = [f"{name} #{n + 1}" for n, name in enumerate(column_names)]
        loop = out_particles.init_loop('', column_names_to_star)
        data_rows = dicts_of_df[element].to_numpy().astype(str).tolist()

        for row in data_rows:
            loop.add_row(row)

    return out_doc


def convert_to_indices(column):
    """Convert a pandas series to indices."""
    unique_vals = column.unique()
    val_to_index = {val: idx for idx, val in enumerate(unique_vals)}
    return column.map(val_to_index)


def create_download_button(file_path, label, file_name):
    with open(file_path, "rb") as fp:
        st.download_button(label=label, data=fp, file_name=file_name)


def get_file_mod_time(file, rln_folder):
    file_path = os.path.join(rln_folder, file)
    if os.path.exists(file_path):
        try:
            return datetime.fromtimestamp(os.path.getmtime(file_path))
        except Exception as e:
            print(e)
            return None
    else:
        return None


def plot_import_streamlit(rln_folder, node_files):
    STA = False
    
    star_data = parse_star(os.path.join(rln_folder, node_files[0]))
    
    if any("tomograms" in node for node in node_files) or any(
        "movies" in node for node in node_files
    ):
        try:
            star_data = star_data["movies"]
        except KeyError:
            star_data = star_data["global"]

        try:
            file_names = star_data["_rlnMicrographMovieName"]
            STA = False
        except KeyError:
            try:
                file_names = star_data["_rlnTomoTiltSeriesName"]
                STA = True
            except:
                return

        if file_names is None:
            st.write("No Micrographs / Movies found")
            return

        # Import by time
        file_mod_times = []
        
        st.warning('Checking files timestamps can take a while')
        if st.checkbox('Show import times?'):
            for file in stqdm(file_names):
                file_path = os.path.join(rln_folder, file)
                if os.path.exists(file_path):
                    try:
                    
                        file_mod_times.append(
                            datetime.fromtimestamp(os.path.getmtime(file_path))
                        )
                    except Exception as e:
                        print(e)
                else:
                    pass

            if not file_mod_times:
                st.write("No valid files found")
                return
            
            #file_mod_times.sort(key=lambda x: x[0])
            file_mod_times.sort()
            
            st.subheader(f"Found {len(file_mod_times)} movies")
            fig = go.Figure()

            fig.add_scatter(
                x=np.arange(len(file_mod_times)),
                y=file_mod_times,
                mode="markers",
                name="Time stamp",
            )

            fig.update_xaxes(title_text="Index")
            fig.update_yaxes(title_text="Time stamp")

            fig.update_layout(title="Imported micrographs timeline")
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Displaying the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
        if STA:
            if st.checkbox('Show tomograms?'):
                process_tomograms(file_names, FOLDER)
            

    if any("particles" in node for node in node_files):
        star_data = star_data["particles"]
        # tomo particles import
        if "_rlnCoordinateZ" in star_data.columns:
            unique_mics = np.unique(star_data["_rlnTomoName"])
            file_idx = st.slider("Tomogram index", 0, len(unique_mics), 0)

            file = os.path.join(FOLDER, unique_mics[file_idx])
            coords_sel = star_data[star_data["_rlnTomoName"] == unique_mics[file_idx]][
                ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]
            ]
            all_coords = star_data[star_data["_rlnTomoName"] == unique_mics[file_idx]][
                ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]
            ]

            merged_df = pd.merge(
                all_coords,
                coords_sel,
                on=["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"],
                how="outer",
                indicator=True,
            )

            # Filter to keep rows where the merge indicator is 'left_only'
            coords_rej = merged_df[merged_df["_merge"] == "left_only"][
                ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]
            ]

            fig = plot_tomogram_picks(file, file_idx, coords_sel, coords_rej, [2, 2])
            st.plotly_chart(fig)

        # SPA
        else:
            pass
    print(f'{datetime.now()}: plot_import_streamlit done')
    

#@st.cache_data(experimental_allow_widgets=True)
def process_tomograms(tilt_series, rlnFOLDER):
    
    tomo_idx = st.slider('Tomogram index', 0, len(tilt_series)-1, 0)
    gaussian_blur = st.slider('Gaussian blur sdev', 0., 5., 0.1, 0.1)
    seleced_tomo = tilt_series[tomo_idx]
    
    process_single_z_slice(os.path.join(rlnFOLDER, seleced_tomo), gaussian_blur)


def plot_motioncorr_stats_streamlit(star_path):
    star = parse_star(star_path)
    
    # Style for seaborn
    sns.set_style("whitegrid")

    star_data = star["micrographs"]
    meta_names = ["_rlnAccumMotionTotal", "_rlnAccumMotionEarly", "_rlnAccumMotionLate"]

    # Streamlit widget to choose between Seaborn and Plotly
    plot_type = st.radio("Choose plotting library", ("Seaborn", "Plotly"), 1)

    if plot_type == "Seaborn":
        fig, ax = plt.subplots()
        for meta in meta_names:
            data_list = np.clip(star_data[meta].astype(float), None, 2000)
            sns.lineplot(
                x=np.arange(0, len(data_list)),
                y=data_list,
                label=meta.replace("_rln", ""),
                linewidth=1,
                ax=ax,
            )

        ax.set_xlabel("Index")
        ax.set_ylabel("Motion")
        ax.set_title("MotionCorr statistics")
        ax.legend(loc="upper right")
        st.pyplot(fig)
        plt.close()

        fig, ax = plt.subplots()
        colors = sns.color_palette(n_colors=len(meta_names), palette="crest")

        for idx, meta in enumerate(meta_names):
            data_list = np.clip(star_data[meta].astype(float), None, 100)
            sns.histplot(
                data_list,
                alpha=0.5,
                bins=np.arange(min(data_list), max(data_list), 0.2),
                label=meta.replace("_rln", ""),
                kde=False,
                ax=ax,
                linewidth=1.5,
                element="step",
                color=colors[idx],
            )

        ax.set_xlabel("Motion")
        ax.set_ylabel("Number")
        ax.set_title("Motion histograms")
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1.02))
        st.pyplot(fig)
        plt.close()

    elif plot_type == "Plotly":
        fig = go.Figure()
        for meta in meta_names:
            data_list = np.clip(star_data[meta].astype(float), None, 2000)
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, len(data_list)),
                    y=data_list,
                    mode="lines",
                    name=meta.replace("_rln", ""),
                )
            )

        fig.update_layout(
            title="MotionCorr statistics",
            xaxis_title="Index",
            yaxis_title="Motion",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        for idx, meta in enumerate(meta_names):
            data_list = np.clip(star_data[meta].astype(float), None, 100)
            fig.add_trace(
                go.Histogram(
                    x=data_list, name=meta.replace("_rln", ""), opacity=0.5, nbinsx=40
                )
            )

        fig.update_layout(
            title="Motion histograms (Plotly)",
            xaxis_title="Motion",
            yaxis_title="Number",
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    ### Motion plots
    st.subheader("Motion plots per micrograph")
    
    mics = star_data['_rlnMicrographName']
    mic_idx = st.slider('Micrograph index', 0, len(mics)-1, 0)

    selected_mic = mics[mic_idx]
    
    show_motion(FOLDER, selected_mic)
    
    if st.checkbox('Show corrected micrographs?'):
        gaussian_blur_stdev = st.slider("Gaussian blur sdev", 0.0, 5.0, 0.1)
        
        # Display the selected micrograph
        if mic_idx < len(mics):
            process_and_display_micrograph(
                mics[mic_idx],
                '',
                '',
                gaussian_blur_stdev,
                FOLDER,
                1,
            )
    
    if st.checkbox('Plot metadata'):
        interactive_scatter_plot(star_path)

    print(f'{datetime.now()}: plot_motioncorr_streamlit done')
    


def show_motion(rln_folder, mic_path, showlegend=True):
    showlegend=True
    
    try:
        fold_local_motion = 200

        motion_star_path = os.path.join(rln_folder, mic_path.replace('mrc', 'star'))
        motion_star = parse_star(motion_star_path)

        global_shift = motion_star['global_shift'].astype(float)
        local_shift = motion_star['local_shift'].astype(float)

        center_local_x = np.mean(local_shift['_rlnCoordinateX'])
        center_local_y = np.mean(local_shift['_rlnCoordinateY'])

        try:
            fold_global_motion_x = int(np.max(local_shift['_rlnCoordinateX'].astype(float))/2/np.max(global_shift['_rlnMicrographShiftX']))
            fold_global_motion_y = int(np.max(local_shift['_rlnCoordinateY'].astype(float))/2/np.max(global_shift['_rlnMicrographShiftY']))
            fold_global_motion = np.min([fold_global_motion_x, fold_global_motion_y])
        except Exception as e:
            print(e)
            fold_global_motion = 1

        global_shift['_rlnMicrographShiftX_adjusted'] = global_shift['_rlnMicrographShiftX']*fold_global_motion+center_local_x
        global_shift['_rlnMicrographShiftY_adjusted'] = global_shift['_rlnMicrographShiftY']*fold_global_motion+center_local_y

        # Create the plot for global shifts
        fig = go.Figure()

        if 'global_shift' in motion_star:
            fig.add_trace(go.Scatter(
                x=global_shift['_rlnMicrographShiftX_adjusted'],
                y=global_shift['_rlnMicrographShiftY_adjusted'],
                mode='markers+lines',
                name=f'Global Motion*{fold_global_motion} fold',
                line=dict(color='#6495ED'),
                showlegend=showlegend
            ))

        # Plot local shifts if available
        if 'local_shift' in motion_star:
            grouped = local_shift.groupby(['_rlnCoordinateX', '_rlnCoordinateY'])
            place_legend = True
            
            for (coordX, coordY), group in grouped:
                group['_rlnCoordinateX_final'] = group['_rlnCoordinateX'] + group['_rlnMicrographShiftX']*fold_local_motion
                group['_rlnCoordinateY_final'] = group['_rlnCoordinateY'] + group['_rlnMicrographShiftY']*fold_local_motion

                if place_legend and showlegend:
                    fig.add_trace(go.Scatter(
                        x=group['_rlnCoordinateX_final'],
                        y=group['_rlnCoordinateY_final'],
                        mode='lines',
                        name=f'Local Motion {fold_local_motion}-fold',
                        marker=dict(color='#6FC381', opacity=0.7),
                        line=dict(color='#6FC381', width=2),
                        showlegend=True
                    ))
                    place_legend = False
                
                else:
                    fig.add_trace(go.Scatter(
                        x=group['_rlnCoordinateX_final'],
                        y=group['_rlnCoordinateY_final'],
                        mode='lines',
                        name=f'Local Motion',
                        marker=dict(color='#6FC381', opacity=0.7),
                        line=dict(color='#6FC381', width=2),
                        showlegend=False
                    ))

        if not fig.data:
            st.write("No data available for plotting.")
            return

        # Update layout
        fig.update_layout(
            title='Global and Local Motions',
            xaxis_title='Shift X',
            yaxis_title='Shift Y',
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                x=0.02, 
                y=0.98, 
                bordercolor="Black",
                borderwidth=2,
                orientation="h"
            ),
            #height=300,
        )
        fig.update_layout(
            scene=dict(aspectmode="cube"),
            #height=600,  # Height of the plot
            margin=dict(l=0, r=0, t=0, b=0),
        )
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        print(f"An error occurred: {e}")
        st.write("No data available for plotting.")


def plot_ctf_stats_streamlit(FOLDER, star_path):
    starctf_path = os.path.join(FOLDER, star_path)
    starctf = parse_star(starctf_path)
    
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Define column names for clarity
    defocus_v = "_rlnDefocusV"
    ctf_max_res = "_rlnCtfMaxResolution"
    ctf_astigmatism = "_rlnCtfAstigmatism"
    ctf_figure_of_merit = "_rlnCtfFigureOfMerit"

    # Get relevant data
    reduced_ctfcorrected = starctf["micrographs"][
        [defocus_v, ctf_max_res, ctf_astigmatism, ctf_figure_of_merit]
    ].astype(float)

    plot_type = st.radio("Choose plotting library", ("Seaborn", "Plotly"), 1)

    if plot_type == "Seaborn":
        # Defocus / index
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            range(0, reduced_ctfcorrected.shape[0]), reduced_ctfcorrected[defocus_v]
        )
        ax.set_xlabel("Index")
        ax.set_ylabel(defocus_v)
        ax.set_title("Defocus per Micrograph")
        plt.legend([defocus_v.replace("_rln", "")], loc="upper right")
        st.pyplot(fig)

        # Max Res / index
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            range(0, reduced_ctfcorrected.shape[0]), reduced_ctfcorrected[ctf_max_res]
        )
        ax.set_xlabel("Index")
        ax.set_ylabel(ctf_max_res)
        ax.set_title("Max Resolution per Micrograph")
        plt.legend([ctf_max_res.replace("_rln", "")], loc="upper right")
        st.pyplot(fig)

        # Defocus histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=reduced_ctfcorrected, x=defocus_v, kde=True, ax=ax, bins=50)
        ax.set_xlabel(defocus_v.replace("_rln", ""))
        ax.set_ylabel("Number")
        ax.set_title("Defocus Histogram")
        st.pyplot(fig)

        # Astigmatism histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            data=reduced_ctfcorrected, x=ctf_astigmatism, kde=True, ax=ax, bins=50
        )
        ax.set_xlabel(ctf_astigmatism.replace("_rln", ""))
        ax.set_ylabel("Number")
        ax.set_title("Astigmatism Histogram")
        st.pyplot(fig)

        # Max Resolution histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=reduced_ctfcorrected, x=ctf_max_res, kde=True, ax=ax, bins=50)
        ax.set_xlabel(ctf_max_res.replace("_rln", ""))
        ax.set_ylabel("Number")
        ax.set_title("Max Resolution Histogram")
        st.pyplot(fig)

        # Defocus / Max resolution
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.histplot(
            data=reduced_ctfcorrected,
            x=defocus_v,
            y=ctf_max_res,
            ax=ax,
            cmap="crest",
            bins=50,
        )
        ax.set_xlabel(defocus_v.replace("_rln", ""))
        ax.set_ylabel(ctf_max_res)
        ax.set_title("Defocus / Max Resolution Histogram")
        st.pyplot(fig)

        # Defocus / Figure of merit
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.histplot(
            data=reduced_ctfcorrected,
            x=defocus_v,
            y=ctf_figure_of_merit,
            ax=ax,
            cmap="crest",
            bins=50,
        )
        ax.set_xlabel(defocus_v.replace("_rln", ""))
        ax.set_ylabel(ctf_figure_of_merit)
        ax.set_title("Defocus / Figure of Merit Histogram")
        st.pyplot(fig)

    if plot_type == "Plotly":
        # Defocus / index
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(reduced_ctfcorrected.shape[0]),
                y=reduced_ctfcorrected[defocus_v],
                mode="lines",
                name="Defocus",
            )
        )
        fig.update_layout(
            title="Defocus per Micrograph", xaxis_title="Index", yaxis_title=defocus_v
        )
        st.plotly_chart(fig, use_container_width=True)

        # Max Res / index
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(reduced_ctfcorrected.shape[0]),
                y=reduced_ctfcorrected[ctf_max_res],
                mode="lines",
                name="Max Resolution",
            )
        )
        fig.update_layout(
            title="Max Resolution per Micrograph",
            xaxis_title="Index",
            yaxis_title=ctf_max_res,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Histograms for Defocus, Astigmatism, and Max Resolution
        for column, title in [
            (defocus_v, "Defocus Histogram"),
            (ctf_astigmatism, "Astigmatism Histogram"),
            (ctf_max_res, "Max Resolution Histogram"),
        ]:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=reduced_ctfcorrected[column]))
            fig.update_layout(title=title, xaxis_title=column, yaxis_title="Number")
            st.plotly_chart(fig, use_container_width=True)

        # Defocus / Max resolution
        fig = go.Figure(
            data=go.Histogram2d(
                x=reduced_ctfcorrected[defocus_v], y=reduced_ctfcorrected[ctf_max_res]
            )
        )
        fig.update_layout(
            title="Defocus / Max Resolution Histogram",
            xaxis_title=defocus_v,
            yaxis_title=ctf_max_res,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Defocus / Figure of merit
        fig = go.Figure(
            data=go.Histogram2d(
                x=reduced_ctfcorrected[defocus_v],
                y=reduced_ctfcorrected[ctf_figure_of_merit],
            )
        )
        fig.update_layout(
            title="Defocus / Figure of Merit Histogram",
            xaxis_title=defocus_v,
            yaxis_title=ctf_figure_of_merit,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    #plot per micrograph CTF 1D average
    st.write('**1D CTF fit per micrograph**')
    ctf_images = starctf['micrographs']['_rlnCtfImage'].str.replace('.ctf:mrc', '_avrot.txt')
    idx = st.slider('Micrograph index', 0, len(starctf['micrographs']))
    ctf_image_path = os.path.join(FOLDER, ctf_images[idx])
    plot_CTF_average(FOLDER, ctf_image_path)
    
    if st.checkbox('Show detailed statistics?'):
        interactive_scatter_plot(starctf_path)
        
    print(f'{datetime.now()}: plot_ctffind_streamlit done')
    


def plot_2d_histogram(defocus_u, figure_of_merit):
    data = pd.DataFrame({"DefocusU": defocus_u, "AutopickFOM": figure_of_merit})
    fig = px.density_heatmap(
        data,
        x="DefocusU",
        y="AutopickFOM",
        nbinsx=40,
        nbinsy=20,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        title="Defocus U vs Autopick Figure of Merit",
        xaxis_title="Defocus U",
        yaxis_title="Autopick Figure of Merit",
    )
    return fig


def plot_line_graph(micrographs, particles_per_mic):
    fig = px.line(
        x=micrographs,
        y=particles_per_mic,
        labels={"x": "Micrographs", "y": "Particles"},
    )
    fig.update_layout(title="Particles per Micrograph")
    return fig


def plot_histogram_particles_per_mic(particles_per_mic):
    fig = px.histogram(particles_per_mic, nbins=30)
    fig.update_layout(
        title="Histogram of Particles Per Micrograph",
        xaxis_title="Particles",
        yaxis_title="Micrographs Count",
    )
    return fig


def plot_histogram_fom(figure_of_merit):
    fig = px.histogram(figure_of_merit, nbins=30)
    fig.update_layout(
        title="Histogram of Autopick Figure of Merit",
        xaxis_title="Autopick Figure of Merit",
        yaxis_title="Count",
    )
    return fig


def process_extract_streamlit(node_files, folder):
    # Check if 'particles.star' is present in node_files
    particles_star = next(
        (
            node_file
            for node_file in node_files
            if "particles.star" == os.path.basename(node_file)
        ),
        None,
    )

    if particles_star:
        star_path = os.path.join(folder, particles_star)
        particles = show_random_particles_streamlit(star_path, folder)

        if st.checkbox('Plot all statistics?'):
            try:
                # Histogram plot
                defocus_u = particles["_rlnDefocusU"]
                autopick_fom = particles["_rlnAutopickFigureOfMerit"]
                fom_histogram_fig = plot_histogram_fom(autopick_fom)
                st.plotly_chart(fom_histogram_fig, use_container_width=True)

                micrographs = np.unique(particles["_rlnMicrographName"])

                particles_per_mic = [
                    particles[particles["_rlnMicrographName"] == mic].shape[0]
                    for mic in micrographs
                ]
                particles_per_mic_histogram_fig = plot_histogram_particles_per_mic(
                    particles_per_mic
                )
                st.plotly_chart(particles_per_mic_histogram_fig, use_container_width=True)

                hist_fig = plot_2d_histogram(defocus_u, autopick_fom)
                st.plotly_chart(hist_fig, use_container_width=True)

                # Line plot
                particles_per_mic = [
                    particles[particles["_rlnMicrographName"] == mic].shape[0]
                    for mic in micrographs
                ]
                line_fig = plot_line_graph(range(micrographs.shape[0]), particles_per_mic)
                st.plotly_chart(line_fig, use_container_width=True)
            except Exception as e:
                print(e)
        
        
        if st.checkbox('Show detailed statistics?'):    
            interactive_scatter_plot(star_path)
    else:
        st.write("No particles found")
    
    print(f'{datetime.now()}: plot_extract_streamlit done')
    


def show_random_particles_streamlit(star_path, project_path, adj_contrast=False):
    star = parse_star(star_path)["particles"]  # Ensure parse_star is defined
    data_shape = star.shape[0]

    num_images, image_width, adj_contrast, r = get_user_inputs()

    # Ensure the number of images requested does not exceed the available data
    num_images = min(num_images, data_shape)

    if f"selected_particles_{star_path}_{num_images}" not in st.session_state:
        random_int = np.random.randint(0, data_shape, num_images)
        st.session_state[f"selected_particles_{star_path}_{num_images}"] = random_int
    # else:
    #     # Ensure existing selected particles indices are within the new data shape
    #     st.session_state['selected_particles'] = [i for i in st.session_state['selected_particles'] if i < data_shape]

    selected = star.iloc[
        st.session_state[f"selected_particles_{star_path}_{num_images}"]
    ]

    particle_array = [
        process_particle(project_path, element, r, adj_contrast)
        for element in selected["_rlnImageName"]
    ]

    display_particles(particle_array, image_width)
    return star


def get_user_inputs():
    col1, col2 = st.columns(2)
    with col1:
        num_images = st.slider(
            "Number of images to plot", min_value=50, max_value=500, value=40, step=5
        )
        image_width = st.slider(
            "Select image width", min_value=50, max_value=300, value=150, step=10
        )
    with col2:
        adj_contrast = st.checkbox("Adjust contrast?")
        r = st.slider(
            "Gaussian blur stdev", min_value=0.1, max_value=5.0, value=0.0, step=0.05
        )
    return num_images, image_width, adj_contrast, r


def process_particle(project_path, element, r, adj_contrast):
    particle_data = element.split("@")
    img_path = os.path.join(project_path, particle_data[1])
    try:
        particle = mrcfile.mmap(img_path).data[int(particle_data[0])]
        particle = normalize_particle(particle)
        if r != 0:
            particle = gaussian_filter(particle, sigma=r)
        if adj_contrast:
            particle = adjust_contrast(particle)  # Ensure adjust_contrast is defined
        return Image.fromarray(particle)
    except IndexError:
        return None


def normalize_particle(particle):
    particle = (particle - particle.min()) / (particle.max() - particle.min())
    return (particle * 255).astype(np.uint8)


def display_particles(particle_array, image_width):
    col_width = max(
        1, 600 // image_width
    )  # Assuming a max container width of 300 for each column
    cols = st.columns(col_width)
    for n, particle in enumerate(particle_array):
        if particle:
            with cols[n % col_width]:
                st.image(particle, width=image_width)


def reverse_FFT(fft_image):
    fft_img_mod = np.fft.ifftshift(fft_image)

    img_mod = np.fft.ifft2(fft_img_mod)
    return img_mod.real


def mask_particle(particle, radius=None):
    if radius == 1:
        return particle

    h, w = particle.shape[:2]
    radius = np.min([h, w]) / 2 * radius
    mask = create_circular_mask(h, w, radius)
    masked_img = particle.copy()
    masked_img[~mask] = 0

    return masked_img


def create_circular_mask(h, w, radius=None):
    center = [int(w / 2), int(h / 2)]

    # use the smallest distance between the center and image walls
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def apply_window(image):
    # Precompute and reuse window if image size is constant across calls
    if not hasattr(apply_window, "hann_window") or apply_window.hann_window.shape != image.shape:
        apply_window.hann_window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    return image * apply_window.hann_window

def plot_power_spectrum(rln_folder, img_path):
    with mrcfile.mmap(os.path.join(rln_folder, img_path), mode='r', permissive=True) as mrc:
        image = mrc.data.copy()

    image = apply_window(image)

    if image.shape[0] != image.shape[1]:
        max_dim = max(image.shape)
        image = np.pad(image, [(0, max_dim - s) for s in image.shape], mode='constant')

    # Compute the Fourier Transform using rfft2
    fft_image = rfft2(image)

    # Create the full power spectrum by mirroring
    powerspec = np.abs(fft_image)**2
    powerspec_full = np.concatenate((powerspec, np.flip(powerspec, axis=1)), axis=1)

    powerspec_shifted = fftshift(powerspec_full)
    powerspec_shifted = normalize_image(powerspec_shifted, resize_shape=None)

    st.image(powerspec_shifted)

def normalize_image(image, clip_limit_low=0.5, clip_limit_high=99.5, lowpass=None, resize_shape=None):
    # Convert to magnitude squared if image is complex
    if np.iscomplexobj(image):
        image = np.abs(image) ** 2

    # Normalize the image
    image = image.astype(np.float32)
    p_low, p_high = np.percentile(image, [clip_limit_low, clip_limit_high])
    normalized_image = rescale_intensity(image, in_range=(p_low, p_high))

    # Resize the image if resize_shape is specified
    if resize_shape is not None:
        normalized_image = resize(normalized_image, (resize_shape, resize_shape), anti_aliasing=True)

    # Apply lowpass filter after resizing if specified
    if lowpass is not None:
        normalized_image = gaussian_filter(normalized_image, sigma=lowpass)

    return normalized_image
    

def adjust_contrast(img, p1=2, p2=98):
    p1, p2 = np.percentile(img, (p1, p2))
    img = exposure.rescale_intensity(img, in_range=(p1, p2))
    return img


def plot_2d_streamlit(rln_folder, job_name, data_star):
    model_files = sorted(
        glob.glob(os.path.join(rln_folder, job_name, "*model.star")),
        key=os.path.getmtime,
    )
    if not model_files:
        st.write("No model files found.")
        return

    class_paths, n_classes, iter_, class_dist, class_res, _, _ = get_classes(
        os.path.join(rln_folder, job_name), [model_files[-1]]
    )
    class_dist = np.squeeze(class_dist)

    sort_classes = st.checkbox("Sort classes?", True)
    if not class_paths:
        st.write("No classes found.")
        return

    for class_path in class_paths:
        display_classes(class_path, class_dist, sort_classes, data_star)

    class_paths, n_classes, iter_, class_dist, class_res, _, _ = get_classes(
        os.path.join(rln_folder, job_name), model_files
    )
    class_dist_plot = plot_class_distribution_pyplot(class_dist)
    st.plotly_chart(class_dist_plot)
    

def get_classes(path_, model_star_files):
    class_dist_per_run = []
    class_res_per_run = []
    fsc_per_run = []
    fsc_res_per_run = []

    for iter, file in enumerate(model_star_files):
        # for the refinement, plot only half2 stats
        star_file = parse_star(file)
        if not "half1" in file:
            class_dist_per_run.append(
                star_file["model_classes"]["_rlnClassDistribution"]
            )

            class_res_per_run.append(
                star_file["model_classes"]["_rlnEstimatedResolution"]
            )
            
            try:
                # Get FSC data
                fsc_per_run.append(
                    star_file["model_class_1"]["_rlnGoldStandardFsc"]
                )
                
                fsc_res_per_run.append(
                    star_file["model_class_1"]["_rlnAngstromResolution"]
                )
            except Exception as e:
                print(f'No FSC found in model file {e}')
                pass

    # stack all data together
    try:
        class_dist_per_run = np.stack(class_dist_per_run)
        class_res_per_run = np.stack(class_res_per_run)
        
        try:
            # Keep only last FSC
            fsc_per_run = fsc_per_run[-1]
            fsc_res_per_run = fsc_res_per_run[-1]
        except:
            pass
        
        # rotate matrix so the there is class(iteration) not iteration(class) and starting from iter 0 --> iter n
        class_dist_per_run = np.flip(np.rot90(class_dist_per_run), axis=0)
        class_res_per_run = np.flip(np.rot90(class_res_per_run), axis=0)

        # Find the class images (3D) or stack (2D)
        class_files = parse_star(model_star_files[-1])["model_classes"][
            "_rlnReferenceImage"
        ]

        class_path = []
        for class_name in class_files:
            class_name = os.path.join(path_, os.path.basename(class_name))

            # Insert only new classes, in 2D only single file
            if class_name not in class_path:
                class_path.append(class_name)

        n_classes = class_dist_per_run.shape[0]
        iter_ = class_dist_per_run.shape[1] - 1

    except ValueError:
        class_path, n_classes, iter_, class_dist_per_run, class_res_per_run, fsc_res_per_run, fsc_per_run = (
            [],
            [],
            [],
            [],
            [],
            [],
            []
        )

    return class_path, n_classes, iter_, class_dist_per_run, class_res_per_run, fsc_res_per_run, fsc_per_run


# def display_classes(class_path, class_distribution, sort_by_distribution=False):
#     if type(class_path) == str:
#         classes = mrcfile.mmap(class_path).data
#     else:
#         # allow accepting mrc stack
#         classes = class_path

#     class_distribution = class_distribution.astype(float)

#     if sort_by_distribution:
#         # Sort classes based on distribution and apply the same order to class images
#         sorted_indices = np.argsort(class_distribution)[::-1]  # Descending order
#         classes = classes[sorted_indices]
#         class_distribution = class_distribution[sorted_indices]

#     # Slider to adjust image width
#     image_width = st.slider(
#         "Select image width", min_value=50, max_value=300, value=150, step=10
#     )

#     # Determine the number of columns based on the selected image width
#     col_width = max(
#         1, 600 // image_width
#     )  # Assuming a max container width of 300 for each column
#     cols = st.columns(col_width)

#     for i, class_image in enumerate(classes):
#         distribution_info = float(class_distribution[i]) * 100
#         with cols[i % col_width]:
#             # Using st.image with output_format='PNG' to allow zoom
#             try:
#                 cls_n = sorted_indices[i]
#             except:
#                 cls_n = i
#             try:
#                 st.image(
#                     normalize(class_image),
#                     caption=f"Class {cls_n} Distribution: {distribution_info:.2f}%",
#                     width=image_width,
#                     output_format="PNG",
#                 )
#             except Exception as e:
#                 print(e)
                
def display_classes(class_path, class_distribution, sort_by_distribution=False, raw_data_star=None):
    
    # get the file name
    pattern = r"job\d\d\d"
    # Search for the pattern in the file path
    
    match = re.search(pattern, raw_data_star)
    
    
    # Extract the matched string if found
    job_string = match.group(0) if match else ''
    
    
    if type(class_path) == str:
        classes = mrcfile.mmap(class_path).data
    else:
        classes = class_path

    class_distribution = class_distribution.astype(float)

    classes_unsorted = classes.copy()
    class_distribution_unsorted = class_distribution.copy()
    
    if sort_by_distribution:
        sorted_indices = np.argsort(class_distribution)[::-1]
        classes = classes[sorted_indices]
        class_distribution = class_distribution[sorted_indices]

    image_width = st.slider("Select image width", min_value=50, max_value=300, value=150, step=10)
    
    # Initialize session state for checkboxes if not already present
    if f'selected_indices_{job_string}' not in st.session_state:
        st.session_state[f'selected_indices_{job_string}'] = []

    # Control buttons
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Select All"):
            st.session_state[f'selected_indices_{job_string}'] = [i for i in range(len(classes))]
    with c2:
        if st.button("Unselect All"):
            st.session_state[f'selected_indices_{job_string}'] = []
    with c3:
        if st.button("Invert Selection"):
            st.session_state[f'selected_indices_{job_string}'] = [i for i in range(len(classes)) if i not in st.session_state[f'selected_indices_{job_string}']]
    
    col_width = max(1, 600 // image_width)
    cols = st.columns(col_width)

    for i, class_image in enumerate(classes):
        distribution_info = float(class_distribution[i]) * 100
        cls_n = sorted_indices[i] if sort_by_distribution else i
        with cols[i % col_width]:

            # Checkbox with session state
            checkbox_key = f"checkbox_{cls_n}"
            checked = st.checkbox(f"Select Class {cls_n}", key=checkbox_key, value=cls_n in st.session_state[f'selected_indices_{job_string}'])
            st.image(normalize(class_image), caption=f"Class {cls_n} Distribution: {distribution_info:.2f}%", width=image_width, output_format="PNG")
            
            if checked and cls_n not in st.session_state[f'selected_indices_{job_string}']:
                st.session_state[f'selected_indices_{job_string}'].append(cls_n)
            elif not checked and cls_n in st.session_state[f'selected_indices_{job_string}']:
                st.session_state[f'selected_indices_{job_string}'].remove(cls_n)

    st.divider()
    # Button to show selected indices
    if c4.button("Show Selected"):
        
        st.subheader(f"Selected Classes: (Total {len(st.session_state[f'selected_indices_{job_string}'])})")
        
        classes = classes_unsorted[st.session_state[f'selected_indices_{job_string}']]
        class_distribution = class_distribution_unsorted[st.session_state[f'selected_indices_{job_string}']]
        cls_n = sorted_indices[i] if sort_by_distribution else i
        col_width = max(1, 600 // image_width)
        cols = st.columns(col_width)
        
        for i, class_image in enumerate(classes):
            distribution_info = float(class_distribution[i]) * 100

            with cols[i % col_width]:
                st.image(normalize(class_image), caption=f"Class {cls_n} Distribution: {distribution_info:.2f}%", width=image_width, output_format="PNG")

    if st.button('Save selected particles to STAR?'):
        particles_star = parse_star(os.path.join(FOLDER, raw_data_star))
        optics = particles_star['optics']
        particles = particles_star['particles']
        
        selected_particles = particles[particles['_rlnClassNumber'].astype(int).isin(st.session_state[f'selected_indices_{job_string}'])]
        modified_star = star_from_df({'optics': optics, 'particles': selected_particles})
        
        # Save the modified STAR file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.star') as tmp_file:
            modified_star.write_file(tmp_file.name)
        
        final_star_name = f'{job_string}_selected.star'
        
        # Read the file in binary mode
        with open(tmp_file.name, 'rb') as file:
            binary_star_data = file.read()
        
        # Provide a download button in Streamlit
        st.download_button(
            label=f"**Download selected data as STAR {final_star_name}**",
            data=binary_star_data,
            file_name=final_star_name,
            mime='application/octet-stream'  # MIME type for binary file
        )
        
        st.divider()
        # Optional: Remove the temporary file if desired
        os.remove(tmp_file.name)



def normalize(class_image):
    if np.average(class_image) != 0:
        return (class_image - np.min(class_image)) / (
            np.max(class_image) - np.min(class_image)
        )
    return class_image


def normalize(class_image):
    if np.average(class_image) != 0:
        return (class_image - np.min(class_image)) / (
            np.max(class_image) - np.min(class_image)
        )
    return class_image


def plot_class_distribution_pyplot(class_dist):
    class_dist = class_dist.astype(np.float16) * 100
    n_classes, n_iterations = class_dist.shape

    fig = go.Figure()

    # Use a Plotly built-in color scale
    colors = px.colors.qualitative.Plotly

    # Cumulative distribution for stackplot effect
    cumulative = np.zeros(n_iterations)
    for i in range(n_classes):
        # Cycle through the color list
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_iterations),
                y=cumulative + class_dist[i],
                mode="lines",
                line=dict(width=0.5, color=color),
                fill="tonexty",
                name=f"Class {i + 1}",
            )
        )
        cumulative += class_dist[i]

    fig.update_layout(
        title="Class Distribution Over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Class Distribution (%)",
        legend_title="Classes",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, range=[0, 100]),
    )

    return fig


def plot_picks_streamlit(rln_folder, job_name, img_resize_fac=0.2):
    path_data = os.path.join(rln_folder, job_name)

    # Determine the source of coordinates (autopick.star, manualpick.star, or coords_suffix_*)
    coord_paths, mics_paths = get_coord_paths(path_data, job_name)

    # Most likely topaz training
    if len(coord_paths) == 0:
        if glob.glob(os.path.join(path_data, "model_training.txt")) != []:
            topaz_training_txt = glob.glob(
                os.path.join(path_data, "model_training.txt")
            )[0]

            data = pd.read_csv(topaz_training_txt, delimiter="\t")
            data_test = data[data["split"] == "test"]

            x = data_test["epoch"]
            data_test = data_test.drop(["iter", "split", "ge_penalty"], axis=1)

            fig_ = go.Figure()

            for n, column in enumerate(data_test.columns):
                if column != "epoch":
                    y = data_test[column]
                    fig_.add_scatter(
                        x=x,
                        y=y,
                        name=f"{column}",
                        hovertemplate=f"{column}<br>Epoch: %{{x}}<br>Y: %{{y:.2f}}<extra></extra>",
                    )

            fig_.update_xaxes(title_text="Epoch")
            fig_.update_yaxes(title_text="Statistics")

            fig_.update_layout(
                title="Topaz training stats. Best model: {}".format(
                    data_test[
                        data_test["auprc"].astype(float)
                        == np.max(data_test["auprc"].astype(float))
                    ]["epoch"].values
                )
            )

            fig_.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                )
            )
            fig_.update_layout(hovermode="x unified")

            st.plotly_chart(fig_)
            print(f'{datetime.now()}: plot_picks_streamlit done')
            return
            

        st.write("No coordinate files found.")
        print(f'{datetime.now()}: plot_picks_streamlit done')
        return

    # Slider to select a micrograph
    if len(mics_paths) > 1:
        mic_index = st.slider("Select Micrograph", 0, len(mics_paths) - 1, 0)
    else:
        mic_index = 0

    gaussian_blur_stdev = st.slider("Gaussian blur sdev", 0.0, 5.0, 0.1)
    plot_all = st.checkbox('Plot all statistics? (Might be slow for huge datasets)', value=False)
    
    if not "ManualPick" in job_name:
        pick_stats = parse_star(os.path.join(rln_folder, coord_paths[mic_index]))
        pick_stats = list(pick_stats.values())[0]["_rlnAutopickFigureOfMerit"].astype(
            float
        )
        fom_slider = st.slider(
            "FOM limit", 0.0, max(pick_stats), [0.0, max(pick_stats)]
        )

        coords_df = pd.DataFrame()
        fom_all_mics = []
        picks_per_mic = []
        
        # only calculate when selected
        if plot_all:
            for idx in range(len(mics_paths)):
                fom_stats = list(parse_star(os.path.join(rln_folder, coord_paths[idx])).values())[0]
                fom_stats['_rlnMicrographName'] = mics_paths[idx]
                coords_df = pd.concat([fom_stats, coords_df])
                fom_stats = fom_stats["_rlnAutopickFigureOfMerit"].astype(float)
                fom_all_mics.extend(fom_stats)
                picks_per_mic.append(fom_stats.shape[0])

        
            modified_star = star_from_df({'particles': coords_df})

            # Save the modified STAR file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.star') as tmp_file:
                modified_star.write_file(tmp_file.name)
        

    else:
        fom_slider = [-10000, 10000]
        fom_all_mics = None

    # Display the selected micrograph
    if mic_index < len(mics_paths):
        process_and_display_micrograph(
            mics_paths[mic_index],
            coord_paths[mic_index],
            fom_slider,
            gaussian_blur_stdev,
            rln_folder,
            img_resize_fac,
        )

    if not "ManualPick" in job_name:
        if plot_all:
            fom_all_mics = np.array(fom_all_mics)
            fom_histogram_fig = plot_histogram_fom(fom_all_mics)
            st.plotly_chart(fom_histogram_fig, use_container_width=True)
        
            if st.checkbox("Show statistics?"):
                interactive_scatter_plot(tmp_file.name)
                
    print(f'{datetime.now()}: plot_picks_streamlit done')


def get_coord_paths(job_folder, rln_folder):
    # Case 1: autopick.star
    autopick_star_path = os.path.join(job_folder, "autopick.star")
    if os.path.exists(autopick_star_path) and os.path.getsize(autopick_star_path) > 0:
        autopick_star = parse_star(autopick_star_path)["coordinate_files"]
        mics_paths = autopick_star["_rlnMicrographName"].to_numpy()
        coord_paths = autopick_star["_rlnMicrographCoordinates"].to_numpy()
        return coord_paths, mics_paths

    # Case 2: manualpick.star
    manualpick_star_path = os.path.join(job_folder, "manualpick.star")
    if (
        os.path.exists(manualpick_star_path)
        and os.path.getsize(manualpick_star_path) > 0
    ):
        manpick_star = parse_star(manualpick_star_path)["coordinate_files"]
        mics_paths = manpick_star["_rlnMicrographName"].to_numpy()
        coord_paths = manpick_star["_rlnMicrographCoordinates"].to_numpy()
        return coord_paths, mics_paths

    # Case 3: coords_suffix_*
    suffix_files = glob.glob(os.path.join(job_folder, "coords_suffix_*"))
    if suffix_files:
        suffix_file = suffix_files[0]
        suffix = (
            os.path.basename(suffix_file)
            .replace("coords_suffix_", "")
            .replace(".star", "")
        )
        mics_data_path = open(suffix_file).readline().strip()

        all_mics_paths = parse_star(os.path.join(rln_folder, mics_data_path))[
            "micrographs"
        ]["_rlnMicrographName"]
        mics_paths = [os.path.join(rln_folder, name) for name in all_mics_paths]

        coord_paths = [
            os.path.join(
                job_folder,
                f"coords_{suffix}",
                os.path.basename(mic_path).replace(".mrc", f"_{suffix}.star"),
            )
            for mic_path in mics_paths
        ]
        return coord_paths, mics_paths

    # Case 4: No matching pattern found
    return [], []


def get_coord_paths(job_folder, rln_folder):
    # Case 1: autopick.star
    autopick_star_path = os.path.join(job_folder, "autopick.star")
    if os.path.exists(autopick_star_path) and os.path.getsize(autopick_star_path) > 0:
        autopick_star = parse_star(autopick_star_path)["coordinate_files"]
        mics_paths = autopick_star["_rlnMicrographName"].to_numpy()
        coord_paths = autopick_star["_rlnMicrographCoordinates"].to_numpy()
        return coord_paths, mics_paths

    # Case 2: manualpick.star
    manualpick_star_path = os.path.join(job_folder, "manualpick.star")
    if (
        os.path.exists(manualpick_star_path)
        and os.path.getsize(manualpick_star_path) > 0
    ):
        manpick_star = parse_star(manualpick_star_path)["coordinate_files"]
        mics_paths = manpick_star["_rlnMicrographName"].to_numpy()
        coord_paths = manpick_star["_rlnMicrographCoordinates"].to_numpy()
        return coord_paths, mics_paths

    # Case 3: coords_suffix_*
    suffix_files = glob.glob(os.path.join(job_folder, "coords_suffix_*"))
    if suffix_files:
        suffix_file = suffix_files[0]
        suffix = (
            os.path.basename(suffix_file)
            .replace("coords_suffix_", "")
            .replace(".star", "")
        )
        mics_data_path = open(suffix_file).readline().strip()

        all_mics_paths = parse_star(os.path.join(rln_folder, mics_data_path))[
            "micrographs"
        ]["_rlnMicrographName"]
        mics_paths = [os.path.join(rln_folder, name) for name in all_mics_paths]

        coord_paths = [
            os.path.join(
                job_folder,
                f"coords_{suffix}",
                os.path.basename(mic_path).replace(".mrc", f"_{suffix}.star"),
            )
            for mic_path in mics_paths
        ]
        return coord_paths, mics_paths

    # Case 4: No matching pattern found
    return [], []


def process_single_z_slice(mic_path, gaussian_blur_stdev):
    with mrcfile.mmap(mic_path, permissive=True) as mrc:
        z_slice = st.slider('Frame index', 0, mrc.data.shape[0]-1, 0)
        micrograph = mrc.data[z_slice].copy()  # Copy the slice to a numpy array
        micrograph, _ = process_micrograph(micrograph, 1, gaussian_blur_stdev)
        plot_micrograph(micrograph, '', '', 1)
    

def process_and_display_micrograph(mic_path, coord_path, fom, gaussian_blur_stdev, rln_folder, img_resize_fac):
    try:
        try:
            micrograph = mrcfile.mmap(mic_path, permissive=True).data
        except Exception as e:
            # Try with a different path
            mic_path = os.path.join(rln_folder, mic_path)
            micrograph = mrcfile.mmap(mic_path, permissive=True).data
        
        # Resize, apply Gaussian blur, and adjust contrast
        micrograph, img_resize_fac = process_micrograph(micrograph, img_resize_fac, gaussian_blur_stdev)

        # Initialize coordinates
        coords_x, coords_y = [], []

        # Try to load coordinates
        try:
            coord_path = os.path.join(rln_folder, coord_path)
            coords, picks_idx = process_coordinates(coord_path, fom)
            coords_x = coords["_rlnCoordinateX"].astype(float)[picks_idx]
            coords_y = coords["_rlnCoordinateY"].astype(float)[picks_idx]
        except Exception as e:
            pass
            #print(f"Coordinate file could not be processed: {e}")

        # Plot and display
        plot_micrograph(micrograph, coords_x, coords_y, img_resize_fac)

    except Exception as e:
        st.error(f"Error processing micrograph: {e}")

def process_micrograph(micrograph, img_resize_fac, gaussian_blur_stdev):
    if micrograph.shape[0] > 500:
        img_resize_fac = 500 / micrograph.shape[0]
        micrograph = rescale(micrograph.astype(float), img_resize_fac)
    else:
        micrograph = micrograph.astype(float)

    micrograph = gaussian_filter(micrograph, gaussian_blur_stdev)
    p1, p2 = np.percentile(micrograph, (0.1, 99.8))
    return exposure.rescale_intensity(micrograph, in_range=(p1, p2)), img_resize_fac

def process_coordinates(coord_path, fom):
    coords = parse_star(coord_path)
    coords = list(coords.values())[0]
    fom_stats = coords["_rlnAutopickFigureOfMerit"].astype(float)
    lower_bound, upper_bound = fom
    picks_idx = np.where((fom_stats >= lower_bound) & (fom_stats <= upper_bound))[0]
    return coords, picks_idx

def plot_micrograph(micrograph, coords_x, coords_y, img_resize_fac):
    fig = plt.figure(dpi=200, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(micrograph, cmap="gray")
    if len(coords_x) > 0 and len(coords_y) > 0:
        ax.scatter(
            coords_x * img_resize_fac,
            coords_y * img_resize_fac,
            edgecolor="limegreen",
            facecolors="none",
            s=120,
            linewidth=0.5,
        )
    ax.axis("off")
    fig.tight_layout(pad=0)
    st.pyplot(fig, bbox_inches="tight")


    
    

def get_note(path_):
    if os.path.exists(path_):
        file = open(path_)
        file_data = str(file.read())
        file_data = file_data.replace("++++", "\n")
        file_data = file_data.replace("`", "")

        file_data = file_data.replace("which", "\nwhich")

        file.close()

    else:
        file_data = "Process queuing. Waiting for the run."
    return file_data


def get_angles(path_, limit=None):
    """
    Euler angles: (rot,tilt,psi) = (?,?,?). Positive rotations of object are clockwise. Projection direction is
    defined by (rot,tilt). Psi is in-plane rotation for tilted image. For untilted rot=psi=in-plane rotation.
    Angles in a STAR file rotate the reference into observations (i.e. particle image), while translations shift
    observations into the reference projection.

    :param path_:
    :return:
    """

    data_star = glob.glob(os.path.join(path_, "", "*data.star"))
    data_star.sort(key=os.path.getmtime)

    last_data_star = data_star[-1]
    
    data_star = parse_star(last_data_star)["particles"]
    if limit:
        data_star.sample(limit)
    
    rot_angles = data_star["_rlnAngleRot"].astype(
        float
    )
    tilt_angles = data_star["_rlnAngleTilt"].astype(
        float
    )
    psi_angles = data_star["_rlnAnglePsi"].astype(
        float
    )

    return rot_angles, tilt_angles, psi_angles


def plot_cls3d_stats_streamlit(rln_folder, job_name, nodes):
    path_data = os.path.join(rln_folder, job_name)

    # Get model files
    model_files = glob.glob(os.path.join(path_data, "*model.star"))
    model_files.sort(key=os.path.getmtime)
    n_inter = len(model_files)

    if n_inter != 0:
        class_paths, n_classes_, iter_, class_dist_, class_res_, fsc_res, fsc = get_classes(
            path_data, model_files
        )

        # Display combined classes
        st.write(f"Combined Classes: found {len(class_paths)}")
        plot_combined_classes_streamlit(class_paths, class_dist_[:, -1])

        # Display class projections
        plot_projections_streamlit(class_paths, class_dist_)

        # Display class distribution
        if n_classes_ > 1:
            plot_class_distribution_streamlit(class_dist_)
        
        # Plot FSC stats
        if 'Refine3D' in job_name:
            fsc_x = fsc_res.astype(float)
            fsc = fsc.astype(float)
            fsc_x_min, fsc_x_max = np.min(fsc_x), np.max(fsc_x)
            
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x= 1 / fsc_x,
                    y=fsc,
                    mode="lines",
                    name='GoldStandardFsc',
                    customdata=fsc_x,
                    hovertemplate=f"Resolution: %{{customdata:.2f}}<br>FSC: %{{y:.2f}}{'%'}<extra></extra>",
                )
            )
            fig.update_layout(hovermode="x unified")
            
            # Add horizontal lines for FSC thresholds
            fig.add_hline(
                y=0.143, line_dash="dash", line_color="black", annotation_text="0.143"
            )
            fig.add_hline(
                y=0.5, line_dash="dash", line_color="black", annotation_text="0.5"
            )

            # Custom ticks
            start_res = 1 / 50  # Starting resolution (low resolution)
            end_res = 1 / fsc_x_min  # Ending resolution (high resolution)
            custom_ticks = np.linspace(start_res, end_res, num=10)
            custom_tickvals = custom_ticks
            custom_ticktext = [f"{round(1/res, 2)}" for res in custom_ticks]

            fig.update_layout(
                xaxis=dict(
                    title="Resolution, Å",
                    tickvals=custom_tickvals,
                    ticktext=custom_ticktext,
                    range=[start_res, end_res],
                ),
                yaxis=dict(title="FSC", range=[-0.05, 1.05]),
                title="Fourier Shell Correlation Curve",
                legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0)"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig)

            fsc143 = 0.143
            resolution_143_idx = np.argmin(
                np.abs(fsc - fsc143)
            )

            fsc05 = 0.5
            resolution_05_idx = np.argmin(
                np.abs(fsc - fsc05)
            )

            st.markdown(
                f"Reported resolution @ FSC=0.143: __{round(fsc_x[resolution_143_idx],2)} Å__"
            )
            st.markdown(
                f"Reported resolution @ FSC=0.5: __{round(fsc_x[resolution_05_idx],2)} Å__"
            )

        
        
        # Display class resolution
        plot_class_resolution_streamlit(class_res_)

        # Plot Angular distribution
        try:
            rot, tilt, psi = get_angles(path_data)
            star_data = parse_star(os.path.join(rln_folder, nodes[0]))
            
            if n_classes_ > 1:
                star_data = star_data["particles"]
                cls_idx = star_data["_rlnClassNumber"]
            else:
                star_data = star_data["particles"]
                cls_idx = None
            plot_angular_distribution_streamlit(psi, rot, tilt, cls_idx)

            if st.checkbox("Show statistics?"):
                interactive_scatter_plot(os.path.join(rln_folder, nodes[0]))
        
        except Exception as e:
            print(e)
            st.write("#### No particles.star file found")

    elif n_inter == 0 and any("mrc" in node for node in nodes):
        st.write("No *_model.star files found in the folder")
        try:
            mrc_idx = [
                index for index, element in enumerate(nodes) if "mrc" in element
            ][0]
            plot_combined_classes_streamlit(
                [os.path.join(rln_folder, nodes[mrc_idx])], [100]
            )
        except Exception as e:
            print(e)
    print(f'{datetime.now()}: plot_cls3d_stats_streamlit done')
    


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized_array = (array - min_value) / (max_value - min_value)
    return normalized_array


def plot_3dclasses(files, conc_axis=0):
    class_averages = []

    if len(files) == 1:
        conc_axis = 1

    for n, class_ in enumerate(files):
        with mrcfile.mmap(class_) as mrc_stack:
            mrcs_file = mrc_stack.data
            z, x, y = mrc_stack.data.shape

            average_top = np.mean(mrcs_file, axis=0)
            try:
                average_top = normalize_array(average_top)
            except:
                pass

            average_front = np.mean(mrcs_file, axis=1)

            try:
                average_front = normalize_array(average_front)

            except:
                pass

            average_side = np.mean(mrcs_file, axis=2)

            try:
                average_side = normalize_array(average_side)
            except:
                pass

            # join 3 views together if joined in 3D classification or Ab Initio, otherwise return projections (for mask)
            if conc_axis == 0:
                try:
                    average_class = np.concatenate(
                        (average_top, average_front, average_side), conc_axis
                    )

                except ValueError:
                    max_size = max(
                        average_top.shape[1],
                        average_front.shape[1],
                        average_side.shape[1],
                    )

                    padded_top = np.pad(
                        average_top, ((0, 0), (0, max_size - average_top.shape[1]))
                    )
                    padded_front = np.pad(
                        average_front, ((0, 0), (0, max_size - average_front.shape[1]))
                    )
                    padded_side = np.pad(
                        average_side, ((0, 0), (0, max_size - average_side.shape[1]))
                    )

                    average_class = np.concatenate(
                        (padded_top, padded_front, padded_side), conc_axis
                    )

                class_averages.append(average_class)
            else:
                class_averages = [average_top, average_front, average_side]

    try:
        if conc_axis == 0:
            final_average = np.concatenate(class_averages, axis=1)
        else:
            final_average = class_averages

    except ValueError:
        final_average = []

    return final_average


def plot_projections_streamlit(class_paths, class_dist_=None, string="Class", cmap="gray"):
    if class_paths is not None and len(class_paths) > 1:
        projections = plot_3dclasses(class_paths)  # Ensure plot_3dclasses is defined

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(projections, cmap=cmap)
        ax.grid(False)

        # Set X-axis ticks and labels
        labels_positions_x = (
            np.linspace(
                1 / len(class_paths) * projections.shape[1],
                projections.shape[1],
                len(class_paths),
            )
            - 0.5 * 1 / len(class_paths) * projections.shape[1]
        )
        labels_x = [f"{string} {x+1}" for x in range(len(class_paths))]

        if class_dist_ is not None:
            class_dist_ = np.array(class_dist_)
            try:
                labels_x = [
                    f"{string} {x+1} ({round(float(class_dist_[:, -1][x]) * 100, 2)}%)"
                    for x in range(len(class_paths))
                ]
            except:
                labels_x = [
                    f"{string} {x+1} ({round(float(class_dist_[x]) * 100, 2)}%)"
                    for x in range(len(class_paths))
                ]

        ax.set_xticks(labels_positions_x)
        ax.set_xticklabels(labels_x)
        ax.xaxis.tick_top()

        # Set Y-axis ticks and labels
        labels_positions_y = (
            np.linspace(1 / 3 * projections.shape[0], projections.shape[0], 3)
            - 0.5 * 1 / 3 * projections.shape[0]
        )
        ax.set_yticks(labels_positions_y)
        ax.set_yticklabels(["Z", "X", "Y"], fontweight="bold")

        st.pyplot(fig)

    else:
        projections = plot_3dclasses(
            class_paths, conc_axis=1
        )  # Ensure plot_3dclasses is defined

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        projection_titles = ["Z Projection", "X Projection", "Y Projection"]

        for idx, ax in enumerate(axes):
            ax.imshow(projections[idx], cmap=cmap)
            ax.set_title(projection_titles[idx], fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

        st.pyplot(fig)


def resize_3d(volume_, new_size=100):
    # Otherwise is read only
    volume_ = volume_.copy()

    if new_size % 2 != 0:
        print("Box size has to be even!")
        return

    original_size = volume_.shape

    # Skip if volume is less than 100
    if original_size[0] <= 100:
        return volume_.copy()

    fft = np.fft.fftn(volume_)
    fft_shift = np.fft.fftshift(fft)

    # crop this part of the fft
    x1, x2 = int((volume_.shape[0] - new_size) / 2), volume_.shape[0] - int(
        (volume_.shape[0] - new_size) / 2
    )

    fft_shift_new = fft_shift[x1:x2, x1:x2, x1:x2]

    # Apply spherical mask
    lx, ly, lz = fft_shift_new.shape
    X, Y, Z = np.ogrid[0:lx, 0:ly, 0:lz]
    dist_from_center = np.sqrt(
        (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 + (Z - lz / 2) ** 2
    )
    mask = dist_from_center <= lx / 2
    fft_shift_new[~mask] = 0

    fft_new = np.fft.ifftshift(fft_shift_new)
    new = np.fft.ifftn(fft_new)

    # Return only real part
    return new.real


def plot_volume(volume, threshold, max_size=150):
    # Resize volume for performance optimization
    volume = volume.copy()
    if np.any(np.array(volume.shape) > max_size):
        resize_factor = max_size / np.max(volume.shape)
        volume = resize(
            volume,
            np.round(np.array(volume.shape) * resize_factor).astype(int),
            anti_aliasing=True,
        )

    # Calculate threshold based on percentile
    min_val, max_val = np.min(volume), np.max(volume)
    actual_threshold = min_val + (max_val - min_val) * threshold

    # Use marching cubes to create a mesh
    try:
        verts, faces, _, _ = marching_cubes(
            volume, level=actual_threshold, spacing=(1, 1, 1)
        )
    except RuntimeError:
        return None  # Return if marching cubes fails

    # Create a Plotly trisurf figure
    fig_volume = ff.create_trisurf(
        x=verts[:, 2],
        y=verts[:, 1],
        z=verts[:, 0],
        plot_edges=False,
        colormap=["rgb(150,150,150)"],
        simplices=faces,
        showbackground=False,
        show_colorbar=False,
    )

    # Customize lighting and light position
    fig_volume.data[0].update(
        lighting=dict(
            ambient=0.5,
            diffuse=0.9,
            fresnel=0.5,
            specular=0.2,
            roughness=0.8,
            facenormalsepsilon=0,
            vertexnormalsepsilon=0,
        ),
        lightposition=dict(x=-100, y=-200, z=-300),
    )

    # Configure the layout and camera
    fig_volume.update_layout(
        title="Overlay of Cα Atoms and Volume Data",
        scene=dict(
            camera=dict(eye=dict(x=3, y=3, z=3)),
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False),  # Hide y-axis
            zaxis=dict(visible=False),  # Hide z-axis
            aspectmode="data",
        ),
        height=600,  # Height of the plot
        margin=dict(l=0, r=0, t=0, b=0),  # Margins
        hovermode=False,
    )  # Disable hover mode

    return fig_volume


def plot_combined_classes_streamlit(class_paths, class_dist):
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(
            "Select Volume Threshold (Fraction)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
        map_resize = st.slider(
            "Map size (px)", min_value=64, max_value=256, value=150, step=2
        )
    with col2:
        n_columns = st.slider(
            "Number of columns", min_value=1, max_value=5, value=3, step=1
        )
        row_height = st.slider(
            "Plot height", min_value=200, max_value=1000, value=400, step=100
        )

    if map_resize > 150:
        st.warning("Larger maps can be very slow!")

    num_classes = len(class_paths)
    cols = min(n_columns, num_classes)
    rows = math.ceil(num_classes / n_columns)
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "scene"}] * cols] * rows)

    # Annotations list to store class labels
    annotations = []
    aspect_ratio = dict(x=1, y=1, z=1)

    for n, cls_path in enumerate(class_paths):
        with mrcfile.open(cls_path, permissive=True) as mrc:
            mrc_cls_data = mrc.data
            fig_ = plot_volume(mrc_cls_data, threshold, max_size=map_resize)

            if fig_ is not None:
                row, col = divmod(n, cols)
                fig.add_trace(fig_["data"][0], row=row + 1, col=col + 1)

                # Calculate the position for the annotation based on row and column
                x = (col + 0.5) / cols
                y = 1 - (row / rows) - 0.05

                cls_dist = round(float(class_dist[n]) * 100, 2)
                # Add annotation for the class number
                annotations.append(
                    dict(
                        x=x,
                        y=y,
                        xref="paper",
                        yref="paper",
                        text=f"Class {n + 1}<br>Distribution {cls_dist}%",
                        showarrow=False,
                        xanchor="center",
                        yanchor="bottom",  # Anchor bottom of text at annotation point
                        font=dict(size=12),
                    )
                )

    # Update layout with annotations
    fig.update_layout(
        hovermode=False,
        annotations=annotations,
        height=rows * row_height,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Update layout for each subplot
    for n in range(num_classes):
        fig.update_layout(
            **{
                f"scene{n + 1}": dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=dict(eye=dict(x=3, y=3, z=3)),
                )
            }
        )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_class_distribution_streamlit(class_dist_, PLOT_HEIGHT=500):
    fig = go.Figure()
    for n, class_ in enumerate(class_dist_):
        class_ = class_.astype(float) * 100
        x = np.arange(0, class_dist_.shape[1])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=class_,
                name=f"Class {n + 1}",
                showlegend=True,
                hovertemplate=f"Class {n + 1}<br>inter: %{{x}}<br>Cls dist: %{{y:.2f}}{'%'}<extra></extra>",
                mode="lines",
                stackgroup="one",
            )
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Class distribution")

    fig.update_layout(title="Class distribution")
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True, height=PLOT_HEIGHT)


def plot_class_resolution_streamlit(class_res_, PLOT_HEIGHT=500):
    fig = go.Figure()
    for n, class_ in enumerate(class_res_):
        class_ = class_.astype(float)
        x = np.arange(0, class_res_.shape[1])

        fig.add_scatter(
            x=x,
            y=class_,
            name="Class {}".format(n + 1),
            showlegend=True,
            hovertemplate=f"Class {n + 1}<br>inter: %{{x}}<br>Cls res:%{{y:.2f}}A<extra></extra>",
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Class Resolution [A]")

    fig.update_layout(title="Class Resolution [A]")
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True, height=PLOT_HEIGHT)


def plot_angular_distribution_streamlit(psi, rot, tilt, cls_data):
    angles_data = pd.DataFrame({"Psi": psi, "Rotation": rot, "Tilt": tilt})

    if cls_data is not None and any(cls_data != None):
        angles_data["Class"] = cls_data
        unique_cls = np.unique(cls_data)
    else:
        unique_cls = [None]

    for cls in unique_cls:
        if cls is not None:
            # Filter data for the current class
            class_data = angles_data[angles_data["Class"] == cls]
        else:
            # Use all data if class is None
            class_data = angles_data

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Psi vs Rotation", "Psi vs Tilt", "Rotation vs Tilt"],
        )

        # Add a subplot for each pair of angles
        angle_pairs = [("Psi", "Rotation"), ("Psi", "Tilt"), ("Rotation", "Tilt")]
        for i, (x_angle, y_angle) in enumerate(angle_pairs, 1):
            fig.add_trace(
                go.Histogram2d(
                    x=class_data[x_angle],
                    y=class_data[y_angle],
                    colorscale="Viridis",
                    nbinsx=40,
                    nbinsy=20,
                ),
                row=1,
                col=i,
            )

        # Update layout for each subplot
        class_label = f" for Class {cls}" if cls is not None else ""
        fig.update_layout(
            title=f"Angular Distribution Heatmap{class_label}",
            height=300,  # Height of the figure in pixels
            width=600,  # Width of the figure in pixels
            showlegend=False,
        )

        # Display in Streamlit with full width
        st.plotly_chart(fig, use_container_width=True)


def plot_mask_streamlit(mask_path):
    # path_data = os.path.join(rln_folder, job_name)
    mask_projections = plot_3dclasses([mask_path], conc_axis=1)
    shortcodes = []

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    projection_titles = ["Z Projection", "X Projection", "Y Projection"]

    for idx, ax in enumerate(axes):
        ax.imshow(mask_projections[idx], cmap="gray")
        ax.set_title(projection_titles[idx], fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)

    view_volume = st.checkbox("View mask?")
    if view_volume:
        st.plotly_chart(plot_volume(mrcfile.mmap(mask_path).data, 0.5))

    print(f'{datetime.now()}: plot_mask_streamlit done')
    

def normalize_and_scale(data):
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return (data_normalized * 255).astype(np.uint8)


def plot_locres_streamlit(node_files, FOLDER, job_name):
    RELION4 = True

    # Relion4
    # nodes 0 PDF, 1 localresfiltered, 2 localresmap

    # Relion3
    # nodes 0 localresfiltered, 1 localresmap, 2 flowchart?

    # Relion4 has PDF as node[0]
    if "pdf" not in node_files[0]:
        RELION4 = False

    # localresfiltered = mrcfile.mmap(os.path.join(FOLDER, node_files[1])).data
    if RELION4:
        localresmap = mrcfile.mmap(os.path.join(FOLDER, node_files[2])).data

    else:
        localresmap = mrcfile.mmap(os.path.join(FOLDER, node_files[1])).data

    data_shape = localresmap.shape

    note = get_note(os.path.join(FOLDER, job_name, "note.txt"))

    # get mask
    try:
        mask_path = re.search(r"--mask\s([\w\d/]+\.mrc)", note).group(1)
    except:
        mask_path = ""

    if mask_path != "":
        mask = mrcfile.open(os.path.join(FOLDER, "", mask_path)).data.copy()
        mask[mask > 0] = 1
        localresmap = localresmap.copy() * mask

    # Streamlit widget to choose between Seaborn and Plotly
    plot_choice = st.radio("Choose plotting library", ["Seaborn", "Plotly"])

    # Plotting using selected library
    if plot_choice == "Seaborn":
        # Seaborn plotting logic
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), dpi=150)

        axes[0].imshow(localresmap[int(data_shape[0] / 2), :, :])
        cbar0 = fig.colorbar(mappable=axes[0].images[0], ax=axes[0])
        cbar0.set_label("Local Resolution, Å")
        axes[0].set_title("Slice Z")
        axes[0].axis("off")

        axes[1].imshow(localresmap[:, int(data_shape[0] / 2)])
        cbar1 = fig.colorbar(mappable=axes[1].images[0], ax=axes[1])
        cbar1.set_label("Local Resolution, Å")
        axes[1].set_title("Slice X")
        axes[1].axis("off")

        axes[2].imshow(localresmap[:, :, int(data_shape[0] / 2)])
        cbar2 = fig.colorbar(mappable=axes[2].images[0], ax=axes[2])
        cbar2.set_label("Local Resolution, Å")
        axes[2].set_title("Slice Y")
        axes[2].axis("off")

        plt.tight_layout()

        st.pyplot(fig)

    else:
        # Calculate slice indices
        z_slice_idx = int(data_shape[0] / 2)
        x_slice_idx = int(data_shape[1] / 2)
        y_slice_idx = int(data_shape[2] / 2)

        # Create subplots for each slice
        fig = make_subplots(
            rows=1, cols=3, subplot_titles=["Slice Z", "Slice X", "Slice Y"]
        )

        # Add heatmap for each slice
        fig.add_trace(
            go.Heatmap(z=localresmap[z_slice_idx, :, :], colorscale="Viridis"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(z=localresmap[:, x_slice_idx, :], colorscale="Viridis"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Heatmap(z=localresmap[:, :, y_slice_idx], colorscale="Viridis"),
            row=1,
            col=3,
        )

        fig.update_layout(title_text="Local Resolution Slices")

        # Update axes to maintain aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=3)

        # Display in Streamlit with full width
        st.plotly_chart(fig, use_container_width=True)

    data = localresmap.flatten()
    data = data[data != 0]

    if plot_choice == "Seaborn":
        # Seaborn histogram
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.histplot(data, kde=False, bins=100, ax=ax)
        ax.set_xlabel("Resolution [Å]", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Histogram of local resolution", fontweight="bold")
        st.pyplot(fig)
    else:
        # Plotly histogram
        fig = go.Figure(data=[go.Histogram(x=data, nbinsx=100)])
        fig.update_layout(
            title="Histogram of local resolution",
            xaxis_title="Resolution [Å]",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)
    
    print(f'{datetime.now()}: plot_locres_streamlit done')
    


def plot_ctf_refine_streamlit(node_files, FOLDER, job_name):
    ctf_data = glob.glob(os.path.join(FOLDER, job_name, "*.mrc"))
    n_images = len(ctf_data)

    # Streamlit widget to choose between Seaborn and Plotly
    #plot_choice = st.selectbox("Choose plotting library", ["Seaborn", "Plotly"])
    plot_choice = "Seaborn"

    if n_images == 0:
        note = get_note(os.path.join(FOLDER, "", job_name, "note.txt"))

        # get last refine job
        try:
            refine_path = re.search(r"--i\s([\w\d/]+\.star)", note).group(1)

            # If CtfRefine is used as in tutorial, then 3 runs are done. The last one should have Refine3D path
            if "CtfRefine" in refine_path:
                refine_path = refine_path.split("/")

                # search another job
                note = get_note(
                    os.path.join(FOLDER, refine_path[0], refine_path[1], "note.txt")
                )

                refine_path = re.search(r"--i\s([\w\d/]+\.star)", note).group(1)

                if "CtfRefine" in refine_path:
                    refine_path = refine_path.split("/")
                    note = get_note(
                        os.path.join(FOLDER, refine_path[0], refine_path[1], "note.txt")
                    )

                    refine_path = re.search(r"--i\s([\w\d/]+\.star)", note).group(1)

                    if "Refine3D" in refine_path:
                        refine_path = refine_path
                else:
                    refine_path = ""

        except Exception as e:
            refine_path = ""
            print(e)

        if refine_path != "":
            refine_data = parse_star(os.path.join(FOLDER, refine_path))[
                "particles"
            ]
            refine_data_U = refine_data["_rlnDefocusU"].values.astype(float)
            refine_data_V = refine_data["_rlnDefocusV"].values.astype(float)

            # node 1 is ctf refined star
            ctf_refine_data = parse_star(os.path.join(FOLDER, node_files[1]))[
                "particles"
            ]
            ctf_refine_data_U = ctf_refine_data["_rlnDefocusU"].values.astype(float)
            ctf_refine_data_V = ctf_refine_data["_rlnDefocusV"].values.astype(float)

            ctf_U = refine_data_U - ctf_refine_data_U
            ctf_V = refine_data_V - ctf_refine_data_V

        if plot_choice == "Seaborn":
            # Seaborn plotting logic
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100)
            sns.histplot(ctf_U, bins=100, ax=axes[0], color="tab:blue")
            axes[0].set_title("DefocusU Change")
            axes[0].set_ylabel("Count")
            axes[0].set_xlabel("Defocus change, Å")

            sns.histplot(ctf_V, bins=100, ax=axes[1], color="tab:orange")
            axes[1].set_title("DefocusV Change")
            axes[1].set_ylabel("Count")
            axes[1].set_xlabel("Defocus change, Å")

            h = axes[2].hist2d(ctf_U, ctf_V, bins=100, cmap="viridis")
            axes[2].set_title("DefocusU/V Change")
            axes[2].set_xlabel("DefocusU change, Å")
            axes[2].set_ylabel("DefocusV change, Å")
            cb = plt.colorbar(h[3], ax=axes[2])
            cb.set_label("Count")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            # Plotly plotting logic
            fig = make_subplots(rows=1, cols=3)
            fig.add_trace(go.Histogram(x=ctf_U, nbinsx=100), row=1, col=1)
            fig.add_trace(go.Histogram(x=ctf_V, nbinsx=100), row=1, col=2)
            fig.add_trace(
                go.Histogram2d(x=ctf_U, y=ctf_V, nbinsx=100, nbinsy=100), row=1, col=3
            )
            fig.update_layout(title="CTF Refine Changes", showlegend=False)
            st.plotly_chart(fig)
    else:
        param_names = [os.path.basename(file).replace(".mrc", "") for file in ctf_data]
        ctf_images = [mrcfile.open(img).data for img in ctf_data]
        display_ctf_stats(ctf_images, param_names, plot_choice)
    print(f'{datetime.now()}: plot_ctf_refine_streamlit done')


def display_ctf_stats(
    images,
    file_names,
    plot_choice,
    columns=2,
    width=8,
    height=5,
    label_wrap_length=50,
    label_font_size=8,
    label=True,
):
    max_images = len(images)
    num_rows = math.ceil(max_images / columns)
    height = max(height, num_rows * height)

    if plot_choice == "Seaborn":
        # Seaborn plotting logic
        fig, axes = plt.subplots(nrows=num_rows, ncols=columns, figsize=(width, height))
        sns.set_style("whitegrid")

        for i, image in enumerate(images):
            ax = axes[i // columns, i % columns]
            ax.imshow(image, cmap="magma")

            if label:
                title = textwrap.shorten(
                    file_names[i], width=label_wrap_length, placeholder="..."
                )
                ax.set_title(title, fontsize=label_font_size)
            ax.axis("off")

        # Hide any empty subplots
        for i in range(len(images), num_rows * columns):
            axes[i // columns, i % columns].axis("off")

        fig.tight_layout()
        st.pyplot(fig)

    else:
        # Plotly plotting logic
        fig = make_subplots(rows=num_rows, cols=columns)
        annotations = []  # Initialize a list for annotations

        for i, image in enumerate(images):
            row = (i // columns) + 1
            col = (i % columns) + 1
            fig.add_trace(
                go.Heatmap(z=image, colorscale="magma"),
                row=(i // columns) + 1,
                col=(i % columns) + 1,
            )

            if label:
                title = textwrap.shorten(
                    file_names[i], width=label_wrap_length, placeholder="..."
                )
                # Append each annotation to the list
                annotations.append(
                    go.layout.Annotation(
                        text=title,
                        xref="paper",
                        yref="paper",
                        x=(i % columns) / columns,
                        y=1 - (i // columns) / num_rows,
                        showarrow=False,
                        font=dict(size=label_font_size),
                        align="center",
                    )
                )

            # Set aspect ratio for each subplot
            fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
            fig.update_yaxes(scaleanchor="x", scaleratio=1, row=row, col=col)

        fig.update_layout(
            height=height * 100, width=width * 100, margin=dict(l=0, r=0, t=30, b=0)
        )
        fig.layout.annotations = (
            annotations  # Assign the list to fig.layout.annotations
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_polish_streamlit(node_files, FOLDER, job_name):
    if any("opt_params_all_groups.txt" in element for element in node_files):
        with open(os.path.join(FOLDER, node_files[0]), "r") as f:
            parameters = f.readline().strip().split(" ")
            st.write("### Optimal parameters")
            st.code(
                f"--s_vel {parameters[0]} --s_div {parameters[1]} --s_acc {parameters[2]}"
            )

    elif any("shiny.star" in element for element in node_files):
        try:
            bfactors_star_path = os.path.join(FOLDER, job_name, "bfactors.star")
            bfactors_data = parse_star(bfactors_star_path)["perframe_bfactors"]

            # Plotly plot setup
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bfactors_data["_rlnMovieFrameNumber"],
                    y=bfactors_data["_rlnBfactorUsedForSharpening"],
                    mode="lines",
                    name="Bfactor Used For Sharpening",
                    line=dict(color="darkgoldenrod"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bfactors_data["_rlnMovieFrameNumber"],
                    y=bfactors_data["_rlnFittedInterceptGuinierPlot"],
                    mode="lines",
                    name="Fitted Intercept Guinier Plot",
                    line=dict(color="lightseagreen"),
                    yaxis="y2",
                )
            )

            # Set up layout
            fig.update_layout(
                title="Polish Job Statistics",
                xaxis_title="Movie Frame Number",
                yaxis_title="Bfactor Used For Sharpening",
                yaxis2=dict(
                    title="Fitted Intercept Guinier Plot", overlaying="y", side="right"
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading B-factors data: {e}")

    else:
        st.write("No relevant data found for the Polish job.")

    print(f'{datetime.now()}: plot_polish_streamlit done')
    

def plot_postprocess_streamlit(nodes, rln_folder):
    postprocess_star_path = os.path.join(rln_folder, nodes[3])

    if not os.path.exists(postprocess_star_path):
        st.write("No postprocess.star file found")
        return

    plot_choice = st.selectbox("Choose plotting library", ["Seaborn", "Plotly"])

    # Read PostProcess star file data
    postprocess_star_data = parse_star(postprocess_star_path)
    fsc_data = postprocess_star_data["fsc"].astype(float)
    guinier_data = postprocess_star_data["guinier"].astype(float)

    # Plot FSC Curve
    plot_fsc_curve(fsc_data, plot_choice)

    # Plot Guinier Curve
    plot_guinier_curve(guinier_data, plot_choice)

    # Display Masked Volume Slices
    display_volume_slices(rln_folder, nodes)
    
    print(f'{datetime.now()}: plot_postprocess_streamlit done')
    


def plot_fsc_curve(fsc_data, plot_choice, plot_relevant=False, dpi=150):
    fsc_x = fsc_data["_rlnAngstromResolution"].astype(float)
    fsc_x_min, fsc_x_max = np.min(fsc_x), np.max(fsc_x)

    if not plot_relevant:
        fsc_to_plot = [
            "_rlnFourierShellCorrelationCorrected",
            "_rlnFourierShellCorrelationUnmaskedMaps",
            "_rlnFourierShellCorrelationMaskedMaps",
            "_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps",
        ]
    else:
        fsc_to_plot = [
            "_rlnFourierShellCorrelationUnmaskedMaps",
            "_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps",
        ]

    if plot_choice == "Seaborn":
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        for meta in fsc_to_plot:
            ax.plot(
                1 / fsc_x, fsc_data[meta].astype(float), label=meta.replace("_rlnFourierShellCorrelation", "").replace("_rlnCorrectedFourierShellCorrelation",'')
            )

        ax.set_xlabel("Resolution, Å", fontsize=18)
        ax.set_ylabel("FSC", fontsize=18)
        ax.tick_params(axis="both", labelsize=12)

        # Set custom ticks based on fsc_x range
        # Start from low resolution (~50 Å) to high resolution (min value from fsc_x)
        start_res = 1 / 50  # Starting resolution (low resolution)
        end_res = 1 / fsc_x_min  # Ending resolution (high resolution)
        custom_ticks = np.linspace(
            start_res, end_res, num=10
        )  # Generate 10 ticks between start and end
        ax.set_xticks(custom_ticks)  # Convert resolutions to frequency for x-axis
        ax.set_xticklabels([f"{round(1/res, 2)}" for res in custom_ticks])

        ax.axhline(0.5, linestyle="--", c="black")
        ax.axhline(0.143, linestyle="--", c="black")
        ax.annotate("0.143", (start_res, 0.16))
        ax.annotate("0.5", (start_res, 0.52))
        ax.tick_params(axis="both", labelsize=12)

        ax.set_xlim(start_res, end_res)
        ax.set_ylim(-0.05, 1.05)

        ax.legend(loc="upper right", fontsize=8)
        sns.despine()

        st.pyplot(fig)

    else:
        fig = go.Figure()

        for meta in fsc_to_plot:
            reciprocal_x = 1 / fsc_x
            fig.add_trace(
                go.Scatter(
                    x=reciprocal_x,
                    y=fsc_data[meta].astype(float),
                    mode="lines",
                    customdata=fsc_x,
                    name=meta.replace("_rlnFourierShellCorrelation", "").replace("_rlnCorrectedFourierShellCorrelation",''),
                    hovertemplate = 'Resolution: %{customdata:.2f} Å<br>FSC: %{y}',
                )
            )

        # Add horizontal lines for FSC thresholds
        fig.add_hline(
            y=0.143, line_dash="dash", line_color="black", annotation_text="0.143"
        )
        fig.add_hline(
            y=0.5, line_dash="dash", line_color="black", annotation_text="0.5"
        )

        # Custom ticks
        start_res = 1 / 50  # Starting resolution (low resolution)
        end_res = 1 / fsc_x_min  # Ending resolution (high resolution)
        custom_ticks = np.linspace(start_res, end_res, num=10)
        custom_tickvals = custom_ticks
        custom_ticktext = [f"{round(1/res, 2)}" for res in custom_ticks]

        fig.update_layout(
            xaxis=dict(
                title="Resolution, Å",
                tickvals=custom_tickvals,
                ticktext=custom_ticktext,
                range=[start_res, end_res],
            ),
            yaxis=dict(title="FSC", range=[-0.05, 1.05]),
            title="Fourier Shell Correlation Curve",
            legend=dict(x=1, y=1, bgcolor="rgba(255,255,255,0)"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig)

    fsc143 = 0.143
    resolution_143_idx = np.argmin(
        np.abs(fsc_data["_rlnFourierShellCorrelationCorrected"].astype(float) - fsc143)
    )

    fsc05 = 0.5
    resolution_05_idx = np.argmin(
        np.abs(fsc_data["_rlnFourierShellCorrelationCorrected"].astype(float) - fsc05)
    )

    st.markdown(
        f"Reported resolution @ FSC=0.143: __{round(fsc_x[resolution_143_idx],2)} Å__"
    )
    st.markdown(
        f"Reported resolution @ FSC=0.5: __{round(fsc_x[resolution_05_idx],2)} Å__"
    )


def plot_guinier_curve(guinier_data, plot_choice):
    guiner_x = guinier_data["_rlnResolutionSquared"].astype(float)
    guinier_to_plot = [
        "_rlnLogAmplitudesOriginal",
        "_rlnLogAmplitudesMTFCorrected",
        "_rlnLogAmplitudesWeighted",
        "_rlnLogAmplitudesSharpened",
        "_rlnLogAmplitudesIntercept",
    ]

    if plot_choice == "Seaborn":
        sns.set(style="whitegrid", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(10, 6))

        for meta in guinier_to_plot:
            y_data = guinier_data[meta].astype(float)
            y_data[y_data == -99] = float("nan")
            ax.plot(guiner_x, y_data, label=meta.replace("_rlnLogAmplitudes", ""))

        ax.set_xlabel("Resolution Squared, [1/Å²]")
        ax.set_ylabel("Ln(Amplitudes)")
        ax.set_title("Guinier plot")
        ax.legend(loc="upper right", fontsize=8)

        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        fig = go.Figure()
        for meta in guinier_to_plot:
            y_data = guinier_data[meta].astype(float)
            y_data[y_data == -99] = float("nan")
            fig.add_trace(
                go.Scatter(
                    x=guiner_x, y=y_data, mode="lines", name=meta.replace("_rlnLogAmplitudes", "")
                )
            )

        fig.update_layout(
            title="Guinier plot",
            xaxis_title="Resolution Squared, [1/Å²]",
            yaxis_title="Ln(Amplitudes)",
        )
        st.plotly_chart(fig)


def display_volume_slices(rln_folder, nodes):
    map_path = os.path.join(rln_folder, nodes[1])
    with mrcfile.mmap(map_path) as mrc:
        map_data = mrc.data

    mode = st.radio("Viewing mode:", ["Slice", "Max project", "View volume"])

    if mode == "Max project":
        max_project_axis = st.radio("Projection axis", [0, 1, 2])
        projection = np.max(map_data, axis=max_project_axis)
        st.image(normalize(projection), use_column_width="always")
    elif mode == "View volume":
        threshold = st.slider(
            "Select Volume Threshold (Fraction)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
        fig = plot_volume(map_data, threshold)
        fig.update_scenes(aspectmode="data")
        st.plotly_chart(fig)
    else:
        z_slider = st.slider(
            "Z slide", 0, map_data.shape[0] - 1, int(map_data.shape[0] // 2)
        )
        st.image(normalize(map_data[z_slider]), use_column_width="always")


def extract_v_parameter(text):
    # Regular expression pattern to match the -v parameter and capture its value
    pattern = r"-v\s+([^\s]+)"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # Check if a match is found
    if match:
        # Return the captured group, which is the value of the -v parameter
        return match.group(1)
    else:
        # Return None if no match is found
        return None


def plot_pdb_streamlit(nodes, rln_folder, job_name):
    cif_path = os.path.join(rln_folder, nodes[0])
    job_folder = os.path.join(rln_folder, job_name)
    # threshold = st.slider("Select Volume Threshold (Fraction)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    show_volume = st.checkbox("Show with volume? (Careful, will be very slow)")

    if show_volume:
        volume_path = extract_v_parameter(
            get_note(os.path.join(job_folder, "note.txt"))
        )
        volume = mrcfile.mmap(os.path.join(rln_folder, volume_path))
        px_size = volume.voxel_size["x"]
        volume = volume.data

    ca_atoms, sequence = read_cif_file(cif_path)

    if show_volume:
        for atom in ca_atoms:
            atom["x"] /= px_size
            atom["y"] /= px_size
            atom["z"] /= px_size

        st.plotly_chart(
            overlay_ca_atoms_and_volume(ca_atoms, volume, 0.5, volume.shape[0])
        )

    else:
        fig = plot_ca_atoms(ca_atoms)
        st.plotly_chart(fig)

    for chain in sequence.keys():
        st.text_area(f"Sequence: chain {chain}", sequence[chain])

    print(f'{datetime.now()}: plot_pdb_streamlit done')
    

def three_to_one(residue):
    """Convert three-letter amino acid codes to one-letter codes"""
    conversion = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    return conversion.get(residue, "?")


def read_cif_file(cif_path):
    doc = cif.read_file(cif_path)
    block = doc.sole_block()

    ca_atoms = []
    sequences = {}

    for row in block.find(
        [
            "_atom_site.group_PDB",
            "_atom_site.label_atom_id",
            "_atom_site.label_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.label_asym_id",
            "_atom_site.label_comp_id",
        ]
    ):
        if row[1] == "CA":  # Only consider CA atoms
            ca_atoms.append(
                {
                    "atom": row[1],
                    "seq_id": row[2],
                    "x": float(row[3]),
                    "y": float(row[4]),
                    "z": float(row[5]),
                    "chain": row[6],
                    "residue": row[7],
                }
            )
            chain = row[6]
            residue = row[7]

            if chain not in sequences:
                sequences[chain] = []
            sequences[chain].append(residue)

    # Convert the sequences to one-letter code
    for chain in sequences:
        sequences[chain] = "".join([three_to_one(res) for res in sequences[chain]])

    return ca_atoms, sequences


def plot_ca_atoms(ca_atoms):
    """Plot Cα atoms in 3D using Plotly with lines and markers."""
    fig = go.Figure()
    ca_atoms = pd.DataFrame(ca_atoms)

    # Check if 'chain' column exists
    if "chain" not in ca_atoms.columns:
        raise ValueError("The 'chain' column is missing from the DataFrame.")

    # Ensure that x, y, z columns exist
    for coord in ["x", "y", "z"]:
        if coord not in ca_atoms.columns:
            raise ValueError(f"The '{coord}' column is missing from the DataFrame.")

    # Iterate over each unique chain
    for chain in ca_atoms["chain"].unique():
        chain_data = ca_atoms[ca_atoms["chain"] == chain]

        fig.add_trace(
            go.Scatter3d(
                x=chain_data["x"],
                y=chain_data["y"],
                z=chain_data["z"],
                mode="lines",
                marker=dict(size=5, opacity=0.8),
                line=dict(width=2),
                name=f"Chain {chain}",
            )
        )

    fig.update_layout(
        title="3D Visualization of Cα Atoms",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
        height=600,
        hovermode=False,
    )
    fig.update_scenes(aspectmode="data")
    return fig


def overlay_ca_atoms_and_volume(ca_atoms, volume, threshold, max_size=150):
    # First, generate the plot for Cα atoms
    ca_atoms_fig = plot_ca_atoms(ca_atoms)

    # Then, generate the volume plot
    volume_fig = plot_volume(volume, threshold, max_size)

    # Check if volume_fig is not None
    if volume_fig is not None and "data" in volume_fig:
        # Extract the trace from the volume figure
        volume_trace = volume_fig["data"][0]

        # Set the transparency of the volume trace
        volume_trace.opacity = 0.3  # 90% transparent

        # Add the volume trace to the Cα atoms figure
        ca_atoms_fig.add_trace(volume_trace)

    # Set the layout properties
    ca_atoms_fig.update_layout(
        title="Overlay of Cα Atoms and Volume Data",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
        height=600,
        hovermode=False,  # Adjust height if needed
    )
    ca_atoms_fig.update_scenes(aspectmode="data")

    return ca_atoms_fig


def plot_selection_streamlit(node_files, FOLDER, job_name):
    # For 2D class selection

    if any("class_averages.star" in element for element in node_files):
        # Selected classes star
        selected_particles_star_path = node_files[0]
        selected_class_star_path = os.path.join(FOLDER, selected_particles_star_path)
        particles_selected = parse_star(selected_class_star_path)["particles"]
        num_particles_selected = particles_selected.shape[0]

        # Get note to get the original star file
        note = get_note(os.path.join(FOLDER, job_name, "note.txt"))
        source_job = extract_source_job(note)

        # Get the original star file
        particles_source = parse_star(os.path.join(FOLDER, source_job))[
            "particles"
        ]
        num_particles_source = particles_source.shape[0]

        # Preparing the DataFrame
        df = pd.DataFrame(
            {
                "status": ["Selected", "Rejected"],
                "count": [
                    num_particles_selected,
                    num_particles_source - num_particles_selected,
                ],
            }
        )

        # Creating a pie chart
        fig = px.pie(
            df,
            values="count",
            names="status",
            title="Selected vs Rejected Particles",
            hole=0.3,  # Add a hole in the center for a donut-like appearance
            labels={"status": "Status", "count": "Count"},
        )

        # Customizing the hover data
        fig.update_traces(hoverinfo="label+percent+value", textinfo="percent+value")

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # Get the selected class averages
        try:
            cls_averages_meta_path = node_files[1]
            cls_averages_data = list(parse_star(os.path.join(FOLDER, cls_averages_meta_path)).values())[0]

            cls_images = []
            selected_classes = []
            for cls_ in cls_averages_data["_rlnReferenceImage"]:
                cls_data = cls_.split("@")
                cls_images.append(
                    mrcfile.mmap(os.path.join(FOLDER, cls_data[1])).data[
                        int(cls_data[0]) - 1
                    ]
                )
                selected_classes.append(int(cls_data[0]))

            cls_images = np.stack(cls_images)
            cls_distribution = cls_averages_data["_rlnClassDistribution"]
            cls_res = cls_averages_data["_rlnEstimatedResolution"]

            display_classes(cls_images, cls_distribution, sort_by_distribution=True, raw_data_star=selected_particles_star_path)

            # Show selected and rejected particles on micrographs

            unique_mics = np.unique(particles_selected["_rlnMicrographName"])
            file_idx = st.slider("Micrograph index", 0, len(unique_mics), 0)
            st.write(f"Particles: :green[selected] :red[rejected]")

            file = os.path.join(FOLDER, unique_mics[file_idx])
            coords_sel = particles_selected[
                particles_selected["_rlnMicrographName"] == unique_mics[file_idx]
            ][["_rlnCoordinateX", "_rlnCoordinateY"]]
            all_coords = particles_source[
                particles_source["_rlnMicrographName"] == unique_mics[file_idx]
            ][["_rlnCoordinateX", "_rlnCoordinateY"]]

            merged_df = pd.merge(
                all_coords,
                coords_sel,
                on=["_rlnCoordinateX", "_rlnCoordinateY"],
                how="outer",
                indicator=True,
            )

            # Filter to keep rows where the merge indicator is 'left_only'
            coords_rej = merged_df[merged_df["_merge"] == "left_only"][
                ["_rlnCoordinateX", "_rlnCoordinateY"]
            ]

            fig = plot_micrograph_picks(file, file_idx, coords_sel, coords_rej)
            st.pyplot(fig)
        except Exception as e:
            print(f'Selection error: {e}')

    elif not any("class_averages.star" in element for element in node_files) and any(
        "split" in element for element in node_files
    ):
        st.subheader("**Split job generated files:**")
        st.write(node_files)

    elif (
        any("particles.star" in element for element in node_files)
        and len(node_files) == 1
    ):
        note = get_note(os.path.join(FOLDER, job_name, "note.txt"))

        try:
            # standard select 3D
            source_job = (
                re.search(r"--i\s([\w\d/]+\.star)", note)
                .group(1)
                .replace("optimiser", "data")
            )

        except:
            return [f"No Data star found for job {job_name}"]

        particles_selected = parse_star(os.path.join(FOLDER, node_files[0]))[
            "particles"
        ]
        particles_source = parse_star(os.path.join(FOLDER, source_job))[
            "particles"
        ]

        num_particles_selected = particles_selected.shape[0]
        num_particles_source = particles_source.shape[0]

        # Preparing the DataFrame
        df = pd.DataFrame(
            {
                "status": ["Selected", "Rejected"],
                "count": [
                    num_particles_selected,
                    num_particles_source - num_particles_selected,
                ],
            }
        )

        # Creating a pie chart
        fig = px.pie(
            df,
            values="count",
            names="status",
            title="Selected vs Rejected Particles",
            hole=0.3,  # Add a hole in the center for a donut-like appearance
            labels={"status": "Status", "count": "Count"},
        )

        # Customizing the hover data
        fig.update_traces(hoverinfo="label+percent+value", textinfo="percent+value")

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        try:
            ### here micrographs
            selected_classes = np.unique(particles_selected["_rlnClassNumber"]).astype(int)

            source_job_folder = source_job.split("/")
            model_files = glob.glob(
                os.path.join(
                    FOLDER, source_job_folder[0], source_job_folder[1], "*model.star"
                )
            )
            model_files.sort(key=os.path.getmtime)
            last_model_star_path = model_files[-1]

            parent_star = parse_star(os.path.join(FOLDER, last_model_star_path))[
                "model_classes"
            ]

            mrcs_paths = parent_star["_rlnReferenceImage"]
            class_dist = parent_star["_rlnClassDistribution"]

            selected_rows = mrcs_paths.str.contains(
                "|".join(f"class{class_num:03d}" for class_num in selected_classes)
            )

            selected_classes_volumes = mrcs_paths[selected_rows]
            class_dist = class_dist[selected_rows].tolist()

            class_paths = []
            for n, class_name in enumerate(selected_classes_volumes.values):
                class_name = os.path.join(FOLDER, class_name)

                # Insert only new classes, in 2D only single file
                if class_name not in class_paths:
                    class_paths.append(class_name)

            plot_combined_classes_streamlit(class_paths, class_dist)

            SPA = True

            if not "_rlnTomoName" in particles_source.columns:
                SPA = True

            else:
                SPA = False

            if SPA:
                # Show selected and rejected particles on micrographs
                unique_mics = np.unique(particles_selected["_rlnMicrographName"])
                file_idx = st.slider("Micrograph index", 0, len(unique_mics), 0)
                st.write(f"Particles: :green[selected] :red[rejected]")

                file = os.path.join(FOLDER, unique_mics[file_idx])
                coords_sel = particles_selected[
                    particles_selected["_rlnMicrographName"] == unique_mics[file_idx]
                ][["_rlnCoordinateX", "_rlnCoordinateY"]]
                all_coords = particles_source[
                    particles_source["_rlnMicrographName"] == unique_mics[file_idx]
                ][["_rlnCoordinateX", "_rlnCoordinateY"]]

                merged_df = pd.merge(
                    all_coords,
                    coords_sel,
                    on=["_rlnCoordinateX", "_rlnCoordinateY"],
                    how="outer",
                    indicator=True,
                )

                # Filter to keep rows where the merge indicator is 'left_only'
                coords_rej = merged_df[merged_df["_merge"] == "left_only"][
                    ["_rlnCoordinateX", "_rlnCoordinateY"]
                ]

                fig = plot_micrograph_picks(file, file_idx, coords_sel, coords_rej)
                st.pyplot(fig)

            else:
                # Show selected and rejected particles on micrographs
                unique_mics = np.unique(particles_selected["_rlnTomoName"])
                file_idx = st.slider("Tomogram index", 0, len(unique_mics), 0)

                file = os.path.join(FOLDER, unique_mics[file_idx])
                coords_sel = particles_selected[
                    particles_selected["_rlnTomoName"] == unique_mics[file_idx]
                ][["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]]
                all_coords = particles_source[
                    particles_source["_rlnTomoName"] == unique_mics[file_idx]
                ][["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]]

                merged_df = pd.merge(
                    all_coords,
                    coords_sel,
                    on=["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"],
                    how="outer",
                    indicator=True,
                )

                # Filter to keep rows where the merge indicator is 'left_only'
                coords_rej = merged_df[merged_df["_merge"] == "left_only"][
                    ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]
                ]

                fig = plot_tomogram_picks(file, file_idx, coords_sel, coords_rej)
                st.plotly_chart(fig)
        except Exception as e:
            st.write('No other files found')
            print(e)
    
    print(f'{datetime.now()}: plot_select_streamlit done')
    


def extract_source_job(note):
    try:
        if "--i" in note:
            return (
                re.search(r"--i\s([\w\d/]+\.star)", note)
                .group(1)
                .replace("optimiser", "data")
            )
        elif "--opt" in note:
            return (
                re.search(r"--opt\s([\w\d/]+\.star)", note)
                .group(1)
                .replace("optimiser", "data")
            )
    except Exception as e:
        return ""


def plot_tomogram_picks(file, n, coods_sel, coords_rej, scatter_size=[5, 2]):
    # micrograph = mrcfile.mmap(file, permissive=True).data
    coords_sel_x = coods_sel["_rlnCoordinateX"]
    coords_sel_y = coods_sel["_rlnCoordinateY"]
    coords_sel_z = coods_sel["_rlnCoordinateZ"]

    coords_rej_x = coords_rej["_rlnCoordinateX"]
    coords_rej_y = coords_rej["_rlnCoordinateY"]
    coords_rej_z = coords_rej["_rlnCoordinateZ"]

    # Creating the figure
    fig = go.Figure()

    # Adding the selected coordinates in green
    fig.add_trace(
        go.Scatter3d(
            x=coords_sel_x,
            y=coords_sel_y,
            z=coords_sel_z,
            mode="markers",
            marker=dict(
                size=scatter_size[0],
                color="green",  # Green for selected coordinates
                opacity=0.8,
            ),
            name="Selected Points",
        )
    )

    # Adding the rejected coordinates in red
    fig.add_trace(
        go.Scatter3d(
            x=coords_rej_x,
            y=coords_rej_y,
            z=coords_rej_z,
            mode="markers",
            marker=dict(
                size=scatter_size[1],
                color="red",  # Red for rejected coordinates
                opacity=0.8,
            ),
            name="Rejected Points",
        )
    )

    # Updating the layout of the figure
    fig.update_layout(
        title="3D Scatter Plot of Coordinates",
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis"),
        legend_title="Points",
        hovermode="closest",
        height=600,
    )
    fig.update_scenes(aspectmode="data")

    return fig


def plot_micrograph_picks(file, n, coods_sel, coords_rej):
    micrograph = mrcfile.mmap(file, permissive=True).data
    coords_sel_x = coods_sel["_rlnCoordinateX"]
    coords_sel_y = coods_sel["_rlnCoordinateY"]
    coords_rej_x = coords_rej["_rlnCoordinateX"]
    coords_rej_y = coords_rej["_rlnCoordinateY"]

    if micrograph.shape[0] > 500:
        img_resize_fac = 500 / micrograph.shape[0]
        mic_red = rescale(micrograph.astype(float), img_resize_fac)
    else:
        img_resize_fac = 1
        mic_red = micrograph.astype(float)
    p1, p2 = np.percentile(mic_red, (0.1, 99.8))
    mic_red = np.array(exposure.rescale_intensity(mic_red, in_range=(p1, p2)))

    sns.set_style("white")
    fig = plt.figure(dpi=200, frameon=False)
    ax = fig.add_axes([0., 0., 1., 1.], xticks=[], yticks=[], frame_on=False)

    ax.imshow(mic_red, cmap="gray")
    ax.axis("off")

    marker_style = dict(
        linestyle="-", marker="o", s=120, facecolors="none", linewidths=1
    )

    ax.scatter(
        coords_sel_x.astype(float) * img_resize_fac,
        coords_sel_y.astype(float) * img_resize_fac,
        edgecolor="limegreen",
        **marker_style,
    )

    ax.scatter(
        coords_rej_x.astype(float) * img_resize_fac,
        coords_rej_y.astype(float) * img_resize_fac,
        edgecolor="#b14e80",
        **marker_style,
    )

    return fig


def plot_pseudosubtomo_streamlit(rlnFOLDER, node_files, job_name):

    if any('particles.star' in element for element in node_files):
        particles_path = os.path.join(rlnFOLDER, node_files[1])
        star = parse_star(particles_path)['particles']
        st.write(f'Generated Pseudosubtomo particles: {star.shape[0]}')

        selection = np.random.randint(0, star['_rlnImageName'].shape[0], 10)
        pseudotomo_data_names = star['_rlnImageName'][selection]
        pseudotomo_ctf_names = star['_rlnCtfImage'][selection]

        data_paths = [os.path.join(rlnFOLDER, name)
                      for name in pseudotomo_data_names]
        ctf_paths = [os.path.join(rlnFOLDER, name)
                     for name in pseudotomo_ctf_names]

        plot_projections_streamlit(data_paths, class_dist_=None, string="Class", cmap="gray")
        plot_projections_streamlit(ctf_paths, class_dist_=None, string="Class", cmap="gray")
        
        interactive_scatter_plot(particles_path)
    
    print(f'{datetime.now()}: plot_pseudosubtomo_streamlit done')
    

def create_network(pipeline_star):
    
    job_edges = pipeline_star['pipeline_input_edges']
    # # Extract the folder names from the _rlnPipeLineEdgeFromNode column
    job_edges["_rlnPipeLineEdgeFromNode"] = job_edges["_rlnPipeLineEdgeFromNode"].apply(lambda s: '/'.join(s.split('/')[:2]))
    job_edges["_rlnPipeLineEdgeProcess"] = job_edges["_rlnPipeLineEdgeProcess"].apply(lambda s: '/'.join(s.split('/')[:2]))

    # # Given array
    job_array = [(x["_rlnPipeLineEdgeFromNode"], x["_rlnPipeLineEdgeProcess"]) for _, x in job_edges.iterrows()]

    # Create a network
    net = Network(notebook=True, height="800px", width="100%")

    # Define color and shape for job types
    job_type_styles = {
    'Import': {'color': '#F94144', 'shape': 'diamond'},
    'MotionCorr': {'color': '#F3722C', 'shape': 'circle'},
    'CtfFind': {'color': '#F8961E', 'shape': 'circle'},
    'ManualPick': {'color': '#F9C74F', 'shape': 'hexagon'},
    'Select': {'color': '#90BE6D', 'shape': 'star'},
    'AutoPick': {'color': '#43AA8B', 'shape': 'hexagon'},
    'Extract': {'color': '#4D908E', 'shape': 'triangleDown'},
    'Class2D': {'color': '#577590', 'shape': 'square'},
    'InitialModel': {'color': '#734fc2', 'shape': 'circle'},
    'Class3D': {'color': '#a088d6', 'shape': 'circle'},
    'Refine3D': {'color': '#c788d6', 'shape': 'circle'}
    }
    
    job_type_styles = {
    'Import': {'color': '#F94144', 'shape': 'diamond'},  # Keeping consistent with reference
    'MotionCorr': {'color': '#F3722C', 'shape': 'circle'},  # Keeping consistent with reference
    'CtfFind': {'color': '#F8961E', 'shape': 'circle'},  # Keeping consistent with reference
    'ManualPick': {'color': '#F9C74F', 'shape': 'hexagon'},  # Keeping consistent with reference
    'Select': {'color': '#90BE6D', 'shape': 'star'},  # Keeping consistent with reference
    'AutoPick': {'color': '#43AA8B', 'shape': 'hexagon'},  # Keeping consistent with reference
    'Extract': {'color': '#4D908E', 'shape': 'triangleDown'},  # Keeping consistent with reference
    'Class2D': {'color': '#577590', 'shape': 'square'},  # Keeping consistent with reference
    'InitialModel': {'color': '#734fc2', 'shape': 'circle'},  # Keeping consistent with reference
    'Class3D': {'color': '#a088d6', 'shape': 'circle'},  # Keeping consistent with reference
    'Refine3D': {'color': '#c788d6', 'shape': 'circle'},  # Keeping consistent with reference
    'MaskCreate': {'color': '#9d54b5', 'shape': 'pentagon'},  # Deeper purple, distinct shape
    'PostProcess': {'color': '#85d5ab', 'shape': 'rectangle'},  # Refreshing green, rectangular
    'CtfRefine': {'color': '#ffe54a', 'shape': 'octagon'},  # Vibrant yellow, unique octagon
    'Polish': {'color': '#54a0c5', 'shape': 'parallelogram'},  # Blue-green, distinctive parallelogram
    'LocalRes': {'color': '#e54a9d', 'shape': 'trapezoid'},  # Pinkish-red, trapezoid for grounding
    'ModelAngelo': {'color': '#c5a054', 'shape': 'ellipse'},  # Earthy brown, smooth ellipse
    'DynaMight': {'color': '#4a9de5', 'shape': 'rhombus'},  # Energetic blue, dynamic rhombus
    'ImportTomo': {'color': '#6a1b9a', 'shape': 'diamond'},  # Dark purple for depth, diamond for precision
    'PseudoSubtomo': {'color': '#00bcd4', 'shape': 'ellipse'},  # Bright turquoise, sleek ellipse
    'ReconstructParticleTomo': {'color': '#8bc34a', 'shape': 'hexagon'},  # Lively green, complex hexagon
    'CtfRefineTomo': {'color': '#ff9800', 'shape': 'parallelogram'},  # Bright orange, parallelogram for stability
    'FrameAlignTomo': {'color': '#9c27b0', 'shape': 'trapezoid'},  # Vivid purple, trapezoid for foundation
    'MultiBody': {'color': '#3f51b5', 'shape': 'star'},  # Deep blue, star for multiplicity
}

    # Add nodes and edges from the array
    for edge in job_array:
        src, dest = edge
        src_job_type = src.split('/')[0]
        dest_job_type = dest.split('/')[0]
        src = src.replace('/', '\n')
        dest = dest.replace('/', '\n')
        src_style = job_type_styles.get(src_job_type, {'color': 'gray', 'shape': 'circle'})
        dest_style = job_type_styles.get(dest_job_type, {'color': 'gray', 'shape': 'circle'})

        net.add_node(src, label=src, color=src_style['color'], shape=src_style['shape'])
        net.add_node(dest, label=dest, color=dest_style['color'], shape=dest_style['shape'])
        net.add_edge(src, dest)

    # Set visualization options
    net.set_options("""
    var options = {
        "nodes": {
            "font": {
                "size": 20,
                "strokeWidth": 0
            }
        },
        "edges": {
            "color": "lightgray",
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "continuous"
            }
        },
        "layout": {
            "hierarchical": {
                "enabled": true,
                "sortMethod": "directed",
                "direction": "DU",
                "nodeSpacing": 50,
                "levelSeparation": 100,
                "seed": 0.5
            }
        },
        "physics": {
            "enabled": true
        }
    }""")
    

    # Generate and return HTML
    return net.generate_html()


def plot_individual_live_ctf(data, meta_list, selected_ranges, color_scheme, units, FOLDER):
    st.session_state['global_mask'] = np.ones(len(data), dtype=bool)

    missing_meta = []
    
    for i, meta in enumerate(meta_list):
        try:
            # Create mask based on the selected range for current meta
            mask = (data[meta] >= selected_ranges[meta][0]) & (data[meta] <= selected_ranges[meta][1])
            combined_mask = mask & st.session_state['global_mask']
            st.session_state['global_mask'] = combined_mask
        except KeyError:
            missing_meta.append(meta)
            pass
    
    # if last_mic_path not in st.session_state or last_ctf_file not in st.session_state:
    #     last_mic_path = data['_rlnMicrographName'].iloc[-1]
    #     last_ctf_file = data['_rlnCtfImage'].iloc[-1]
    #     st.session_state['last_mic_path'] = last_mic_path
    #     st.session_state['last_ctf_file'] = last_ctf_file
    last_mic_path = data['_rlnMicrographName'].iloc[-1]
    last_ctf_file = data['_rlnCtfImage'].iloc[-1]
    
    for i, meta in enumerate(meta_list):
        if meta not in missing_meta:
            try:
                # Update DataFrame with the combined mask for current meta
                data['Selected'] = combined_mask
                
                # Creating the scatter plot with histogram for current meta
                fig = px.scatter(
                    data_frame=data,
                    y=meta,
                    color='Selected',
                    color_discrete_sequence=[color_scheme[i], 'rgba(128, 128, 128, 0.5)'],  # Adjusted for transparency
                    color_discrete_map={False: 'rgba(128, 128, 128, 0.5)', True: color_scheme[i]},
                    marginal_y="histogram",
                    hover_data=[meta],
                    orientation='h',
                    render_mode='webgl',
                    size_max=10,
                    size=np.ones(data.shape[0]),
                    template='plotly_white',
                    height=300
                )
                fig.update_traces(marker=dict(line_width=0))

                # Update layout and add rectangle highlights
                min_val, max_val = data[meta].min(), data[meta].max()
                unit = units[i]
                
                try:
                    fig.update_layout(
                        showlegend=False,
                        xaxis={"showgrid": False},
                        yaxis={"showgrid": True, "title_text": f"{unit}"},
                        height=300,
                        yaxis_range=[min_val - min_val * 0.05, max_val + max_val * 0.05],
                        title=meta.replace('_rln','')
                    )
                    fig.add_hrect(y0=selected_ranges[meta][1], y1=max_val + max_val * 0.05, line_width=0, fillcolor="grey", opacity=0.2)
                    fig.add_hrect(y0=min_val - min_val * 0.05, y1=selected_ranges[meta][0], line_width=0, fillcolor="grey", opacity=0.2)
                
                except:
                    fig.update_layout(
                        showlegend=False,
                        xaxis={"showgrid": False},
                        yaxis={"showgrid": True, "title_text": f"{unit}"},
                        height=300,
                        title=meta.replace('_rln','')
                    )

                # Display the plot for current meta
                st.plotly_chart(fig, use_container_width=True)
            
            except KeyError or ValueError:
                pass

    # Update the global mask with the combined mask of the last meta
    st.session_state['global_mask'] = combined_mask
    
    mark_done =  ':white_check_mark:' if combined_mask.iloc[-1] == 1 else ':x:'
    
    st.subheader(f'Last micrograph preview')
    st.write(f'{last_mic_path}, **status**: {mark_done}')
    c1, c2 = st.columns(2)
    container1 = st.container()
    
    with container1:    
        with c1:
            c1.write('Last Micrograph')
            try:
                process_and_display_micrograph(last_mic_path, '', 0, 0.2, FOLDER, 1)
            except Exception as e:
                print(e)
                st.write('No Data')

        with c2:
            c2.write('Last Powerspectrum')
            try:
                plot_power_spectrum(FOLDER, last_mic_path)
            except Exception as e:
                print(e)
                st.write('No Data')
    
        st.write('Motion model')
        try:
            show_motion(FOLDER, last_mic_path, False)
        except Exception as e:
            print(e)
            st.write('No Data')

        st.write('CTF FIT')
        try:
            plot_CTF_average(FOLDER, last_ctf_file)
        except Exception as e:
            print(e)
            st.write('No Data')
            
    
    return data[combined_mask]

    c11, c22 = st.columns(2)

    #container2 = st.container(border=True)
    #with container1:    
        


def get_footer(show=True):
    footer = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;700&display=swap');

    a:link, a:visited {
        color: #007bff;  /* Nicer shade of blue */
        background-color: transparent;
        text-decoration: none;  /* Removing underline */
    }

    a:hover, a:active {
        color: #0056b3;  /* Darker shade of blue for hover */
        background-color: transparent;
        text-decoration: underline;  /* Underline on hover */
    }

    .footer {
    position: adaptive;
    left: 0;
    bottom: 1rem;
    width: 100%;
    text-align: left;
    padding: 10px;
    font-size: 16px;
    font-family: 'Segoe UI', sans-serif;
    border-top: 0px solid #e7e7e7;
    }

    .footer p {
        margin: 10px 0;
        color: #606060;  /* Gray color for visibility on both dark and light backgrounds */
    }

    .container {
        position: relative;
        padding-bottom: 50px; /* Height of the footer */
    }

    .bug-report {
        color: #28a745;  /* Green color for bug report link */
    }

    .bold {
        font-weight: 500;
    }
    </style>
    <div class="container">
        <div class="footer">
            <p class="bold">Developed by <a class="bold" href="mailto:dzyla@lji.org" target="_blank">Dawid Zyla</a><br></p> 
            <p><a class="bug-report" href="https://github.com/dzyla/Follow_Relion_gracefully" target="_blank">Report bugs and suggestions</a></p>
        </div>
    </div>
    """
    if show:
        return footer
    else:
        return ''

#'''backdrop-filter: blur(5px); /* Apply blur effect */
# background-color: rgba(255, 255, 255, 0.5); /* Semi-transparent white */'''


def get_subfolders(parent_folder, job_type):
    subfolders = []
    folder_path = os.path.join(parent_folder, job_type)
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, item)):
                subfolders.append(item)
    return subfolders

def get_job_files(job_folder, job_type, file_suffix):
    files = []
    for item in os.listdir(job_folder):
        if item.endswith(file_suffix):
            files.append(item)
    return files

def display_job_selection(job_type, file_suffix):
    subfolders = get_subfolders(FOLDER, job_type)
    if subfolders:
        selected_job = st.selectbox(f'Select {job_type} job', subfolders)
        job_folder = os.path.join(FOLDER, job_type, selected_job)
        files = get_job_files(job_folder, job_type, file_suffix)
        if files:
            return os.path.join(job_folder, files[0])
    return None

#@st.cache_data(experimental_allow_widgets=True)

def get_modification_time(file_path):
    """
    Get the last modification time of a file.
    Returns the modification time if file exists, otherwise None.
    """
    if os.path.exists(file_path):
        return os.path.getmtime(file_path)
    return None

def process_data_new_run(FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files, calculate_ice, add_movie_dates):
    
    # Store modification times in session_state
    st.session_state['node_files_mod_times'] = {
        'import': get_modification_time(os.path.join(FOLDER, import_node_files[0])),
        'motioncorr': get_modification_time(os.path.join(FOLDER, motioncorr_node_files[0])),
        'ctffind': get_modification_time(os.path.join(FOLDER, ctffind_node_files[0]))
    }

    import_data_star = process_import_data(FOLDER, import_node_files, add_movie_dates)
    motioncorr_data_star = parse_star(os.path.join(FOLDER, motioncorr_node_files[0]))['micrographs']
    ctffind_data_star_whole = parse_star(os.path.join(FOLDER, ctffind_node_files[0]))
    ctffind_data_star = ctffind_data_star_whole['micrographs']
    ctffind_optics_star = ctffind_data_star_whole['optics']

    # Merge data and calculate additional metrics
    final_data_star = merge_and_calculate_metrics(FOLDER, import_data_star, motioncorr_data_star, ctffind_data_star)

    if calculate_ice:
        final_data_star = calculate_ice_thickness(FOLDER, final_data_star)
    
    return final_data_star, ctffind_optics_star
    
    
    
def process_data(FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files, calculate_ice, add_movie_dates):
    
    if 'live_settings' not in st.session_state:
        st.session_state['live_settings'] = []
    else:
        current_settings = [FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files, calculate_ice, add_movie_dates]
        settings_changed = st.session_state['live_settings'] != current_settings
        print(f'{datetime.now()} Live settings changed: {settings_changed}')
    
    current_settings = [FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files, calculate_ice, add_movie_dates]
    files_changed = have_modification_times_changed(FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files)
        
    # Check if final_data_star and ctffind_optics_star are not in session_state or if file modification times have changed
    if ('final_data_star' not in st.session_state or 
            'ctffind_optics_star' not in st.session_state or 
            files_changed or 
            st.session_state['live_settings'] != current_settings):
        
        # Read and process data
        final_data_star, ctffind_optics_star = process_data_new_run(FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files, calculate_ice, add_movie_dates)
        
        # Store the data and current modification times in session state
        st.session_state['final_data_star'] = final_data_star
        st.session_state['ctffind_optics_star'] = ctffind_optics_star
        update_modification_times_in_state(FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files)

    final_data_star = st.session_state['final_data_star']
    ctffind_optics_star = st.session_state['ctffind_optics_star']
    
    # Prepare and display plots
    prepare_and_display_plots(final_data_star, ctffind_optics_star, FOLDER)

def have_modification_times_changed(FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files):
    # Get current modification times
    try:
        current_mod_times = {
            'import': get_modification_time(os.path.join(FOLDER, import_node_files[0])),
            'motioncorr': get_modification_time(os.path.join(FOLDER, motioncorr_node_files[0])),
            'ctffind': get_modification_time(os.path.join(FOLDER, ctffind_node_files[0]))
        }

        # Compare with stored modification times
        stored_mod_times = st.session_state.get('node_files_mod_times', {})
        for key in current_mod_times:
            if current_mod_times[key] != stored_mod_times.get(key):
                return True  # Modification times have changed
        return False  # No changes in modification times
    except:
        return False

def update_modification_times_in_state(FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files):
    # Update modification times in session state
    st.session_state['node_files_mod_times'] = {
        'import': get_modification_time(os.path.join(FOLDER, import_node_files[0])),
        'motioncorr': get_modification_time(os.path.join(FOLDER, motioncorr_node_files[0])),
        'ctffind': get_modification_time(os.path.join(FOLDER, ctffind_node_files[0]))
    }

def process_import_data(FOLDER, import_node_files, add_movie_dates):
    # Read 'Import' data star file
    import_data_star = parse_star(os.path.join(FOLDER, import_node_files[0]))['movies']
    
    add_modification_dates_and_base_names(FOLDER, import_data_star, add_movie_dates)

    return import_data_star

def add_modification_dates_and_base_names(FOLDER, import_data_star, add_movie_dates):
    file_mod_times = []
    if add_movie_dates:
        for file in import_data_star["_rlnMicrographMovieName"]:
            file_path = os.path.join(FOLDER, file)
            if os.path.exists(file_path):  # Check if the file_path exists
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            else:
                mod_time = datetime.today()
            file_mod_times.append(mod_time)

    if len(file_mod_times) == import_data_star.shape[0]:
        import_data_star['_rlnMicrograph_date'] = file_mod_times

    # Regex to match different file extensions and then replace with 'mrc'
    import_data_star['_rlnMicrographBase'] = import_data_star['_rlnMicrographMovieName'].str.replace(
        r'\.(tif|tiff|TIFF|TIF|MRC|mrc)$', '.mrc', regex=True).apply(os.path.basename)
    
    print(file_mod_times)
    return import_data_star

def merge_and_calculate_metrics(FOLDER, import_data_star, motioncorr_data_star, ctffind_data_star):
    final_data_star = pd.merge(ctffind_data_star, motioncorr_data_star, on='_rlnMicrographName', how='inner')
    final_data_star['_rlnMicrographBase'] = final_data_star['_rlnMicrographName'].apply(os.path.basename)
    final_data_star = pd.merge(import_data_star, final_data_star, on='_rlnMicrographBase', how='inner')
    return final_data_star

def calculate_ice_thickness(FOLDER, final_data_star):
    ice_per_mic = []
    for file in stqdm(final_data_star['_rlnCtfImage']):
        power_spectrum_ave_rot_txt_paths = os.path.join(FOLDER, file.replace('.ctf:mrc', '_avrot.txt'))
        if os.path.exists(power_spectrum_ave_rot_txt_paths):
            ave_rot = pd.read_csv(power_spectrum_ave_rot_txt_paths, skiprows=[0,1,2,3,4,7,8,9,10,11,12], header=None, delim_whitespace=True).transpose()
            ice_ring = ave_rot[0].between(0.25, 0.28, inclusive="both")
            ice_per_mic.append(round(sum(np.abs(ave_rot[ice_ring][1])), 6))
        else:
            ice_per_mic.append(0)
    
    if np.sum(ice_per_mic)>0:
        final_data_star['_rlnMicrographIceThickness'] = np.stack(ice_per_mic)
    
    return final_data_star


def plot_CTF_average(FOLDER, ctf_file_path):
    power_spectrum_ave_rot_txt_paths = os.path.join(FOLDER, ctf_file_path.replace('.ctf:mrc', '_avrot.txt'))
    
    if os.path.exists(power_spectrum_ave_rot_txt_paths):
        ave_rot = pd.read_csv(power_spectrum_ave_rot_txt_paths, skiprows=[0,1,2,3,4,6,10], header=None, delim_whitespace=True).transpose()
        
        ave_rot.columns = ['Spatial_freq', '1D_Ave', 'Fit', 'Fit_CC']

        # Create Plotly figure
        fig = go.Figure()

        # Add traces for each column
        fig.add_trace(go.Scatter(x=ave_rot['Spatial_freq'], 
                                 y=ave_rot['1D_Ave'], mode='lines', 
                                 customdata=1/ave_rot['Spatial_freq'],
                                 hovertemplate=f"Resolution: %{{customdata:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                                 name='1D Average'))
        fig.add_trace(go.Scatter(x=ave_rot['Spatial_freq'],
                                 y=ave_rot['Fit'],
                                 mode='lines',
                                 customdata=1/ave_rot['Spatial_freq'],
                                 name='Fit'))
        fig.add_trace(go.Scatter(x=ave_rot['Spatial_freq'],
                                 y=ave_rot['Fit_CC'],
                                 mode='lines',
                                 customdata=1/ave_rot['Spatial_freq'],
                                 name='Fit CC'))

        # Update layout
        fig.update_layout(
            #title="CTF Fit",
            xaxis_title="Spatial Frequency, 1/Å",
            yaxis_title="CTF",
            template="plotly_white",
            legend=dict(
                x=0.02,  # Adjust x position (0 is far left, 1 is far right)
                y=0.98,  # Adjust y position (0 is bottom, 1 is top)
                bordercolor="Black",
                borderwidth=2,
                orientation="h"  # Horizontal layout of legend items
            ),
            yaxis_range=[-0.1,1.1]
        )
        fig.update_layout(hovermode="x unified")
        
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("File not found: " + power_spectrum_ave_rot_txt_paths)



def prepare_and_display_plots(final_data_star, optics, FOLDER):
    color_hex_list = ['#6FC381', '#9AD5A7', '#6495ED', '#ADD8E6', '#FA8072', '#F6AE2D', '#F9CB76']
    meta_to_plot = ['_rlnMicrograph_date', '_rlnAccumMotionTotal', '_rlnDefocusU', '_rlnCtfAstigmatism', '_rlnCtfMaxResolution', '_rlnCtfFigureOfMerit', '_rlnMicrographIceThickness']
    units = ['Date', '_rlnAccumMotionTotal', 'Å', 'nm', 'Å', '_rlnCtfFigureOfMerit', 'Relative Ice thickness']

    convert_to_float(final_data_star, meta_to_plot)
    slider_values = create_sliders(final_data_star, meta_to_plot)
    final_df = plot_individual_live_ctf(final_data_star, meta_to_plot, slider_values, color_hex_list, units, FOLDER)
    display_selected_micrographs_count()
    
    meta_to_drop = ['_rlnMicrograph_date', 'Selected', '_rlnMicrographBase']
    # Clean df from unwanted columns
    for meta_ in meta_to_drop:
        try:
            final_df = final_df.drop(meta_, axis=1)
        except:
            pass
    modified_star = star_from_df({'optics': optics, 'micrographs': final_df})

    # Save the modified STAR file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.star') as tmp_file:
        modified_star.write_file(tmp_file.name)

    # Read the file in binary mode
    with open(tmp_file.name, 'rb') as file:
        binary_star_data = file.read()

    # Provide a download button in Streamlit
    st.download_button(
        label=f"**Download selected data as STAR selected_micrographs.star**",
        data=binary_star_data,
        file_name='selected_micrographs.star',
        mime='application/octet-stream'  # MIME type for binary file
    )
    st.divider()
    # Optional: Remove the temporary file if desired
    os.remove(tmp_file.name)
    
def convert_to_float(final_data_star, meta_to_plot):
    for meta in meta_to_plot:
        if meta != '_rlnMicrograph_date':
            try:
                final_data_star[meta] = final_data_star[meta].astype(float)
            except KeyError:
                pass

def create_sliders(final_data_star, meta_to_plot):
    slider_values = {}
    for meta in meta_to_plot:
        try:
            min_val, max_val = final_data_star[meta].min(), final_data_star[meta].max()
            slider_key = f"{meta}_slider"
            slider_values[meta] = st.sidebar.slider(f"Select {meta} range", min_value=min_val, max_value=max_val, value=(min_val, max_val), key=slider_key) \
                                if meta != '_rlnMicrograph_date' else [min_val, max_val]
        except KeyError:
            pass
    return slider_values

def display_selected_micrographs_count():
    selected_mics = np.sum(st.session_state['global_mask'])
    percent = round(selected_mics/st.session_state['global_mask'].shape[0]*100, 2)
    st.sidebar.subheader(f'Number of selected Micrographs: {selected_mics} ({percent}%)')

@st.cache_resource
def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()

def show_output(process):
    stdout_file, stderr_file, full_command = process['stdout_file'], process['stderr_file'], process['full_command']
    st.write(f"**Running Command:**")
    st.code(full_command)

    pid_to_kill = process['pid']
    
    
    # Add a button to kill the job
    if st.button("Kill Job"):
        if pid_to_kill:
            try:
                # Convert PID to an integer
                pid_to_kill = int(pid_to_kill)
                # Use the 'kill' command to terminate the process
                os.kill(pid_to_kill, signal.SIGTERM)
                st.success(f"Process with PID {pid_to_kill} has been terminated.")
            except ValueError:
                st.error("Invalid PID. Please enter a valid PID.")
            except ProcessLookupError:
                st.error(f"No process found with PID {pid_to_kill}.")
        else:
            st.warning("Please enter a PID to kill a job.")
    
    
    # Containers for output and error
    stdout_container = st.empty()
    stderr_container = st.empty()
    
    # Generate unique keys for widgets using process_id
    stdout_key = f"stdout_{process['pid']}"
    stderr_key = f"stderr_{process['pid']}"
    refresh_key = f"refresh_{process['pid']}"

    # Function to update the containers
    def update_output():
        with open(stdout_file, "r") as file:
            stdout_lines = file.readlines()
        with open(stderr_file, "r") as file:
            stderr_lines = file.readlines()

        # Reverse the lines to show latest first
        stdout_content = ''.join(stdout_lines[::-1])
        stderr_content = ''.join(stderr_lines[::-1])
        
            # Generate unique keys for text_area widgets
        stdout_key = f"stdout_{pid_to_kill}_{full_command}"
        stderr_key = f"stderr_{pid_to_kill}_{full_command}"
        
        # Generate unique keys for widgets using process_id
        stdout_key = f"stdout_{process['pid']}"
        stderr_key = f"stderr_{process['pid']}"
        refresh_key = f"refresh_{process['pid']}"
        
        stdout_container.text_area("Standard Output", stdout_content, height=300, key=stdout_key)
        stderr_container.text_area("Standard Error", stderr_content, height=300, key=stderr_key)

    # Initial update
    update_output()

    # Refresh button
    if st.button('Refresh Output', key=refresh_key):
        st.rerun()

    # Automatic refresh every 5 seconds
    #st.experimental_rerun()
    
def run_command(command):
    with tempfile.NamedTemporaryFile(delete=False) as temp_stdout, \
         tempfile.NamedTemporaryFile(delete=False) as temp_stderr:
        process = subprocess.Popen(command, stdout=temp_stdout, stderr=temp_stderr)
        return process.pid, temp_stdout.name, temp_stderr.name

def run_command_get_output(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error_message = stderr.decode().strip()
        raise RuntimeError(f"Error in command execution: {error_message}")
    return stdout.decode().strip()

def run_command_with_cwd(command, cwd):
    with tempfile.NamedTemporaryFile(delete=False) as temp_stdout, \
         tempfile.NamedTemporaryFile(delete=False) as temp_stderr:
        process = subprocess.Popen(command, stdout=temp_stdout, stderr=temp_stderr, cwd=cwd)
        return process.pid, temp_stdout.name, temp_stderr.name

def parse_help_text(help_text):
    pattern = r"--(\w+)(?: \(([^)]+)\))?"
    matches = re.findall(pattern, help_text)
    return {f"--{match[0]}": match[1] for match in matches}

def generate_widgets(params, command):
    
    num_cpus = os.cpu_count()
    
    if 'mpi' in command:
        use_mpi = True
        new_params = {'mpi': '1', 'threads': str(num_cpus)}
        new_params.update(params)
        params = new_params
    else:
        use_mpi = False
    
    print(f'params {type(params)}: {params}')
    
    
    inputs = {}
    defaults = {}
    include_param = {}

    for param, default in params.items():
        c1, c2 = st.columns([4,6])

        with c1:
            include = st.checkbox(param, value=False, key=f"include_{param}")
            include_param[param] = include

        with c2:
            if include:
                if "--gpu" in param:
                    value = st.text_input("", value=default, key=f"value_{param}")
                # elif param == 'mpi':
                #     value = st.number_input("", min_value=1, max_value=int(num_cpus), value=int(default), key=f"value_{param}", step=1)
                # elif param == 'threads':
                #     max_threads = num_cpus // int(inputs.get('mpi', 1))
                #     value = st.number_input("", min_value=1, max_value=max_threads, value=int(default), key=f"value_{param}", step=1)
                elif default in ["true", "false"] and not "--gpu" in param:
                    value = st.checkbox("", value=default == "true", key=f"value_{param}")
                elif re.match(r"^-?\d+\.?\d*$", default):
                    value = st.number_input("", value=float(default) if '.' in default else int(default), key=f"value_{param}")
                else:
                    value = st.text_input("", value=default, key=f"value_{param}")
                inputs[param] = value
            else:
                inputs[param] = default  # Store the default value even if not included

        defaults[param] = default

    # Filter inputs based on include_param
    filtered_inputs = {param: inputs[param] for param in include_param if include_param[param]}
    
    return filtered_inputs, defaults, include_param


def create_temp_directory(directory_path='temp'):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def folder_selector(root_path):
    if 'current_path' not in st.session_state or not os.path.exists(st.session_state.current_path):
        st.session_state.current_path = root_path

    if not os.path.isdir(st.session_state.current_path):
        st.sidebar.error(f"Error accessing the folder: {st.session_state.current_path}. Resetting to the default path.")
        st.session_state.current_path = root_path

    folders = sorted([f for f in os.listdir(st.session_state.current_path) if os.path.isdir(os.path.join(st.session_state.current_path, f))])

    st.sidebar.write(f"Current Path: {st.session_state.current_path}")

    # Radio buttons for folder navigation
    selected_folder = st.sidebar.radio("Select a folder", ['.', "(Go Up)"] + folders, 0)

    if selected_folder == "(Go Up)":
        new_path = os.path.dirname(st.session_state.current_path)
        if new_path and os.path.exists(new_path):
            st.session_state.current_path = new_path
            st.rerun()
    
    elif selected_folder in folders:
        st.session_state.current_path = os.path.join(st.session_state.current_path, selected_folder)
        st.rerun()

    # Choose the current folder
    if st.sidebar.button('Choose Current Folder'):
        st.sidebar.write('Selected Folder:', st.session_state.current_path)
        return st.session_state.current_path


# Initialize or load the job database
def init_job_db():
    if 'job_db' not in st.session_state:
        st.session_state.job_db = pd.DataFrame(columns=['job_id', 'full_command', 'status', 'pid', 'stdout_file', 'stderr_file', 'visible'])
    return st.session_state.job_db

# Save job to the job database
def save_job(command):
    job_db = init_job_db()
    new_job_id = f"job{len(job_db) + 1:03d}"
    job_data = {
        'job_id': new_job_id,
        'command': command,
        'status': 'Ready',
        'pid': None,
        'stdout_file': None,
        'stderr_file': None,
        'visible': True
    }
    job_db = job_db.append(job_data, ignore_index=True)
    st.session_state.job_db = job_db

# Clear job history
def clear_job_history():
    if 'job_db' in st.session_state:
        st.session_state.job_db['visible'] = False

def custom_css():
    # Custom CSS to modify padding of the sidebar and main panel
    css = """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
        /* Target the sidebar using the data-testid attribute */
        div[data-testid="stSidebarUserContent"] {
            padding-top: 0rem;  /* Adjust the padding-top value as needed */
        }
        div[data-testid="block-container"] {
            padding-top: 2rem;  /* Adjust the padding-top value as needed */
        }
    </style>
    """
    return css


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.session_state["password_args"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.title(':blue[Follow Relion Gracefully] :snowflake: :microscope:')
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error(":see_no_evil: Password incorrect")
    return False


def reset_relion_folder_path():
    st.session_state.folder_selected = False

# Function to list directories/files in the current folder
# def list_dirs_files(searchterm: str) -> List[str]:
#     if not searchterm:
#         return []
    
#     # List all files and directories in the current folder
#     all_items = os.listdir(searchterm)
    
#     # Filter items based on the search term
#     filtered_items = [item for item in all_items if searchterm.lower() in item.lower()]
    
#     return filtered_items





####################################################################################
############################# Here we go ###########################################
####################################################################################




# Set Streamlit config
st.set_page_config(
    page_title="Follow Relion Gracefully",
    page_icon=":microscope:",#:microscope:
    initial_sidebar_state="expanded"
)

# Inject custom CSS with markdown
st.markdown(custom_css(), unsafe_allow_html=True)
footer = get_footer()


# Create argument parser
parser = argparse.ArgumentParser(description="Process some paths and a password.")

# Add arguments
parser.add_argument("--i","--folder", type=str, help="Path to the default folder", default="~/")
parser.add_argument("--path","--relion-path", type=str, help="Path to the RELION bin folder", default="/")
parser.add_argument("--p", "--password", type=str, help="Password is you want to secure your instance", required=False, default='')

# Parse arguments
args = parser.parse_args()

st.session_state["password_args"] = args.p
if 'default_job_folder' not in st.session_state:
    st.session_state['default_job_folder'] = args.i

relion_path = args.path

if st.session_state["password_args"] != '':
    if not check_password():
        st.stop()

st.sidebar.title(':blue[Follow Relion Gracefully] :snowflake: :microscope: ')

mode = st.sidebar.selectbox("Choose Mode:", ["View Jobs", "Live processing", "Run Relion (Experimental)"])

# Classic Follow Relion mode with job preview    
if mode == "View Jobs":
    c1, c2 = st.sidebar.columns(2)
    
    with c1:
        FOLDER_root = st.text_input("Root Folder Path", value=st.session_state['default_job_folder'])
        st.session_state['default_job_folder'] = FOLDER_root

    # Initialize the checkbox value
    use_default_star = False
    
    # Check if default_pipeliner.star exists in the root folder
    default_pipeliner_path = os.path.join(FOLDER_root, "default_pipeline.star")
    use_default_pipeliner = os.path.exists(default_pipeliner_path)

    # Option to use default_pipeliner.star if it exists
    if use_default_pipeliner:
        with c1:
            use_default_star = st.checkbox("Use default_pipeliner.star in Root Folder")

    # Retrieve folders in the specified path
    folders = get_folders(FOLDER_root)
    selected_folder="."
    
    # If there are folders, let the user select one
    if folders and not use_default_star:
        with c2:
            selected_folder = st.selectbox("Select a Project Folder", folders)
        
        # Construct the full path to the selected folder
        job_folder_path = os.path.join(FOLDER_root, selected_folder)
    else:
        # Use default_pipeliner.star if checkbox is selected
        if use_default_star:
            job_folder_path = FOLDER_root
        else:
            job_folder_path = None

    if job_folder_path:
        FOLDER = job_folder_path
        
        if FOLDER != os.path.join(st.session_state['default_job_folder'], selected_folder) or FOLDER != job_folder_path:
            print(FOLDER)
        
        # Further processing with the selected job folder
        try:
            # Use default pipeline if selected, else use pipeline in the selected folder
            pipeliner_path = default_pipeliner_path if use_default_star else os.path.join(job_folder_path, "default_pipeline.star")
            pipeline_star = parse_star(pipeliner_path)
        except:
            pipeline_star = None

        if pipeline_star:
            data_results = pipeline_star["pipeline_nodes"]
            pipeline_processes = pipeline_star["pipeline_processes"]

            # Ensure the label is present in the DataFrame
            if "_rlnPipeLineProcessStatusLabel" not in pipeline_processes:
                pipeline_processes[
                    "_rlnPipeLineProcessStatusLabel"
                ] = pipeline_processes.get("_rlnPipeLineProcessType", "")

            # Create DataFrame
            df = pd.DataFrame(pipeline_processes)


            # Unique process types
            try:
                try:
                    process_types = list(df["_rlnPipeLineProcessTypeLabel"].unique())
                except:
                    st.error('This folder contains Relion 3 process types. Relion 3 metadata handling is not supported. Try updating to newer Relion version')
                    st.stop()
                    
                process_types.append("relion.flowchart")
                process_types.append("relion.InteractivePlot")

                st.sidebar.title("Process Types")
                selected_process = st.sidebar.radio(
                    "Choose a process type:", process_types)

                if selected_process == "relion.flowchart":
                    st.title("Network Graph")
                    job_network_html = create_network(pipeline_star)
                    components.html(job_network_html, height=800)

                if selected_process == "relion.InteractivePlot":
                    # File path input
                    create_temp_directory()
                    
                    star_path = st.text_input('Provide STAR file path')

                    # File uploader
                    uploaded_file = st.file_uploader("Or upload STAR file", type=['star'])

                    star_file = None
                    temp_file_path = None

                    if star_path and os.path.exists(star_path):
                        # Load from file path
                        star_file = parse_star(star_path)
                        file_name = os.path.basename(star_path)
                    if uploaded_file is not None:
                        # Save uploaded file to a temporary file
                        temp_file_path = os.path.join("temp", uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load from uploaded file
                        star_file = parse_star(temp_file_path)
                        file_name = uploaded_file.name

                    if star_file:
                        dict_keys = star_file.keys()
                        key_selected = st.selectbox('Select STAR block:', dict_keys)
                        interactive_scatter_plot(star_file, max_rows=None, own=True, selected_block=key_selected, own_name=file_name)

                        # Clean up: delete the temporary file if it was created
                        if temp_file_path and os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

                # Jobs for selected process type
                jobs = df[df["_rlnPipeLineProcessTypeLabel"] == selected_process][
                    "_rlnPipeLineProcessName"
                ].tolist()
                
                # Job selection
                if jobs:
                    st.sidebar.title(f"{selected_process} Jobs")
                    selected_job = st.sidebar.radio("Choose a job:", jobs, index=len(jobs)-1)

                    display_job_info(
                        selected_job, FOLDER, pipeline_processes, pipeline_star
                    )
                    
                else:
                    st.sidebar.write("No jobs available for this process type.")
                    pass
            
            except Exception as e:
                print(e)
                st.error(f"No jobs available for this process type. {e}")

        else:
            st.error(f"No default_pipeline.star found in the directory.")
            
            interactive_plot = st.sidebar.radio('Job selection', ['','relion.InteractivePlot'])
            if interactive_plot == 'relion.InteractivePlot':
                # File path input
                create_temp_directory()
                
                star_path = st.text_input('Provide STAR file path')

                # File uploader
                uploaded_file = st.file_uploader("Or upload STAR file", type=['star'])

                star_file = None
                temp_file_path = None

                if star_path and os.path.exists(star_path):
                    # Load from file path
                    star_file = parse_star(star_path)
                    file_name = os.path.basename(star_path)
                if uploaded_file is not None:
                    # Save uploaded file to a temporary file
                    temp_file_path = os.path.join("temp", uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load from uploaded file
                    star_file = parse_star(temp_file_path)
                    file_name = uploaded_file.name

                if star_file:
                    dict_keys = star_file.keys()
                    key_selected = st.selectbox('Select STAR block:', dict_keys)
                    interactive_scatter_plot(star_file, max_rows=None, own=True, selected_block=key_selected, own_name=file_name)

                    # Clean up: delete the temporary file if it was created
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
        
    st.sidebar.markdown(footer, unsafe_allow_html=True)
            
            
elif mode == "Live processing":
    #Code inpired and partially borrowed from https://github.com/cryoEM-CNIO/CNIO_Relion_Tools
    # Sidebar layout for inputs
    c1, c2 = st.sidebar.columns(2)
    
    with c1:
        FOLDER_root = st.text_input("Root Folder Path", value=st.session_state['default_job_folder'])
        st.session_state['default_job_folder'] = FOLDER_root
        use_root_folder = st.checkbox("Use Root Folder", value=False)
    
    # Initialize job_folder_path
    job_folder_path = None

    # If using root folder directly, check for default_pipeline.star
    if use_root_folder:
        default_pipeliner_path = os.path.join(FOLDER_root, "default_pipeline.star")
        use_default_pipeliner = os.path.exists(default_pipeliner_path)
        if use_default_pipeliner:
            job_folder_path = FOLDER_root
        else:
            st.sidebar.error("default_pipeline.star not found in Root Folder.")
            # If the checkbox is ticked but default_pipeline.star is not found,
            # it implies the user may want to untick and look for specific folders.

    # If not using root folder directly, let user select a folder
    if not use_root_folder or not job_folder_path:
        folders = get_folders(FOLDER_root)
        if folders:
            with c2:
                selected_folder = st.sidebar.selectbox("Select a Project Folder", folders)
            job_folder_path = os.path.join(FOLDER_root, selected_folder)
        else:
            st.sidebar.write("No folders found in the specified path.")


    if job_folder_path:
        FOLDER = job_folder_path
        
        if FOLDER != st.session_state['default_job_folder']:
            print(FOLDER)
        
        # Further processing with the selected job folder
        try:
            pipeliner_path = os.path.join(job_folder_path, "default_pipeline.star")
            pipeline_star = parse_star(pipeliner_path)

        except:
            pipeline_star = None

        if pipeline_star:
            data_results = pipeline_star["pipeline_nodes"]
            pipeline_processes = pipeline_star["pipeline_processes"]
            
            # Ensure the label is present in the DataFrame
            if "_rlnPipeLineProcessStatusLabel" not in pipeline_processes:
                pipeline_processes[
                    "_rlnPipeLineProcessStatusLabel"
                ] = pipeline_processes.get("_rlnPipeLineProcessType", "")

            # Create DataFrame
            df = pd.DataFrame(pipeline_processes)
            
            metadata_to_monitor = ['Import', 'MotionCorr', 'CtfFind']
            
            
            st.sidebar.divider()
            
            jobs_to_monitor = {}
            for meta in metadata_to_monitor:
                jobs = df[df['_rlnPipeLineProcessName'].str.contains(meta)]
                if jobs.shape[0]>0:
                    jobs_to_monitor[meta] = jobs
            try:
                import_job_select = st.sidebar.selectbox('Select Import Job:', jobs_to_monitor['Import']['_rlnPipeLineProcessName'])
            except Exception as e:
                print(e)
                pass
            
            try:
                motionCorr_job_select = st.sidebar.selectbox('Select MotionCorr Job:', jobs_to_monitor['MotionCorr']['_rlnPipeLineProcessName'])
            except Exception as e:
                print(e)
                pass
            
            try:
                ctffind_job_select = st.sidebar.selectbox('Select Ctffind Job:', jobs_to_monitor['CtfFind']['_rlnPipeLineProcessName'])
            except Exception as e:
                print(e)
                pass
            
            
            continue_with_live=True
            
            process_nodes = pipeline_star["pipeline_nodes"]["_rlnPipeLineNodeName"]
            try:
                import_node_files = [node for node in process_nodes if import_job_select in node]
                motioncorr_node_files = [node for node in process_nodes if motionCorr_job_select in node]
                ctffind_node_files = [node for node in process_nodes if ctffind_job_select in node]
            
            except NameError:
                try:
                    # Display selectboxes and get selected files for each job type
                    import_file = display_job_selection('Import', 'movies.star')
                    motioncorr_file = display_job_selection('MotionCorr', 'corrected_micrographs.star')
                    ctffind_file = display_job_selection('CtfFind', 'micrographs_ctf.star')

                    # Combine selected files into lists
                    import_node_files = [import_file] if import_file else []
                    motioncorr_node_files = [motioncorr_file] if motioncorr_file else []
                    ctffind_node_files = [ctffind_file] if ctffind_file else []
                    
                    if not (import_node_files or motioncorr_node_files or ctffind_node_files):
                        st.warning('Necessary folders not found')
                        continue_with_live = False
                    
                    pass
                except Exception as e:
                    st.write('Necessery folders not found')
                    continue_with_live=False
                    print(e)
            
            if continue_with_live:
                col1, col2 = st.sidebar.columns(2)
               
                # Rerun by cleaning states
                with col1:
                    if st.button('Clear and Rerun'):
                        #clear_cache()
                        del st.session_state['final_data_star']
                        del st.session_state['ctffind_optics_star']
                        
                        #st.experimental_rerun()
                    add_movie_dates_option = st.sidebar.checkbox('Add Movie Modification Dates (Might be slow)', value=False)
                    calculate_ice_thickness_option = st.sidebar.checkbox('Calculate Ice Thickness (Might be slow)', value=False)
                
                if 'live_run' not in st.session_state:
                    st.session_state['live_run'] = False
                
                if 'run_counter' not in st.session_state:
                    st.session_state['run_counter'] = 0

                # Create a checkbox that directly modifies st.session_state['live_run']
                st.session_state['live_run'] = st.sidebar.checkbox('Continues update?', value=st.session_state['live_run'], key='unique_live_checkbox')

                # Run the data processing function
                process_data(FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files, calculate_ice_thickness_option, add_movie_dates_option)
                
                #update live_settings
                st.session_state['live_settings'] = [FOLDER, import_node_files, motioncorr_node_files, ctffind_node_files, calculate_ice_thickness_option, add_movie_dates_option]

                # Continuously rerun if live_run is True
                if st.session_state['live_run']:
                    time.sleep(10)  # Sleep time to prevent too frequent updates, adjust as needed
                    st.session_state['run_counter'] += 1
                    run_couter = st.session_state['run_counter']
                    print(f'Running live for {10*run_couter} s')
                    st.rerun()
    
    print(f'{datetime.now()}: Relion Live calculation done')
    st.sidebar.markdown(footer, unsafe_allow_html=True)


elif mode == "Run Relion (Experimental)":
    # Initializing the job database
    job_db = init_job_db()

    if os.name == 'nt':
        st.error('Detected Windows operating system. Running Relion requires Linux/Mac.')
        st.stop()
        
    # Initialize session states for various purposes
    if 'processes' not in st.session_state:
        st.session_state.processes = []
    if 'selected_process' not in st.session_state:
        st.session_state.selected_process = None
    if 'folder_selected' not in st.session_state:
        st.session_state.folder_selected = False
    if 'use_path' not in st.session_state:
        st.session_state.use_path = False
    if 'relion_directory' not in st.session_state:
        st.session_state.relion_directory = args.path
    if 'command' not in st.session_state:
        st.session_state.command = None

    # Input for working directory
    working_dir = st.text_input("**Enter the working directory:**", value=st.session_state['default_job_folder'])
    if working_dir != st.session_state['default_job_folder']:
        print(working_dir)
        
    st.session_state['default_job_folder'] = working_dir

    # Radio button for selecting mode - either run a new program or view running processes
    mode = st.sidebar.radio("Choose Mode", ["Run Program", "View Running Processes"])

    if mode == "Run Program":
        # Checkbox to decide if Relion should be searched in the system's PATH environment variable
        use_path = st.sidebar.checkbox('Search for Relion in $PATH?', value=st.session_state.use_path)
        st.session_state.use_path=use_path
        
        # List to store paths to Relion programs
        relion_programs = []

        # If 'use_path' is True, search for Relion executables in system's PATH
        if use_path:
            for directory in os.environ['PATH'].split(os.pathsep):
                relion_programs.extend(glob.glob(os.path.join(directory, 'relion*')))
            if len(relion_programs) == 0:
                st.sidebar.warning('Relion executable not found in PATH. Please provide the path below.')
                directory = st.sidebar.text_input('Provide Relion bin PATH:')
                relion_programs.extend(glob.glob(os.path.join(directory, 'relion*')))
                if len(relion_programs) == 0:
                    st.sidebar.warning(f'Relion executable not found in {directory}. Please provide the path below.')
        else:
            directory = st.sidebar.text_input('Provide Relion bin PATH:', value=st.session_state.relion_directory)
            relion_programs.extend(glob.glob(os.path.join(directory, 'relion*')))
            if len(relion_programs) == 0:
                st.sidebar.warning(f'Relion executable not found in {directory}. Please provide the path below.')
            else:
                st.session_state.relion_directory = directory

        # If Relion executables are found, allow user to select a command
        if st.session_state.folder_selected or len(relion_programs) != 0:
            relion_programs = sorted(set(relion_programs))
            selected_command = st.selectbox("Select Command", relion_programs, index=relion_programs.index(st.session_state.command) if st.session_state.command in relion_programs else 0)
            if selected_command != st.session_state.command:
                st.session_state.command = selected_command

            # If a command is selected, display its help text and generate widgets for parameters
            if st.session_state.command:
                # Execute help command only if a new command is selected
                try:
                    if 'motioncorr' in st.session_state.command:
                        st.session_state.command = st.session_state.command+' --use_own'
                    help_command = f"{st.session_state.command} --help"
                    help_text = run_command_get_output(help_command)
                    with st.expander('Parameters description:'):
                        st.code(help_text)
                    parameters = parse_help_text(help_text)
                    user_inputs, defaults, included_params = generate_widgets(parameters, st.session_state.command)

                    # Separate MPI and threads parameters if they exist
                    mpi_value = user_inputs.pop('mpi', None)
                    threads_value = user_inputs.pop('threads', None)

                    # Construct the full command
                    command_parts = []

                    if mpi_value:
                        command_parts.append(f"mpirun -n {mpi_value}")

                    command_parts.append(st.session_state.command)

                    command_parts += [f"{key} {value}" for key, value in user_inputs.items() if str(value) != str(defaults[key]) or included_params[key]]

                    if threads_value:
                        command_parts.append(f"--j {threads_value}")

                    full_command_str = " ".join(command_parts)

                    # Button to show the full command and add it to the processes list
                    if st.button("Show Command"):
                        st.session_state.processes.append({
                            'full_command': full_command_str,
                            'status': 'Ready',
                            'pid': None,
                            'stdout_file': None,
                            'stderr_file': None
                        })
                        st.code(full_command_str)
                except RuntimeError as e:
                    st.error(str(e))


        # Display button for the latest process only
        if st.session_state['processes']:
            latest_process = st.session_state['processes'][-1]
            if latest_process['status'] == 'Ready':
                basename = os.path.basename(latest_process['full_command'].split()[0])
                display_name = f"{basename}, job{len(st.session_state['processes']):03d}"
                if st.button(f"Run Command: {display_name}", key="run_latest"):
                    with st.spinner('Running...'):
                        # Change working directory if specified
                        cwd = working_dir if working_dir else None
                        pid, stdout_file, stderr_file = run_command_with_cwd(shlex.split(latest_process['full_command']), cwd)
                        latest_process.update({
                            'pid': pid,
                            'stdout_file': stdout_file,
                            'stderr_file': stderr_file,
                            'status': 'Running'
                        })
                        st.success(f"Command is running with PID: {pid}")
                        
    elif mode == "View Running Processes":
        process_options = [f"PID: {p['pid']} - {p['full_command']}" for p in st.session_state['processes'] if p['status'] == 'Running']
        selected_process_option = st.sidebar.radio("Select process to view", process_options)

        for process in st.session_state['processes']:
            process_label = f"PID: {process['pid']} - {process['full_command']}"
            if process_label == selected_process_option:
                st.session_state['selected_process'] = process
                show_output(process)
                break

