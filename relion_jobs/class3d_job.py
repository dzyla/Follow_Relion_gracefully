# class3d_job.py
import os
import glob
import logging
import traceback
import math
from datetime import datetime
from typing import List
import re

import numpy as np
import pandas as pd
import mrcfile
import streamlit as st
from plotly.subplots import make_subplots

# Shared utilities from your project.
from lib.utils import (
    parse_star,
    interactive_scatter_plot,
    report_error,
    get_angles,
    get_classes, 
    get_note,
)
from lib.image_utils import (downsample_volume, plot_angular_distribution_sphere, plot_fsc_stats, plot_class_resolution, 
                         plot_class_distribution, plot_projections, plot_volume, plot_angular_distribution_heatmap)


logger = logging.getLogger("main_app")



def _load_and_cache_job_data(rln_folder: str, job_name: str, nodes: List[str]) -> dict:
    """
    Load all volumes, class distributions, resolutions, etc. for the given job_name.
    Downsample volumes and store everything in st.session_state[data_key].
    """
    data_key = f"class3d_data_{job_name}"
    if data_key not in st.session_state:
        st.session_state[data_key] = {}

    # If already loaded, just return
    if st.session_state[data_key].get("loaded", False):
        logger.debug(f"Data for job '{job_name}' found in session state.")
        return st.session_state[data_key]

    with st.spinner("Loading job data from disk..."):
        # 1) Construct path to job folder
        path_data = os.path.join(rln_folder, job_name)

        # 2) Gather model.star files
        model_files = glob.glob(os.path.join(path_data, "*model.star"))
        model_files.sort(key=os.path.getmtime)
        logger.debug(f"Found {len(model_files)} model.star files for job '{job_name}': {model_files}")

        # get symmetry information from the note
        symmetry = 'C1'
        note = get_note(os.path.join(path_data, 'note.txt'))
        logger.debug(f"Note file content: {note}")
        
        # find --sym XXX in the note using re
        if note:
            match = re.search(r'--sym\s+([A-Z]\d+)', note)
            if match:
                symmetry = match.group(1)
                logger.debug(f"Found symmetry: {symmetry}")

        # 3) If no model.star, fallback to MRC in the nodes
        if len(model_files) == 0:
            logger.info("No *_model.star files found. Will try to read MRC volumes directly.")
            mrc_list = [f for f in nodes if f.lower().endswith(".mrc")]
            logger.debug(f"original MRC files: {mrc_list}")
            
            if any("merged" in f for f in mrc_list):
                mrc_list = [f for f in mrc_list if "merged" in f]
            elif any("half1" in f for f in mrc_list):
                mrc_list = [f for f in mrc_list if "half1" in f]
            
            logger.debug(f"Filtered MRC files: {mrc_list}")
            
            volumes = []
            for mrc_file in mrc_list:
                volume_path = os.path.join(rln_folder, mrc_file)
                try:
                    logger.debug(f"Loading MRC volume: {volume_path}")
                    with mrcfile.mmap(volume_path, permissive=True) as mrcf:
                        volumes.append(mrcf.data)
                except Exception:
                    error = traceback.format_exc()
                    logger.error(f"Error displaying volume {mrc_file}:\n{error}")
            st.session_state[data_key] = {
                "volumes_downsampled": volumes,
                "class_dist": [1.0] * len(volumes),
                "class_res": np.array([]),
                "fsc_res": np.array([]),
                "fsc_vals": np.array([]),
                "angles": ([], [], []),
                "star_block": None,
                "n_classes": len(volumes),
                "class_paths": mrc_list,
                "loaded": True,
                "symmetry": symmetry,
            }
            return st.session_state[data_key]

        # 4) If we have model.star, parse them
        try:
            (class_paths, n_classes, _iter_count,
             class_dist, class_res, fsc_res, fsc_vals) = get_classes(path_data, model_files)
        except Exception as e:
            report_error(e)
            logger.error(f"Error in get_classes: {e}")
            st.session_state[data_key] = {}
            return {}

        # 5) Load volumes
        volumes_raw = []
        for cls_path in class_paths:
            full_path = os.path.join(path_data, os.path.basename(cls_path))
            logger.debug(f"Loading volume: {full_path}")
            with mrcfile.mmap(full_path, permissive=True) as mrcf:
                volumes_raw.append(mrcf.data)

        # 6) Angles
        try:
            rot, tilt, psi = get_angles(path_data)
        except Exception as e:
            logger.debug(f"Failed to load angles from {path_data}: {e}")
            rot, tilt, psi = ([], [], [])

        # 7) Attempt to parse star file for star_block
        star_block = None
        try:
            if nodes:
                star_main = parse_star(os.path.join(path_data, os.path.basename(nodes[0])))
                star_block = star_main.get("particles", pd.DataFrame())
        except Exception as e:
            logger.debug(f"Failed to parse star file from nodes[0]: {e}")

        # 8) Store in st.session_state
        st.session_state[data_key] = {
            "volumes_raw": volumes_raw,
            "volumes_downsampled": [],
            "class_dist": class_dist,
            "class_res": class_res,
            "fsc_res": fsc_res,
            "fsc_vals": fsc_vals,
            "angles": (rot, tilt, psi),
            "star_block": star_block,
            "n_classes": n_classes,
            "class_paths": class_paths,
            "loaded": True,
            "symmetry": symmetry,
        }

    return st.session_state[data_key]


def _ensure_downsampled(job_data: dict, map_resize: int) -> List[np.ndarray]:
    """
    Ensure the volumes in session_state are downsampled to 'map_resize' size.
    Only redo if the stored version does not match the requested size.
    """
    if not job_data.get("volumes_raw", []):
        return job_data.get("volumes_downsampled", [])

    # Check if we already have downsampled volumes for this size
    # For clarity, we store them in job_data["volumes_downsampled_{size}"] or a single key with the size
    cached_size = job_data.get("cached_map_resize", None)
    if cached_size == map_resize and job_data.get("volumes_downsampled", []):
        logger.debug(f"Using cached volumes downsampled to {map_resize}.")
        return job_data["volumes_downsampled"]

    # Otherwise, downsample now
    new_downsampled = []
    for vol in job_data["volumes_raw"]:
        dvol = downsample_volume(vol, map_resize)
        new_downsampled.append(dvol)

    job_data["volumes_downsampled"] = new_downsampled
    job_data["cached_map_resize"] = map_resize
    return new_downsampled


def plot_combined_classes(volumes, class_dist, session_key="plot_combined_classes"):
    """
    Display multiple 3D volumes as isosurfaces in a grid of Plotly subplots,
    with user controls for threshold, resizing, columns, and row height.
    Uses the `plot_volume` function to generate iso-surfaces, caching the
    results in st.session_state to avoid repeated computation.

    'session_key' can incorporate a job-specific identifier to keep caches separate.
    The background for each subplot is set to black.
    """
    logger.debug(f"{datetime.now()}: plot_combined_classes called with {len(volumes)} volumes.")

    # Initialize session state cache for this session_key if needed.
    if session_key not in st.session_state:
        st.session_state[session_key] = {}
    
    # Retrieve local iso-surface cache.
    iso_cache = st.session_state.get("iso_surface_cache", {})
    #logger.debug(f"Current iso_surface_cache: {iso_cache}")

    volumes_n = len(volumes)
    if volumes_n < 5:
        columns_to_show = volumes_n
        if volumes_n == 1:
            plot_height = 700
        else:
            plot_height = 600
    else:
        columns_to_show = 5
        plot_height = 300
        
    

    # Basic UI controls.
    c1, c2 = st.columns(2)
    with c1:
        threshold = st.slider("Select Volume Threshold (Fraction)", 0.0, 1.0, 0.5, 0.01)
        map_resize = st.slider("Map size (px)", 64, 256, 150, 2)
    with c2:
        n_columns = st.slider("Number of columns", min_value=1, max_value=5, value=columns_to_show, step=1)
        row_height = st.slider("Plot height", min_value=100, max_value=1000, value=plot_height, step=100)
    
    show_class = st.checkbox("Show class volumes?", value=True)
    if not show_class:
        st.info("Class volumes are hidden. Check the box to show them.")
        return

    num_classes = len(volumes)
    cols = min(n_columns, num_classes)
    rows = math.ceil(num_classes / n_columns)

    # Create subplots with type 'scene' for 3D.
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)])
    
    annotations = []
    for idx, volume in enumerate(volumes):
        row_idx, col_idx = divmod(idx, cols)
        # Create a cache key from session_key, volume index, threshold, and map_resize.
        cache_key = (session_key, idx, threshold, map_resize)
        if cache_key in iso_cache:
            fig_ = iso_cache[cache_key]
            logger.debug(f"Reusing cached iso-surface for volume {idx+1}.")
        else:
            with st.spinner(f"Creating iso-surface for class {idx+1}..."):
                fig_ = plot_volume(volume, threshold, max_size=map_resize)
            iso_cache[cache_key] = fig_
        
        if fig_ and len(fig_.data) > 0:
            fig.add_trace(fig_.data[0], row=row_idx+1, col=col_idx+1)
            dist_percent = 0.0
            if idx < len(class_dist):
                dist_percent = round(float(class_dist[idx]) * 100, 2)
            ann_text = f"Class {idx+1}<br>Dist: {dist_percent}%"
            x_ = (col_idx + 0.5) / cols
            y_ = 1 - (row_idx / rows) - 0.05
            annotations.append(
                dict(
                    x=x_,
                    y=y_,
                    xref="paper",
                    yref="paper",
                    text=ann_text,
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=12),
                )
            )
    
    # Set overall layout with black background.
    fig.update_layout(
        hovermode=False,
        annotations=annotations,
        height=rows * row_height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black",  # overall background color
    )
    fig.update_layout(scene_dragmode='orbit', title='3D Volume Isosurfaces')
    
    # Configure each subplot's scene with a black background.
    for n in range(num_classes):
        scene_id = f"scene{n+1}" if n > 0 else "scene"
        fig.update_layout(
            **{
                scene_id: dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=dict(eye=dict(x=2.5, y=2.5, z=2.5)),
                    bgcolor="black"  # set each scene's background to black.
                )
            }
        )
    
    # Store the updated cache back into session state.
    st.session_state["iso_surface_cache"] = iso_cache
    
    st.plotly_chart(fig, use_container_width=True)


def plot_class3d(rln_folder: str, nodes: List[str]) -> None:
    """
    Main function to plot the results of a 3D classification or refinement job,
    caching data in session state to avoid repeated I/O and repeated downsampling.

    This function:
      1) Clears/sets a job-specific session state if the job changed.
      2) Loads data for the job, downsampling volumes as needed.
      3) Displays combined classes, volume projections, class distributions, etc.
    """
    logger.debug(f"{datetime.now()}: plot_class3d called with nodes={nodes}")
    if not nodes:
        st.warning("No files provided in 'nodes'.")
        return

    job_name = os.path.dirname(nodes[0])  # Basic extraction of job folder name
    if not job_name:
        st.warning("Could not determine job name from nodes.")
        return

    # Check if this is a new job
    if "current_class3d_job" not in st.session_state or st.session_state["current_class3d_job"] != job_name:
        # It's a different job => reset iso_surface_cache or use a job-specific key
        st.session_state["current_class3d_job"] = job_name
        st.session_state["iso_surface_cache"] = {}  # Clear old caches for iso surfaces

    if 'Class3D' in job_name:
        job_type = 'Class3D'
    elif 'Refine3D' in job_name:
        job_type = 'Refine3D'
    elif 'Reconstruct' in job_name:
        job_type = 'Reconstruct'
    else:
        job_type = '3D Classification'
    
    st.subheader(f"{job_type} job: {job_name}")
    
    # 1) Load or retrieve data from session state
    job_data = _load_and_cache_job_data(rln_folder, job_name, nodes)
    if not job_data.get("loaded", False):
        st.warning("Could not load the job data properly.")
        return

    # 2) If user modifies map_resize, only re-downsample volumes if needed
    #  Optional: you can store 'plot_combined_classes_map_resize' in st.session_state
    map_resize_request = 150
    if "plot_combined_classes_map_resize" in st.session_state:
        map_resize_request = st.session_state["plot_combined_classes_map_resize"]
    with st.spinner("Downsampling Volumes", show_time=True):
        volumes_downsampled = _ensure_downsampled(job_data, map_resize_request)
    n_classes = job_data["n_classes"]
    class_dist = job_data["class_dist"]
    class_res = job_data["class_res"]

    # Ensure FSC data is NumPy arrays
    fsc_res = np.array(job_data["fsc_res"], dtype=float) if isinstance(job_data["fsc_res"], (list, tuple)) else job_data["fsc_res"]
    fsc_vals = np.array(job_data["fsc_vals"], dtype=float) if isinstance(job_data["fsc_vals"], (list, tuple)) else job_data["fsc_vals"]

    rot, tilt, psi = job_data["angles"]
    # Ensure angles are NumPy arrays not pandas Series
    rot = np.array(rot, dtype=float) if isinstance(rot, (pd.Series)) else rot
    tilt = np.array(tilt, dtype=float) if isinstance(tilt, (pd.Series)) else tilt
    psi = np.array(psi, dtype=float) if isinstance(psi, (pd.Series)) else psi
    
    
    star_block = job_data["star_block"]
    
    logger.debug(f'rot: {rot}, tilt: {tilt}, psi: {psi}')

    # Determine final distribution if multiple iterations
    if isinstance(class_dist, np.ndarray) and class_dist.ndim == 2:
        class_dist_final = class_dist[:, -1]  # last iteration
    else:
        class_dist_final = class_dist

    if n_classes == 0 and len(volumes_downsampled) == 0:
        st.warning("No 3D class references found.")
        return

    st.write(f"Found {n_classes} classes in final iteration.")
    if len(volumes_downsampled) < n_classes:
        pass

    # 3) Show 3D volumes
    with st.expander("Combined 3D Volumes", expanded=True):
        # use a job-specific session key => "plot_combined_classes_{job_name}"
        job_session_key = f"plot_combined_classes_{job_name}"
        with st.spinner("plotting combined classes..."):
            plot_combined_classes(volumes_downsampled, class_dist_final, session_key=job_session_key)

    # 4) Volume projections
    with st.expander("Volume Projections", expanded=True):
        plot_projections(volumes_downsampled, class_dist_final)

    # 5) Class distribution
    if n_classes > 1 and isinstance(class_dist, np.ndarray) and class_dist.ndim == 2:
        with st.expander("Class Distribution Over Iterations", expanded=True):
            plot_class_distribution(class_dist)

    # 6) FSC for "Refine3D" jobs
    if "Refine3D" in job_name and fsc_res.size > 0 and fsc_vals.size > 0:
        col1, col2 = st.columns(2)
        with col1.expander("Fourier Shell Correlation (FSC) Stats", expanded=True):
            plot_fsc_stats(fsc_res, fsc_vals)

    # 7) Class resolution
    if isinstance(class_res, np.ndarray) and class_res.size > 0:
        if "Refine3D" in job_name:
            with col2.expander("Class Resolution", expanded=True):
                plot_class_resolution(class_res)
        else:
            with st.expander("Class Resolution", expanded=True):
                plot_class_resolution(class_res)

    # 8) Angular distribution
    try:
        if n_classes > 1 and star_block is not None and "_rlnClassNumber" in star_block.columns:
            cls_idx = star_block["_rlnClassNumber"]
        else:
            cls_idx = None

        if not (rot.all() and tilt.all() and psi.all()):
            logger.info("No angles found for angular distribution.")
            return
        else:
            with st.expander("Angular Distribution", expanded=True):
                plot_type = st.radio("Select plot type:", ("3D", "2D"), index=0, horizontal=True)
                # Example: 3D sphere representation
                # or you can call plot_angular_distribution(...) for 2D histograms
                if plot_type == "3D":
                    plot_angular_distribution_sphere(psi, rot, tilt,  cls_idx, symmetry=job_data["symmetry"])
                else:
                    plot_angular_distribution_heatmap(psi, rot, tilt, cls_idx, symmetry=job_data["symmetry"])

        # Show star file scatter
        if st.checkbox("Show star file interactive scatter?"):
            interactive_scatter_plot(os.path.join(rln_folder, nodes[0]))

    except Exception as e:
        st.warning("No angle or particle data found for angular distribution.")
        logger.debug(f"Angular distribution error: {report_error(e)}")

    logger.info(f"{datetime.now()}: plot_class3d done.")