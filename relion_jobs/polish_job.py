#polish_job.py

import os
import traceback
from datetime import datetime
from typing import List
import logging

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Import shared utilities (ensure these functions are defined in your project)
from lib.utils import parse_star
from lib.image_utils import report_error
from relion_jobs.extract_job import show_random_particles

logger = logging.getLogger("main_app")


def plot_polish(FOLDER: str, node_files: List[str]) -> None:
    """
    Main function to plot polish job results in Streamlit.
    
    Depending on the contents of node_files, this function handles:
      1) Optimal parameter reporting (if "opt_params_all_groups.txt" is found),
      2) B-factor and Guinier analysis (if "shiny.star" is found),
      3) Tomogram motion visualization (if "tomograms.star" is found).
    
    The module uses Plotly for plotting and caches loaded data in session state.
    """
    train_job = False
    
    logger.debug(f"{datetime.now()}: plot_polish called with folder: {FOLDER} and node_files: {node_files}")
    for file in node_files:
        if 'star' in file:
            star_file_path = os.path.join(FOLDER, file)
        elif 'opt_params_all_groups.txt' in file:
            
            star_file_path = os.path.join(FOLDER, file)
            train_job = True
    
    job_path = os.path.dirname(star_file_path)
    logger.debug(f"Job path: {job_path}")
    
    # Create a job-specific session state key.
    polish_key = f"polish_data_{job_path}"
    if polish_key not in st.session_state:
        st.session_state[polish_key] = {}
    cache = st.session_state[polish_key]

    # Case 1: Optimal parameters available.
    if any("opt_params_all_groups.txt" in element for element in node_files):
        logger.debug("Optimal parameters found.")
        params_path = star_file_path
        try:
            with open(params_path, "r") as f:
                parameters = f.readline().strip().split()
            st.markdown("### Optimal Parameters")
            st.code(f"--s_vel {parameters[0]} --s_div {parameters[1]} --s_acc {parameters[2]}")
        except Exception as exc:
            st.error(f"Error reading optimal parameters: {exc}")
            logger.error(f"Optimal parameters error: {exc}\n{traceback.format_exc()}")
        return

    # Case 2: B-factor analysis (shiny.star found).
    elif any("shiny.star" in element for element in node_files):
        try:
            bfactors_star_path = os.path.join(job_path, "bfactors.star")
            logger.debug(f"Loading B-factors from: {bfactors_star_path}")
            # Cache parsed bfactors data.
            bfactors_cache_key = f"bfactors_data_{job_path}"
            logger.debug(f"Cache key for bfactors: {bfactors_cache_key}")
            if bfactors_cache_key in cache:
                bfactors_data = cache[bfactors_cache_key]
                logger.debug("Using cached bfactors data.")
            else:
                bfactors_data = parse_star(bfactors_star_path)["perframe_bfactors"]
                cache[bfactors_cache_key] = bfactors_data

            # Build a Plotly figure with two traces.
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bfactors_data["_rlnMovieFrameNumber"],
                    y=bfactors_data["_rlnBfactorUsedForSharpening"],
                    mode="lines",
                    name="Bfactor Used For Sharpening",
                    line=dict(color="darkgoldenrod", width=3),
                    hovertemplate="Movie Frame: %{x}<br>Bfactor: %{y}<extra></extra>"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bfactors_data["_rlnMovieFrameNumber"],
                    y=bfactors_data["_rlnFittedInterceptGuinierPlot"],
                    mode="lines",
                    name="Fitted Intercept Guinier Plot",
                    line=dict(color="lightseagreen", width=3),
                    yaxis="y2",
                    hovertemplate="Movie Frame: %{x}<br>Intercept: %{y}<extra></extra>"
                )
            )
            fig.update_layout(
                title="Polish Job Statistics",
                xaxis=dict(title="Movie Frame Number"),
                yaxis=dict(title="Bfactor Used For Sharpening"),
                yaxis2=dict(
                    title="Fitted Intercept Guinier Plot", overlaying="y", side="right"
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            if st.checkbox(":gem: Show random shiny particles?"):
                # Assume show_random_particles is defined.
                c1, c2 = st.columns([1,4])
                show_random_particles(star_file_path, FOLDER, c1, c2)

        except Exception as exc:
            logger.error(f"{datetime.now()}: Error in bfactors branch:\n{report_error(exc)}")
            st.error("Error loading B-factors data")
        return

    # Case 3: Tomogram motion analysis.
    elif any("tomograms.star" in element for element in node_files):
        try:
            # Paths: use node_files[2] for particle star and node_files[3] for motion star.
            particle_star_path = os.path.join(FOLDER, node_files[2])
            motion_star_path = os.path.join(FOLDER, node_files[3])
            
            # Load particle star data and cache it.
            particle_cache_key = f"particle_star_{job_path}"
            if particle_cache_key in cache:
                particle_star = cache[particle_cache_key]
                logger.debug("Using cached particle star data.")
            else:
                particle_star = parse_star(particle_star_path)["particles"]
                # Keep only the relevant columns.
                particle_star = particle_star[
                    ["_rlnTomoName", "_rlnCenteredCoordinateXAngst",
                     "_rlnCenteredCoordinateYAngst", "_rlnCenteredCoordinateZAngst"]
                ]
                for col in particle_star.columns:
                    if col.startswith("_rln"):
                        try:
                            particle_star[col] = particle_star[col].astype(float)
                        except ValueError:
                            logger.warning(f"Column {col} cannot be converted to float.")
                
                #particle_star = convert_to_float(particle_star)
                cache[particle_cache_key] = particle_star

            unique_tomo_names = np.unique(particle_star["_rlnTomoName"])
            idx = 0
            
            c1, c2 = st.columns([1, 5])
            
            
            if unique_tomo_names.size > 1:
                idx = c1.slider("Tomogram index", 0, unique_tomo_names.size - 1, 0)
            tomo_name = unique_tomo_names[idx]
            particle_star_selected = particle_star[particle_star["_rlnTomoName"] == tomo_name]
            st.markdown(
                f"**Selected Tomogram:** {tomo_name}. **Number of particles:** {particle_star_selected.shape[0]}"
            )

            # Load or cache motion data.
            temp_folder = os.path.join(os.path.dirname(particle_star_path), "temp")
            temp_motion_file_path = os.path.join(temp_folder, f"{tomo_name}_motion.star")
            if os.path.exists(temp_motion_file_path):
                motion_cache_key = f"motion_data_{tomo_name}"
                if motion_cache_key in cache:
                    motion_file = cache[motion_cache_key]
                    logger.debug("Using cached motion star data.")
                else:
                    motion_file = parse_star(temp_motion_file_path)
                    cache[motion_cache_key] = motion_file

                # Calculate total motion per tomogram block.
                all_motion = []

                for key in motion_file.keys():
                    try:
                        motion_df = motion_file[key].astype(float, errors='ignore')
                        x_trace = np.mean(motion_df["_rlnOriginXAngst"])
                        y_trace = np.mean(motion_df["_rlnOriginYAngst"])
                        z_trace = np.mean(motion_df["_rlnOriginZAngst"])
                        total_motion = np.abs(x_trace + y_trace + z_trace)
                        all_motion.append(total_motion)
                    except Exception as exc:
                        logger.error(f"Error processing motion data for key {key}: {exc}")
                        all_motion.append(0)
                
                # Create a 3D scatter plot for particle positions, colored by total motion.
                scatter = go.Scatter3d(
                    x=particle_star_selected["_rlnCenteredCoordinateXAngst"],
                    y=particle_star_selected["_rlnCenteredCoordinateYAngst"],
                    z=particle_star_selected["_rlnCenteredCoordinateZAngst"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=all_motion,
                        colorscale="Inferno",
                        colorbar=dict(title="Total Motion"),
                    ),
                    name="Particles",
                    hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>"
                )
                fig_motion = go.Figure(data=[scatter])
                fig_motion.update_layout(
                    scene=dict(
                        xaxis_title="X (Å)",
                        yaxis_title="Y (Å)",
                        zaxis_title="Z (Å)",
                        aspectmode="data",
                    ),
                    height=600,
                    title="Total Motion of Particles in Tomogram (summed XYZ motion)",
                    showlegend=False,
                )
                c2.plotly_chart(fig_motion, use_container_width=True)
            else:
                # Fallback: if the temp motion file does not exist, call process_tomo_motion.
                st.warning("No temporary motion file found; please run the motion correction step.")
                # You may call: process_tomo_motion(motion_star_path)

        except Exception as exc:
            logger.error(f"{datetime.now()}: Error in tomogram branch: {exc}\n{traceback.format_exc()}")
            st.error("Error processing tomogram motion data.")
        return

    else:
        st.write("No relevant data found for the Polish job.")

    logger.info(f"{datetime.now()}: plot_polish done")


def process_tomo_motion(filename: str):
    """
    Example function to process a tomogram motion star file.
    Splits the file into blocks and returns a list of (block_name, DataFrame).
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        blocks = []
        block = []
        for line in lines:
            if line.startswith('data_'):
                if block:
                    blocks.append(block)
                block = [line]
            else:
                block.append(line)
        if block:
            blocks.append(block)
        
        dataframes = []
        for block in blocks:
            block_name = block[0].strip()
            data_section_started = False
            columns = []
            data_rows = []
            for line in block:
                line = line.strip()
                if line.startswith('_rln'):
                    columns.append(line.split()[0])
                elif data_section_started:
                    if line.startswith('#') or 'None' in line:
                        continue
                    data_row = line.split()
                    data_rows.append(data_row)
                elif line.startswith('loop_'):
                    data_section_started = True
            df = pd.DataFrame(data_rows, columns=columns).dropna()
            df = df.astype(float, errors='ignore')
            dataframes.append((block_name, df))
        return dataframes
    except Exception as exc:
        logger.error(f"Error processing tomogram motion file: {exc}")
        st.error("Error processing tomogram motion file.")
        return []
