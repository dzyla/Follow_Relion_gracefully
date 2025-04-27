# picking_job.py

import os
import glob
import tempfile
import logging
from datetime import datetime
from typing import List, Tuple, Union

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from lib.utils import parse_star, star_from_df, interactive_scatter_plot, report_error
from lib.image_utils import micrograph_viewer

logger = logging.getLogger("main_app")

def plot_histogram_particles_per_mic(
    particles_per_mic: Union[pd.Series, np.ndarray]
) -> go.Figure:
    """
    Create a histogram of particles per micrograph.
    
    Parameters:
        particles_per_mic: Series or array of particle counts.
    
    Returns:
        A Plotly Figure.
    """
    try:
        fig = px.histogram(particles_per_mic, nbins=30)
        fig.update_layout(
            title="Histogram of Particles Per Micrograph",
            xaxis_title="Particles",
            yaxis_title="Micrographs Count",
        )
        return fig
    except Exception as exc:
        report_error(exc)
        logger.error("Error in plot_histogram_particles_per_mic.")
        return go.Figure()


def plot_histogram_fom(
    figure_of_merit: Union[pd.Series, np.ndarray]
) -> go.Figure:
    """
    Create a histogram of autopick figure-of-merit.
    
    Parameters:
        figure_of_merit: Series or array of FOM values.
    
    Returns:
        A Plotly Figure.
    """
    try:
        fig = px.histogram(figure_of_merit, nbins=50)
        fig.update_layout(
            title="Histogram of Autopick Figure of Merit",
            xaxis_title="Autopick Figure of Merit",
            yaxis_title="Count",
        )
        return fig
    except Exception as exc:
        report_error(exc)
        logger.error("Error in plot_histogram_fom.")
        return go.Figure()


def get_coord_paths(job_folder: str, rln_folder: str) -> Tuple[List[str], List[str]]:
    """
    Retrieve coordinate file paths and corresponding micrograph paths.
    
    Checks in order:
        - autopick.star
        - manualpick.star
        - files matching "coords_suffix_*"
    
    Parameters:
        job_folder (str): Path to the job folder.
        rln_folder (str): Base folder for micrographs.
    
    Returns:
        A tuple (coord_paths, mics_paths).
    """
    try:
        # Case 1: autopick.star
        autopick_star_path = os.path.join(job_folder, "autopick.star")
        if os.path.exists(autopick_star_path) and os.path.getsize(autopick_star_path) > 0:
            autopick_star = parse_star(autopick_star_path)["coordinate_files"]
            mics_paths = autopick_star["_rlnMicrographName"].to_numpy().tolist()
            coord_paths = autopick_star["_rlnMicrographCoordinates"].to_numpy().tolist()
            return coord_paths, mics_paths

        # Case 2: manualpick.star
        manualpick_star_path = os.path.join(job_folder, "manualpick.star")
        if os.path.exists(manualpick_star_path) and os.path.getsize(manualpick_star_path) > 0:
            manpick_star = parse_star(manualpick_star_path)["coordinate_files"]
            mics_paths = manpick_star["_rlnMicrographName"].to_numpy().tolist()
            coord_paths = manpick_star["_rlnMicrographCoordinates"].to_numpy().tolist()
            return coord_paths, mics_paths

        # Case 3: coords_suffix_* pattern
        suffix_files = glob.glob(os.path.join(job_folder, "coords_suffix_*"))
        if suffix_files:
            suffix_file = suffix_files[0]
            suffix = os.path.basename(suffix_file).replace("coords_suffix_", "").replace(".star", "")
            with open(suffix_file, "r") as f:
                mics_data_path = f.readline().strip()
            all_mics_paths = parse_star(os.path.join(rln_folder, mics_data_path))["micrographs"]["_rlnMicrographName"]
            mics_paths = [os.path.join(rln_folder, name) for name in all_mics_paths]
            coord_paths = [
                os.path.join(
                    job_folder,
                    f"coords_{suffix}",
                    os.path.basename(mic_path).replace(".mrc", f"_{suffix}.star")
                )
                for mic_path in mics_paths
            ]
            return coord_paths, mics_paths

        # Case 4: No matching pattern found.
        return [], []
    except Exception as exc:
        report_error(exc)
        logger.error("Error in get_coord_paths.")
        return [], []





def plot_picks(
    rln_folder: str, job_name: str, img_resize_fac: float = 0.2
) -> None:
    """
    Display picking statistics and overlay picks on the selected micrograph using micrograph_viewer.
    Determines the coordinate source (autopick.star, manualpick.star, or coords_suffix_*)
    and passes the computed picks to micrograph_viewer (which now supports optional picks overlay).
    Also supports Topaz training statistics if no coordinate files are found.
    
    Parameters:
        rln_folder (str): Base folder for job and micrograph data.
        job_name (str): Name of the job folder.
        img_resize_fac (float): Initial resize factor.
    
    Returns:
        None.
    """
    
    logger.debug(f"{datetime.now()}: plot_picks started with job_name: {job_name}")
    try:
        path_data = os.path.join(rln_folder, job_name)
        coord_paths, mics_paths = get_coord_paths(path_data, job_name)
        
        logger.debug(f"coord_paths: {coord_paths}")
        
        # Fallback for Topaz training statistics.
        if not coord_paths:
            topaz_training_files = glob.glob(os.path.join(path_data, "model_training.txt"))
            if topaz_training_files:
                topaz_training_txt = topaz_training_files[0]
                data = pd.read_csv(topaz_training_txt, delimiter="\t")
                data_test = data[data["split"] == "test"]
                x = data_test["epoch"]
                data_test = data_test.drop(["iter", "split", "ge_penalty"], axis=1)
                fig = go.Figure()
                for column in data_test.columns:
                    if column != "epoch":
                        y = data_test[column]
                        fig.add_scatter(
                            x=x,
                            y=y,
                            name=column,
                            hovertemplate=f"{column}<br>Epoch: %{{x}}<br>Y: %{{y:.2f}}<extra></extra>",
                        )
                fig.update_xaxes(title_text="Epoch")
                fig.update_yaxes(title_text="Statistics")
                best_epoch = data_test[data_test["auprc"].astype(float) == np.max(data_test["auprc"].astype(float))]["epoch"].values
                fig.update_layout(title=f"Topaz training stats. Best model: {best_epoch}")
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                logger.info(f"{datetime.now()}: plot_picks_streamlit done (Topaz training)")
                return
            st.write("No coordinate files found.")
            logger.info(f"{datetime.now()}: plot_picks_streamlit done (No coordinate files)")
            return
        
        # Select a micrograph.
        col1, col2 = st.columns([1, 3])
        
        # For autopick data (non-manual), get FOM slider and compute picks overlay.
        if "ManualPick" not in job_name:
            # problem with indexing and passing it to micrograph_viewer######
            
            
            plot_all = col1.checkbox("Plot FOM statistics? (Might be slow for huge datasets)", value=False)
            fom_all_mics = []
            tmp_file_path = None
            if plot_all:
                coords_df = pd.DataFrame()
                for idx in range(len(mics_paths)):
                    try:
                        star_data = parse_star(os.path.join(rln_folder, coord_paths[idx]))
                        star_data = list(star_data.values())[0]
                        star_data["_rlnMicrographName"] = mics_paths[idx]
                        coords_df = pd.concat([coords_df, star_data], ignore_index=True)
                        fom_stats = star_data["_rlnAutopickFigureOfMerit"].astype(float)
                        fom_all_mics.extend(fom_stats)
                    except Exception as exc:
                        report_error(exc)
                        logger.error(f"Error processing picking stats for micrograph index {idx}")
                modified_star = star_from_df({"particles": coords_df})
                with tempfile.NamedTemporaryFile(delete=False, suffix=".star") as tmp_file:
                    modified_star.write_file(tmp_file.name)
                tmp_file_path = tmp_file.name
        else:
            fom_slider = [-10000, 10000]
            picks_overlay = None
            plot_all = False

        # Use micrograph_viewer to display the micrograph with picks overlay.
        # If picks_overlay is None, micrograph_viewer behaves as before.
        micrograph_viewer(
            rln_folder=rln_folder,
            image_files=mics_paths,
            selected_filter="gaussian",
            default_gaussian=0.2,
            coord_paths=coord_paths,
        )

        # For autopick, if "plot all" is enabled, show FOM histogram and detailed stats.
        if "ManualPick" not in job_name and plot_all and fom_all_mics:
            fom_all_mics = np.array(fom_all_mics)
            fom_histogram_fig = plot_histogram_fom(fom_all_mics)
            col2.plotly_chart(fom_histogram_fig, use_container_width=True)
            
            if st.checkbox("Show detailed statistics?") and tmp_file_path:
                interactive_scatter_plot(tmp_file_path)

        logger.info(f"{datetime.now()}: plot_picks_streamlit done")
    except Exception as exc:
        report_error(exc)
        logger.error("Error in plot_picks_streamlit function.")