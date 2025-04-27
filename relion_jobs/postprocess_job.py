#postprocess_job.py

import os
import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from skimage.transform import resize
import mrcfile
import seaborn as sns


# Import your shared utilities.
from lib.utils import parse_star, interactive_scatter_plot
from lib.image_utils import normalize, display_volume_slices, plot_volume
from relion_jobs.select_job import display_classes  

# Setup logger
logger = logging.getLogger("main_app")


def _load_postprocess_data(rln_folder: str, postprocess_star_path: str) -> dict:
    """
    Load the postprocess.star file using parse_star and cache the result in session state.
    If the current postprocess file is the same as the cached one, return the cached data.
    """
    cache_key = "postprocess_data"
    current_file_key = "current_postprocess_file"
    if (
        current_file_key in st.session_state 
        and st.session_state[current_file_key] == postprocess_star_path 
        and cache_key in st.session_state
    ):
        logger.debug("Using cached postprocess star data.")
        return st.session_state[cache_key]

    try:
        logger.debug(f"Loading postprocess star file: {postprocess_star_path}")
        postprocess_data = parse_star(postprocess_star_path)
        st.session_state[cache_key] = postprocess_data
        st.session_state[current_file_key] = postprocess_star_path
        return postprocess_data
    except Exception as e:
        logger.error(f"Error loading postprocess star file: {e}")
        return {}


def plot_postprocess(rln_folder, nodes):
    """
    Main function to plot postprocess data in Streamlit.
    This function:
      1) Loads (or retrieves from session state) the postprocess.star file.
      2) Plots the FSC and Guinier curves side by side.
      3) Displays masked volume slices.
    """
    logger.debug(f"{datetime.now()}: plot_postprocess started for {rln_folder} with {nodes}")
    
    # Assume postprocess.star file is the 4th node.
    postprocess_star_path = os.path.join(rln_folder, nodes[3])
    if not os.path.exists(postprocess_star_path):
        st.write("No postprocess.star file found")
        return

    # Load and cache postprocess data.
    postprocess_data = _load_postprocess_data(rln_folder, postprocess_star_path)
    if not postprocess_data:
        st.write("Failed to load postprocess.star data")
        return

    # Directly use Plotly for all plotting (remove Seaborn option).
    try:
        fsc_data = postprocess_data["fsc"].astype(float)
        guinier_data = postprocess_data["guinier"].astype(float)
    except Exception as e:
        st.write("Error processing postprocess.star data")
        logger.error(f"Error converting FSC/Guinier data to float: {e}")
        return

    st.write("## Postprocess Data Visualization")
    # Arrange FSC and Guinier plots into two columns.
    col_fsc, col_guinier = st.columns(2)
    with col_fsc:
        st.markdown("### Fourier Shell Correlation (FSC) Curve")
        plot_fsc_curve(fsc_data)
    with col_guinier:
        st.markdown("### Guinier Plot")
        plot_guinier_curve(guinier_data)

    # Display Masked Volume Slices.
    with st.expander("Volume preview"):
        display_volume_slices(rln_folder, nodes)

    logger.info(f"{datetime.now()}: plot_postprocess done")


def plot_fsc_curve(fsc_data, dpi=150):
    """
    Plot the Fourier Shell Correlation (FSC) curve using Plotly.
    FSC curves for different map types are colored using Seaborn's 'deep' palette.
    """
    # Retrieve FSC resolution data and convert to float.
    fsc_x = fsc_data["_rlnAngstromResolution"].astype(float)
    fsc_x_min = np.min(fsc_x)

    # FSC fields to plot.
    fsc_to_plot = [
        "_rlnFourierShellCorrelationCorrected",
        "_rlnFourierShellCorrelationUnmaskedMaps",
        "_rlnFourierShellCorrelationMaskedMaps",
        "_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps",
    ]

    # Use Seaborn's default "deep" palette for colors.
    palette = sns.color_palette("deep").as_hex()

    fig = go.Figure()
    for i, meta in enumerate(fsc_to_plot):
        color = palette[i % len(palette)]
        reciprocal_x = 1 / fsc_x
        fig.add_trace(
            go.Scatter(
                x=reciprocal_x,
                y=fsc_data[meta].astype(float),
                mode="lines",
                line=dict(color=color, width=3),
                customdata=fsc_x,
                name=meta.replace("_rlnFourierShellCorrelation", "").replace("_rlnCorrectedFourierShellCorrelation", ""),
                hovertemplate="Resolution: %{customdata:.2f} Å<br>FSC: %{y:.2f}<extra></extra>",
            )
        )

    # Add horizontal threshold lines.
    fig.add_hline(y=0.143, line_dash="dash", line_color="black", annotation_text="0.143")
    fig.add_hline(y=0.5, line_dash="dash", line_color="black", annotation_text="0.5")

    # Determine axis range and custom ticks.
    start_res = 1 / 50
    end_res = 1 / fsc_x_min
    custom_ticks = np.linspace(start_res, end_res, num=10)
    fig.update_layout(
        xaxis=dict(
            title="Resolution, Å",
            tickvals=custom_ticks,
            ticktext=[f"{round(1/res, 2)}" for res in custom_ticks],
            range=[start_res, end_res],
        ),
        yaxis=dict(title="FSC", range=[-0.05, 1.05]),
        title="Fourier Shell Correlation Curve",
        legend=dict(x=1, y=1, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0)"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Resolution annotations.
    fsc143 = 0.143
    idx_143 = np.argmin(np.abs(fsc_data["_rlnFourierShellCorrelationCorrected"].astype(float) - fsc143))
    fsc05 = 0.5
    idx_05 = np.argmin(np.abs(fsc_data["_rlnFourierShellCorrelationCorrected"].astype(float) - fsc05))
    st.markdown(f"Reported resolution @ FSC=0.143: **{round(fsc_x[idx_143], 2)} Å**")
    st.markdown(f"Reported resolution @ FSC=0.5: **{round(fsc_x[idx_05], 2)} Å**")

def plot_guinier_curve(guinier_data):
    """
    Plot the Guinier curve using Plotly.
    """
    guiner_x = guinier_data["_rlnResolutionSquared"].astype(float)
    guinier_to_plot = [
        "_rlnLogAmplitudesOriginal",
        "_rlnLogAmplitudesMTFCorrected",
        "_rlnLogAmplitudesWeighted",
        "_rlnLogAmplitudesSharpened",
        "_rlnLogAmplitudesIntercept",
    ]
    
    palette = sns.color_palette("deep").as_hex()
    
    fig = go.Figure()
    for i, meta in enumerate(guinier_to_plot):
        color = palette[i % len(palette)]
        try:
            y_data = guinier_data[meta].astype(float)
            # Replace invalid values (-99) with NaN.
            y_data[y_data == -99] = float("nan")
            fig.add_trace(
                go.Scatter(
                    x=guiner_x[1:],
                    y=y_data[1:],
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=meta.replace("_rlnLogAmplitudes", ""),
                )
            )
        except Exception:
            pass

    fig.update_layout(
        title="Guinier Plot",
        xaxis_title="Resolution Squared, [1/Å²]",
        yaxis_title="Ln(Amplitudes)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)



