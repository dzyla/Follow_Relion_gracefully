# class2d_job.py
"""
Module handling 2D classification visualization and class selection.
"""

import os
import glob
import logging
from datetime import datetime
from typing import List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Import your existing shared utilities. Adjust imports as needed.
# ------------------------------------------------------------------
from lib.utils import (
    interactive_scatter_plot,
    get_classes,
    report_error,
)
from relion_jobs.select_job import display_classes  # or wherever display_classes is defined


logger = logging.getLogger("main_app")


@st.fragment
def plot_class_distribution(class_dist_: np.ndarray, PLOT_HEIGHT=500):
    """
    Plots iteration-by-iteration distribution for 2D classes.
    Expects class_dist_ shape = (num_classes, num_iterations).
    """
    import plotly.graph_objects as go

    try:
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
                    hovertemplate=(
                        f"Class {n + 1}<br>Iteration: %{{x}}"
                        "<br>Cls dist: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                    mode="lines",
                    stackgroup="one",
                )
            )

        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Class distribution (%)")
        fig.update_layout(title="Class distribution over iterations")
        fig.update_layout(hovermode="x unified", height=PLOT_HEIGHT)

        st.plotly_chart(fig, use_container_width=True, height=PLOT_HEIGHT)
    except Exception as exc:
        report_error(exc)
        st.error("Error plotting class distribution.")


def plot_class2d(rln_folder: str, nodes: List[str]) -> None:
    """
    Show 2D classes for a given job, let user select classes, generate new star file,
    and optionally plot iteration-by-iteration distribution or do interactive plotting.

    Parameters:
        rln_folder: Root folder for the job data.
        nodes: List of relevant star/mrc/etc. files produced by the job.
    """
    logger.debug(f"{datetime.now()}: plot_class2d started with {nodes}")
    st.subheader("2D Classification Job")

    try:
        # 1) Derive the job base folder from the first node path
        job_basefolder = os.path.join(rln_folder, os.path.dirname(nodes[0]))
        data_star = os.path.join(job_basefolder, nodes[0])
        logger.debug(f"Job base folder: {job_basefolder}")

        # 2) Gather model.star files
        model_files = sorted(
            glob.glob(os.path.join(job_basefolder, "*model.star")),
            key=os.path.getmtime,
        )
        if not model_files:
            st.write("No model files found.")
            return

        # 3) Basic info about classes from the last model
        class_paths, n_classes, iter_, class_dist, class_res, _, _ = get_classes(
            job_basefolder, [model_files[-1]]
        )
        class_dist = np.squeeze(class_dist)
        logger.debug(f"Class paths: {class_paths}")

        if not len(class_paths):
            st.write("No classes found in the last model.")
            return

        # We place the class display in an expander so it is collapsible
        with st.expander("Class Averages and Selections", expanded=True):
            # This controls the UI for sorting classes
            col_sort, _ = st.columns([1, 10])
            sort_classes = col_sort.checkbox("Sort classes?", True)

            # Show each class path
            for class_path in class_paths:
                # `display_classes` is from relion_jobs.select_job
                # Keep the call arguments as is for backward compatibility
                logger.debug(f"2D Class path: {class_path}")
                display_classes(
                    class_path=class_path,
                    class_distribution=class_dist,
                    sort_by_distribution=sort_classes,
                    raw_data_star=nodes[0],  # pass the first node star
                    rln_folder=job_basefolder
                )

        # Now gather all model files for iteration-based distribution
        with st.expander("Distribution Over Iterations", expanded=False):
            class_paths_all, n_classes_all, iter_all, class_dist_all, class_res_all, _, _ = get_classes(
                job_basefolder, model_files
            )
            # Plot distribution if it is not empty
            if class_dist_all.size > 0:
                plot_class_distribution(class_dist_all, PLOT_HEIGHT=500)
            else:
                st.write("No iteration-based distribution data available.")

        if st.checkbox("Show detailed statistics?"):
            interactive_scatter_plot(os.path.join(rln_folder, nodes[0]))
            st.info("Displaying interactive scatter plot for the main STAR file.")

        plt.close()
        logger.info(f"{datetime.now()}: plot_class2d done.")

    except Exception as exc:
        report_error(exc)
        st.error("An error occurred during 2D classification visualization.")
