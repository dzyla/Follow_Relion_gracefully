# aligntilt_job.py

import os
import logging
from typing import List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from lib.utils import parse_star, get_values_from_first_key, report_error, interactive_scatter_plot
from lib.image_utils import micrograph_viewer

logger = logging.getLogger("main_app")


def plot_align_tilt_series(rln_folder: str, node_file: str) -> None:
    """Plot alignment tilt series for a RELION node star file.

    Args:
        rln_folder (str): Path to the RELION folder containing star files.
        node_file (str): Name of the primary star file for the tilt series.
    """
    try:
        logger.info("Plotting alignment tilt series for node: %s", node_file)
        star_path = os.path.join(rln_folder, node_file)

        if not os.path.isfile(star_path):
            logger.error("Star file not found: %s", star_path)
            st.warning(f"Star file not found: {star_path}")
            return

        logger.debug("Parsing star file at: %s", star_path)
        try:
            star = parse_star(star_path)
        except Exception as exc:
            report_error(exc, f"Failed to parse star file: {star_path}")
            st.warning("Failed to parse the main star file.")
            return

        if "global" not in star:
            logger.error("Missing 'global' section in star file: %s", star_path)
            st.warning("The star file does not contain a 'global' section.")
            return

        tomo_files = star["global"].get("_rlnTomoTiltSeriesStarFile", [])
        if not isinstance(tomo_files, list) or not tomo_files:
            logger.error("No tilt-series references found in 'global' section.")
            st.warning("No tilt-series data found in the star file.")
            return

        dfs_tilt: List[pd.DataFrame] = []
        for tomo_file in tomo_files:
            path = os.path.join(rln_folder, tomo_file)
            try:
                star_data = parse_star(path)
                df = get_values_from_first_key(star_data)
                df["FileSource"] = os.path.basename(path)
                dfs_tilt.append(df)
            except Exception as exc:
                logger.error("Error parsing star file %s: %s", path, exc)
                report_error(exc, f"Error parsing tilt-series star file: {path}")

        if not dfs_tilt:
            logger.error("Parsed tilt-series data is empty.")
            st.warning("No tilt-series data found. Cannot plot tilt angles.")
            return

        try:
            combined_df = pd.concat(dfs_tilt, ignore_index=True)

            # Group by Tilt Series (using FileSource or _rlnTomoName if available)
            # FileSource is reliable as it comes from the loop above
            ts_options = combined_df["FileSource"].unique().tolist()

            if not ts_options:
                st.info("No tilt series identifiers found.")
            else:
                col1, col2 = st.columns([1, 3])

                with col1:
                    selected_ts = st.selectbox("Select Tilt Series:", ts_options)

                with col2:
                    # Filter data for selected TS
                    ts_data = combined_df[combined_df["FileSource"] == selected_ts].copy()

                    # Ensure we have image paths
                    if "_rlnMicrographName" in ts_data.columns:
                        image_col = "_rlnMicrographName"
                    elif "_rlnImageName" in ts_data.columns:
                         image_col = "_rlnImageName"
                    else:
                        image_col = None
                        st.warning("Column for image paths (e.g., _rlnMicrographName) not found.")

                    if image_col:
                        # Sort by Tilt Angle if available
                        if "_rlnTomoTiltAngle" in ts_data.columns:
                            ts_data["_rlnTomoTiltAngle"] = pd.to_numeric(ts_data["_rlnTomoTiltAngle"], errors='coerce')
                            ts_data = ts_data.sort_values("_rlnTomoTiltAngle")

                        # Get list of images
                        image_paths = ts_data[image_col].tolist()

                        if image_paths:
                            st.info(f"Loaded {len(image_paths)} images for {selected_ts}. Sorted by tilt angle.")
                            # Use the existing efficient micrograph viewer
                            micrograph_viewer(
                                rln_folder=rln_folder,
                                image_files=image_paths,
                                selected_filter="gaussian",
                                default_gaussian=0.0
                            )
                        else:
                            st.warning("No image paths found for this tilt series.")

                    # Alignment Statistics Plot
                    st.markdown("---")
                    st.subheader("Tilt Alignment Statistics")

                    if "_rlnTomoTiltAngle" in ts_data.columns:
                        fig = go.Figure()

                        # Helper to add trace if column exists
                        def add_trace(col_name, color, name):
                            if col_name in ts_data.columns:
                                fig.add_trace(go.Scatter(
                                    x=ts_data["_rlnTomoTiltAngle"],
                                    y=ts_data[col_name],
                                    mode="markers",
                                    marker=dict(color=color, size=10),
                                    name=name
                                ))

                        add_trace("_rlnTomoXTilt", "#007acc", "X Tilt")
                        add_trace("_rlnTomoYTilt", "#cc3333", "Y Tilt")
                        add_trace("_rlnTomoZRot", "#ffd354", "Z Rot")
                        add_trace("_rlnTomoXShiftAngst", "#b850c8", "X Shift (Å)")
                        add_trace("_rlnTomoYShiftAngst", "#45ab84", "Y Shift (Å)")

                        fig.update_layout(
                            title=f"Alignment Stats: {selected_ts}",
                            xaxis_title="Tilt Angle (deg)",
                            yaxis_title="Value",
                            hovermode="x unified",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Tilt angle data not available for plotting.")

        except Exception as e:
            logger.error("Error combining or plotting tilt series data: %s", e)
            st.error("Failed to combine tilt series data for plotting.")

    except Exception as exc:
        report_error(exc, "Unexpected error in plot_align_tilt_series")
        st.warning("An unexpected error occurred while plotting tilt series.")
