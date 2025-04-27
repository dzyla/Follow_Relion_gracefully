# motioncorr_job.py

import logging
import os
from typing import List, Optional

# Third-Party Imports
import altair as alt # Restore Altair import
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Local Imports
from lib.image_utils import micrograph_viewer 
from lib.utils import (
    get_first_key,
    get_values_from_first_key,
    interactive_scatter_plot,
    parse_star,
    report_error, 
)

# =============================================================================
# Logger Setup
# =============================================================================
logger = logging.getLogger("main_app") # Use logger from main app


def show_motion(rln_folder: str, mic_paths: List[str]) -> None:
    """
    Plots global and local motion shifts for a selected micrograph using Plotly.

    Args:
        rln_folder: Base folder containing the project data.
        mic_paths: List of relative paths to micrograph metadata STAR files.
    """
    if not mic_paths:
        st.info("No micrograph metadata files provided to display motion.")
        return

    try:
        # --- Micrograph Selection ---
        col1, col2 = st.columns([1, 3])
        with col1:
            if len(mic_paths) == 1:
                idx = 0
                st.caption(f"Displaying motion for: {os.path.basename(mic_paths[0])}")
            else:
                idx = st.slider("Select Micrograph Index:", 0, len(mic_paths) - 1, 0, key="motioncorr_mic_slider")
            mic_metadata_rel_path = mic_paths[idx]
            st.caption(f"File: {os.path.basename(mic_metadata_rel_path)}")

        mic_metadata_abs_path = os.path.join(rln_folder, mic_metadata_rel_path)
        logger.debug(f"Processing motion from file: {mic_metadata_abs_path}")

        if not os.path.exists(mic_metadata_abs_path):
             st.warning(f"Metadata file not found: {mic_metadata_rel_path}")
             return

        # --- Parse STAR File ---
        motion_star = parse_star(mic_metadata_abs_path)
        if not motion_star:
             st.warning(f"Could not parse or empty motion metadata file: {mic_metadata_rel_path}")
             return

        # --- Extract Motion Data ---
        global_shift_df = motion_star.get("global_shift")
        local_shift_df = motion_star.get("local_shift")

        if global_shift_df is None and local_shift_df is None:
            st.info("No global or local motion data found in this metadata file.")
            return

        fig = go.Figure()
        traces_added = False

        # --- Process and Plot Local Motion ---
        if isinstance(local_shift_df, pd.DataFrame) and not local_shift_df.empty:
            try:
                local_shift = local_shift_df.astype(float)
                coord_x_col, coord_y_col = "_rlnCoordinateX", "_rlnCoordinateY"
                shift_x_col, shift_y_col = "_rlnMicrographShiftX", "_rlnMicrographShiftY"
                required_local_cols = [coord_x_col, coord_y_col, shift_x_col, shift_y_col]

                if all(col in local_shift.columns for col in required_local_cols):
                    fold_local_motion = 200 #col1.slider(
                        # "Local Motion Scale:", 1, 500, 100, 10, key="local_motion_scale",
                         #help="Magnifies local shifts for visibility.")

                    local_shift["X_start"] = local_shift[coord_x_col]
                    local_shift["Y_start"] = local_shift[coord_y_col]
                    local_shift["X_end"] = local_shift[coord_x_col] + local_shift[shift_x_col] * fold_local_motion
                    local_shift["Y_end"] = local_shift[coord_y_col] + local_shift[shift_y_col] * fold_local_motion

                    grouped = local_shift.groupby([coord_x_col, coord_y_col])
                    logger.debug(f"Plotting {len(grouped)} local motion tracks.")
                    all_x_local, all_y_local = [], []
                    for _, group_df in grouped:
                         all_x_local.extend([group_df['X_start'].iloc[0], group_df['X_end'].iloc[-1], None])
                         all_y_local.extend([group_df['Y_start'].iloc[0], group_df['Y_end'].iloc[-1], None])

                    if all_x_local:
                        fig.add_trace(go.Scatter(
                            x=all_x_local, y=all_y_local, mode="lines", name="Local Shifts",
                            line=dict(color="#039d83", width=1), hoverinfo='none'
                        ))
                        traces_added = True
                else: logger.warning(f"Missing required local motion columns in {mic_metadata_rel_path}")
            except Exception as exc:
                 report_error(exc, f"Error processing local shift data for {mic_metadata_rel_path}")
                 st.warning(f"Could not process local motion data: {exc}")

        # --- Process and Plot Global Motion ---
        if isinstance(global_shift_df, pd.DataFrame) and not global_shift_df.empty:
            try:
                global_shift = global_shift_df.astype(float)
                shift_x_col, shift_y_col = "_rlnMicrographShiftX", "_rlnMicrographShiftY"
                if all(col in global_shift.columns for col in [shift_x_col, shift_y_col]):
                    center_x, center_y = 0.0, 0.0
                    if isinstance(local_shift_df, pd.DataFrame) and not local_shift_df.empty and \
                       all(col in local_shift_df.columns for col in ["_rlnCoordinateX", "_rlnCoordinateY"]):
                        try:
                             center_x = local_shift_df["_rlnCoordinateX"].astype(float).mean()
                             center_y = local_shift_df["_rlnCoordinateY"].astype(float).mean()
                        except Exception: pass # Ignore errors, default to 0,0

                    if isinstance(local_shift_df, pd.DataFrame) and not local_shift_df.empty and \
                       all(col in local_shift_df.columns for col in ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnMicrographShiftX", "_rlnMicrographShiftY"]):
                          try:
                                max_coord_x = local_shift_df["_rlnCoordinateX"].astype(float).max()
                                max_coord_y = local_shift_df["_rlnCoordinateY"].astype(float).max()
                                max_gshift_x = global_shift[shift_x_col].abs().max()
                                max_gshift_y = global_shift[shift_y_col].abs().max()
                                scale_x = (max_coord_x / (2 * max_gshift_x)) if max_gshift_x > 1e-6 else 100
                                scale_y = (max_coord_y / (2 * max_gshift_y)) if max_gshift_y > 1e-6 else 100
                                fold_global_motion = max(1, int(min(scale_x, scale_y, 500)))
                          except Exception as e:
                               logger.warning(f"Could not calculate global fold factor: {e}. Using default.")
                               fold_global_motion = 100 # Default scale if local data missing/problematic
                    else:
                         fold_global_motion = st.slider(
                             "Global Motion Scale:", 1, 1000, 100, 10, key="global_motion_scale",
                             help="Magnifies global shifts for visibility when local data is absent.")

                    global_x = global_shift[shift_x_col] * fold_global_motion + center_x
                    global_y = global_shift[shift_y_col] * fold_global_motion + center_y

                    fig.add_trace(go.Scatter(
                        x=global_x, y=global_y, mode="lines+markers", name="Global Shift",
                        line=dict(color="#007acc", width=2), marker=dict(size=5),
                        hovertemplate="Frame: %{customdata}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>",
                        customdata=np.arange(len(global_x))
                    ))
                    traces_added = True
                else: logger.warning(f"Missing required global motion columns in {mic_metadata_rel_path}")
            except Exception as exc:
                 report_error(exc, f"Error processing global shift data for {mic_metadata_rel_path}")
                 st.warning(f"Could not process global motion data: {exc}")

        # --- Display Combined Plot ---
        if traces_added:
            fig.update_layout(
                title=f"Motion Track: {os.path.basename(mic_metadata_rel_path)}",
                xaxis_title="X Coordinate (px)", yaxis_title="Y Coordinate (px)",
                width=500, height=500, showlegend=True, hovermode="closest",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                xaxis=dict(scaleanchor="y", scaleratio=1),
                margin=dict(l=10, r=10, t=50, b=10)
            )
            with col2:
                st.plotly_chart(fig, use_container_width=True)
        else:
            with col2:
                st.info("No plottable motion data found.")

    except FileNotFoundError:
         st.warning(f"Motion metadata file not found: {mic_metadata_rel_path}")
    except Exception as exc:
        report_error(exc, f"Error in show_motion for {mic_metadata_rel_path}")
        st.warning(f"An error occurred while displaying motion: {exc}")


# =============================================================================
# Main Job Function
# =============================================================================

def plot_motioncorr(rln_folder: str, node: str) -> None:
    """
    Processes and plots MotionCorr job results, including Altair statistics plots.

    Args:
        rln_folder: Base directory of the RELION project.
        node: Specific node file for this job (e.g., "MotionCorr/.../corrected_micrographs.star").
    """
    star_path = os.path.join(rln_folder, node)
    logger.info(f"Processing MotionCorr job: {star_path}")

    if not os.path.exists(star_path):
        st.warning(f"MotionCorr STAR file not found: {node}")
        logger.error(f"File not found: {star_path}")
        return

    try:
        star = parse_star(star_path)
        if not star:
             st.warning(f"Could not parse or empty STAR file: {node}")
             return
    except Exception as exc:
        report_error(exc, f"Failed to parse MotionCorr STAR file: {node}")
        st.warning(f"Error parsing {node}. See logs.")
        return

    star_data: Optional[pd.DataFrame] = None
    # Handle different potential block names
    if 'micrographs' in star:
        star_data = star["micrographs"]
        logger.debug("Using 'micrographs' block.")
    elif 'global' in star: # Relion 5 Tomo case
        logger.debug("Found 'global' block (likely Relion 5 Tomo MotionCorr).")
        try:
            tomo_star_files = star["global"]['_rlnTomoTiltSeriesStarFile']
            tomo_star_data_list = []
            failed_loads = 0
            for tomo_star in tomo_star_files:
                tomo_star_path = os.path.join(rln_folder, tomo_star)
                if os.path.exists(tomo_star_path):
                    parsed_tomo_star = parse_star(tomo_star_path)
                    first_block_data = get_values_from_first_key(parsed_tomo_star)
                    if isinstance(first_block_data, pd.DataFrame):
                        tomo_star_data_list.append(first_block_data)
                    else: failed_loads += 1; logger.warning(f"No DataFrame in {tomo_star}")
                else: failed_loads += 1; logger.warning(f"Tilt series STAR not found: {tomo_star_path}")
            if failed_loads > 0: st.warning(f"Failed to load data from {failed_loads} tilt series STAR(s).")
            if tomo_star_data_list:
                star_data = pd.concat(tomo_star_data_list, ignore_index=True)
                logger.info(f"Concatenated data from {len(tomo_star_data_list)} tilt series.")
            else: st.warning("Could not load motion data from linked tilt series STAR files."); return
        except KeyError as e: report_error(e, "Missing key in Relion 5 Tomo MotionCorr"); st.warning(f"Error: Missing key {e}."); return
        except Exception as e: report_error(e, "Error processing Relion 5 Tomo MotionCorr"); st.warning(f"Error processing Tomo data: {e}."); return
    else:
         first_key = get_first_key(star)
         if first_key and isinstance(star[first_key], pd.DataFrame):
             star_data = star[first_key]
             logger.warning(f"Using first block '{first_key}' as fallback data source.")
         else: st.warning("No 'micrographs' or 'global' block found."); return

    if star_data is None or star_data.empty:
        st.warning("No valid micrograph data found for MotionCorr.")
        return

    # --- Motion Statistics Plotting (Altair) ---
    st.subheader("Motion Statistics")
    meta_names = ["_rlnAccumMotionTotal", "_rlnAccumMotionEarly", "_rlnAccumMotionLate"]
    df_line = pd.DataFrame()
    df_hist = pd.DataFrame()
    percentile_99 = None # Initialize percentile

    try:
        valid_meta_found = False
        for meta in meta_names:
            if meta in star_data.columns:
                 data_array = pd.to_numeric(star_data[meta], errors='coerce').dropna()
                 if not data_array.empty:
                     valid_meta_found = True
                     # Calculate percentile for clipping (only once, based on Total if possible)
                     if meta == "_rlnAccumMotionTotal" and percentile_99 is None:
                          percentile_99 = np.percentile(data_array, 99.5)

                     # Use the calculated percentile, or fallback if Total wasn't available first
                     clip_value = percentile_99 if percentile_99 is not None else np.percentile(data_array, 99.5)
                     data_clipped_line = np.clip(data_array, None, clip_value)
                     # Use a fixed clip for histogram for better comparison range? e.g., 100 Angstroms
                     data_clipped_hist = np.clip(data_array, None, 100)

                     series_name = meta.replace("_rln", "").replace("AccumMotion", "") # Shorter name
                     temp_df_line = pd.DataFrame({"Index": np.arange(len(data_clipped_line)), "Motion": data_clipped_line, "Series": series_name})
                     temp_df_hist = pd.DataFrame({"Motion": data_clipped_hist, "Series": series_name})
                     df_line = pd.concat([df_line, temp_df_line], ignore_index=True)
                     df_hist = pd.concat([df_hist, temp_df_hist], ignore_index=True)
                 else: logger.warning(f"Column '{meta}' is empty or has no numeric data.")
            else: logger.warning(f"Column '{meta}' not found in STAR data.")

        if not valid_meta_found:
             st.caption("No motion statistics columns found to plot.")
        else:
            # Create Altair charts
            clip_info = f" (line plot clipped at {percentile_99:.2f} Å)" if percentile_99 is not None else ""
            line_chart = alt.Chart(df_line).mark_line(point=False, opacity=0.8).encode( # point=False for large datasets
                x=alt.X("Index:Q", title="Micrograph Index"),
                y=alt.Y("Motion:Q", title="Accumulated Motion (Å)"),
                color=alt.Color("Series:N", title="Motion Type"),
                tooltip=["Index", "Motion", "Series"]
            ).properties(
                title=f"Motion Statistics{clip_info}"
            ).interactive() # Add interactivity

            hist_chart = alt.Chart(df_hist).mark_bar(opacity=0.6, binSpacing=0).encode( # Adjust binSpacing
                x=alt.X("Motion:Q", bin=alt.Bin(maxbins=50), title="Accumulated Motion (Å, clipped at 100)"),
                y=alt.Y("count()", title="Frequency", stack=None), # stack=None for overlaid bars
                color=alt.Color("Series:N", title="Motion Type"),
                tooltip=[alt.Tooltip("Motion:Q", bin=True), "count()", "Series:N"]
            ).properties(
                title="Motion Histograms"
            ).interactive()

            # Display side-by-side
            col1, col2 = st.columns(2)
            with col1: st.altair_chart(line_chart, use_container_width=True)
            with col2: st.altair_chart(hist_chart, use_container_width=True)

    except Exception as exc:
        report_error(exc, "Error generating MotionCorr statistics plots.")
        st.warning(f"Could not generate motion statistics plots: {exc}")

    st.divider()

    # --- Motion Tracks and Micrograph Display ---
    # Checkbox to show motion tracks plot
    metadata_col = '_rlnMicrographMetadata'
    if metadata_col in star_data.columns:
        # Use default value True to show motion by default if available
        if st.checkbox("Show Motion Tracks per Micrograph?", value=True, key="motioncorr_show_tracks"):
            st.subheader("Motion Tracks")
            motion_star_files = star_data[metadata_col].dropna().tolist() # Drop missing values
            valid_motion_files = [f for f in motion_star_files if isinstance(f, str) and f]
            if valid_motion_files:
                show_motion(rln_folder, valid_motion_files)
            else: st.info("No valid micrograph metadata files found.")
    else: st.caption(f"'{metadata_col}' column not found - cannot display individual motion tracks.")

    st.divider()

    # Display corrected micrographs
    st.subheader("Corrected Micrographs")
    mics_col = '_rlnMicrographName'
    if mics_col in star_data.columns:
        mics = star_data[mics_col].dropna().tolist()
        valid_mics = [m for m in mics if isinstance(m, str) and m]
        if valid_mics:
            try:
                micrograph_viewer(rln_folder, valid_mics)
            except Exception as exc:
                report_error(exc, "Error calling micrograph_viewer for corrected micrographs")
                st.warning(f"Error displaying corrected micrographs: {exc}")
        else: st.info("No valid corrected micrograph paths found.")
    else: st.caption(f"'{mics_col}' column not found - cannot display corrected micrographs.")

    st.divider()

    # Option to plot metadata interactively
    if st.checkbox("Plot Metadata Interactively?", key="motioncorr_plot_meta"):
        st.subheader("Metadata Distribution")
        try:
            interactive_scatter_plot(data_source=star_path, title_prefix="MotionCorr Metadata")
        except Exception as exc:
            report_error(exc, "Error calling interactive_scatter_plot for MotionCorr metadata")
            st.warning(f"Error plotting metadata: {exc}")

    logger.info(f"Finished processing MotionCorr job: {node}")