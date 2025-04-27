#ctfrefine_job.py


import os
import re
import math
import traceback
import logging
import glob
import textwrap
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import mrcfile
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from lib.utils import (
    parse_star,
    get_note,
    interactive_scatter_plot,
    report_error
)

logger = logging.getLogger("main_app")

# --- Constants ---
RLN_DEFOCUS_U = "_rlnDefocusU"
RLN_DEFOCUS_V = "_rlnDefocusV"
RLN_COORD_X = "_rlnCoordinateX"
RLN_COORD_Y = "_rlnCoordinateY"
RLN_MICROGRAPH_NAME = "_rlnMicrographName"
RLN_TOMO_TILT_STAR = "_rlnTomoTiltSeriesStarFile"
RLN_TOMO_TILT_ANGLE = "_rlnTomoNominalStageTiltAngle"
RLN_CTF_SCALEFACTOR = "_rlnCtfScalefactor"

# --- Helper Functions ---

def _load_and_cache_star_data(file_path: str, cache_key: str) -> Optional[pd.DataFrame]:
    """Loads particle data from a STAR file, using Streamlit caching."""
    if cache_key in st.session_state:
        logger.debug(f"Using cached STAR data for key: {cache_key}")
        return st.session_state[cache_key]
    else:
        try:
            data = parse_star(file_path)
            # Assuming 'particles' table is the relevant one for refinement data
            particles_df = data.get("particles")
            if particles_df is not None and not particles_df.empty:
                st.session_state[cache_key] = particles_df
                logger.debug(f"Loaded and cached STAR data from: {file_path}")
                return particles_df
            else:
                logger.warning(f"No 'particles' data found or empty in STAR file: {file_path}")
                return None
        except Exception as e:
            report_error(e, f"Failed to parse STAR file: {file_path}")
            logger.error(f"Failed to parse STAR file: {file_path}\n{traceback.format_exc()}")
            return None

def _get_refine_star_path_from_note(note_content: str, base_folder: str) -> Optional[str]:
    """Extracts the input STAR file path (--i argument) from note.txt content."""
    try:
        refine_match = re.search(r"--i\s+([\w\d/\\\.\-\_]+\.star)", note_content)
        if not refine_match:
            logger.warning("Could not find '--i <star_file>' pattern in note.txt.")
            return None

        refine_path = refine_match.group(1)

        # Handle potential nesting common in tutorials (e.g., CtfRefine job using output of previous)
        # This assumes a specific structure like 'CtfRefine/jobXXX/particles.star'
        if "CtfRefine" in refine_path and len(refine_path.split(os.sep)) > 1:
            parts = refine_path.split(os.sep)
            # Construct path to the *parent* job's note.txt
            # This assumes the path is relative like 'CtfRefine/job001/particles.star'
            # Adjust index/logic if path structure is different
            parent_note_path = os.path.join(base_folder, parts[0], parts[1], "note.txt")
            if os.path.exists(parent_note_path):
                logger.debug(f"Found nested CtfRefine path, checking parent note: {parent_note_path}")
                parent_note_content = get_note(parent_note_path)
                parent_match = re.search(r"--i\s+([\w\d/\\\.\-\_]+\.star)", parent_note_content)
                if parent_match:
                    logger.debug("Using refine path from parent job's note.")
                    return parent_match.group(1) # Return the path from the parent note
                else:
                    logger.warning(f"Nested CtfRefine path found, but could not extract --i from parent note: {parent_note_path}")
            else:
                 logger.warning(f"Nested CtfRefine path found, but parent note does not exist: {parent_note_path}")
        
        # If not nested or parent parsing failed, return the original path
        return refine_path

    except Exception as e:
        report_error(e, "Error extracting refine star path from note content")
        logger.error(f"Error processing note content: {e}\n{traceback.format_exc()}")
        return None

# --- Plotting Functions ---

def display_ctf_stats_grid(
    images: List[np.ndarray],
    file_names: List[str],
    columns: int = 4,
    label_wrap_length: int = 40,
    label_font_size: int = 10,
) -> None:
    """
    Displays a grid of 2D heatmap images (like CTF diagnostic images) using Plotly.

    Args:
        images: List of 2D NumPy arrays to display.
        file_names: List of file names corresponding to the images for titles.
        columns: Number of columns in the grid layout.
        label_wrap_length: Maximum length for subplot titles before wrapping.
        label_font_size: Font size for subplot titles.
    """
    if not images:
        st.warning("No images provided to display_ctf_stats_grid.")
        return

    max_images = len(images)
    num_rows = math.ceil(max_images / columns)

    # Prepare shortened titles
    subplot_titles = [
        textwrap.shorten(name, width=label_wrap_length, placeholder="...")
        for name in file_names
    ]
    # Pad with empty strings if file_names is shorter than images
    subplot_titles.extend([""] * (max_images - len(subplot_titles)))

    fig = make_subplots(rows=num_rows, cols=columns, subplot_titles=subplot_titles)

    for i, image in enumerate(images):
        row = (i // columns) + 1
        col = (i % columns) + 1

        fig.add_trace(
            go.Heatmap(z=image, colorscale="thermal_r", showscale=False),
            row=row, col=col
        )
        # Ensure aspect ratio is maintained for each heatmap
        fig.update_xaxes(scaleanchor=f"y{i+1}", scaleratio=1, showticklabels=False, row=row, col=col)
        fig.update_yaxes(scaleanchor=f"x{i+1}", scaleratio=1, showticklabels=False, row=row, col=col)
        # remove grid lines
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)

    fig.update_layout(
        height=num_rows * 400, # Dynamically adjust height based on rows, or let Streamlit handle it
        # width=columns * 300, # Let Streamlit manage width primarily
        margin=dict(l=10, r=10, t=40 if any(subplot_titles) else 10, b=10), # Adjust top margin for titles
        font=dict(size=label_font_size),
        title="CTF Diagnostic Images", # Overall title
    )
    # Update title font size specifically
    for annotation in fig.layout.annotations:
        annotation.font.size = label_font_size

    st.plotly_chart(fig, use_container_width=True)


def generate_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    axis_right_col: Optional[str] = None,
    keep_aspect_ratio: bool = False, # Note: Aspect ratio primarily for 3D/images
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    y_label_right: Optional[str] = None,
    col_to_plot: Optional[str] = None, # Not used in this version, but kept for context
) -> None:
    """
    Generates a Plotly line plot from a DataFrame, optionally with a secondary y-axis.

    Args:
        df: The DataFrame containing the plot data.
        x_col: Column name for the x-axis. Use "index" to plot against the DataFrame index.
        y_cols: List of column names for the primary y-axis.
        axis_right_col: Column name for the secondary y-axis (optional).
        keep_aspect_ratio: If True, tries to maintain data aspect ratio (less common for line plots).
        x_label: Custom label for the x-axis. Defaults to x_col name.
        y_label: Custom label for the primary y-axis. Defaults to joined y_cols names.
        y_label_right: Custom label for the secondary y-axis. Defaults to axis_right_col name.
    """
    if df.empty:
        st.warning("Cannot generate plot: DataFrame is empty.")
        return

    plot_df = df.copy() # Work on a copy

    # Handle plotting against index
    if x_col == "index":
        plot_df["index"] = plot_df.index
        actual_x_col = "index"
    else:
        actual_x_col = x_col

    # Verify columns exist
    required_cols = [actual_x_col] + y_cols + ([axis_right_col] if axis_right_col else [])
    missing_cols = [col for col in required_cols if col not in plot_df.columns]
    if missing_cols:
        st.error(f"Missing required columns for plotting: {', '.join(missing_cols)}")
        logger.error(f"Missing columns in DataFrame for generate_line_plot: {missing_cols}")
        return

    # Attempt to convert relevant columns to numeric, report errors
    cols_to_convert = required_cols
    error_cols = []
    for col in cols_to_convert:
        try:
            # Use pd.to_numeric for more robust conversion
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        except Exception as e:
            # This catch might be less likely with errors='coerce' but kept for safety
            error_cols.append(col)
            logger.warning(f"Could not convert column '{col}' to numeric: {e}")

    # Drop rows where conversion failed for critical columns
    plot_df.dropna(subset=[actual_x_col] + y_cols + ([axis_right_col] if axis_right_col else []), inplace=True)

    if plot_df.empty:
        st.error(f"DataFrame is empty after attempting numeric conversion and dropping invalid rows. Initial conversion issues in columns: {error_cols}")
        return
    elif error_cols:
         st.warning(f"Could not convert data to numeric in columns: {', '.join(error_cols)}. Plotting with valid rows only.")


    # Create the main plot
    try:
        fig = px.line(plot_df, x=actual_x_col, y=y_cols, title="Tomogram Tilt Series Statistics")
    except Exception as e:
        report_error(e, "Failed to create Plotly Express line plot.")
        logger.error(f"Error creating px.line plot: {e}\n{traceback.format_exc()}")
        st.error("An error occurred while creating the plot.")
        return

    # Add secondary axis if specified
    if axis_right_col:
        fig.add_trace(
            go.Scatter(
                x=plot_df[actual_x_col],
                y=plot_df[axis_right_col],
                mode="lines",
                name=y_label_right if y_label_right else axis_right_col, # Use label for name if provided
                yaxis="y2" # Assign to the secondary y-axis
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title=y_label_right if y_label_right else axis_right_col,
                overlaying="y",
                side="right",
            )
        )

    # Set axis labels
    fig.update_layout(
        xaxis_title=x_label if x_label else actual_x_col,
        yaxis_title=y_label if y_label else ", ".join(y_cols),
        legend_title_text="Parameters"
    )

    # Apply aspect ratio if requested (mainly for scatter/heatmap)
    if keep_aspect_ratio:
        fig.update_layout(yaxis_scaleanchor="x") # Common way for 2D aspect ratio

    if col_to_plot:
        col_to_plot.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig, use_container_width=True)

# --- Main Function ---

def plot_ctf_refine(folder: str, node_files: List[str]) -> None:
    """
    Main function to plot CTF refinement statistics for a given job.

    Determines if the job involves tomograms or standard micrographs and
    routes to the appropriate plotting logic.

    For non-tomogram jobs:
      - Attempts to find and display CTF diagnostic MRC images first.
      - If no MRCs are found, parses STAR files (refined and original CTF estimates)
        extracted via note.txt to calculate and plot defocus changes (U, V).
      - Offers an optional 3D scatter plot of particle positions vs defocus per micrograph.
      - Offers an optional link to more detailed statistics via `interactive_scatter_plot`.

    For tomogram jobs:
      - Loads the tomogram series STAR file.
      - Allows selection of a specific tomogram via a slider.
      - Plots tilt-dependent parameters (e.g., Defocus U/V, CTF Scalefactor) vs tilt angle.

    Args:
        folder: The base directory containing the Relion project structure.
        node_files: A list of relevant output filenames for this job node
                    (e.g., ['job005/ctffind.star', 'job005/corrected_micrographs.star']).
                    The exact content depends on the pipeline graph structure.
                    Conventionally, node_files[0] might be the primary job identifier path part.
                    node_files[1] is often the main output star file for CtfRefine.
    """
    if not node_files:
        st.error("No node files provided for CTF Refine job.")
        logger.error("plot_ctf_refine called with empty node_files list.")
        return

    # Construct a base job path, assuming node_files[0] gives relative job dir
    # Example: node_files[0] = 'CtfRefine/job005'
    # Needs adjustment if node_files structure is different
    job_base_path_part = os.path.dirname(node_files[0]) # Gets 'CtfRefine' if input is 'CtfRefine/job005/output.star'
    if not job_base_path_part: # If node_files[0] is just 'output.star'
         job_base_path_part = os.path.dirname(folder) # Fallback, might not be correct
         logger.warning(f"Could not determine job base path from node_files[0]: {node_files[0]}. Using folder parent: {job_base_path_part}")
    
    job_path = os.path.join(folder, job_base_path_part) # Path like /path/to/project/CtfRefine/job005
    job_name_short = job_base_path_part # e.g., CtfRefine/job005


    logger.info(f"{datetime.now()}: plot_ctf_refine called for job '{job_name_short}'")
    logger.debug(f"Job path: {job_path}, Node files: {node_files}")

    try:
        # --- Determine Job Type (Tomogram vs. Non-Tomogram) ---
        # Heuristic: check if 'tomo' appears in node filenames or job path parts
        is_tomogram_job = any("tomo" in node.lower() for node in node_files) or \
                          any("tomo" in part.lower() for part in job_path.split(os.sep))

        if not is_tomogram_job:
            logger.debug(f"Processing as non-tomogram job: {job_name_short}")

            # --- Non-Tomogram: Try loading diagnostic MRC images first ---
            # These are often output by CtfFind but might be present in CtfRefine dir
            try:
                # Look for common diagnostic patterns
                mrc_patterns = ["*.mrc", "diagnostic_plots/*.mrc"]
                ctf_mrc_files = []
                for pattern in mrc_patterns:
                    ctf_mrc_files.extend(glob.glob(os.path.join(job_path, pattern)))
                
                # Filter out potentially large micrograph MRCs if possible, keep smaller diagnostics
                ctf_mrc_files = [f for f in ctf_mrc_files if os.path.getsize(f) < 50 * 1024 * 1024] # Example size threshold: 50MB
                
                ctf_mrc_files = sorted(list(set(ctf_mrc_files))) # Unique and sorted

                if ctf_mrc_files:
                    logger.info(f"Found {len(ctf_mrc_files)} potential CTF diagnostic MRC files.")
                    mrc_images = []
                    mrc_names = []
                    for mrc_path in ctf_mrc_files:
                        try:
                            with mrcfile.open(mrc_path, permissive=True) as mrc:
                                # Take only 2D images
                                if mrc.data is not None and mrc.data.ndim == 2:
                                    mrc_images.append(mrc.data)
                                    mrc_names.append(os.path.basename(mrc_path))
                                else:
                                     logger.debug(f"Skipping non-2D or empty MRC: {mrc_path}")
                        except Exception as img_err:
                            logger.warning(f"Could not read MRC file {mrc_path}: {img_err}")
                    
                    if mrc_images:
                         display_ctf_stats_grid(mrc_images, mrc_names)
                         # Stop here if we displayed MRCs
                         logger.info(f"{datetime.now()}: Displayed CTF MRC grid for {job_name_short}. plot_ctf_refine done.")
                         return 
                    else:
                        logger.info("No valid 2D diagnostic MRC images found to display.")

            except Exception as glob_err:
                 logger.error(f"Error searching for MRC files: {glob_err}")
                 # Continue to STAR file processing if MRC search fails

            # --- Non-Tomogram: Process STAR files if no MRCs were displayed ---
            logger.info("No diagnostic MRCs displayed, proceeding with STAR file analysis.")
            
            # Assume node_files[1] is the CTF refined star file for this job
            if len(node_files) < 2:
                 st.error(f"Cannot process CTF Refine job {job_name_short}: Expected at least two node files (input and output star), got {len(node_files)}.")
                 logger.error(f"Insufficient node files for job {job_name_short}. Got: {node_files}")
                 return
            
            ctf_refined_star_path = os.path.join(folder, node_files[1])
            if not os.path.exists(ctf_refined_star_path):
                 st.error(f"CTF Refined STAR file not found: {ctf_refined_star_path}")
                 logger.error(f"CTF Refined STAR file not found: {ctf_refined_star_path}")
                 return

            # Get the input star file path from note.txt
            note_path = os.path.join(job_path, "note.txt")
            refine_input_star_rel_path = None
            if os.path.exists(note_path):
                 note_content = get_note(note_path)
                 refine_input_star_rel_path = _get_refine_star_path_from_note(note_content, folder)
            else:
                st.warning(f"note.txt not found at {note_path}. Cannot determine input STAR file for comparison.")
                logger.warning(f"note.txt not found at {note_path}")


            if not refine_input_star_rel_path:
                st.error(f"Could not determine the input STAR file used for job {job_name_short} from note.txt. Cannot calculate defocus changes.")
                # Attempt to show stats from the output file only
                if st.checkbox(f"Show statistics for output file ({os.path.basename(ctf_refined_star_path)}) only?", key=f"stats_only_{job_name_short}"):
                     interactive_scatter_plot(ctf_refined_star_path)
                return # Cannot proceed with comparison plots

            refine_input_star_abs_path = os.path.join(folder, refine_input_star_rel_path)
            if not os.path.exists(refine_input_star_abs_path):
                 st.error(f"Input STAR file specified in note.txt not found: {refine_input_star_abs_path}")
                 logger.error(f"Input STAR file from note.txt not found: {refine_input_star_abs_path}")
                 return

            # Load data using caching
            refine_input_cache_key = f"star_data_{job_name_short}_input_{os.path.basename(refine_input_star_abs_path)}"
            ctf_refined_cache_key = f"star_data_{job_name_short}_output_{os.path.basename(ctf_refined_star_path)}"

            refine_input_data = _load_and_cache_star_data(refine_input_star_abs_path, refine_input_cache_key)
            ctf_refined_data = _load_and_cache_star_data(ctf_refined_star_path, ctf_refined_cache_key)

            if refine_input_data is None or ctf_refined_data is None:
                st.error("Failed to load required STAR file data. Check logs for details.")
                return
            if RLN_DEFOCUS_U not in refine_input_data or RLN_DEFOCUS_V not in refine_input_data \
               or RLN_DEFOCUS_U not in ctf_refined_data or RLN_DEFOCUS_V not in ctf_refined_data:
                st.error(f"Missing required defocus columns ('{RLN_DEFOCUS_U}', '{RLN_DEFOCUS_V}') in one or both STAR files.")
                logger.error("Missing required defocus columns in STAR files.")
                return

            # Compute defocus differences - wrap in try-except for robustness
            try:
                refine_input_U = pd.to_numeric(refine_input_data[RLN_DEFOCUS_U], errors='coerce')
                refine_input_V = pd.to_numeric(refine_input_data[RLN_DEFOCUS_V], errors='coerce')
                ctf_refined_U = pd.to_numeric(ctf_refined_data[RLN_DEFOCUS_U], errors='coerce')
                ctf_refined_V = pd.to_numeric(ctf_refined_data[RLN_DEFOCUS_V], errors='coerce')
                
                # Align data before subtracting (assuming particle order might change, use MicrographName/Coordinates as keys if needed)
                # Simple subtraction assumes particles are in the same order and number.
                # A more robust approach would merge dataframes on particle identifiers if available.
                # For now, assume matching order and handle potential length mismatches or NaNs.
                common_len = min(len(refine_input_U), len(ctf_refined_U))
                if len(refine_input_U) != len(ctf_refined_U):
                    st.warning(f"Input ({len(refine_input_U)}) and output ({len(ctf_refined_U)}) STAR files have different numbers of particles. Comparing the first {common_len} particles.")
                    logger.warning(f"Particle count mismatch: Input={len(refine_input_U)}, Output={len(ctf_refined_U)}. Using common length {common_len}.")

                delta_U = refine_input_U[:common_len] - ctf_refined_U[:common_len]
                delta_V = refine_input_V[:common_len] - ctf_refined_V[:common_len]
                
                # Remove NaNs that might result from conversion errors or subtraction
                valid_indices = ~np.isnan(delta_U) & ~np.isnan(delta_V)
                delta_U = delta_U[valid_indices]
                delta_V = delta_V[valid_indices]
                
                if len(delta_U) == 0:
                    st.error("Could not compute valid defocus differences. Check data integrity.")
                    logger.error("No valid defocus differences computed after handling NaNs.")
                    return

            except KeyError as ke:
                 st.error(f"Missing expected column in STAR file: {ke}")
                 logger.error(f"KeyError during defocus difference calculation: {ke}")
                 return
            except Exception as calc_err:
                 report_error(calc_err, "Error calculating defocus differences")
                 logger.error(f"Error calculating defocus differences: {calc_err}\n{traceback.format_exc()}")
                 st.error("An error occurred while calculating defocus differences.")
                 return


            # Plotting Defocus Change Histograms
            st.subheader("Distribution of Defocus Changes (Input - Refined)")
            fig_hist = make_subplots(rows=1, cols=3, subplot_titles=["Δ DefocusU", "Δ DefocusV", "Δ DefocusU vs Δ DefocusV"])

            fig_hist.add_trace(go.Histogram(x=delta_U, nbinsx=100, name="ΔU", marker_color="cornflowerblue"), row=1, col=1)
            fig_hist.add_trace(go.Histogram(x=delta_V, nbinsx=100, name="ΔV", marker_color="tomato"), row=1, col=2)
            fig_hist.add_trace(go.Histogram2d(x=delta_U, y=delta_V, nbinsx=50, nbinsy=50, colorscale="Viridis", colorbar_title="Count"), row=1, col=3)

            fig_hist.update_layout(
                 title="CTF Refinement Defocus Changes",
                 showlegend=False,
                 xaxis1_title="Δ Defocus U (Å)", yaxis1_title="Count",
                 xaxis2_title="Δ Defocus V (Å)", yaxis2_title="Count",
                 xaxis3_title="Δ Defocus U (Å)", yaxis3_title="Δ Defocus V (Å)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Optional: Per-micrograph 3D scatter plot
            st.subheader("Per-Micrograph Particle View")
            if st.checkbox("Show 3D scatter plot of particles per micrograph?", key=f"show_scatter_{job_name_short}"):
                if not all(k in ctf_refined_data.columns for k in [RLN_MICROGRAPH_NAME, RLN_COORD_X, RLN_COORD_Y, RLN_DEFOCUS_U]):
                    st.warning(f"Cannot generate scatter plot: Missing required columns ({RLN_MICROGRAPH_NAME}, {RLN_COORD_X}, {RLN_COORD_Y}, {RLN_DEFOCUS_U}) in {node_files[1]}")
                else:
                    try:
                         # Prepare data - convert relevant columns safely
                         scatter_data = ctf_refined_data[[RLN_MICROGRAPH_NAME, RLN_DEFOCUS_U, RLN_COORD_X, RLN_COORD_Y]].copy()
                         scatter_data[RLN_DEFOCUS_U] = pd.to_numeric(scatter_data[RLN_DEFOCUS_U], errors='coerce')
                         scatter_data[RLN_COORD_X] = pd.to_numeric(scatter_data[RLN_COORD_X], errors='coerce')
                         scatter_data[RLN_COORD_Y] = pd.to_numeric(scatter_data[RLN_COORD_Y], errors='coerce')
                         scatter_data.dropna(inplace=True) # Remove rows where conversion failed

                         c1, c2 = st.columns([1, 5])
                         unique_mics = sorted(scatter_data[RLN_MICROGRAPH_NAME].unique())
                         if not unique_mics:
                             st.warning("No valid micrograph data found for scatter plot after cleaning.")
                         else:
                             mic_idx = c1.slider("Select Micrograph Index:", 0, len(unique_mics)-1, 0, key=f"mic_slider_{job_name_short}")
                             selected_mic_name = unique_mics[mic_idx]

                             selected_particles = scatter_data[scatter_data[RLN_MICROGRAPH_NAME] == selected_mic_name]

                             if selected_particles.empty:
                                 st.warning(f"No valid particles found for micrograph: {selected_mic_name}")
                             else:
                                 fig_scatter = go.Figure(data=[
                                     go.Scatter3d(
                                         x=selected_particles[RLN_COORD_X],
                                         y=selected_particles[RLN_COORD_Y],
                                         z=selected_particles[RLN_DEFOCUS_U],
                                         mode='markers',
                                         marker=dict(
                                             size=5, # Reduced size slightly
                                             color=selected_particles[RLN_DEFOCUS_U],
                                             colorscale='Viridis',
                                             opacity=0.7,
                                             colorbar=dict(title='Defocus U (Å)')
                                         ),
                                         hovertemplate=f"<b>Micrograph:</b> {os.path.basename(selected_mic_name)}<br>" +
                                                       "X: %{x:.1f} px<br>" +
                                                       "Y: %{y:.1f} px<br>" +
                                                       "Defocus U: %{z:.1f} Å<extra></extra>"
                                     )
                                 ])
                                 fig_scatter.update_layout(
                                     title=f"Particles on Micrograph {os.path.basename(unique_mics[mic_idx])}",
                                     scene=dict(
                                         xaxis_title='X (px)',
                                         yaxis_title='Y (px)',
                                         zaxis_title='Defocus U (Å)'
                                     ),
                                     margin=dict(l=10, r=10, b=10, t=40), # Adjusted margins
                                     # width=700, # Let streamlit manage width
                                     height=600 # Fixed height might be useful for 3D
                                 )
                                 c2.plotly_chart(fig_scatter, use_container_width=True)

                    except KeyError as ke:
                        st.error(f"Missing column required for scatter plot: {ke}")
                        logger.error(f"KeyError preparing data for scatter plot: {ke}")
                    except Exception as scatter_err:
                        report_error(scatter_err, "Error generating 3D scatter plot")
                        logger.error(f"Error in 3D scatter plot section: {scatter_err}\n{traceback.format_exc()}")
                        st.error("An error occurred while generating the scatter plot.")


            # Optional: Link to detailed interactive scatter plot utility
            st.subheader("Detailed Statistics")
            if st.checkbox("Show interactive statistics plot?", key=f"show_istats_{job_name_short}"):
                 interactive_scatter_plot(ctf_refined_star_path) # Call the utility

        else:
            # --- Tomogram Job Processing ---
            logger.info(f"Processing as tomogram job: {job_name_short}")
            st.subheader("Tomogram CTF Refinement Analysis")

            # Assume node_files[1] is the tomograms_ctf_refined.star file
            if len(node_files) < 2:
                 st.error(f"Cannot process Tomo CTF Refine job {job_name_short}: Expected at least two node files, got {len(node_files)}.")
                 logger.error(f"Insufficient node files for tomo job {job_name_short}. Got: {node_files}")
                 return

            tomo_summary_star_path = os.path.join(folder, node_files[1])
            if not os.path.exists(tomo_summary_star_path):
                st.error(f"Tomogram summary STAR file not found: {tomo_summary_star_path}")
                logger.error(f"Tomogram summary STAR file not found: {tomo_summary_star_path}")
                return

            # Load the summary star file containing paths to individual tilt series stars
            tomo_summary_cache_key = f"tomo_summary_{job_name_short}_{os.path.basename(tomo_summary_star_path)}"
            tomo_summary_data = _load_and_cache_star_data(tomo_summary_star_path, tomo_summary_cache_key) # Expects 'global' or similar table

            # Attempt to load from 'global' or fallback to 'particles' if structure varies
            global_data = parse_star(tomo_summary_star_path).get("global")

            if global_data is None or RLN_TOMO_TILT_STAR not in global_data.columns:
                 st.error(f"Could not find 'global' table or '{RLN_TOMO_TILT_STAR}' column in tomogram summary STAR: {tomo_summary_star_path}")
                 logger.error(f"Invalid tomogram summary STAR format: {tomo_summary_star_path}")
                 return

            tomo_tilt_star_paths_rel = global_data[RLN_TOMO_TILT_STAR].tolist()
            tomo_tilt_star_paths_abs = [os.path.join(folder, p) for p in tomo_tilt_star_paths_rel]

            if not tomo_tilt_star_paths_abs:
                st.error(f"No tomogram tilt series STAR files listed in {tomo_summary_star_path}.")
                logger.error(f"No tilt series paths found in {tomo_summary_star_path}")
                return
            
            c1, c2 = st.columns([1, 4])
            # Slider to select tomogram
            tomogram_idx = c1.slider(
                "Select Tomogram Tilt Series:", 0, len(tomo_tilt_star_paths_abs) - 1, 0,
                key=f"tomo_select_{job_name_short}" # Display filename in slider
            )
            selected_tomo_star_path = tomo_tilt_star_paths_abs[tomogram_idx]
            st.write(f"Displaying data for: **{os.path.basename(selected_tomo_star_path)}**")


            if not os.path.exists(selected_tomo_star_path):
                st.error(f"Selected tomogram tilt series STAR file not found: {selected_tomo_star_path}")
                logger.error(f"Selected tomo tilt STAR not found: {selected_tomo_star_path}")
                return

            # Load the actual tilt series data
            tomo_tilt_cache_key = f"tomo_tilt_data_{job_name_short}_{os.path.basename(selected_tomo_star_path)}"
            # Tilt series data is usually in the first table (often 'global' or 'optics', sometimes 'tilts')
            try:
                tomo_tilt_data_dict = parse_star(selected_tomo_star_path)
                # Heuristic to find the correct table (often the first non-empty one)
                tomo_tilt_df = None
                for key, df in tomo_tilt_data_dict.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        tomo_tilt_df = df
                        logger.debug(f"Using table '{key}' from tomogram tilt STAR: {selected_tomo_star_path}")
                        break
                
                if tomo_tilt_df is None:
                    st.error(f"Could not find a valid data table in the selected tomogram tilt STAR file: {selected_tomo_star_path}")
                    logger.error(f"No valid DataFrame found in {selected_tomo_star_path}")
                    return

                # Cache the loaded DataFrame
                st.session_state[tomo_tilt_cache_key] = tomo_tilt_df
                logger.debug(f"Loaded and cached tomogram tilt data from: {selected_tomo_star_path}")

            except Exception as e:
                 report_error(e, f"Failed to parse tomogram tilt series STAR: {selected_tomo_star_path}")
                 logger.error(f"Failed to parse tomogram tilt series STAR: {selected_tomo_star_path}\n{traceback.format_exc()}")
                 st.error(f"Error parsing STAR file: {os.path.basename(selected_tomo_star_path)}")
                 return

            # Generate the plot for the selected tomogram
            required_tomo_cols = [RLN_TOMO_TILT_ANGLE, RLN_DEFOCUS_U, RLN_DEFOCUS_V]
            optional_tomo_cols = [RLN_CTF_SCALEFACTOR]
            
            missing_req = [col for col in required_tomo_cols if col not in tomo_tilt_df.columns]
            if missing_req:
                 st.error(f"Selected tomogram STAR file is missing required columns: {', '.join(missing_req)}")
                 logger.error(f"Missing required columns in {selected_tomo_star_path}: {missing_req}")
                 return

            # Check for optional column for secondary axis
            secondary_axis_col = RLN_CTF_SCALEFACTOR if RLN_CTF_SCALEFACTOR in tomo_tilt_df.columns else None
            if not secondary_axis_col:
                logger.info(f"'{RLN_CTF_SCALEFACTOR}' column not found. Plotting without secondary axis.")


            generate_line_plot(
                df=tomo_tilt_df,
                x_col=RLN_TOMO_TILT_ANGLE,
                y_cols=[RLN_DEFOCUS_U, RLN_DEFOCUS_V],
                axis_right_col=secondary_axis_col,
                x_label="Nominal Stage Tilt Angle (°)",
                y_label="Defocus U/V (Å)",
                y_label_right="CTF Scalefactor" if secondary_axis_col else None,
                keep_aspect_ratio=False, # Aspect ratio doesn't make sense here
                col_to_plot=c2 # Plot in the second column of the layout
            )

        logger.info(f"{datetime.now()}: plot_ctf_refine for {job_name_short} completed.")

    except FileNotFoundError as fnf_err:
        st.error(f"File not found during CTF Refine processing: {fnf_err}")
        logger.error(f"FileNotFoundError in plot_ctf_refine for job {job_name_short}: {fnf_err}\n{traceback.format_exc()}")
        report_error(fnf_err, "File not found")
    except Exception as exc:
        st.error(f"An unexpected error occurred in CTF Refine plotting: {exc}")
        logger.error(f"Unhandled exception in plot_ctf_refine for job {job_name_short}: {exc}\n{traceback.format_exc()}")
        report_error(exc, f"Error plotting CTF Refine job {job_name_short}")