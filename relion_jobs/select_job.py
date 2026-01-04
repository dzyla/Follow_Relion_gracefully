#select_job.py

import os
import re
import glob
import tempfile
import traceback
import logging
import math
from datetime import datetime
from typing import List, Union, Optional
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import mrcfile

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from skimage.transform import rescale
from skimage import exposure
import seaborn as sns
import altair as alt
from scipy.stats import gaussian_kde


from lib.utils import (
    parse_star,
    star_from_df,
    report_error,
    get_note, extract_source_job
)
from lib.image_utils import normalize, blur, plot_volume

logger = logging.getLogger("main_app")


# ---------------------
# HELPER FUNCTIONS
# ---------------------



def load_classes(class_path: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load class data from a file or use the provided NumPy array.
    Classes are cached in Streamlit session state to avoid re-calculation.
    """
    cache_key = (
        f"loaded_classes_{class_path}" if isinstance(class_path, str) else "loaded_classes_in_memory"
    )
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    if isinstance(class_path, str):
        # Load from file
        classes = mrcfile.mmap(class_path, permissive=True).data
    else:
        classes = class_path

    # Cache the classes in session state
    st.session_state[cache_key] = classes
    return classes


def create_plot_overlay(
    image_array: np.ndarray, overlay_text: str, display_width: int, font_size: int = 8, padding: int = 5, use_labels: bool = True
):
    """
    Create a matplotlib figure from an image array with text overlay for Streamlit.
    A semi-transparent black rectangle is drawn at the bottom with white overlay text.
    """
    if image_array.dtype != "uint8":
        image_array = (np.clip(image_array, 0, 1) * 255).astype("uint8")

    from PIL import Image
    img = Image.fromarray(image_array)
    orig_width, orig_height = img.size
    scale_factor = display_width / orig_width
    display_height = int(orig_height * scale_factor)

    dpi = 150
    fig_width = display_width / dpi
    fig_height = display_height / dpi
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image_array, cmap="gray")
    ax.axis("off")

    overlay_height_ratio = (font_size * 1.5 + padding) / display_height
    overlay_bottom = 0

    if use_labels:
        rect = patches.Rectangle((0, overlay_bottom), 1, overlay_height_ratio,
                                 transform=fig.transFigure, figure=fig,
                                 color="black", alpha=0.5)
        fig.patches.append(rect)
        text_y = overlay_bottom + (overlay_height_ratio / 2)
        fig.text(
            0.5,
            text_y,
            overlay_text,
            color="white",
            fontsize=font_size,
            ha="center",
            va="center",
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    return fig


@st.fragment
def display_classes(
    class_path: Union[str, np.ndarray], # Can be path or actual data
    class_distribution: np.ndarray,
    sort_by_distribution: bool = True,
    raw_data_star: Optional[str] = None, # Filename within rln_folder
    rln_folder: Optional[str] = None,
) -> None:
    """
    Display class images with selection UI and download functionality.
    """
    try:
        # --- Generate a unique identifier for this specific instance/job ---
        # Use hash of path or job string if available for more stable keys
        pattern = r"job\d\d\d"
        match = re.search(pattern, raw_data_star) if raw_data_star else None
        job_string = match.group(0) if match else hash(str(class_path)) # Fallback identifier

        logger.debug(f"Job string for display_classes: {job_string}. Raw data star: {raw_data_star}")

        # --- State Keys ---
        # Key for the set of selected class *indices* (original indices before sorting)
        selection_state_key = f"selected_indices_{job_string}"
        # Key to store download info: tuple (filename, data_bytes)
        download_info_key = f"download_info_{job_string}"
        # Key for the flag to show the download button
        show_download_key = f"show_download_{job_string}"

        # --- Initialize Session State ---
        if selection_state_key not in st.session_state:
            st.session_state[selection_state_key] = set()
        # Ensure it's always a set
        if not isinstance(st.session_state[selection_state_key], set):
             try:
                 st.session_state[selection_state_key] = set(st.session_state[selection_state_key])
             except TypeError: # Handle case where it might be non-iterable
                 st.session_state[selection_state_key] = set()

        st.session_state.setdefault(download_info_key, None)
        st.session_state.setdefault(show_download_key, False)

        # --- Load and Prepare Class Data ---
        # Assuming load_classes handles loading from path or using passed array
        # And ideally includes caching if loading from path is expensive
        classes = load_classes(class_path) # Replace with actual load/cache logic
        class_distribution = np.asarray(class_distribution).astype(float) # Ensure numpy array

        if classes is None or len(classes) == 0:
             st.warning("No class averages found or empty file.")
             return

        if len(classes) != len(class_distribution):
             st.error("Mismatch between number of classes and distribution data, or failed to load classes.")
             logger.error(f"Class/Distribution mismatch: {len(classes) if classes is not None else 'None'} vs {len(class_distribution)}")
             return

        class_count = len(classes)

        # --- Sort classes if needed ---
        if sort_by_distribution and class_count > 0:
            # argsort gives indices that would sort the array in ascending order
            # [::-1] reverses it for descending order (highest distribution first)
            sorted_indices = np.argsort(class_distribution)[::-1]
        else:
            sorted_indices = np.arange(class_count) # Original 0 to N-1 order

        # --- UI Controls (Select/Unselect/Invert) ---
        ctrl_cols = st.columns(4)
        dynamic_columns = st.slider("Columns for classes:", 4, 12, 8, key=f"cols_slider_{job_string}")
        image_width = 250 # Adjust as needed

        # Function to reset download state (call when selection changes)
        def reset_download_state():
            st.session_state[show_download_key] = False
            st.session_state[download_info_key] = None
            logger.debug(f"Reset download state for {job_string}")

        with ctrl_cols[0]:
            if st.button("Select All", key=f"select_all_{job_string}", on_click=reset_download_state):
                # Select based on the *original* indices
                st.session_state[selection_state_key] = set(np.arange(class_count))
                st.rerun() # Rerun needed to show updated checkboxes immediately

        with ctrl_cols[1]:
            if st.button("Unselect All", key=f"unselect_all_{job_string}", on_click=reset_download_state):
                st.session_state[selection_state_key] = set()
                st.rerun()

        with ctrl_cols[2]:
            if st.button("Invert Selection", key=f"invert_{job_string}", on_click=reset_download_state):
                current_selection = st.session_state[selection_state_key]
                st.session_state[selection_state_key] = {
                    i for i in range(class_count) if i not in current_selection
                }
                st.rerun()

        # --- Display Classes with Checkboxes ---
        container = st.container()
        cols = container.columns(dynamic_columns, gap="small")
        selection_changed_in_loop = False

        # Pagination/Lazy Loading for 2D classes
        MAX_CLASSES_PER_PAGE = 200
        total_pages = (len(sorted_indices) + MAX_CLASSES_PER_PAGE - 1) // MAX_CLASSES_PER_PAGE

        current_page = 0
        if total_pages > 1:
            current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1

        start_idx = current_page * MAX_CLASSES_PER_PAGE
        end_idx = min((current_page + 1) * MAX_CLASSES_PER_PAGE, len(sorted_indices))

        visible_indices = sorted_indices[start_idx:end_idx]

        for display_idx, original_class_idx in enumerate(visible_indices):
            col = cols[display_idx % dynamic_columns]
            with col:
                 try:
                     if original_class_idx >= len(classes):
                         st.warning(f"Index {original_class_idx} out of bounds")
                         continue

                     class_data = classes[original_class_idx]
                     if class_data is None or class_data.size == 0:
                         st.warning(f"Class {original_class_idx+1}: Empty data")
                         continue

                     # Normalize class image for display
                     img_normalized = normalize(class_data)

                     dist_val = 0.0
                     if original_class_idx < len(class_distribution):
                         dist_val = class_distribution[original_class_idx]

                     # Prepare overlay text
                     overlay_txt = f"Class {original_class_idx+1} | {dist_val*100:.2f}%"
                     img_fig = create_plot_overlay(
                         img_normalized, overlay_txt, display_width=image_width, font_size=8
                     )

                     checkbox_key = f"select_class_{job_string}_{original_class_idx}"
                     is_checked_in_state = original_class_idx in st.session_state[selection_state_key]

                     new_checked_value = st.checkbox(
                         f"Select Class {original_class_idx+1}",
                         value=is_checked_in_state,
                         key=checkbox_key,
                         label_visibility="hidden"
                     )

                     # Render image below checkbox
                     # Optimized: Convert Matplotlib figure to image array
                     try:
                         img_fig.canvas.draw()
                         # Use buffer_rgba() as tostring_rgb() is removed in Matplotlib 3.8+
                         width, height = img_fig.canvas.get_width_height()
                         img_array = np.frombuffer(img_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
                         # Use first 3 channels (RGB), ignoring Alpha
                         st.image(img_array[:, :, :3], use_container_width=True)
                     finally:
                         plt.close(img_fig) # Explicitly close figure to save memory

                     # --- Update session state based on checkbox interaction ---
                     # This check runs *after* the checkbox is rendered and potentially interacted with
                     if new_checked_value != is_checked_in_state:
                         if new_checked_value:
                             st.session_state[selection_state_key].add(original_class_idx)
                         else:
                             st.session_state[selection_state_key].discard(original_class_idx)
                         # If the selection changed via checkbox, reset download state and flag rerun
                         reset_download_state()
                         selection_changed_in_loop = True
                 except Exception as e:
                     logger.error(f"Error displaying class {original_class_idx}: {e}")
                     st.error("Error displaying class")


        # If any checkbox changed the state, rerun to ensure consistency and hide download button
        if selection_changed_in_loop:
             logger.debug("Checkbox selection changed, rerunning...")
             st.rerun()


        st.divider()

        # --- Show selected classes section (Optional, can be removed if not needed) ---
        show_selected_button_key = f"show_selected_{job_string}"
        if st.button("Show Selected Classes", key=show_selected_button_key):
            st.session_state[f'show_selected_area_{job_string}'] = True # Use state to keep area visible

        if st.session_state.get(f'show_selected_area_{job_string}', False):
             selected_indices_original = sorted(
                 list(st.session_state[selection_state_key]),
                 key=lambda idx: -class_distribution[idx] # Sort by distribution desc
             )
             st.subheader(f"Selected Classes: (Total {len(selected_indices_original)})")
             if not selected_indices_original:
                 st.write("No classes selected.")
             else:
                 sel_cols = st.columns(dynamic_columns)
                 for idx, class_idx in enumerate(selected_indices_original):
                     overlay_txt = f"Class {class_idx+1} | {class_distribution[class_idx]*100:.2f}%"
                     img_fig_sel = create_plot_overlay(
                         normalize(classes[class_idx]), overlay_txt, display_width=image_width
                     )
                     with sel_cols[idx % dynamic_columns]:
                         st.pyplot(img_fig_sel, use_container_width=True, clear_figure=True)
                     plt.close(img_fig_sel) # Close figure


        # --- Save and Prepare Download ---
        save_button_key = f"save_star_{job_string}"
        if st.button("Prepare STAR file from selected particles", key=save_button_key):
            selected_class_numbers = {idx + 1 for idx in st.session_state[selection_state_key]} # Relion uses 1-based class numbers

            if not selected_class_numbers:
                 st.warning("No classes selected. Cannot generate STAR file.")
                 reset_download_state() # Ensure no old download button shows
            elif rln_folder is None or raw_data_star is None:
                st.error("Missing Relion project folder or source particle STAR file path. Cannot generate file.")
                reset_download_state()
            else:
                try:
                    full_star_path = os.path.join(rln_folder, os.path.basename(raw_data_star))
                    logger.debug(f"Full STAR path: {full_star_path}")
                    if not os.path.exists(full_star_path):
                         st.error(f"Source STAR file not found: {full_star_path}")
                         reset_download_state()
                    else:
                        with st.spinner(f"Parsing {raw_data_star} and filtering particles..."):
                             # --- Parsing and Filtering ---
                             # (Ensure parse_star and star_from_df are imported or available)
                             particles_star = parse_star(full_star_path)
                             if "particles" not in particles_star or "optics" not in particles_star:
                                 st.error(f"Could not find 'optics' or 'particles' data in {raw_data_star}")
                                 reset_download_state()
                             else:
                                 optics_df = particles_star["optics"]
                                 particles_df = particles_star["particles"]

                                 if "_rlnClassNumber" not in particles_df.columns:
                                     st.error("'_rlnClassNumber' column not found in particles data.")
                                     reset_download_state()
                                 else:
                                     # Filter based on the 1-based class numbers
                                     selected_particles_df = particles_df[
                                         particles_df["_rlnClassNumber"].astype(int).isin(selected_class_numbers)
                                     ].copy() # Use copy to avoid SettingWithCopyWarning

                                     st.success(f"Filtered {len(selected_particles_df)} particles from selected classes.")

                                     if selected_particles_df.empty:
                                         st.warning("Selection resulted in zero particles.")
                                         reset_download_state()
                                     else:
                                         # --- Generate STAR file content in memory ---
                                         modified_star_obj = star_from_df({"optics": optics_df, "particles": selected_particles_df})

                                         # Try writing to BytesIO first (more efficient)
                                         try:
                                             bytes_io = BytesIO()
                                             modified_star_obj.write_file(bytes_io) # Assumes write_file can handle buffer
                                             binary_star_data = bytes_io.getvalue()
                                             logger.info("Generated STAR file content in memory.")
                                         except TypeError: # If write_file strictly needs a path
                                             logger.warning("write_file needs path, using temporary file.")
                                             with tempfile.NamedTemporaryFile(delete=False, suffix=".star", mode='wb') as tmp_file:
                                                 # If write_file expects text mode, adjust NamedTemporaryFile mode and reading
                                                 modified_star_obj.write_file(tmp_file.name)
                                                 tmp_file_path = tmp_file.name
                                             with open(tmp_file_path, "rb") as file:
                                                 binary_star_data = file.read()
                                             os.remove(tmp_file_path) # Clean up temp file
                                             logger.info("Generated STAR file content via temporary file.")


                                         final_star_name = f"{job_string}_selected_classes.star"

                                         # --- Store data in session state for download button ---
                                         st.session_state[download_info_key] = (final_star_name, binary_star_data)
                                         st.session_state[show_download_key] = True
                                         # Rerun needed to display the download button in the next step
                                         st.rerun()

                except Exception as e:
                    st.error(f"Failed to generate STAR file: {e}")
                    report_error(e)
                    reset_download_state() # Clear download state on error


        # --- Display Download Button (if data is ready) ---
        if st.session_state.get(show_download_key, False):
            download_info = st.session_state.get(download_info_key)
            if download_info:
                final_star_name, binary_star_data = download_info
                st.download_button(
                    label=f"⬇️ Download {final_star_name} ({len(binary_star_data)/1024:.1f} KB)",
                    data=binary_star_data,
                    file_name=final_star_name,
                    mime="application/octet-stream", # Generic binary MIME type, usually works fine
                    key=f'download_button_{job_string}',
                    help="Click to download the STAR file containing only particles from the selected classes.",
                    # on_click=reset_download_state # Optionally reset state AFTER download click? (Might interfere)
                )
            else:
                 # This case should ideally not happen if show_download_key is True
                 logger.error(f"Inconsistent state: show_download_key is True but download_info_key is None for {job_string}")
                 st.error("Download data not found. Please try preparing the file again.")
                 st.session_state[show_download_key] = False # Reset flag

        st.divider()

    except ImportError as ie:
         st.error(f"Missing necessary library for STAR file handling or plotting: {ie}. Please install it.")
         logger.error(f"ImportError in display_classes: {ie}")
    except AttributeError as ae:
        st.error(f"An attribute error occurred, possibly related to STAR file structure or plotting: {ae}")
        logger.error(f"AttributeError in display_classes: {ae}", exc_info=True)
    except FileNotFoundError as fnf:
        st.error(f"File not found: {fnf}")
        logger.error(f"FileNotFoundError in display_classes: {fnf}")
        reset_download_state() # Reset download if source file gone
    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")
        report_error(exc) # Use your global error reporter
        logger.error("Unexpected error in display_classes", exc_info=True)
        # Attempt to reset download state in case of unexpected error during processing
        if job_string: # Check if job_string was defined
            reset_download_state()


@st.fragment
def plot_combined_classes(volume_paths: List[str], class_dist: List[float]) -> None:
    """
    Display multiple 3D class volumes in a grid using Plotly subplots.
    Expects MRC file paths and a matching list of distribution percentages (0.0–1.0).
    """
    try:
        logger.debug(
            f"plot_combined_classes called with {len(volume_paths)} volume paths and distribution of length {len(class_dist)}"
        )

        with st.expander("3D Volume Plotting Options :gear:", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider(
                    "Select Volume Threshold (Fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                )
                map_size = st.slider(
                    "Map size (px)",
                    min_value=64,
                    max_value=256,
                    value=150,
                    step=2,
                )
            with col2:
                n_columns = st.slider(
                    "Number of columns",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                )
                row_height = st.slider(
                    "Plot height",
                    min_value=200,
                    max_value=1000,
                    value=400,
                    step=100,
                )

            if map_size > 150:
                st.warning("Larger maps can be very slow, especially for many volumes.")

        num_classes = len(volume_paths)
        cols = min(n_columns, num_classes)
        rows = math.ceil(num_classes / n_columns)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        )

        annotations = []
        if st.checkbox("Show class volumes?"):
            with st.spinner("Loading volumes...", show_time=True):
                for idx, path in enumerate(volume_paths):
                    # Load volume from file
                    with mrcfile.mmap(path, permissive=True) as mrc_in:
                        volume_data = mrc_in.data  # 3D NumPy array

                    fig_vol = plot_volume(volume_data, threshold, max_size=map_size)
                    if fig_vol is not None and len(fig_vol.data) > 0:
                        row_idx, col_idx = divmod(idx, cols)
                        fig.add_trace(fig_vol.data[0], row=row_idx + 1, col=col_idx + 1)

                        # Place annotation
                        x = (col_idx + 0.5) / cols
                        y = 1 - (row_idx / rows) - 0.05
                        cls_dist_percent = round(float(class_dist[idx]) * 100, 2)
                        text_str = f"Class {idx + 1}<br>Distribution {cls_dist_percent}%"

                        annotations.append(
                            dict(
                                x=x,
                                y=y,
                                xref="paper",
                                yref="paper",
                                text=text_str,
                                showarrow=False,
                                xanchor="center",
                                yanchor="bottom",
                                font=dict(size=12),
                            )
                        )

                fig.update_layout(
                    hovermode=False,
                    annotations=annotations,
                    height=rows * row_height,
                    margin=dict(l=0, r=0, t=0, b=0),
                )

                # Update each subplot's scene layout
                for n in range(num_classes):
                    scene_id = f"scene{n+1}" if n > 0 else "scene"
                    fig.update_layout(
                        **{
                            scene_id: dict(
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                zaxis=dict(visible=False),
                                camera=dict(eye=dict(x=3, y=3, z=3)),
                            )
                        }
                    )

                st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        report_error(exc)
        logger.error("Error in plot_combined_classes: %s", exc)
        st.error(f"An error occurred while plotting volumes: {exc}")


# ------------------------------------------------------------------
# TOMOGRAM PLOTTING (3D picks) + MICROGRAPH (2D picks)
# ------------------------------------------------------------------
def plot_tomogram_picks(
    file: str,
    n: int,
    coords_sel: pd.DataFrame,
    coords_rej: pd.DataFrame,
    scatter_size: List[int] = [5, 2],
) -> go.Figure:
    """
    Create a 3D scatter plot of selected and rejected picks using Plotly.
    For tomography data, these picks have X, Y, Z coordinates.
    """
    try:
        try:
            coords_sel_x = coords_sel["_rlnCoordinateX"]
            coords_sel_y = coords_sel["_rlnCoordinateY"]
            coords_sel_z = coords_sel.get("_rlnCoordinateZ", np.zeros_like(coords_sel_x))

            coords_rej_x = coords_rej["_rlnCoordinateX"]
            coords_rej_y = coords_rej["_rlnCoordinateY"]
            coords_rej_z = coords_rej.get("_rlnCoordinateZ", np.zeros_like(coords_rej_x))
        except KeyError:
            coords_sel_x = coords_sel["_rlnCenteredCoordinateXAngst"]
            coords_sel_y = coords_sel["_rlnCenteredCoordinateYAngst"]
            coords_sel_z = coords_sel.get("_rlnCenteredCoordinateZAngst", np.zeros_like(coords_sel_x))

            coords_rej_x = coords_rej["_rlnCenteredCoordinateXAngst"]
            coords_rej_y = coords_rej["_rlnCenteredCoordinateYAngst"]
            coords_rej_z = coords_rej.get("_rlnCenteredCoordinateZAngst", np.zeros_like(coords_rej_x))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=coords_sel_x,
                y=coords_sel_y,
                z=coords_sel_z,
                mode="markers",
                marker=dict(size=scatter_size[0], color="green", opacity=0.8),
                name="Selected Points",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=coords_rej_x,
                y=coords_rej_y,
                z=coords_rej_z,
                mode="markers",
                marker=dict(size=scatter_size[1], color="red", opacity=0.8),
                name="Rejected Points",
            )
        )
        fig.update_layout(
            title="3D Scatter Plot of Coordinates",
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
            ),
            legend_title="Points",
            hovermode="closest",
            height=600,
        )
        fig.update_scenes(aspectmode="data")
        return fig
    except Exception as exc:
        report_error(exc)
        logger.error("Error in plot_tomogram_picks: %s", exc)
        return go.Figure()


def plot_micrograph_picks(
    file: str,
    n: int,
    coords_sel: pd.DataFrame,
    coords_rej: pd.DataFrame,
    show_picks: bool = True,
    gaussian_blur_sdev: float = 0.2,
    marker_size: int = 120,
    plot_as_points: bool = False,
    img_size: int = 1024,
):
    """
    Plot micrograph picks in 2D (SPA approach).
    Returns (matplotlib Figure, Altair KDE Chart).
    """
    try:
        micrograph = mrcfile.mmap(file, permissive=True).data
        coords_sel_x = coords_sel["_rlnCoordinateX"].astype(float)
        coords_sel_y = coords_sel["_rlnCoordinateY"].astype(float)
        coords_rej_x = coords_rej["_rlnCoordinateX"].astype(float)
        coords_rej_y = coords_rej["_rlnCoordinateY"].astype(float)

        # Downsample large images if needed
        if micrograph.shape[0] > img_size:
            img_resize_fac = img_size / micrograph.shape[0]
            mic_red = rescale(micrograph.astype(float), img_resize_fac)
        else:
            img_resize_fac = 1
            mic_red = micrograph.astype(float)

        # Apply Gaussian blur and rescale intensity
        mic_red = blur(mic_red, sigma=gaussian_blur_sdev)
        p1, p2 = np.percentile(mic_red, (0.1, 99.8))
        mic_red = exposure.rescale_intensity(mic_red, in_range=(p1, p2))

        sns.set_style("white")
        fig = plt.figure(dpi=200, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frame_on=False)
        ax.imshow(mic_red, cmap="gray")
        ax.axis("off")

        # Decide how to draw the picks
        if plot_as_points:
            sel_kwargs = dict(s=marker_size, color="limegreen", alpha=0.3, marker="o")
            rej_kwargs = dict(s=marker_size, color="#b14e80", alpha=0.3, marker="o")
        else:
            sel_kwargs = dict(
                linestyle="-",
                marker="o",
                s=marker_size,
                facecolors="none",
                edgecolors="limegreen",
                linewidths=1,
            )
            rej_kwargs = dict(
                linestyle="-",
                marker="o",
                s=marker_size,
                facecolors="none",
                edgecolors="#b14e80",
                linewidths=1,
            )

        if show_picks:
            ax.scatter(coords_sel_x * img_resize_fac, coords_sel_y * img_resize_fac, **sel_kwargs)
            ax.scatter(coords_rej_x * img_resize_fac, coords_rej_y * img_resize_fac, **rej_kwargs)

        # Compute intensity distribution for a KDE plot
        radius = int(marker_size // 2)
        height, width = mic_red.shape

        def extract_intensity(xval, yval):
            ix = int(round(xval * img_resize_fac))
            iy = int(round(yval * img_resize_fac))
            x_min = max(0, ix - radius)
            x_max = min(width, ix + radius + 1)
            y_min = max(0, iy - radius)
            y_max = min(height, iy + radius + 1)
            window = mic_red[y_min:y_max, x_min:x_max]
            return np.mean(window) if window.size > 0 else np.nan

        intensities_sel = [extract_intensity(x, y) for x, y in zip(coords_sel_x, coords_sel_y)]
        intensities_rej = [extract_intensity(x, y) for x, y in zip(coords_rej_x, coords_rej_y)]
        intensities_sel = np.array([i for i in intensities_sel if not np.isnan(i)])
        intensities_rej = np.array([i for i in intensities_rej if not np.isnan(i)])
        all_intensities = np.concatenate([intensities_sel, intensities_rej])

        if all_intensities.size == 0:
            x_grid = np.linspace(0, 1, 100)
        else:
            x_min_val, x_max_val = np.min(all_intensities), np.max(all_intensities)
            x_grid = np.linspace(x_min_val, x_max_val, 100)

        kde_sel = gaussian_kde(intensities_sel) if intensities_sel.size > 0 else None
        kde_rej = gaussian_kde(intensities_rej) if intensities_rej.size > 0 else None

        density_sel = kde_sel(x_grid) if kde_sel else np.zeros_like(x_grid)
        density_rej = kde_rej(x_grid) if kde_rej else np.zeros_like(x_grid)

        df_sel = pd.DataFrame({"intensity": x_grid, "density": density_sel, "group": "Selected"})
        df_rej = pd.DataFrame({"intensity": x_grid, "density": density_rej, "group": "Rejected"})
        df_kde = pd.concat([df_sel, df_rej], ignore_index=True)

        kde_chart = (
            alt.Chart(df_kde)
            .mark_line()
            .encode(
                x=alt.X("intensity:Q", title="Average Intensity"),
                y=alt.Y("density:Q", title="Density"),
                color=alt.Color(
                    "group:N",
                    scale=alt.Scale(domain=["Selected", "Rejected"], range=["limegreen", "#b14e80"]),
                ),
            )
            .properties(title="Intensity around Peaks", width=300, height=350)
        )

        return fig, kde_chart
    except Exception:
        error_info = traceback.format_exc()
        print(f"An error occurred in plot_micrograph_picks:\n{error_info}")
        return None, None


@st.fragment
def show_micrograph_picks_ui(
    particles_selected: pd.DataFrame,
    particles_source: pd.DataFrame,
    folder: str,
    micrograph_column: str = "_rlnMicrographName",
    section_label: str = "Micrograph index",
    col_ratios: list = [1, 4],
):
    """
    Display UI for micrograph picks using the same code for both 2D and 3D classification.

    Parameters:
        particles_selected (DataFrame): Selected particles subset with columns
                                        ["_rlnCoordinateX", "_rlnCoordinateY"] and micrograph_column.
        particles_source (DataFrame): Full source particles with the same columns.
        folder (str): Path to the folder containing micrographs.
        micrograph_column (str): Column name identifying the micrograph (or tomogram). Defaults to '_rlnMicrographName'.
        section_label (str): Label for the index slider ('Micrograph index' or 'Tomogram index').
        col_ratios (list): Layout ratio for Streamlit columns.
    """
    try:
        unique_mics = np.unique(particles_selected[micrograph_column])
        if len(unique_mics) == 0:
            st.write("No micrographs found.")
            return

        c1, c2 = st.columns(col_ratios)
        file_idx = c1.slider(section_label, 0, len(unique_mics), 0)
        gaussian_blur_sdev = c1.slider("Gaussian blur (sdev)", 0.0, 5.0, 0.2, 0.1)
        marker_size = c1.slider("Marker size", 10, 300, 120, 10)
        plot_as_points = c1.checkbox("Plot as points?", value=True)
        show_picks = c1.checkbox("Show picks?", value=True)

        st.write("Particles: :green[selected] :red[rejected]")
        file_mic = os.path.join(folder, unique_mics[file_idx])

        coords_sel = particles_selected[
            particles_selected[micrograph_column] == unique_mics[file_idx]
        ][["_rlnCoordinateX", "_rlnCoordinateY"]]
        all_coords = particles_source[
            particles_source[micrograph_column] == unique_mics[file_idx]
        ][["_rlnCoordinateX", "_rlnCoordinateY"]]

        merged_df = pd.merge(
            all_coords,
            coords_sel,
            on=["_rlnCoordinateX", "_rlnCoordinateY"],
            how="outer",
            indicator=True,
        )
        coords_rej = merged_df[merged_df["_merge"] == "left_only"][["_rlnCoordinateX", "_rlnCoordinateY"]]

        # Call your existing plot_micrograph_picks function, returning (matplotlib Figure, Altair chart).
        fig_mic, fig_stats = plot_micrograph_picks(
            file_mic,
            file_idx,
            coords_sel,
            coords_rej,
            show_picks,
            gaussian_blur_sdev,
            marker_size,
            plot_as_points,
        )

        if fig_mic is not None:
            c2.pyplot(fig_mic, use_container_width=False)
        if fig_stats is not None:
            c1.altair_chart(fig_stats, use_container_width=True)

    except Exception as exc:
        report_error(exc)
        logger.error("Error in show_micrograph_picks_ui: %s", exc)
        st.error("Error displaying micrograph picks UI.")




# ------------------------------------------------------------------
# MAIN "plot_selection" EXAMPLE
# ------------------------------------------------------------------
def plot_selection(node_files: list, FOLDER: str, job_name: str) -> None:
    """
    Display class selection and particle picking statistics based on the provided job files.
    Depending on node_files, it displays different plots:
      - If "class_averages.star" is present, a pie chart and class images with checkboxes are shown.
      - If a single "particles.star" is present, a pie chart of selected vs rejected particles is shown.
      - If files contain "split", a simple file list is displayed.
      - For 3D classification, it also plots 3D volumes (isosurface) and micrograph/tomogram picks.
    """
    logger.debug(f"plot_selection called with node_files: {node_files}, FOLDER: {FOLDER}, job_name: {job_name}")

    try:
        # --------------------------------------------------------------
        # 1) 2D class selection
        # --------------------------------------------------------------
        if any("class_averages.star" in f for f in node_files):
            selected_particles_star_path = node_files[0]
            selected_class_star_path = os.path.join(FOLDER, selected_particles_star_path)
            particles_selected = parse_star(selected_class_star_path).get("particles", pd.DataFrame())
            num_particles_selected = len(particles_selected)

            note = get_note(os.path.join(FOLDER, job_name, "note.txt"))
            source_job = extract_source_job(note)
            logger.debug(f"Source job: {source_job}")

            source_star_path = os.path.join(FOLDER, source_job)
            particles_source = parse_star(source_star_path).get("particles", pd.DataFrame())
            num_particles_source = len(particles_source)

            # Display pie chart
            df_count = pd.DataFrame(
                {
                    "status": ["Selected", "Rejected"],
                    "count": [num_particles_selected, num_particles_source - num_particles_selected],
                }
            )
            fig_pie = px.pie(
                df_count,
                values="count",
                names="status",
                title="Selected vs Rejected Particles",
                hole=0.3,
                labels={"status": "Status", "count": "Count"},
            )
            fig_pie.update_traces(hoverinfo="label+percent+value", textinfo="percent+value")
            st.plotly_chart(fig_pie)
            st.divider()

            try:
                # Show 2D class images with checkboxes
                cls_averages_meta_path = node_files[1]
                cls_averages_data = list(parse_star(os.path.join(FOLDER, cls_averages_meta_path)).values())[0]
                cls_images = []
                selected_classes = []
                for cls_ in cls_averages_data["_rlnReferenceImage"]:
                    cls_data = cls_.split("@")
                    class_img = mrcfile.mmap(os.path.join(FOLDER, cls_data[1]), permissive=True).data
                    cls_images.append(class_img[int(cls_data[0]) - 1])
                    selected_classes.append(int(cls_data[0]))

                cls_images = np.stack(cls_images)
                cls_distribution = cls_averages_data["_rlnClassDistribution"]
                display_classes(
                    cls_images,
                    cls_distribution,
                    sort_by_distribution=True,
                    raw_data_star=selected_particles_star_path,
                    rln_folder=FOLDER,
                )

                # Show micrograph picks
                show_micrograph_picks_ui(
                    particles_selected,
                    particles_source,
                    folder=FOLDER,
                    micrograph_column="_rlnMicrographName",
                    section_label="Micrograph index",
                    col_ratios=[1, 4],
                )


            except Exception as exc:
                report_error(exc)
                logger.error("Error in class selection display: %s", exc)
                st.write("No additional class selection data found.")

        # --------------------------------------------------------------
        # 2) Split job
        # --------------------------------------------------------------
        elif not any("class_averages.star" in f for f in node_files) and any("split" in f for f in node_files) or any("join" in f for f in node_files):
            st.subheader("**Split job generated files:**")
            star_paths = [os.path.join(FOLDER, x) for x in node_files if x.endswith(".star")]
            star_content = {}
            for star_file in star_paths:
                star_data = parse_star(star_file)
                if len(star_data) == 1:
                    key = list(star_data.keys())[0]
                    size = len(star_data[key])
                    star_content[os.path.basename(star_file)] = {key: size}
                else:
                    max_key = max(star_data, key=lambda k: len(star_data[k]))
                    size = len(star_data[max_key])
                    star_content[os.path.basename(star_file)] = {max_key: size}
            st.write(star_content)

        # --------------------------------------------------------------
        # 3) 3D classification logic
        # --------------------------------------------------------------
        elif any("particles.star" in f for f in node_files) and len(node_files) == 1:
            logger.info("3D classification selection")
            note = get_note(os.path.join(FOLDER, job_name, "note.txt"))
            try:
                source_job = re.search(r"--i\s([\w\d/]+\.star)", note).group(1).replace("optimiser", "data")
            except Exception as exc:
                report_error(exc)
                st.error(f"No Data star found for job {job_name}")
                return

            particles_selected = parse_star(os.path.join(FOLDER, node_files[0])).get("particles", pd.DataFrame())
            particles_source = parse_star(os.path.join(FOLDER, source_job)).get("particles", pd.DataFrame())
            num_particles_selected = len(particles_selected)
            num_particles_source = len(particles_source)

            # Pie chart
            df_count = pd.DataFrame(
                {
                    "status": ["Selected", "Rejected"],
                    "count": [num_particles_selected, num_particles_source - num_particles_selected],
                }
            )
            fig_pie = px.pie(
                df_count,
                values="count",
                names="status",
                title="Selected vs Rejected Particles",
                hole=0.3,
                labels={"status": "Status", "count": "Count"},
            )
            fig_pie.update_traces(hoverinfo="label+percent+value", textinfo="percent+value")
            st.plotly_chart(fig_pie)
            st.divider()


            # Identify selected classes
            try:
                selected_classes = np.unique(particles_selected["_rlnClassNumber"]).astype(int)
                source_job_folder = source_job.split("/")
                model_files = glob.glob(os.path.join(FOLDER, source_job_folder[0], source_job_folder[1], "*model.star"))
                model_files.sort(key=os.path.getmtime)
                last_model_star_path = model_files[-1]

                parent_star = parse_star(os.path.join(FOLDER, last_model_star_path))["model_classes"]
                mrcs_paths = parent_star["_rlnReferenceImage"]
                class_dist = parent_star["_rlnClassDistribution"]

                # Filter volumes for selected classes
                selected_rows = mrcs_paths.str.contains(
                    "|".join(f"class{class_num:03d}" for class_num in selected_classes)
                )
                selected_classes_volumes = mrcs_paths[selected_rows]
                class_dist = class_dist[selected_rows].tolist()

                class_paths = []
                for path_str in selected_classes_volumes.values:
                    full_path = os.path.join(FOLDER, path_str)
                    if full_path not in class_paths:
                        class_paths.append(full_path)

                # Plot combined 3D volumes
                plot_combined_classes(class_paths, class_dist)

                st.divider()

                # Distinguish SPA vs Tomography
                if "_rlnTomoName" not in particles_source.columns:
                    show_micrograph_picks_ui(
                        particles_selected,
                        particles_source,
                        folder=FOLDER,
                        micrograph_column="_rlnMicrographName",
                        section_label="Micrograph index",
                        col_ratios=[1, 4],
                    )
                else:
                    # Tomography job: 3D picks
                    unique_mics = np.unique(particles_selected["_rlnTomoName"])

                    col1, col2 = st.container().columns([1, 4])
                    file_idx = col1.slider("Tomogram index", 0, len(unique_mics)-1, 0)

                    logger.debug(f'particles_selected: {particles_selected}')

                    file_tomo = os.path.join(FOLDER, unique_mics[file_idx])
                    logger.debug(f"file_tomo: {file_tomo}")

                    try:
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
                        coords_rej = merged_df[merged_df["_merge"] == "left_only"][
                            ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]
                        ]

                    except KeyError:
                        try:
                            coords_sel = particles_selected[
                            particles_selected["_rlnTomoName"] == unique_mics[file_idx]
                            ][["_rlnCenteredCoordinateXAngst", "_rlnCenteredCoordinateYAngst", "_rlnCenteredCoordinateZAngst"]]
                            all_coords = particles_source[
                                particles_source["_rlnTomoName"] == unique_mics[file_idx]
                            ][["_rlnCenteredCoordinateXAngst", "_rlnCenteredCoordinateYAngst", "_rlnCenteredCoordinateZAngst"]]

                            merged_df = pd.merge(
                                    all_coords,
                                    coords_sel,
                                    on=["_rlnCenteredCoordinateXAngst", "_rlnCenteredCoordinateYAngst", "_rlnCenteredCoordinateZAngst"],
                                    how="outer",
                                    indicator=True,
                                    )
                            coords_rej = merged_df[merged_df["_merge"] == "left_only"][
                                ["_rlnCenteredCoordinateXAngst", "_rlnCenteredCoordinateYAngst", "_rlnCenteredCoordinateZAngst"]
                            ]


                        except Exception as exc:
                            report_error(exc)
                            logger.error(f"Error: {exc}")
                            st.write("No coordinates found for this tomogram.")
                            return

                    col1.markdown(f"**Tomogram:** `{unique_mics[file_idx]}`")
                    fig_tomo = plot_tomogram_picks(file_tomo, file_idx, coords_sel, coords_rej)
                    if fig_tomo is not None:
                        col2.plotly_chart(fig_tomo)

            except Exception:
                error_info = traceback.format_exc()
                logger.error(f"{datetime.now()}: An unexpected error occurred:\n{error_info}")
                st.write("No other files found.")

        logger.info(f"{datetime.now()}: plot_selection done")

    except Exception as exc:
        report_error(exc)
        logger.error("Error in plot_selection: %s", exc)


    plt.close()