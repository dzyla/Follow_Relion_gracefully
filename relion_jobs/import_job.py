#import_job.py

# Standard Library Imports
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Third-Party Imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Local Imports
from lib.image_utils import micrograph_viewer 
from lib.utils import (
    get_first_key,
    get_modification_time,
    parse_star,
    report_error,  
)

# =============================================================================
# Logger Setup
# =============================================================================
logger = logging.getLogger("main_app")  # Use logger from main app


@st.fragment
def plot_tomogram_picks(
    n: int,  # Tomogram index (for context/logging)
    coords_sel: Dict[str, List[float]],
    coords_rej: Dict[str, List[float]],
    scatter_size: Tuple[int, int] = (5, 2),
) -> go.Figure:
    """
    Creates a 3D scatter plot of selected and rejected tomogram coordinates.

    Args:
        n: Tomogram index (used mainly for logging/titles).
        coords_sel: Dict with '_rlnCoordinateX/Y/Z' lists for selected points.
        coords_rej: Dict with '_rlnCoordinateX/Y/Z' lists for rejected points.
        scatter_size: Tuple of marker sizes (selected, rejected).

    Returns:
        A Plotly Figure object containing the 3D scatter plot.
    """
    try:
        fig = go.Figure()
        required_keys = ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]

        # Plot selected points if data exists and is valid
        sel_x, sel_y, sel_z = [], [], []
        if coords_sel and all(k in coords_sel for k in required_keys):
            sel_x, sel_y, sel_z = (
                coords_sel["_rlnCoordinateX"],
                coords_sel["_rlnCoordinateY"],
                coords_sel["_rlnCoordinateZ"],
            )
            if sel_x:  # Check if list is not empty
                fig.add_trace(
                    go.Scatter3d(
                        x=sel_x,
                        y=sel_y,
                        z=sel_z,
                        mode="markers",
                        marker=dict(size=scatter_size[0], color="green", opacity=0.7),
                        name=f"Selected ({len(sel_x)})",
                        customdata=np.arange(len(sel_x)),
                        hovertemplate="<b>Selected Pick</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>",
                    )
                )
            else:
                logger.info(f"No selected coordinates to plot for tomogram index {n}.")
        else:
            logger.info(
                f"Selected coordinates data structure invalid or missing keys for tomogram index {n}."
            )

        # Plot rejected points if data exists and is valid
        rej_x, rej_y, rej_z = [], [], []
        if coords_rej and all(k in coords_rej for k in required_keys):
            rej_x, rej_y, rej_z = (
                coords_rej["_rlnCoordinateX"],
                coords_rej["_rlnCoordinateY"],
                coords_rej["_rlnCoordinateZ"],
            )
            if rej_x:  # Check if list is not empty
                fig.add_trace(
                    go.Scatter3d(
                        x=rej_x,
                        y=rej_y,
                        z=rej_z,
                        mode="markers",
                        marker=dict(size=scatter_size[1], color="red", opacity=0.6),
                        name=f"Rejected ({len(rej_x)})",
                        customdata=np.arange(len(rej_x)),
                        hovertemplate="<b>Rejected Pick</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>",
                    )
                )
            else:
                logger.info(f"No rejected coordinates to plot for tomogram index {n}.")
        else:
            logger.info(
                f"Rejected coordinates data structure invalid or missing keys for tomogram index {n}."
            )

        fig.update_layout(
            title=f"Tomogram {n + 1}: Particle Picks 3D View",
            scene=dict(
                xaxis_title="X (px)",
                yaxis_title="Y (px)",
                zaxis_title="Z (px)",
                aspectmode="data",
            ),
            legend_title="Pick Status",
            hovermode="closest",
            height=600,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    except KeyError as e:
        report_error(
            e,
            f"Missing coordinate key '{e}' while plotting tomogram picks for index {n}",
        )
        st.warning(f"Coordinate data missing key: {e}. Cannot plot picks fully.")
        return go.Figure()
    except Exception as e:
        report_error(e, f"Error plotting tomogram picks for index {n}")
        st.warning(f"Could not plot tomogram picks: {e}")
        return go.Figure()


def plot_import(rln_folder: str, node_files: List[str]) -> None:
    """
    Processes and displays information for a RELION Import job.

    Handles visualization for imported movies, tomograms (including Relion 5
    STAR-based imports), and particle coordinates. Logs errors instead of
    showing st.error messages.

    Args:
        rln_folder: Base directory of the RELION project.
        node_files: List of node file paths associated with this job.
    """
    if not node_files:
        st.warning("No node files provided for Import job.")
        return

    try:
        star_file_path = os.path.join(rln_folder, node_files[0])
        logger.info(f"Processing Import job using STAR file: {star_file_path}")

        if not os.path.exists(star_file_path):
            logger.error(f"Import job STAR file not found: {star_file_path}")
            st.warning(f"Import STAR file '{node_files[0]}' not found.")
            return

        try:
            star_data_all = parse_star(star_file_path)
            if not star_data_all:
                st.warning(f"Could not parse or data empty in: {node_files[0]}")
                return
        except Exception as e:
            report_error(e, f"Failed to parse STAR file {node_files[0]} for Import job")
            st.warning(f"Error parsing '{node_files[0]}'. Check logs.")
            return

        # Determine data type
        is_movie_import = any("movies" in node for node in node_files)
        is_tomo_import = any("tilt_series" in node for node in node_files) or any(
            "tomograms" in node for node in node_files
        )
        is_particle_import = any("particles" in node for node in node_files)

        # --- Movie/Tomogram Data Processing ---
        if is_movie_import or is_tomo_import:
            logger.info("Processing movie/tomogram import.")
            file_names: Optional[List[str]] = None

            try:
                # --- Corrected Logic to find the data block ---
                data_block = None
                movies_block = star_data_all.get("movies")
                if isinstance(movies_block, pd.DataFrame) and not movies_block.empty:
                    data_block = movies_block
                    logger.debug("Using 'movies' data block.")
                else:
                    global_block = star_data_all.get("global")
                    if isinstance(global_block, pd.DataFrame) and not global_block.empty:
                        data_block = global_block
                        logger.debug("Using 'global' data block.")
                    else:
                        first_key = get_first_key(star_data_all)
                        if first_key:
                            first_block = star_data_all.get(first_key)
                            if (
                                isinstance(first_block, pd.DataFrame)
                                and not first_block.empty
                            ):
                                data_block = first_block
                                logger.debug(f"Using first data block found: '{first_key}'")

                if data_block is None:
                    block_names = list(star_data_all.keys())
                    msg = f"Could not find a suitable non-empty data block ('movies', 'global', or first) in STAR file. Blocks: {block_names}"
                    logger.error(msg)
                    raise KeyError(msg)
                # --- End Corrected Logic ---

                # Extract filenames
                if "_rlnMicrographMovieName" in data_block:
                    file_names = data_block["_rlnMicrographMovieName"].tolist()
                    logger.debug("Found movie names in _rlnMicrographMovieName.")
                elif "_rlnTomoTiltSeriesName" in data_block:
                    file_names = data_block["_rlnTomoTiltSeriesName"].tolist()
                    logger.debug("Found tomogram base names in _rlnTomoTiltSeriesName.")
                elif "_rlnTomoTiltSeriesStarFile" in data_block:  # Relion 5+ Tomo
                    tomo_star_files = data_block["_rlnTomoTiltSeriesStarFile"].tolist()
                    logger.info(
                        f"Relion 5 tomo import: {len(tomo_star_files)} tilt series STARs."
                    )
                    all_tilt_movie_names = []
                    failed_parses = 0
                    for ts_star_rel_path in tomo_star_files:
                        ts_star_full_path = os.path.join(rln_folder, ts_star_rel_path)
                        if not os.path.exists(ts_star_full_path):
                            logger.warning(
                                f"Tilt series STAR not found: {ts_star_full_path}"
                            )
                            failed_parses += 1
                            continue
                        ts_data = parse_star(ts_star_full_path)
                        ts_movies_block = ts_data.get(get_first_key(ts_data))
                        if (
                            ts_movies_block is not None
                            and "_rlnMicrographMovieName" in ts_movies_block
                        ):
                            all_tilt_movie_names.extend(
                                ts_movies_block["_rlnMicrographMovieName"].tolist()
                            )
                        else:
                            logger.warning(
                                f"No movie names in tilt series STAR: {ts_star_rel_path}"
                            )
                            failed_parses += 1
                    if failed_parses > 0:
                        st.warning(
                            f"Failed to process {failed_parses} tilt series STAR file(s)."
                        )
                    file_names = all_tilt_movie_names
                else:
                    raise KeyError("No known movie/tomogram filename column found.")

            except KeyError as e:
                report_error(e, f"Missing key in {node_files[0]} for movie/tomo filenames")
                st.warning(
                    f"Could not find movie/tomogram filenames in STAR file: {e}. See logs."
                )
                return
            except Exception as e:
                report_error(
                    e, f"Error processing movie/tomo import data from {node_files[0]}"
                )
                st.warning(f"Error processing movie/tomo data: {e}. See logs.")
                return

            if not file_names:
                st.warning("No movie or tomogram file paths were found in the STAR file.")
                return

            # --- Modification Time Plot ---
            st.subheader(f"Imported {len(file_names)} Files")
            file_names_abs = [os.path.join(rln_folder, fn) for fn in file_names]
            limit_display = 100
            files_to_check = file_names_abs
            if len(file_names_abs) > limit_display:
                if st.checkbox(
                    f"Limit timeline plot to {limit_display} random files?",
                    value=True,
                    key="limit_import_plot",
                ):
                    indices = np.random.choice(
                        len(file_names_abs), limit_display, replace=False
                    )
                    files_to_check = [file_names_abs[i] for i in indices]
                    st.caption(
                        f"Showing timeline for {limit_display}/{len(file_names_abs)} files."
                    )

            with st.spinner(
                f"Loading modification times for {len(files_to_check)} files..."
            ):
                try:
                    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                        mod_times_opt: List[Optional[datetime]] = list(
                            executor.map(get_modification_time, files_to_check)
                        )
                    file_mod_times: List[datetime] = sorted(
                        [mt for mt in mod_times_opt if mt is not None]
                    )
                except Exception as e:
                    report_error(e, "Error fetching modification times.")
                    st.warning(f"Could not fetch file modification times: {e}. See logs.")
                    file_mod_times = []

            if file_mod_times:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(file_mod_times))),
                        y=file_mod_times,
                        mode="markers",
                        name="Timestamp",
                        marker=dict(size=4),
                        hovertemplate="Index: %{x}<br>Time: %{y|%Y-%m-%d %H:%M:%S}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="File Import Timeline (by modification date)",
                    xaxis_title="File Index (sorted by time)",
                    yaxis_title="Modification Timestamp",
                    height=300,
                    margin=dict(l=50, r=20, t=40, b=40),
                )
                with st.expander("Show Import Timeline", expanded=False):
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No valid modification times found.")

            # --- Micrograph Viewer ---
            st.subheader("Micrograph/Tomogram Viewer")
            try:
                micrograph_viewer(rln_folder, file_names)
            except Exception as e:
                report_error(e, "Error displaying micrograph viewer.")
                st.warning(f"Could not display image viewer: {e}. See logs.")

        # --- Particle Data Processing ---
        elif is_particle_import:
            logger.info("Processing particle import.")
            particles_block = star_data_all.get("particles")
            if particles_block is None or particles_block.empty:
                st.warning("No 'particles' data block found in the STAR file.")
                return

            if "_rlnCoordinateZ" in particles_block.columns:
                logger.info("Detected tomogram particle coordinates.")
                st.subheader("Tomogram Particle Picks")
                tomo_name_col = "_rlnTomoName"
                if tomo_name_col not in particles_block.columns:
                    logger.error(
                        f"Missing '{tomo_name_col}' column for grouping particles."
                    )
                    st.warning(
                        f"Missing '{tomo_name_col}' column. Cannot display picks by tomogram."
                    )
                    return

                unique_tomos = particles_block[tomo_name_col].unique()
                if len(unique_tomos) == 0:
                    st.warning("No tomogram names found in particle data.")
                    return

                tomo_idx = st.slider(
                    "Select Tomogram Index:",
                    0,
                    len(unique_tomos) - 1,
                    0,
                    key="import_particle_tomo_idx",
                )
                selected_tomo_name = unique_tomos[tomo_idx]
                st.caption(f"Showing picks for: {selected_tomo_name}")

                tomo_coords_df = particles_block[
                    particles_block[tomo_name_col] == selected_tomo_name
                ]
                # Ensure required columns exist before creating dict
                coord_cols = ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]
                if not all(col in tomo_coords_df.columns for col in coord_cols):
                    logger.error(
                        "Missing coordinate columns in filtered particle data for selected tomogram."
                    )
                    st.warning(
                        "Missing coordinate columns for selected tomogram. Cannot plot picks."
                    )
                    return

                coords_dict = {k: tomo_coords_df[k].tolist() for k in coord_cols}
                coords_rej_empty: Dict[str, List[float]] = {k: [] for k in coord_cols}

                fig_3d = plot_tomogram_picks(tomo_idx, coords_dict, coords_rej_empty)
                st.plotly_chart(fig_3d, use_container_width=True)

            else:  # SPA particle import
                logger.info("Detected SPA particle coordinates.")
                st.info(
                    f"SPA particle coordinates imported ({len(particles_block)} particles). No 3D plot generated."
                )

        else:
            st.warning(
                "Could not determine data type (movies/tomograms/particles) from node files."
            )
            logger.warning(f"Import job type unclear for node files: {node_files}")

        logger.info(f"Finished processing Import job: {node_files[0]}")
    except Exception as exc:
        report_error(exc)
        st.error(f"An error occurred in plot_import: {exc}")
