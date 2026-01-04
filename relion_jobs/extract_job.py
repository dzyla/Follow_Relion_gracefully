# extract_job.py

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import altair as alt
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# Local application imports
from lib.image_utils import blur, clip, normalize_particle
from lib.utils import (get_unique_key, interactive_scatter_plot, parse_star,
                       report_error)

logger = logging.getLogger("main_app")

# --- Constants ---
# RELION Column Names / Keys
PARTICLES_KEY = "particles"
RLN_IMAGE_NAME = "_rlnImageName"
RLN_IMAGE_STACK_NAME = "_rlnImageStackName"
RLN_MICROGRAPH_NAME = "_rlnMicrographName"
RLN_DEFOCUS_U = "_rlnDefocusU"
RLN_AUTOPICK_FOM = "_rlnAutopickFigureOfMerit"
RLN_CTF_IMAGE = "_rlnCtfImage"

# File Names / Patterns
PARTICLES_STAR = "particles.star"
PARTICLES_SUBTRACTED_STAR = "particles_subtracted.star"
OPTIMISATION_SET_STAR = "optimisation_set.star"

# UI Defaults
DEFAULT_N_PARTICLES = 15
DEFAULT_PARTICLE_SIZE = 64
NUM_DISPLAY_COLS = 5
DEFAULT_BLUR_SIGMA = 1.0

# Altair Column Names
ALTAIR_FOM_COL = "AutopickFOM"
ALTAIR_PARTICLES_COL = "Particles"
ALTAIR_MICROGRAPH_INDEX_COL = "MicrographIndex"
ALTAIR_DEFOCUS_U_COL = "DefocusU"


# -----------------------------------------
# Altair-based statistics
# -----------------------------------------
def altair_histogram_fom(figure_of_merit: np.ndarray) -> Optional[alt.Chart]:
    """Creates an Altair histogram for the Autopick Figure of Merit."""
    try:
        df = pd.DataFrame({ALTAIR_FOM_COL: figure_of_merit})
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X(f"{ALTAIR_FOM_COL}:Q", bin=alt.Bin(maxbins=30), title="Autopick FOM"),
                alt.Y("count()", title="Count"),
            )
            .properties(title="Histogram of Autopick Figure of Merit")
        ).interactive()
        return chart
    except Exception as e:
        report_error(e, "Failed to create Autopick FOM histogram")
        return None


def altair_histogram_particles_per_mic(particles_per_mic: List[int]) -> Optional[alt.Chart]:
    """Creates an Altair histogram for the number of particles per micrograph."""
    try:
        df = pd.DataFrame({ALTAIR_PARTICLES_COL: particles_per_mic})
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X(f"{ALTAIR_PARTICLES_COL}:Q", bin=alt.Bin(maxbins=30), title="# of Particles"),
                alt.Y("count()", title="Micrograph Count"),
            )
            .properties(title="Histogram of Particles Per Micrograph")
        ).interactive()
        return chart
    except Exception as e:
        report_error(e, "Failed to create particles per micrograph histogram")
        return None


def altair_heatmap_defocus_fom(defocus_u: np.ndarray, figure_of_merit: np.ndarray) -> Optional[alt.Chart]:
    """Creates an Altair heatmap of Defocus U vs Autopick FOM."""
    try:
        df = pd.DataFrame({ALTAIR_DEFOCUS_U_COL: defocus_u, ALTAIR_FOM_COL: figure_of_merit})
        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                alt.X(f"{ALTAIR_DEFOCUS_U_COL}:Q", bin=alt.Bin(maxbins=40), title="Defocus U"),
                alt.Y(f"{ALTAIR_FOM_COL}:Q", bin=alt.Bin(maxbins=20), title="Autopick FOM"),
                alt.Color("count()", scale=alt.Scale(scheme="viridis"), legend=None, title="Count"),
                tooltip=[
                    alt.Tooltip(f"{ALTAIR_DEFOCUS_U_COL}:Q", title="Defocus U", bin=True),
                    alt.Tooltip(f"{ALTAIR_FOM_COL}:Q", title="Autopick FOM", bin=True),
                    alt.Tooltip("count()", title="Count")
                ]
            )
            .properties(title="Defocus U vs. Autopick FOM")
        ).interactive()
        return chart
    except Exception as e:
        report_error(e, "Failed to create Defocus vs FOM heatmap")
        return None


def altair_line_graph(micrograph_indices: List[int], particles_per_mic: List[int]) -> Optional[alt.Chart]:
    """Creates an Altair line graph of particles per micrograph."""
    try:
        df = pd.DataFrame({ALTAIR_MICROGRAPH_INDEX_COL: micrograph_indices, ALTAIR_PARTICLES_COL: particles_per_mic})
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{ALTAIR_MICROGRAPH_INDEX_COL}:Q", title="Micrograph Index"),
                y=alt.Y(f"{ALTAIR_PARTICLES_COL}:Q", title="# of Particles"),
                tooltip=[ALTAIR_MICROGRAPH_INDEX_COL, ALTAIR_PARTICLES_COL]
            )
            .properties(title="Particles per Micrograph (Sequential)")
        ).interactive()
        return chart
    except Exception as e:
        report_error(e, "Failed to create particles per micrograph line graph")
        return None


# -----------------------------------------
# Image Loading (MRC)
# -----------------------------------------
def load_spa_image(full_info: str) -> Optional[np.ndarray]:
    """
    Loads a single particle image (potentially indexed) from an MRC file.

    Args:
        full_info (str): Path to MRC file, potentially with '#idx=N'.

    Returns:
        Optional[np.ndarray]: The raw 2D numpy array of the particle, or None on error.
    """
    try:
        path_part = full_info
        idx: Optional[int] = None

        if "#idx=" in full_info:
            path_part, idx_part = full_info.split("#idx=", 1)
            path_part = path_part.strip()
            try:
                idx = int(idx_part)
            except ValueError:
                logger.warning(f"Invalid index '{idx_part}' in path: {full_info}")
                return None

        # Use permissive=True for potentially corrupted headers
        with mrcfile.mmap(path_part, mode="r", permissive=True) as mrc:
            if idx is not None:
                if idx < 0 or idx >= mrc.data.shape[0]:
                    logger.warning(f"Index {idx} out of bounds for MRC file {path_part} with shape {mrc.data.shape}")
                    return None
                arr = mrc.data[idx]
            else:
                # If no index, assume it's a single 2D image or maybe a stack (take first image)
                if mrc.data.ndim == 3:
                    logger.debug(f"Loading first slice from 3D MRC file (no index provided): {path_part}")
                    arr = mrc.data[0]
                elif mrc.data.ndim == 2:
                    arr = mrc.data
                else:
                    logger.warning(f"Unexpected MRC data dimension {mrc.data.ndim} for path: {path_part}")
                    return None
        return arr.copy() # Return a copy to avoid issues with memory mapping
    except FileNotFoundError:
        logger.warning(f"MRC file not found: {path_part}")
        return None
    except Exception as e:
        report_error(e, f"Failed to load SPA image from: {full_info}")
        return None


def load_3d_mrc(path: str) -> Optional[np.ndarray]:
    """
    Loads an MRC file, preserving its dimensions (can be 2D or 3D).

    Args:
        path (str): Path to the MRC file.

    Returns:
        Optional[np.ndarray]: The raw numpy array (2D or 3D), or None on error.
    """
    try:
        # Use permissive=True for potentially corrupted headers
        with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
            data = mrc.data.copy() # Return a copy
        return data
    except FileNotFoundError:
        logger.warning(f"MRC file not found: {path}")
        return None
    except Exception as e:
        report_error(e, f"Failed to load 3D MRC from: {path}")
        return None


# -----------------------------------------
# Image Display Helper
# -----------------------------------------
def display_particles(particle_list: List[Optional[np.ndarray]], particle_size: int, title: str = "") -> None:
    """
    Displays a grid of particle images using Matplotlib in Streamlit columns.

    Args:
        particle_list (List[Optional[np.ndarray]]): List of 2D numpy arrays (particles).
                                                    None entries are skipped.
        particle_size (int): Target display size (influences figure size).
        title (str, optional): Title for the expander containing the grid. Defaults to "".
    """
    valid_particles = [p for p in particle_list if p is not None]
    if not valid_particles:
        st.info(f"No {title} images available to display." if title else "No images available to display.")
        return

    if title:
        st.subheader(title)

    num_valid = len(valid_particles)
    num_rows = int(np.ceil(num_valid / NUM_DISPLAY_COLS))

    with st.expander(title if title else "Particle Images", expanded=True):
        fig = None # Initialize fig to None
        try:
            for row in range(num_rows):
                row_cols = st.columns(NUM_DISPLAY_COLS)
                for col_index in range(NUM_DISPLAY_COLS):
                    idx = row * NUM_DISPLAY_COLS + col_index
                    if idx >= num_valid:
                        break # Stop if we run out of images

                    arr = valid_particles[idx]
                    with row_cols[col_index]:
                        try:
                            # Create a small figure for each particle
                            fig, ax = plt.subplots(figsize=(particle_size / 32, particle_size / 32)) # Adjust divisor for desired size
                            ax.imshow(arr, cmap="gray")
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # Turn off axis spines for cleaner look
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            st.pyplot(fig, use_container_width=False)
                            plt.close(fig) # Close figure to free memory
                            fig = None # Reset fig
                        except Exception as e_inner:
                            report_error(e_inner, f"Failed to display particle image at index {idx}")
                            st.caption("Error displaying image.")
                            if fig:
                                plt.close(fig) # Ensure figure is closed on error
                                fig = None
        finally:
             # Ensure the last figure is closed if an error occurred mid-loop
             if fig:
                 plt.close(fig)


# -----------------------------------------
# Single-particle processing (SPA)
# -----------------------------------------
def show_random_particles(
    star_path: str,
    base_folder: str,
    controls_col: DeltaGenerator,
    images_col: DeltaGenerator
) -> pd.DataFrame:
    """
    Loads, processes, and displays random particles from an SPA star file.

    Args:
        star_path (str): Full path to the particles.star file.
        base_folder (str): Base project path to resolve relative paths in the star file.
        controls_col (DeltaGenerator): Streamlit column for UI controls.
        images_col (DeltaGenerator): Streamlit column for displaying images.

    Returns:
        pd.DataFrame: The loaded particle DataFrame, or an empty DataFrame on error.
    """
    particles_df: Optional[pd.DataFrame] = None
    try:
        star_data = parse_star(star_path)
        if not star_data or PARTICLES_KEY not in star_data:
            controls_col.warning(f"No '{PARTICLES_KEY}' data found in {os.path.basename(star_path)}.")
            return pd.DataFrame()
        particles_df = star_data[PARTICLES_KEY]
        data_shape = particles_df.shape[0]
        controls_col.write(f"**Extracted {data_shape} particles**")

    except Exception as e:
        report_error(e, f"Failed to parse star file: {star_path}")
        controls_col.error(f"Error reading star file: {os.path.basename(star_path)}")
        return pd.DataFrame()

    with controls_col:
        try:
            n_particles_input = st.number_input(
                "Number of Particles to Show", min_value=1, value=DEFAULT_N_PARTICLES, key=f"n_part_{star_path}"
            )
            # particle_size = st.number_input("Image Size (px)", min_value=32, value=DEFAULT_PARTICLE_SIZE) # Keep fixed for now
            particle_size = DEFAULT_PARTICLE_SIZE
            apply_contrast = st.checkbox("Apply Contrast Clip (1-99%)?", value=False, key=f"contrast_{star_path}")
            gauss_sdev = st.slider(
                "Apply Gaussian Blur Sigma", min_value=0.0, value=DEFAULT_BLUR_SIGMA, max_value=5.0, step=0.1, key=f"blur_{star_path}"
            )
        except Exception as e:
             report_error(e, "Error creating Streamlit controls for particle display.")
             st.warning("Error setting up display controls.")
             return particles_df # Return df loaded so far

    n_particles_to_display = min(n_particles_input, data_shape)
    if n_particles_to_display <= 0:
         images_col.info("Select 1 or more particles to display.")
         return particles_df

    # Use session state to keep the random selection consistent across reruns
    session_key = get_unique_key("selected_particles", star_path, n_particles_to_display)
    if session_key not in st.session_state:
        try:
            random_indices = np.random.choice(data_shape, size=n_particles_to_display, replace=False)
            st.session_state[session_key] = random_indices
        except ValueError as e:
            report_error(e, f"Error generating random indices (data_shape={data_shape}, n={n_particles_to_display})")
            controls_col.warning("Could not select random particles.")
            return particles_df
    else:
         # Ensure saved indices are valid if n_particles changed
         if len(st.session_state[session_key]) != n_particles_to_display:
             try:
                 random_indices = np.random.choice(data_shape, size=n_particles_to_display, replace=False)
                 st.session_state[session_key] = random_indices
             except ValueError as e:
                report_error(e, f"Error generating random indices (data_shape={data_shape}, n={n_particles_to_display})")
                controls_col.warning("Could not select random particles.")
                return particles_df
         else:
             random_indices = st.session_state[session_key]


    try:
        selected_particles = particles_df.iloc[random_indices]
    except IndexError as e:
        report_error(e, f"Error selecting particles using indices: {random_indices}")
        controls_col.warning("Error accessing selected particle data.")
        # Clear invalid state
        if session_key in st.session_state: del st.session_state[session_key]
        return particles_df
    except Exception as e:
        report_error(e, "Unexpected error selecting particles from DataFrame.")
        controls_col.warning("Error selecting particle data.")
        return particles_df

    # --- Nested Function for Parsing Image Name ---
    def parse_rln_image_name(image_entry: str) -> Optional[str]:
        """Parses RELION _rlnImageName format (e.g., '001@path/to/mrcs') into full path with optional index."""
        try:
            if "@" in image_entry:
                idx_str, relative_path = image_entry.split("@", 1)
                # Validate index part
                try:
                    idx = int(idx_str) - 1 # RELION uses 1-based indexing
                    if idx < 0: raise ValueError("Index must be positive")
                except ValueError:
                    logger.warning(f"Invalid index '{idx_str}' in image entry: {image_entry}")
                    return None
                full_path = os.path.join(base_folder, relative_path)
                return f"{full_path}#idx={idx}"
            else:
                # Assume it's just a relative path to a single-image MRC
                return os.path.join(base_folder, image_entry)
        except Exception as e:
            report_error(e, f"Error parsing image entry: {image_entry}")
            return None
    # --- End Nested Function ---

    try:
        if RLN_IMAGE_NAME not in selected_particles.columns:
             controls_col.warning(f"Column '{RLN_IMAGE_NAME}' not found in star file.")
             return particles_df

        # Generate full paths (or path#idx) for selected particles
        data_paths = [parse_rln_image_name(entry) for entry in selected_particles[RLN_IMAGE_NAME]]
        valid_data_paths = [p for p in data_paths if p is not None]

        if not valid_data_paths:
            images_col.info("No valid particle image paths found.")
            return particles_df

        # Read raw data concurrently
        raw_arrays: List[Optional[np.ndarray]] = []
        with ThreadPoolExecutor() as executor:
            # Map returns results in the order futures are submitted
            results = executor.map(load_spa_image, valid_data_paths)
            raw_arrays = list(results) # Collect results

        # Process loaded arrays sequentially (normalization, blur, contrast)
        processed_particles: List[Optional[np.ndarray]] = []
        for arr in raw_arrays:
            if arr is None:
                processed_particles.append(None) # Keep placeholder for failed loads
                continue
            try:
                # Normalize first
                proc_arr = normalize_particle(arr)
                # Apply blur if sigma > 0
                if gauss_sdev > 0.0:
                    proc_arr = blur(proc_arr, sigma=gauss_sdev)
                # Apply contrast clip if requested
                if apply_contrast:
                    proc_arr = clip(proc_arr, 1, 99)
                processed_particles.append(proc_arr)
            except Exception as e:
                report_error(e, "Error processing particle image array")
                processed_particles.append(None) # Add placeholder on processing error

        with images_col:
            display_particles(processed_particles, particle_size, title="Random SPA Particles")

    except KeyError as e:
         report_error(e, f"Missing expected column for particle display: {e}")
         controls_col.warning(f"Missing required column in star file: {e}")
    except Exception as e:
         report_error(e, "Error during particle loading or processing pipeline.")
         images_col.error("An error occurred displaying particles.")

    return particles_df # Return the original loaded DataFrame


# -----------------------------------------
# Tomography processing (RELION 4/5)
# -----------------------------------------
def detect_relion_version(star_df: pd.DataFrame) -> int:
    """
    Heuristic to identify RELION version based on typical column names.

    Args:
        star_df (pd.DataFrame): The DataFrame from particles.star.

    Returns:
        int: 4 or 5, representing the guessed RELION version.
    """
    columns = set(star_df.columns)
    # RELION 4 typically has _rlnCtfImage for pseudo-subtomo CTFs
    if RLN_CTF_IMAGE in columns:
        logger.debug("Detected RELION version 4 based on _rlnCtfImage column.")
        return 4
    # RELION 5 often uses _rlnImageStackName for direct 2D stacks
    logger.debug("Assuming RELION version 5 (no _rlnCtfImage column found).")
    return 5


def plot_relion4_pseudosubtomos(
    base_folder: str,
    star_df: pd.DataFrame,
    selection_indices: np.ndarray,
    particle_size: int,
    apply_contrast: bool,
    gauss_sdev: float,
    images_col: DeltaGenerator
):
    """Loads and displays RELION 4 pseudo-subtomograms and their CTFs."""
    try:
        logger.info("Processing RELION 4 pseudo-subtomograms.")
        if RLN_IMAGE_NAME not in star_df.columns:
            images_col.warning(f"Missing column '{RLN_IMAGE_NAME}'. Cannot display pseudo-subtomograms.")
            return

        selected_entries = star_df.iloc[selection_indices]
        pseudo_data_names = selected_entries[RLN_IMAGE_NAME].values.tolist()

        ctf_names = []
        has_ctf_column = RLN_CTF_IMAGE in selected_entries.columns
        if has_ctf_column:
            ctf_names = selected_entries[RLN_CTF_IMAGE].values.tolist()
        else:
             logger.info("No '_rlnCtfImage' column found, CTF volumes will not be displayed.")

        data_paths = [os.path.join(base_folder, name) for name in pseudo_data_names if pd.notna(name)]
        ctf_paths = [os.path.join(base_folder, name) for name in ctf_names if pd.notna(name)]

        # Load pseudo-subtomogram volumes concurrently
        raw_data: List[Optional[np.ndarray]] = []
        if data_paths:
            with ThreadPoolExecutor() as executor:
                results = executor.map(load_3d_mrc, data_paths)
                raw_data = list(results)

        # Process each volume (average 3D volumes for 2D display)
        processed_particles: List[Optional[np.ndarray]] = []
        for dat in raw_data:
            if dat is None:
                processed_particles.append(None)
                continue
            try:
                proc_arr = dat
                # If 3D, average along the Z-axis (axis 0)
                if proc_arr.ndim == 3:
                    proc_arr = proc_arr.mean(axis=0)
                elif proc_arr.ndim != 2:
                     logger.warning(f"Skipping pseudo-subtomogram with unexpected dimensions: {dat.ndim}")
                     processed_particles.append(None)
                     continue

                proc_arr = normalize_particle(proc_arr)
                if gauss_sdev > 0.0:
                    proc_arr = blur(proc_arr, sigma=gauss_sdev)
                if apply_contrast:
                    proc_arr = clip(proc_arr, 1, 99)
                processed_particles.append(proc_arr)
            except Exception as e:
                 report_error(e, "Error processing pseudo-subtomogram volume.")
                 processed_particles.append(None)

        with images_col:
            display_particles(processed_particles, particle_size, title="RELION 4 Pseudo-subtomograms (Z-Averaged)")

        # Load and display CTF volumes if available
        if has_ctf_column and ctf_paths:
            raw_ctf_data: List[Optional[np.ndarray]] = []
            with ThreadPoolExecutor() as executor:
                results = executor.map(load_3d_mrc, ctf_paths)
                raw_ctf_data = list(results)

            processed_ctfs: List[Optional[np.ndarray]] = []
            for ctf_dat in raw_ctf_data:
                if ctf_dat is None:
                    processed_ctfs.append(None)
                    continue
                try:
                    proc_ctf = ctf_dat
                    if proc_ctf.ndim == 3:
                        proc_ctf = proc_ctf.mean(axis=0)
                    elif proc_ctf.ndim != 2:
                        logger.warning(f"Skipping CTF volume with unexpected dimensions: {ctf_dat.ndim}")
                        processed_ctfs.append(None)
                        continue

                    proc_ctf = normalize_particle(proc_ctf)
                    # Typically don't blur/contrast CTF, but apply blur if requested globally
                    if gauss_sdev > 0.0:
                         proc_ctf = blur(proc_ctf, sigma=gauss_sdev)
                    processed_ctfs.append(proc_ctf)
                except Exception as e:
                    report_error(e, "Error processing CTF volume.")
                    processed_ctfs.append(None)

            with images_col:
                display_particles(processed_ctfs, particle_size, title="CTF Volumes (Z-Averaged)")

    except KeyError as e:
        report_error(e, f"Missing expected column for RELION 4 processing: {e}")
        images_col.warning(f"Missing required column in star file: {e}")
    except Exception as e:
        report_error(e, "Error during RELION 4 pseudo-subtomogram processing.")
        images_col.error("An error occurred displaying RELION 4 pseudo-subtomograms.")


def plot_relion5_direct2d(
    base_folder: str,
    star_df: pd.DataFrame,
    selection_indices: np.ndarray,
    particle_size: int,
    apply_contrast: bool,
    gauss_sdev: float,
    controls_col: DeltaGenerator,
    images_col: DeltaGenerator
):
    """Loads and displays RELION 5 direct 2D stacks, potentially with tilt selection."""
    try:
        logger.info("Processing RELION 5 direct 2D stacks.")
        # RELION 5 might use _rlnImageStackName or fallback to _rlnImageName
        image_col_name = RLN_IMAGE_STACK_NAME if RLN_IMAGE_STACK_NAME in star_df.columns else RLN_IMAGE_NAME

        if image_col_name not in star_df.columns:
            images_col.warning(f"Missing column '{image_col_name}'. Cannot display RELION 5 particles.")
            return

        selected_entries = star_df.iloc[selection_indices]
        image_names = selected_entries[image_col_name].values.tolist()
        data_paths = [os.path.join(base_folder, name) for name in image_names if pd.notna(name)]

        if not data_paths:
            images_col.info("No valid image paths found for selected RELION 5 particles.")
            return

        # Load all volumes concurrently (can be 2D or 3D)
        raw_volumes: List[Optional[np.ndarray]] = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(load_3d_mrc, data_paths)
            raw_volumes = list(results)

        # Determine the maximum number of tilts present across all loaded volumes
        n_tilts_per_volume: List[int] = []
        max_tilts = 0
        for vol in raw_volumes:
            if vol is None:
                n_tilts_per_volume.append(0) # Mark as 0 tilts if loading failed
            elif vol.ndim == 3:
                num_tilts = vol.shape[0]
                n_tilts_per_volume.append(num_tilts)
                max_tilts = max(max_tilts, num_tilts)
            elif vol.ndim == 2:
                n_tilts_per_volume.append(1) # 2D image counts as 1 tilt
                max_tilts = max(max_tilts, 1)
            else:
                 n_tilts_per_volume.append(0) # Invalid dimension
                 logger.warning(f"Volume has unexpected dimension {vol.ndim}, treating as 0 tilts.")


        # If max_tilts > 1, provide a slider to select tilt index
        tilt_index = 0 # Default to first tilt
        if max_tilts > 1:
             try:
                 tilt_index = controls_col.slider(
                     "Select Tilt Index to Display", min_value=0, max_value=max_tilts - 1, value=0, step=1,
                     key=f"tilt_slider_{base_folder}_{image_col_name}"
                 )
             except Exception as e:
                 # Handle potential errors if slider bounds are invalid somehow
                 report_error(e, f"Error creating tilt slider (max_tilts={max_tilts})")
                 controls_col.warning("Could not create tilt selection slider.")
                 tilt_index = 0 # Fallback
        elif max_tilts == 1:
             controls_col.caption("Data appears to be 2D (single tilt).")
        else: # max_tilts == 0 or less
             images_col.info("No valid tilt data found in loaded volumes.")
             return


        # Process the selected tilt for each volume
        processed_images: List[Optional[np.ndarray]] = []
        for vol, n_tilts in zip(raw_volumes, n_tilts_per_volume):
            if vol is None or n_tilts == 0:
                processed_images.append(None)
                continue

            try:
                if vol.ndim == 2:
                    # Always use the image if it's 2D
                    arr_2d = vol
                elif vol.ndim == 3:
                    # For 3D, select the slice based on slider, clamping if needed
                    current_tilt_index = min(tilt_index, n_tilts - 1) # Clamp index to valid range for *this* volume
                    arr_2d = vol[current_tilt_index]
                else:
                    # Should not happen based on earlier checks, but defensively skip
                     processed_images.append(None)
                     continue

                # Process the extracted 2D slice
                proc_arr = normalize_particle(arr_2d)
                if gauss_sdev > 0.0:
                    proc_arr = blur(proc_arr, sigma=gauss_sdev)
                if apply_contrast:
                    proc_arr = clip(proc_arr, 1, 99)
                processed_images.append(proc_arr)

            except IndexError as e:
                 report_error(e, f"Index error accessing tilt {tilt_index} for volume with {n_tilts} tilts.")
                 processed_images.append(None)
            except Exception as e:
                 report_error(e, "Error processing RELION 5 volume slice.")
                 processed_images.append(None)


        with images_col:
            display_particles(processed_images, particle_size, title=f"RELION 5 Particles (Tilt Index {tilt_index})")

    except KeyError as e:
        report_error(e, f"Missing expected column for RELION 5 processing: {e}")
        images_col.warning(f"Missing required column in star file: {e}")
    except Exception as e:
        report_error(e, "Error during RELION 5 particle processing.")
        images_col.error("An error occurred displaying RELION 5 particles.")


@st.fragment
def plot_pseudosubtomo(
    base_folder: str,
    node_files: List[str]
) -> None:
    """
    Main function to handle subtomogram/pseudo-subtomogram display based on RELION version.

    Args:
        base_folder (str): Path to the job folder.
        node_files (List[str]): List of files in the job folder.
    """
    try:
        controls_col, images_col = st.columns([1, 4])
        particles_star_name: Optional[str] = None
        for nf in node_files:
            if PARTICLES_STAR in nf:
                particles_star_name = nf
                break

        if not particles_star_name:
            controls_col.warning(f"No '{PARTICLES_STAR}' file found. Cannot display subtomograms.")
            return

        particles_path = os.path.join(base_folder, particles_star_name)
        particles_df: Optional[pd.DataFrame] = None

        star_data = parse_star(particles_path)
        if not star_data or PARTICLES_KEY not in star_data:
             controls_col.warning(f"No '{PARTICLES_KEY}' data found in {particles_star_name}.")
             return
        particles_df = star_data[PARTICLES_KEY]
        if particles_df.empty:
            controls_col.info(f"'{PARTICLES_KEY}' data in {particles_star_name} is empty.")
            return

        logger.debug(f"Loaded star file for tomo: {particles_path}, shape={particles_df.shape}")
        relion_version = detect_relion_version(particles_df)

        controls_col.markdown(f"**Detected RELION version:** {relion_version}")
        controls_col.markdown(f"**Extracted particles:** {particles_df.shape[0]}")

        # --- Common Tomo Controls ---
        n_particles_input = controls_col.number_input(
            "Number of Particles to Show", min_value=1, value=DEFAULT_N_PARTICLES, key=f"n_part_tomo_{base_folder}"
        )
        particle_size = DEFAULT_PARTICLE_SIZE # Fixed size
        apply_contrast = controls_col.checkbox(
            "Apply Contrast Clip (1-99%)?", value=True, key=f"contrast_tomo_{base_folder}" # Default True for tomo often helpful
        )
        gauss_sdev = controls_col.slider(
            "Apply Gaussian Blur Sigma", min_value=0.0, value=DEFAULT_BLUR_SIGMA, max_value=5.0, step=0.1, key=f"blur_tomo_{base_folder}"
        )
        # --- End Controls ---

        n_particles_to_display = min(n_particles_input, particles_df.shape[0])
        if n_particles_to_display <= 0:
            images_col.info("Select 1 or more particles to display.")
            return

        # Get random selection using session state
        session_key = get_unique_key("selected_tomo_particles", base_folder, particles_path, n_particles_to_display)
        if session_key not in st.session_state:
             try:
                selection_indices = np.random.choice(particles_df.shape[0], size=n_particles_to_display, replace=False)
                st.session_state[session_key] = selection_indices
             except ValueError as e:
                report_error(e, f"Error generating random indices for tomo (shape={particles_df.shape[0]}, n={n_particles_to_display})")
                controls_col.warning("Could not select random particles.")
                return
        else:
             # Ensure saved indices are valid if n_particles changed
            if len(st.session_state[session_key]) != n_particles_to_display:
                try:
                    selection_indices = np.random.choice(particles_df.shape[0], size=n_particles_to_display, replace=False)
                    st.session_state[session_key] = selection_indices
                except ValueError as e:
                    report_error(e, f"Error generating random indices for tomo (shape={particles_df.shape[0]}, n={n_particles_to_display})")
                    controls_col.warning("Could not select random particles.")
                    return
            else:
                selection_indices = st.session_state[session_key]


        # Branch based on detected RELION version
        if relion_version == 4:
            plot_relion4_pseudosubtomos(
                base_folder, particles_df, selection_indices, particle_size, apply_contrast, gauss_sdev, images_col
            )
        else: # RELION 5
            plot_relion5_direct2d(
                base_folder, particles_df, selection_indices, particle_size, apply_contrast, gauss_sdev, controls_col, images_col
            )

    except FileNotFoundError:
        st.error(f"Particles star file not found.")
    except KeyError as e:
         report_error(e, f"Missing expected key '{PARTICLES_KEY}' in star file")
         st.error(f"Could not find '{PARTICLES_KEY}' data.")
    except Exception as e:
         report_error(e, f"Error processing subtomograms")
         st.error("An unexpected error occurred while processing subtomograms.")


@st.fragment
def display_spa_interface(star_file_path: str, base_folder: str) -> None:
    try:
        controls_col, right_col = st.columns([1, 4])
        # Display random particles and get the DataFrame back
        particles_df = show_random_particles(star_file_path, base_folder, controls_col, right_col)

        if particles_df is not None and not particles_df.empty:
            # --- Optional SPA Plots ---
            if controls_col.checkbox("Show Detailed Stats Plots?", key=f"show_plots_{base_folder}"):
                with right_col:
                    st.markdown("---") # Separator
                    st.subheader("Additional Statistics")
                    plot_cols = particles_df.columns
                    # FOM Histogram
                    if RLN_AUTOPICK_FOM in plot_cols:
                         fom_data = pd.to_numeric(particles_df[RLN_AUTOPICK_FOM], errors='coerce').dropna()
                         if not fom_data.empty:
                             chart = altair_histogram_fom(fom_data.values)
                             if chart: st.altair_chart(chart, use_container_width=True)
                             else: st.caption("Could not generate FOM histogram.")
                         else: st.caption(f"No valid numeric data found for '{RLN_AUTOPICK_FOM}'.")
                    else:
                         st.caption(f"Column '{RLN_AUTOPICK_FOM}' not found for histogram.")

                    # Particles per Micrograph Histogram & Line Plot
                    if RLN_MICROGRAPH_NAME in plot_cols:
                        try:
                            counts = particles_df.groupby(RLN_MICROGRAPH_NAME).size()
                            ppm_values = counts.values.tolist()
                            micrograph_indices = list(range(len(counts)))

                            if ppm_values:
                                chart_hist = altair_histogram_particles_per_mic(ppm_values)
                                if chart_hist: st.altair_chart(chart_hist, use_container_width=True)
                                else: st.caption("Could not generate particles/micrograph histogram.")

                                chart_line = altair_line_graph(micrograph_indices, ppm_values)
                                if chart_line: st.altair_chart(chart_line, use_container_width=True)
                                else: st.caption("Could not generate particles/micrograph line plot.")
                            else:
                                st.caption("No particle counts per micrograph calculated.")
                        except Exception as e:
                             report_error(e, "Error calculating or plotting particles per micrograph.")
                             st.warning("Could not process particles per micrograph plots.")
                    else:
                         st.caption(f"Column '{RLN_MICROGRAPH_NAME}' not found for particles/micrograph plots.")

                    # Defocus vs FOM Heatmap
                    if RLN_DEFOCUS_U in plot_cols and RLN_AUTOPICK_FOM in plot_cols:
                        def_u = pd.to_numeric(particles_df[RLN_DEFOCUS_U], errors='coerce')
                        fom = pd.to_numeric(particles_df[RLN_AUTOPICK_FOM], errors='coerce')
                        valid_idx = def_u.notna() & fom.notna()
                        if valid_idx.any():
                            chart = altair_heatmap_defocus_fom(def_u[valid_idx].values, fom[valid_idx].values)
                            if chart: st.altair_chart(chart, use_container_width=True)
                            else: st.caption("Could not generate Defocus vs FOM heatmap.")
                        else:
                            st.caption(f"No valid paired data for '{RLN_DEFOCUS_U}' and '{RLN_AUTOPICK_FOM}'.")
                    else:
                         st.caption(f"Missing '{RLN_DEFOCUS_U}' or '{RLN_AUTOPICK_FOM}' for heatmap.")

            # --- Optional Interactive Plot ---
            if controls_col.checkbox("Show Interactive Scatter Plot?", key=f"show_interactive_{base_folder}"):
                 with right_col:
                      try:
                          interactive_scatter_plot(data_source=star_file_path)
                      except Exception as e:
                          report_error(e, f"Error calling interactive_scatter_plot for {star_file_path}")
                          st.warning("Could not display the interactive scatter plot.")

        elif particles_df is None:
             # Error handled within show_random_particles
             pass
        else: # Empty DataFrame returned
             right_col.info("No particle data loaded to generate plots.")
    except Exception as e:
        report_error(e, "Error in SPA interface display")
        st.error("Error displaying SPA interface")

# -----------------------------------------
# Main Entry Point
# -----------------------------------------
def process_extract(base_folder: str, node_files: List[str]) -> None:
    """
    Main function to process and display results from a RELION Extract job.

    Determines if the job is SPA or Tomography and calls the appropriate
    visualization functions. Uses a two-column layout for controls and images.

    Args:
        base_folder (str): Path to the RELION job folder.
        node_files (List[str]): List of relevant files found in the folder.
    """
    logger.info(f"Processing Extract job in folder: {base_folder}")

    try:
        # Find the primary particles star file
        particles_star_file: Optional[str] = None
        if any(PARTICLES_STAR in f for f in node_files):
            particles_star_file = next((f for f in node_files if PARTICLES_STAR in f), None)
        elif any(PARTICLES_SUBTRACTED_STAR in f for f in node_files):
             particles_star_file = next((f for f in node_files if PARTICLES_SUBTRACTED_STAR in f), None)

        if not particles_star_file:
            st.warning(f"**No '{PARTICLES_STAR}' or '{PARTICLES_SUBTRACTED_STAR}' file found.**")
            logger.warning(f"No primary particle star file found in {base_folder}")
            return

        star_file_path = os.path.join(base_folder, particles_star_file)

        # Detect if it's a tomography job (heuristic: presence of optimisation_set.star)
        is_tomo = any(OPTIMISATION_SET_STAR in nf for nf in node_files)

        if is_tomo:
            logger.info("Tomography workflow detected.")
            plot_pseudosubtomo(base_folder, node_files)
        else:
            logger.info("Single Particle Analysis workflow detected.")
            display_spa_interface(star_file_path, base_folder)

    except Exception as e:
        # Catch-all for unexpected errors in the main logic
        report_error(e, f"An unexpected error occurred in process_extract for folder {base_folder}")
        st.error("An unexpected error occurred. Please check the application logs.")
        logger.error(f"Fatal error in process_extract for {base_folder}: {e.__class__.__name__}")

    logger.info(f"Finished processing Extract job in folder: {base_folder}")