import os
import traceback
import logging

import streamlit as st
import json

from lib.image_utils import micrograph_viewer

from lib.utils import (
    parse_star,
    get_values_from_first_key,
)

logger = logging.getLogger("main_app")

def report_error(exc: Exception) -> None:
    """
    Report an error using the global error handler if available.
    Otherwise, log the error with full traceback.
    """
    error_info = traceback.format_exc()
    logger.error("An unexpected error occurred:\n%s", error_info)
    

def plot_tomographs(rln_folder: str, node_files: str) -> None:
    logger.info(f"Plotting tomographs... rln_folder= {rln_folder}, node_files= {node_files}")
    
    star_path = os.path.join(rln_folder, node_files)
    logger.debug(f"star_path: {star_path}")
    
    if not os.path.exists(star_path):
        st.error(f"Star file not found: {star_path}")
        return

    try:
        star = parse_star(star_path)
    except Exception as exc:
        report_error(exc)
        st.error("Failed to parse the main star file.")
        return

    if "global" not in star:
        st.error("The star file does not contain a 'global' section.")
        return

    logger.debug(f"Parsed star file: {star}")

    # Gather paths to tilt-series star files from the exclude job
    tomo_star_files = star["global"]["_rlnTomoTiltSeriesStarFile"]
    tomo_mrc_files = star["global"]["_rlnTomoReconstructedTomogramHalf1"]
    tomo_star_files_paths = [os.path.join(rln_folder, f) for f in tomo_star_files]

    # Parse each tilt-series star file
    dfs_tilt = []
    for path_tomo_star in tomo_star_files_paths:
        try:
            star_data = parse_star(path_tomo_star)
            if star_data:
                df_tilt = get_values_from_first_key(star_data)
                df_tilt["FileSource"] = os.path.basename(path_tomo_star)
                dfs_tilt.append(df_tilt)
        except Exception as exc:
            logger.error(f"Error parsing star file {path_tomo_star}: {exc}")
            report_error(exc)

    if not dfs_tilt:
        st.error("No tilt-series data found. Cannot plot tilt angles.")
        return
    
    if 'Denoise' in node_files:
        logger.info("Denoised tomograms detected.")
        
        if '_rlnTomoReconstructedTomogramDenoised' not in star["global"]:
            logger.info("Denoise training job detected")
            config_path = os.path.join(os.path.dirname(star_path), "external/training/train_data_config.json")
            if os.path.exists(config_path):
                config_json = json.load(open(config_path))
                st.write("**Denoising training job config:**")
                st.json(config_json)
                

        else:
            tomo_mrc_files = star["global"]["_rlnTomoReconstructedTomogramDenoised"]
            micrograph_viewer(rln_folder=rln_folder, image_files=tomo_mrc_files)
    else:
        micrograph_viewer(rln_folder=rln_folder, image_files=tomo_mrc_files)