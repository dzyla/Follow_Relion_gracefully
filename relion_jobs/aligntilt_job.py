# aligntilt_job.py

import os
import logging
from typing import List

import streamlit as st
import pandas as pd

from lib.utils import parse_star, get_values_from_first_key, report_error

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

        # TODO: Implement plotting logic here using dfs_tilt
    except Exception as exc:
        report_error(exc, "Unexpected error in plot_align_tilt_series")
        st.warning("An unexpected error occurred while plotting tilt series.")
