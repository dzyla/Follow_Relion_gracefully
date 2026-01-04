#ctffind_job.py

import os
import logging
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import mrcfile

from lib.utils import (
    parse_star,
    get_values_from_first_key,
    report_error,
    interactive_scatter_plot,
)
from lib.image_utils import clip, normalize

logger = logging.getLogger("main_app")

# =============================================================================
# Cached helpers
# =============================================================================
@st.cache_data(show_spinner=False)
def read_ctf_average_data(ctf_avrot_path: str) -> pd.DataFrame:
    if not os.path.exists(ctf_avrot_path):
        raise FileNotFoundError(ctf_avrot_path)
    df = pd.read_csv(
        ctf_avrot_path,
        skiprows=[0, 1, 2, 3, 4, 6, 10],
        sep=r"\s+",
        names=["Spatial_freq", "1D_Ave", "Fit", "Fit_CC"],
        engine="python",
    )
    df["Resolution"] = 1.0 / df["Spatial_freq"]
    return df.melt(
        id_vars=["Spatial_freq", "Resolution"],
        value_vars=["1D_Ave", "Fit", "Fit_CC"],
        var_name="Type",
        value_name="CTF",
    )


@st.cache_data(show_spinner=False)
def load_and_normalize_mrc(mrc_path: str) -> np.ndarray:
    if not os.path.exists(mrc_path):
        raise FileNotFoundError(mrc_path)
    with mrcfile.mmap(mrc_path, permissive=True) as mrc:
        data = np.squeeze(mrc.data)
    return normalize(clip(data, 1, 99))


@st.cache_data(show_spinner=True)
def load_star_data(folder: str, star_file: str) -> pd.DataFrame:
    """
    Return a dataframe with **one row per micrograph or tomogram**.

    For single-particle projects it takes the *micrographs* table.

    For tomography projects (STAR contains only the *global* table) it opens
    every `_rlnTomoTiltSeriesStarFile`, extracts the first table inside each
    of those STAR files, and concatenates them.

    The caller therefore always receives a dataframe with real, per-image
    statistics – never the empty *global* table.
    """
    path = os.path.join(folder, star_file)
    star = parse_star(path)

    # ── single-particle ────────────────────────────────────────────────────
    if "micrographs" in star:
        return star["micrographs"]

    # ── tomography: expand the global list ────────────────────────────────
    if "global" in star and "_rlnTomoTiltSeriesStarFile" in star["global"].columns:
        series_files = star["global"]["_rlnTomoTiltSeriesStarFile"].astype(str)
        dfs: list[pd.DataFrame] = []
        for rel_p in series_files:
            sub_path = os.path.join(folder, rel_p)
            try:
                df = get_values_from_first_key(parse_star(sub_path))
                if isinstance(df, pd.DataFrame):
                    dfs.append(df)
            except Exception as exc:
                report_error(exc, f"Parsing tilt-series STAR {sub_path}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)

    raise ValueError("Could not find a usable table (micrographs / global).")

# =============================================================================
# Plot functions  (unchanged except for minor guards)
# =============================================================================
@st.fragment
def plot_CTF_average(FOLDER: str, ctf_file_path: str) -> None:
    """
    Plot the 1D CTF fit per micrograph from a given text file using Altair.

    Parameters:
        FOLDER (str): Folder containing the CTF file.
        ctf_file_path (str): Path to the CTF file.

    Returns:
        None.
    """
    try:
        col1, col2 = st.columns([3, 1])
        power_spectrum_ave_rot_txt_paths = os.path.join(FOLDER, ctf_file_path.replace(".ctf:mrc", "_avrot.txt"))
        if not os.path.exists(power_spectrum_ave_rot_txt_paths):
            st.error("File not found: " + power_spectrum_ave_rot_txt_paths)
            return

        ave_rot = pd.read_csv(
            power_spectrum_ave_rot_txt_paths,
            skiprows=[0, 1, 2, 3, 4, 6, 10],
            header=None,
            sep=r'\s+'
        ).transpose()
        ave_rot.columns = ["Spatial_freq", "1D_Ave", "Fit", "Fit_CC"]

        ave_rot["Resolution"] = 1 / ave_rot["Spatial_freq"]
        ave_rot_melt = ave_rot.melt(
            id_vars=["Spatial_freq", "Resolution"],
            value_vars=["1D_Ave", "Fit", "Fit_CC"],
            var_name="Type",
            value_name="CTF"
        )

        # Use three distinct colors for these three lines
        chart_ctf = alt.Chart(ave_rot_melt).mark_line().encode(
            x=alt.X("Spatial_freq:Q", title="Spatial Frequency (1/Å)"),
            y=alt.Y("CTF:Q", title="CTF"),
            color=alt.Color("Type:N", scale=alt.Scale(range=["#FA8072", "#6FC381", "#6495ED"])),
            tooltip=["Spatial_freq", "Resolution", "CTF", "Type"]
        ).properties(
            title="CTF Fit per Micrograph",
            width=600,
            height=400
        )

        col1.altair_chart(chart_ctf, use_container_width=True)

        try:
            mrc_file = os.path.join(FOLDER, ctf_file_path.replace("_avrot.txt", ".ctf").replace(":mrc", ""))
            logger.debug(f"CTF file: {mrc_file}")

            mrc_data = np.squeeze(mrcfile.mmap(mrc_file).data)
            mrc_image = normalize(clip(mrc_data, 1, 99))
            col2.image(mrc_image, caption=f"CTF Image: {mrc_file}")
        except Exception as exc:
            report_error(exc)
            col2.error("Error loading CTF image.")

    except Exception as exc:
        report_error(exc)
        logger.error("Error in plot_CTF_average function.")
        st.error("Error plotting CTF average.")

def plot_ctf_stats(folder: str, star_file: str) -> None:
    # ── data load (tomography-aware) ───────────────────────────────────────
    try:
        star_df = load_star_data(folder, star_file)
    except Exception as exc:
        report_error(exc)
        st.error("Failed to load STAR data.")
        return

    try:
        star_df = star_df.apply(lambda c: pd.to_numeric(c, errors="ignore"))
        star_df["Index"] = np.arange(len(star_df))

        cols = {
            "defocus_v": "_rlnDefocusV",
            "max_res": "_rlnCtfMaxResolution",
        }
        numeric_cols = {
            label: col
            for label, col in cols.items()
            if col in star_df.columns and pd.api.types.is_numeric_dtype(star_df[col])
        }
        if not numeric_cols:
            st.error("No numeric CTF columns found.")
            return

        st.subheader("CTF Statistics per Image")

        # Defocus plot
        if "defocus_v" in numeric_cols:
            cv = numeric_cols["defocus_v"]
            c1, c2 = st.columns([4, 1])
            line = (
                alt.Chart(star_df)
                .mark_line(color="#6FC381")
                .encode(x="Index:Q", y=f"{cv}:Q", tooltip=["Index", f"{cv}:Q"])
                .properties(title="Defocus V", height=300)

            )
            kde = (
                alt.Chart(star_df)
                .transform_density(density=cv, as_=[cv, "density"])
                .mark_area(orient="horizontal")
                .encode(y=f"{cv}:Q", x="density:Q")
                .properties(height=300)
            )
            c1.altair_chart(line, use_container_width=True)
            c2.altair_chart(kde, use_container_width=True)

        # Max-resolution plot
        if "max_res" in numeric_cols:
            mr = numeric_cols["max_res"]
            c1, c2 = st.columns([4, 1])
            line = (
                alt.Chart(star_df)
                .mark_line(color="#6495ED")
                .encode(x="Index:Q", y=f"{mr}:Q", tooltip=["Index", f"{mr}:Q"])
                .properties(title="Max Resolution", height=300)

            )
            kde = (
                alt.Chart(star_df)
                .transform_density(density=mr, as_=[mr, "density"])
                .mark_area(orient="horizontal")
                .encode(y=f"{mr}:Q", x="density:Q")
                .properties(height=300)
            )
            c1.altair_chart(line, use_container_width=True)
            c2.altair_chart(kde, use_container_width=True)

        # 2-D histogram
        st.subheader("2-D Parameter Distribution")
        numeric_candidates = [
            c for c in star_df.columns if pd.api.types.is_numeric_dtype(star_df[c]) and star_df[c].nunique() > 1
        ]
        if len(numeric_candidates) >= 2:
            x_col = st.selectbox("X-axis", numeric_candidates, key="2dh_x")
            y_col = st.selectbox(
                "Y-axis", numeric_candidates, key="2dh_y", index=1 if x_col == numeric_candidates[0] else 0
            )
            if x_col != y_col:
                chart2d = (
                    alt.Chart(star_df)
                    .mark_rect()
                    .encode(
                        x=alt.X(f"{x_col}:Q", bin=alt.Bin(maxbins=50)),
                        y=alt.Y(f"{y_col}:Q", bin=alt.Bin(maxbins=50)),
                        color=alt.Color("count()", scale=alt.Scale(scheme="viridis")),
                    )
                    .properties(title=f"2-D Histogram: {x_col} vs {y_col}")

                )
                st.altair_chart(chart2d, use_container_width=True)
            else:
                st.warning("Select two different columns.")
        else:
            st.info("Not enough numeric columns for a 2-D histogram.")

        # 1-D CTF fit per image (only if column exists)
        if "_rlnCtfImage" in star_df.columns and star_df["_rlnCtfImage"].notna().any():
            st.subheader("1-D CTF Fit")
            img_series = star_df["_rlnCtfImage"].astype(str)
            idx = st.slider("Image index", 0, len(img_series) - 1, 0)
            plot_CTF_average(folder, img_series.iloc[idx])

        # optional interactive scatter
        if st.checkbox("Detailed scatter plot?", key="scatter"):
            interactive_scatter_plot(os.path.join(folder, star_file))

        logger.info(f"{datetime.now()}: plot_ctf_stats finished")

    except Exception as exc:
        report_error(exc)
        st.error(f"An error occurred in plot_ctf_stats: {exc}")
