# excludetilt_job.py

import os
import traceback
import logging
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.utils import (
    parse_star,
    get_values_from_first_key,
    get_note,
    extract_source_job,
    report_error,
)

logger = logging.getLogger("main_app")

def plot_exclude_tilt(rln_folder: str, node_files: str) -> None:
    """
    Loads a star file referencing multiple tomography star files. Plots the tilt angles
    (_rlnTomoNominalStageTiltAngle) for a user-selected tilt series as:
      1) Blue diameter lines from -r to +r.
      2) Blue points on the circumference for those angles.

    If a source job star file also exists (extracted from note.txt) and references
    the same tilt series, we identify the angles that are present in the source
    but missing from the selected dataset. We plot those missing angles in red
    (both diameter lines and points).
    """
    logger.info(f"Plotting exclude tilt data with Plotly... {rln_folder}, {node_files}")
    star_path = os.path.join(rln_folder, node_files)
    logger.debug(f"star_path: {star_path}")

    working_dir = os.path.dirname(star_path)
    note = get_note(os.path.join(working_dir, "note.txt"))
    logger.debug(f"Note: {note}")

    source_job = os.path.join(rln_folder, extract_source_job(note))
    logger.debug(f"Source job: {source_job}")

    # Try to parse the source job file, if it exists.
    dfs_source_tilt = []
    if os.path.exists(source_job):
        try:
            source_star = parse_star(source_job)
            if source_star is not None and "global" in source_star:
                tomo_star_files_source = source_star["global"]["_rlnTomoTiltSeriesStarFile"]
                tomo_star_files_source_paths = [
                    os.path.join(rln_folder, f) for f in tomo_star_files_source
                ]

                for path_tomo_source_star in tomo_star_files_source_paths:
                    try:
                        star_data_source = parse_star(path_tomo_source_star)
                        if star_data_source:
                            df_src = get_values_from_first_key(star_data_source)
                            df_src["FileSource"] = os.path.basename(path_tomo_source_star)
                            dfs_source_tilt.append(df_src)
                    except Exception as exc:
                        logger.error(f"Error parsing star file {path_tomo_source_star}: {exc}")
                        report_error(exc)
            else:
                logger.debug(f"Source star is None or missing 'global' key: {source_job}")
        except Exception as exc:
            logger.error(f"Error parsing source star file {source_job}: {exc}")
            report_error(exc)
    else:
        logger.error(f"Source job file not found: {source_job}")
        # If missing, we will just plot the main star data.

    ##############################
    # Now parse the main star file (the exclude job) for the actual tilt data
    ##############################
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

    # Gather paths to tilt-series star files from the exclude job
    tomo_star_files = star["global"]["_rlnTomoTiltSeriesStarFile"]
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

    ##############################
    # UI Layout: slider to pick which tilt series to display
    ##############################
    col1, col2, col3 = st.columns([1, 3, 3])

    with col1:
        selected_idx = st.slider(
            label="Select Tilt Series",
            min_value=0,
            max_value=len(dfs_tilt),
            value=1,
            step=1
        ) - 1

    selected_df = dfs_tilt[selected_idx].copy()
    tilt_col = "_rlnTomoNominalStageTiltAngle"
    if tilt_col not in selected_df.columns:
        st.error(f"Column '{tilt_col}' not found in tilt-series data.")
        return

    # Convert from degrees to radians
    selected_df["TiltAngleDeg"] = selected_df[tilt_col].astype(float)
    selected_df["TiltAngleRad"] = np.deg2rad(selected_df["TiltAngleDeg"])

    ##############################
    # Build lines/points for the main (blue) dataset
    ##############################
    lines_x_blue, lines_y_blue = [], []
    points_x_blue, points_y_blue = [], []
    text_labels_blue = []

    for _, row in selected_df.iterrows():
        angle_deg = row["TiltAngleDeg"]
        angle_rad = row["TiltAngleRad"]
        x_plus = np.cos(angle_rad)
        y_plus = np.sin(angle_rad)
        x_minus = -x_plus
        y_minus = -y_plus

        # lines
        lines_x_blue.extend([x_minus, x_plus, None])
        lines_y_blue.extend([y_minus, y_plus, None])

        # points at +r
        points_x_blue.append(x_plus)
        points_y_blue.append(y_plus)
        text_labels_blue.append(f"{row['FileSource']}<br>Tilt={angle_deg:.1f}°")

    ##############################
    # Check if we have a matching source dataset for this tilt series
    ##############################
    missing_df = pd.DataFrame()
    selected_file_source = selected_df["FileSource"].iloc[0]

    if len(dfs_source_tilt) > 0:
        # Find a source DF that has the same FileSource
        possible_matches = [
            df_s for df_s in dfs_source_tilt
            if (df_s["FileSource"].unique()[0] == selected_file_source)
        ]
        if len(possible_matches) == 1:
            source_df = possible_matches[0].copy()
            if tilt_col in source_df.columns:
                source_df["TiltAngleDeg"] = source_df[tilt_col].astype(float)
                # Identify angles that are in source_df but missing from selected_df
                missing_df = source_df[~source_df["TiltAngleDeg"].isin(selected_df["TiltAngleDeg"])]
                missing_df["TiltAngleRad"] = np.deg2rad(missing_df["TiltAngleDeg"])

    ##############################
    # Build lines/points for the missing (red) dataset
    ##############################
    lines_x_red, lines_y_red = [], []
    points_x_red, points_y_red = [], []
    text_labels_red = []

    if not missing_df.empty:
        for _, row in missing_df.iterrows():
            angle_deg = row["TiltAngleDeg"]
            angle_rad = row["TiltAngleRad"]
            x_plus = np.cos(angle_rad)
            y_plus = np.sin(angle_rad)
            x_minus = -x_plus
            y_minus = -y_plus

            lines_x_red.extend([x_minus, x_plus, None])
            lines_y_red.extend([y_minus, y_plus, None])

            points_x_red.append(x_plus)
            points_y_red.append(y_plus)
            text_labels_red.append(f"{row['FileSource']}<br>Tilt={angle_deg:.1f}°")

    ##############################
    # Plot with Plotly
    ##############################
    fig = go.Figure()

    # 1) Circle boundary
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-1, x1=1,
        y0=-1, y1=1,
        line=dict(color="lightgray")
    )

    # 2) Blue lines
    fig.add_trace(go.Scatter(
        x=lines_x_blue,
        y=lines_y_blue,
        mode="lines",
        line=dict(color="#007acc"),  # blue
        name="Kept tilt lines",      # LEGEND: updated
        hoverinfo="none",
        showlegend=False  # LEGEND: hidden
    ))

    # 3) Blue points
    fig.add_trace(go.Scatter(
        x=points_x_blue,
        y=points_y_blue,
        mode="markers",
        marker=dict(color="#007acc", size=8),
        text=text_labels_blue,
        hovertemplate="%{text}<extra></extra>",
        name="Kept tilt angles"      # LEGEND: updated
    ))

    # 4) Red lines for missing angles
    if not missing_df.empty:
        fig.add_trace(go.Scatter(
            x=lines_x_red,
            y=lines_y_red,
            mode="lines",
            line=dict(color="#cc3333", width=3),  # red
            hoverinfo="none",
            showlegend=False  # LEGEND: hidden
        ))
        # 5) Red points
        fig.add_trace(go.Scatter(
            x=points_x_red,
            y=points_y_red,
            mode="markers",
            marker=dict(color="#cc3333", size=15),
            text=text_labels_red,
            hovertemplate="%{text}<extra></extra>",
            name="Excluded tilt angles"  # LEGEND: updated
        ))

    # Make it square
    fig.update_xaxes(
        range=[-1.1, 1.1],
        scaleanchor="y",
        scaleratio=1,
        showgrid=False,
        zeroline=False,
        visible=False
    )
    fig.update_yaxes(
        range=[-1.1, 1.1],
        showgrid=False,
        zeroline=False,
        visible=False
    )

    fig.update_layout(
        title=f"Tilt Angles: {selected_file_source}",
        showlegend=True,  # LEGEND: now shown
        width=600,
        height=600
    )

    with col2:
        st.write("**Tilt Angles**")
        st.plotly_chart(fig, use_container_width=True)
    
    if "AlignTiltSeries" in node_files:
        logger.info("AlignTiltSeries job detected.")
        
        # plot the statistics from the tilt alignment: _rlnTomoXTilt, _rlnTomoYTilt _rlnTomoZRot _rlnTomoXShiftAngst _rlnTomoYShiftAngst using plotly
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=selected_df["TiltAngleDeg"],
            y=selected_df["_rlnTomoXTilt"],
            mode="markers",
            marker=dict(color="#007acc", size=10),
            name="_rlnTomoXTilt"
        ))
        fig2.add_trace(go.Scatter(
            x=selected_df["TiltAngleDeg"],
            y=selected_df["_rlnTomoYTilt"],
            mode="markers",
            marker=dict(color="#cc3333", size=10),
            name="_rlnTomoYTilt"
        ))
        fig2.add_trace(go.Scatter(
            x=selected_df["TiltAngleDeg"],
            y=selected_df["_rlnTomoZRot"],
            mode="markers",
            marker=dict(color="#ffd354", size=10),
            name="_rlnTomoZRot"
        ))
        fig2.add_trace(go.Scatter(
            x=selected_df["TiltAngleDeg"],
            y=selected_df["_rlnTomoXShiftAngst"],
            mode="markers",
            marker=dict(color="#b850c8", size=10),
            name="_rlnTomoXShiftAngst"
        ))
        fig2.add_trace(go.Scatter(
            x=selected_df["TiltAngleDeg"],
            y=selected_df["_rlnTomoYShiftAngst"],
            mode="markers",
            marker=dict(color="#45ab84", size=10),
            name="_rlnTomoYShiftAngst"
        ))
        fig2.update_layout(
            title=f"Tilt Angles: {selected_file_source}",
            xaxis_title="Tilt Angle (deg)",
            yaxis_title="Tilt Alignment Statistics",
            width=600,
            height=600
        )
        col3.write("**Tilt Alignment Statistics**")
        col3.plotly_chart(fig2, use_container_width=True)

    logger.info(f"{datetime.now()}: plot_exclude_tilt completed successfully with Plotly.")
