#mask_job.py

import os
import logging
import traceback
from datetime import datetime
from typing import List

import numpy as np
import mrcfile
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from skimage.transform import resize

# Import shared utilities.
from lib.utils import get_note, extract_source_job
from lib.image_utils import normalize

logger = logging.getLogger("mask_job")
logger.setLevel(logging.DEBUG)


def plot_volume_custom(volume: np.ndarray, threshold: float, max_size: int = 150,
                         colormap_override: str = None, opacity_override: float = None,
                         title: str = None) -> go.Figure:
    """
    Generate an isosurface plot using marching cubes with Plotly.
    
    Args:
        volume (np.ndarray): 3D volume data.
        threshold (float): Fraction (0 to 1) used to compute the actual threshold intensity.
        max_size (int): Maximum dimension size to which the volume is resized.
        colormap_override (str): If provided, use a single color (e.g. "rgb(100,149,237)")
                                 for all facets instead of the default gray.
        opacity_override (float): If provided, set marker opacity (0 to 1).
        title (str): Title of the figure.
    
    Returns:
        A Plotly Figure containing the isosurface.
    """
    try:
        # Create a writable copy.
        volume = np.array(volume, copy=True)
        volume.flags.writeable = True

        # Resize if needed.
        original_shape = volume.shape
        if np.any(np.array(original_shape) > max_size):
            resize_factor = max_size / np.max(volume.shape)
            new_shape = np.round(np.array(volume.shape) * resize_factor).astype(int)
            volume = resize(volume, new_shape, anti_aliasing=False)

        # Determine threshold intensity.
        min_val, max_val = np.min(volume), np.max(volume)
        actual_threshold = min_val + (max_val - min_val) * threshold

        # Compute mesh via marching cubes using mcubes.
        import mcubes
        verts, faces = mcubes.marching_cubes(volume, actual_threshold)

        # Use default gray color or override.
        if colormap_override is None:
            color_list = ["rgb(180,180,180)"]
        else:
            color_list = [colormap_override]

        fig_volume = ff.create_trisurf(
            x=verts[:, 2],
            y=verts[:, 1],
            z=verts[:, 0],
            simplices=faces,
            colormap=color_list,
            plot_edges=False,
            showbackground=True,
            show_colorbar=False,
        )

        # Modify the Mesh3d trace.
        mesh = fig_volume.data[0]
        mesh.update(
            flatshading=False,
            lighting=dict(
                ambient=0.25,
                diffuse=0.9,
                specular=0.2,
                roughness=0.7,
                fresnel=4,
            ),
            lightposition=dict(x=0, y=500, z=500),
        )
        if opacity_override is not None:
            mesh.opacity = opacity_override

        # Layout: black background.
        fig_volume.update_layout(
            title=title if title is not None else "3D Volume Isosurface",
            scene=dict(
                camera=dict(eye=dict(x=2, y=2, z=2)),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                bgcolor="black",
            ),
            paper_bgcolor="black",
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode=False,
        )

        return fig_volume

    except RuntimeError:
        return None
    except Exception as exc:
        logger.error(f"plot_volume_custom() error: {exc}")
        return None


def overlay_projections(source: np.ndarray, mask: np.ndarray, axis: int = 0, alpha: float = 0.7) -> np.ndarray:
    """
    Compute a maximum intensity projection (MIP) of the source and mask volumes along the given axis,
    then overlay the mask (colored in cornflower blue) on top of the source (in grayscale).

    Args:
        source (np.ndarray): 3D source volume.
        mask (np.ndarray): 3D mask volume.
        axis (int): Axis along which to compute the projection.
        alpha (float): Transparency factor for the mask.
    
    Returns:
        An RGB image (as a NumPy array) combining the source and mask projections.
    """
    # Compute MIPs.
    proj_source = source
    proj_mask = mask

    # Normalize projections.
    norm_source = normalize(proj_source)
    norm_mask = normalize(proj_mask)

    # Convert grayscale source to RGB.
    source_rgb = np.dstack([norm_source] * 3)
    # Define cornflower blue in normalized RGB.
    blue = np.array([100,149,237], dtype=np.float32)/255.0

    # Create a mask image with cornflower blue.
    mask_rgb = np.ones_like(source_rgb)
    mask_rgb[..., 0] = blue[0]
    mask_rgb[..., 1] = blue[1]
    mask_rgb[..., 2] = blue[2]

    # Blend images: raw map + mask overlay.
    mask_intensity = norm_mask[..., np.newaxis]
    overlay = source_rgb * (1 - alpha * mask_intensity) + (alpha * mask_rgb * mask_intensity)
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return overlay_uint8


def plot_mask(rln_folder: str, nodes: List[str]) -> None:
    """
    Plot the mask and the source volume together.
    
    The function:
      1) Retrieves the mask path and source job path.
      2) Caches the volumes in session state (job-specific) so that repeated
         threshold changes do not cause re-reading from disk.
      3) Provides two view modes:
         - 3D: Isosurface overlays using plot_volume_custom.
         - 2D: Maximum intensity projections overlaid (raw map in grayscale and mask in cornflower blue).
    """
    logger.info("Plotting mask for nodes: %s", nodes)

    # Determine job folder from nodes.
    job_folder = os.path.join(rln_folder, os.path.dirname(nodes[0]))
    # Use a unique job key for caching, e.g., based on job folder.
    job_key = f"mask_job_{job_folder}"
    if job_key not in st.session_state:
        st.session_state[job_key] = {}

    note_path = os.path.join(job_folder, "note.txt")
    note = get_note(note_path)
    source_job = extract_source_job(note)

    mask_path = os.path.join(rln_folder, nodes[0])
    source_job_path = os.path.join(rln_folder, source_job)

    # Cache volumes in session state (job-specific).
    cache = st.session_state[job_key]
    if "mask_volume" not in cache:
        try:
            mask_volume = mrcfile.mmap(mask_path, mode='r').data
            cache["mask_volume"] = mask_volume
        except Exception as exc:
            st.error(f"Error loading mask: {exc}")
            return
    else:
        mask_volume = cache["mask_volume"]

    if "source_volume" not in cache:
        try:
            source_volume = mrcfile.mmap(source_job_path, mode='r').data
            cache["source_volume"] = source_volume
        except Exception as exc:
            st.error(f"Error loading source volume: {exc}")
            return
    else:
        source_volume = cache["source_volume"]

    # Display mode selection: 3D or 2D.
    c1, c2 = st.columns([1, 3])
    view_mode = c1.radio("Display mode", ["3D", "2D"], horizontal=True)

    if view_mode == "3D":
        with c1:
            st.markdown("### 3D Rendering Controls")
            threshold_source = st.slider("Original Map Threshold Fraction", 0.0, 1.0, 0.5, 0.01)
            threshold_mask = st.slider("Mask Threshold Fraction", 0.0, 1.0, 0.5, 0.01)
        # Generate isosurface figures.
        fig_source = plot_volume_custom(source_volume, threshold_source, max_size=150,
                                        colormap_override=None, opacity_override=None,
                                        title="Original Map")
        cornflower_blue = "rgb(100,149,237)"
        fig_mask = plot_volume_custom(mask_volume, threshold_mask, max_size=150,
                                      colormap_override=cornflower_blue, opacity_override=0.5,
                                      title="Mask")
        # Overlay the two isosurface traces.
        fig_overlay = go.Figure()
        if fig_source and len(fig_source.data) > 0:
            fig_overlay.add_trace(fig_source.data[0])
        if fig_mask and len(fig_mask.data) > 0:
            fig_overlay.add_trace(fig_mask.data[0])
        fig_overlay.update_layout(
            title="Overlay: Original Map and Mask (3D)",
            scene=dict(
                camera=dict(eye=dict(x=2, y=2, z=2)),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                bgcolor="black",
            ),
            paper_bgcolor="black",
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode=False,
        )
        c2.plotly_chart(fig_overlay, use_container_width=True)
    else:
        c1.markdown("### 2D Projection Controls")
        with c1:
            projection_axis = st.radio("Projection axis", ['XY', 'XZ', 'YZ'], horizontal=True)
            # Map projection axis to axis index: use convention where 'XY' is projection along Z.
            projection_axis = {'XY': 0, 'XZ': 1, 'YZ': 2}[projection_axis]
            st.markdown("Adjust projection settings as needed.")
        # Compute maximum intensity projections for both volumes.
        proj_source = np.mean(source_volume, axis=projection_axis)
        proj_mask = np.max(mask_volume, axis=projection_axis)
        proj_source_norm = normalize(proj_source)
        proj_mask_norm = normalize(proj_mask)
        overlay_image = overlay_projections(proj_source_norm, proj_mask_norm, axis=projection_axis, alpha=0.7)
        c2.image(overlay_image, caption="Overlay of Original Map and Mask", width=400)
    
    logger.info(f"{datetime.now()}: plot_mask done.")
