# dynamight_job.py

import os
import glob
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mrcfile
import starfile
from sklearn.cluster import KMeans

logger = logging.getLogger("main_app")


# --- Dynamight Imports ---
try:
    from dynamight.data.handlers.particle_image_preprocessor import ParticleImagePreprocessor
    from dynamight.data.dataloaders.relion import RelionDataset
    from dynamight.evaluation.utils import (
        compute_dimensionality_reduction,
        compute_latent_space_and_colors,
        compute_max_deviation,
    )
    from dynamight.utils.utils_new import compute_threshold

    from dynamight.models.encoder import HetEncoder
    from dynamight.models.decoder import PointDecoder
    from dynamight.models.poses import PoseModel

    # List of classes to allowlist for torch.load.
    DYNAMIGHT_NEEDED_CLASSES = [HetEncoder, PointDecoder, PoseModel]
    DYNAMIGHT_AVAILABLE = True

except ImportError as e:
    st.error(f"Dynamight library or specific model class import error: {e}. Please install/check Dynamight.")
    DYNAMIGHT_AVAILABLE = False
    DYNAMIGHT_NEEDED_CLASSES = []  # Cannot load safely without classes

    logger.error(f"Dynamight import error: {e}")

def assign_device(model: Any, device: str) -> None:
    """
    Sets the device attribute for a model and its submodules if available.
    
    Args:
        model: The model (or submodel) to assign the device.
        device: The target device, e.g. "cpu" or "cuda:0".
    """
    if hasattr(model, "device"):
        model.device = device
    for attr in ["p2i", "projector", "image_smoother", "p2v"]:
        component = getattr(model, attr, None)
        if component is not None and hasattr(component, "device"):
            component.device = device


def plot_volume_slices(volume: Optional[np.ndarray], title: str, ang_pix: float = 1.0, colormap: str = "gray"):
    """
    Generates a Plotly figure with three orthogonal slices of a 3D volume.

    This function checks for valid volume data and creates sliders for the Z, Y, and X indices.
    """
    if volume is None or not isinstance(volume, np.ndarray) or volume.ndim != 3:
        st.warning(f"Volume data for '{title}' is missing or invalid.")
        return None

    st.write(f"**{title} Slices:**")
    col1, col2, col3 = st.columns(3)
    max_z, max_y, max_x = volume.shape[0] - 1, volume.shape[1] - 1, volume.shape[2] - 1
    step = 1
    val_z = int(max_z / 2) if max_z > 0 else 0
    val_y = int(max_y / 2) if max_y > 0 else 0
    val_x = int(max_x / 2) if max_x > 0 else 0

    with col1:
        z_slice_idx = st.slider(f"Z index", 0, max_z, val_z, step, key=f"z_{title}", disabled=(max_z == 0))
    with col2:
        y_slice_idx = st.slider(f"Y index", 0, max_y, val_y, step, key=f"y_{title}", disabled=(max_y == 0))
    with col3:
        x_slice_idx = st.slider(f"X index", 0, max_x, val_x, step, key=f"x_{title}", disabled=(max_x == 0))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["XY Plane (Z slice)", "XZ Plane (Y slice)", "YZ Plane (X slice)"],
    )

    if volume.shape[0] > 0:
        fig.add_trace(
            go.Heatmap(
                z=volume[z_slice_idx, :, :],
                colorscale=colormap,
                showscale=False,
                name="XY",
                x0=0,
                dx=ang_pix,
                y0=0,
                dy=ang_pix,
            ),
            row=1,
            col=1,
        )
    if volume.shape[1] > 0:
        fig.add_trace(
            go.Heatmap(
                z=volume[:, y_slice_idx, :],
                colorscale=colormap,
                showscale=False,
                name="XZ",
                x0=0,
                dx=ang_pix,
                y0=0,
                dy=ang_pix,
            ),
            row=1,
            col=2,
        )
    if volume.shape[2] > 0:
        fig.add_trace(
            go.Heatmap(
                z=volume[:, :, x_slice_idx],
                colorscale=colormap,
                showscale=False,
                name="YZ",
                x0=0,
                dx=ang_pix,
                y0=0,
                dy=ang_pix,
            ),
            row=1,
            col=3,
        )

    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=350)
    for i in range(1, 4):
        fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=1, col=i, title_text="Pixels")
        fig.update_xaxes(showticklabels=True, ticks="outside", title_text="Pixels", row=1, col=i)
        fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=i)
        fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=i)
    return fig


#@st.cache_data(show_spinner=True, allow_output_mutation=True, suppress_st_warning=True)
def _compute_dynamight_data(
    output_directory_str: str,
    refinement_star_file_str: Optional[str],
    checkpoint_file_str: str,
    half_set: int,
    mask_file_str: Optional[str],
    particle_diameter: Optional[float],
    soft_edge_width: float,
    batch_size: int,
    gpu_id: Optional[int],
    preload_images: bool,
    n_workers: int,
    dimensionality_reduction_method: str,
    inverse_deformation_str: Optional[str],
    atomic_model_str: Optional[str],
    reduce_by_deformation: bool,
    cluster: bool,
) -> Dict[str, Any]:
    """
    Performs the core data loading and computation.
    
    Loads checkpoint and models, sets up the dataset, computes latent space and clusters,
    performs dimensionality reduction, generates volumes and prepares metadata.
    
    Args:
        output_directory_str: The job output directory.
        refinement_star_file_str: Path to the refinement star file.
        checkpoint_file_str: Path to the checkpoint file.
        half_set: Indicates which half set to use.
        mask_file_str: Optional mask file path.
        particle_diameter: Particle diameter value.
        soft_edge_width: Soft edge width value.
        batch_size: Batch size for DataLoader.
        gpu_id: GPU id to use. Use None or -1 for CPU.
        preload_images: Whether to preload images.
        n_workers: Number of worker processes.
        dimensionality_reduction_method: Method for dimensionality reduction.
        inverse_deformation_str: Optional inverse deformation file path.
        atomic_model_str: Optional atomic model file path.
        reduce_by_deformation: Whether to reduce based on deformation.
        cluster: Whether to perform clustering.
        
    Returns:
        A dictionary with data ready for visualization.
    """
    if not DYNAMIGHT_AVAILABLE:
        raise ImportError("Dynamight library not available or failed to import.")
    if not DYNAMIGHT_NEEDED_CLASSES:
        raise ImportError("Cannot load checkpoint safely: Required Dynamight model classes not imported.")

    # --- Setup Paths ---
    output_directory = Path(output_directory_str)
    refinement_star_file = Path(refinement_star_file_str) if refinement_star_file_str else None
    checkpoint_file = Path(checkpoint_file_str)
    mask_file = Path(mask_file_str) if mask_file_str else None
    atomic_model = Path(atomic_model_str) if atomic_model_str else None
    inverse_deformation = Path(inverse_deformation_str) if inverse_deformation_str else None

    logger.info(f"Starting data processing: Checkpoint {checkpoint_file}, Half Set {half_set}")

    # --- Device Setup ---
    if gpu_id is None or gpu_id == -1:
        device = "cpu"
        logger.info("Using CPU device.")
    else:
        if torch.cuda.is_available():
            device = f"cuda:{gpu_id}"
            logger.info(f"Using CUDA device: {device}")
        else:
            device = "cpu"
            logger.warning(f"GPU {gpu_id} requested but CUDA not available. Falling back to CPU.")

    # --- Load Checkpoint Safely ---
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    logger.info(f"Loading checkpoint: {checkpoint_file}")
    try:
        with torch.serialization.safe_globals(DYNAMIGHT_NEEDED_CLASSES):
            cp = torch.load(checkpoint_file, map_location="cpu")
        logger.info("Checkpoint loaded successfully using safe_globals.")
    except (RuntimeError, ValueError, KeyError) as e:
        if "weights_only set to False" in str(e) or "was not an allowed global" in str(e):
            st.error(f"Checkpoint loading failed with safe settings. Error: {e}")
            st.warning(f"Missing classes in allowlist: {DYNAMIGHT_NEEDED_CLASSES}. Consider using weights_only=False (INSECURE) if you TRUST this checkpoint.")
            raise RuntimeError(f"Failed safe checkpoint loading. Needed classes: {DYNAMIGHT_NEEDED_CLASSES}") from e
        else:
            logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_file}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading checkpoint {checkpoint_file}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error loading checkpoint {checkpoint_file}") from e

    # --- Determine Refinement Star File ---
    if refinement_star_file is None:
        cp_ref_dir = cp.get("refinement_directory")
        if cp_ref_dir:
            potential_ref_dir = checkpoint_file.parent.parent.parent / cp_ref_dir
            refinement_star_file = potential_ref_dir
            if refinement_star_file.is_dir():
                potential_star = refinement_star_file / "run_data.star"
                if not potential_star.exists():
                    potential_star = refinement_star_file / "particles.star"
                if potential_star.exists():
                    refinement_star_file = potential_star
                    logger.info(f"Using star file found via checkpoint: {refinement_star_file}")
                else:
                    raise FileNotFoundError(f"Refinement dir '{refinement_star_file}' found via checkpoint, but no 'run_data.star' or 'particles.star' found.")
            elif not refinement_star_file.exists():
                raise FileNotFoundError(f"Refinement path '{refinement_star_file}' from checkpoint not found.")
        else:
            raise ValueError("Refinement star file must be provided or found via checkpoint.")
    if not refinement_star_file or not Path(refinement_star_file).exists():
        raise FileNotFoundError(f"Refinement star file not found or invalid: {refinement_star_file}")
    logger.info(f"Using refinement star file: {refinement_star_file}")

    # --- Load Models ---
    try:
        poses = cp["poses"]
        poses.load_state_dict(cp["poses_state_dict"])
        poses.to(device)
        if half_set == 0:
            encoder_h1 = cp["encoder_half1"].to(device)
            decoder_h1 = cp["decoder_half1"].to(device)
            encoder_h2 = cp["encoder_half2"].to(device)
            decoder_h2 = cp["decoder_half2"].to(device)

            encoder_h1.load_state_dict(cp["encoder_half1_state_dict"])
            decoder_h1.load_state_dict(cp["decoder_half1_state_dict"])
            encoder_h2.load_state_dict(cp["encoder_half2_state_dict"])
            decoder_h2.load_state_dict(cp["decoder_half2_state_dict"])
            for model in [encoder_h1, decoder_h1, encoder_h2, decoder_h2]:
                assign_device(model, device)
            logger.info("Loaded models for half_set=0 onto device.")
        else:
            encoder = cp[f"encoder_half{half_set}"].to(device)
            decoder = cp[f"decoder_half{half_set}"].to(device)
            encoder.load_state_dict(cp[f"encoder_half{half_set}_state_dict"])
            decoder.load_state_dict(cp[f"decoder_half{half_set}_state_dict"])
            for model in [encoder, decoder]:
                assign_device(model, device)
            logger.info(f"Loaded models for half_set={half_set} onto device.")
    except KeyError as e:
        raise KeyError(f"Checkpoint file {checkpoint_file} missing key: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading models from checkpoint: {e}")

    # --- Load Dataset ---
    logger.info("Loading Relion dataset...")
    try:
        relion_dataset = RelionDataset(
            path=refinement_star_file.resolve(),
            circular_mask_thickness=soft_edge_width,
            particle_diameter=particle_diameter,
        )
        dataset = relion_dataset.make_particle_dataset()
        diameter_ang = relion_dataset.particle_diameter
        box_size = relion_dataset.box_size
        ang_pix = relion_dataset.pixel_spacing_angstroms
        logger.info(f"Dataset loaded: {len(dataset)} particles. Box: {box_size}px, AngPix: {ang_pix} Å/px, Diameter: {diameter_ang} Å")
    except Exception as e:
        raise RuntimeError(f"Error initializing RelionDataset: {e}")

    # --- Determine Particle Indices and Create DataLoader ---
    logger.info("Determining particle indices and creating DataLoader...")
    if half_set == 1:
        indices = cp["indices_half1"].cpu().numpy()
    elif half_set == 2:
        inds_half1 = cp["indices_half1"].cpu().numpy()
        indices = np.asarray(list(set(range(len(dataset))) - set(list(inds_half1))))
    elif half_set == 0:
        indices = cp.get("indices_val")
        indices = indices.cpu().numpy() if indices is not None else None
    else:
        raise ValueError(f"Invalid half_set value: {half_set}.")

    if indices is None and half_set == 0:
        raise ValueError("Checkpoint missing validation indices ('indices_val'). Cannot run with half_set=0.")
    if len(indices) == 0:
        raise ValueError(f"No particles selected for half_set {half_set}.")
    logger.info(f"Selected {len(indices)} particles for half set {half_set}.")
    dataset_half = torch.utils.data.Subset(dataset, indices)
    actual_n_workers = min(n_workers, os.cpu_count() // 2) if os.cpu_count() else n_workers
    dataloader_half = DataLoader(
        dataset=dataset_half,
        batch_size=batch_size,
        num_workers=actual_n_workers,
        shuffle=False,
        pin_memory=(device != "cpu"),
    )

    # --- Initialize Preprocessor ---
    try:
        batch = next(iter(dataloader_half))
    except StopIteration:
        raise ValueError("DataLoader is empty after subsetting.")
    data_preprocessor = ParticleImagePreprocessor()
    circular_mask_radius = diameter_ang / (2 * ang_pix) if diameter_ang and ang_pix else None
    circular_mask_thickness = soft_edge_width / ang_pix if soft_edge_width and ang_pix else None
    data_preprocessor.initialize_from_stack(
        stack=batch["image"],
        circular_mask_radius=circular_mask_radius,
        circular_mask_thickness=circular_mask_thickness,
    )
    data_preprocessor.to(device)

    # --- Compute Latent Space ---
    logger.info("Computing latent space embeddings...")
    with torch.no_grad():
        if half_set == 0:
            latent_dim = decoder_h1.latent_dim
            latent_space_h1, latent_colors_h1, point_colors_h1, feature_vec_h1 = compute_latent_space_and_colors(
                encoder_h1, decoder_h1, dataloader_half, poses, data_preprocessor, indices, reduce_by_deformation, device
            )
            latent_space_h2, latent_colors_h2, point_colors_h2, feature_vec_h2 = compute_latent_space_and_colors(
                encoder_h2, decoder_h2, dataloader_half, poses, data_preprocessor, indices, reduce_by_deformation, device
            )
        else:
            latent_dim = decoder.latent_dim
            latent_space, latent_colors, point_colors, feature_vec = compute_latent_space_and_colors(
                encoder, decoder, dataloader_half, poses, data_preprocessor, indices, reduce_by_deformation, device
            )

    # --- Dimensionality Reduction ---
    embedded_latent_space_np = None
    embedded_latent_space_h1_np = None
    embedded_latent_space_h2_np = None
    if half_set == 0:
        target_space_h1 = feature_vec_h1 if reduce_by_deformation else latent_space_h1
        target_space_h2 = feature_vec_h2 if reduce_by_deformation else latent_space_h2
        if target_space_h1.shape[1] > 3:
            logger.info(f"Computing dimensionality reduction ({dimensionality_reduction_method})...")
            embedded_latent_space_h1_np = compute_dimensionality_reduction(target_space_h1, dimensionality_reduction_method)
            embedded_latent_space_h2_np = compute_dimensionality_reduction(target_space_h2, dimensionality_reduction_method)
            logger.info("Dimensionality reduction finished.")
        else:
            embedded_latent_space_h1_np = target_space_h1.cpu().numpy()
            embedded_latent_space_h2_np = target_space_h2.cpu().numpy()
        closest_idx_h1 = np.argmin(latent_colors_h1["amount"].cpu().numpy())
        latent_closest_np = embedded_latent_space_h1_np[closest_idx_h1]
        max_diff, diff_col = compute_max_deviation(latent_space_h1, latent_space_h2, decoder_h1, decoder_h2)
        latent_colors_h1["difference"] = diff_col
        latent_colors_h2["difference"] = diff_col
    else:
        target_space = feature_vec if not reduce_by_deformation else latent_space
        if target_space.shape[1] > 3:
            logger.info(f"Computing dimensionality reduction ({dimensionality_reduction_method})...")
            embedded_latent_space_np = compute_dimensionality_reduction(target_space, dimensionality_reduction_method)
            logger.info("Dimensionality reduction finished.")
        else:
            embedded_latent_space_np = target_space.cpu().numpy()
        closest_idx = np.argmin(latent_colors["amount"].cpu().numpy())
        latent_closest_np = embedded_latent_space_np[closest_idx]

    # --- Clustering (Optional) ---
    if cluster:
        logger.info("Performing K-Means clustering...")
        cluster_input_tensor = feature_vec if half_set != 0 else feature_vec_h1
        cluster_input_np = cluster_input_tensor.cpu().numpy()
        kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(cluster_input_np)
        cluster_labels = kmeans.labels_
        if half_set != 0:
            latent_colors["cluster"] = torch.from_numpy(cluster_labels)
        else:
            latent_colors_h1["cluster"] = torch.from_numpy(cluster_labels)
        logger.info("Clustering finished.")

    # --- Generate Volumes ---
    logger.info("Generating volumes...")
    r = torch.zeros([2, 3], device=device)
    t = torch.zeros([2, 2], device=device)
    cons_volume_np = None
    V0_np = None
    cons_volume_h1_np = None
    V0_h1_np = None
    cons_volume_h2_np = None
    V0_h2_np = None
    with torch.no_grad():
        if half_set == 0:
            cons_volume_h1 = decoder_h1.generate_consensus_volume()[0].detach()
            cons_volume_h2 = decoder_h2.generate_consensus_volume()[0].detach()
            V0_h1 = decoder_h1.generate_volume(torch.zeros(2, latent_dim, device=device), r, t).float()[0].detach()
            V0_h2 = decoder_h2.generate_volume(torch.zeros(2, latent_dim, device=device), r, t).float()[0].detach()
            cons_volume_h1_np = cons_volume_h1.cpu().numpy()
            cons_volume_h2_np = cons_volume_h2.cpu().numpy()
            V0_h1_np = V0_h1.cpu().numpy()
            V0_h2_np = V0_h2.cpu().numpy()
        else:
            cons_volume = decoder.generate_consensus_volume()[0].detach()
            V0 = decoder.generate_volume(torch.zeros(2, latent_dim, device=device), r, t).float()[0].detach()
            cons_volume_np = cons_volume.cpu().numpy()
            V0_np = V0.cpu().numpy()
    logger.info("Volume generation finished.")

    # --- Prepare Particle DataFrame ---
    logger.info("Reading particle metadata...")
    try:
        star_data = starfile.read(refinement_star_file)
        if isinstance(star_data, dict):
            dataframe_all = star_data.get("particles", star_data.get("optics"))
        elif isinstance(star_data, pd.DataFrame):
            dataframe_all = star_data
        else:
            raise TypeError(f"Unexpected data type from starfile: {type(star_data)}")
        if dataframe_all is None:
            raise ValueError("Could not find 'particles' or 'optics' table in star file.")
        valid_indices = indices[indices < len(dataframe_all)]
        if len(valid_indices) < len(indices):
            logger.warning(f"Indices range exceeds dataframe length. Using {len(valid_indices)} valid particles.")
        dataframe_half = dataframe_all.iloc[valid_indices].copy()
        logger.info("Particle metadata prepared.")
    except Exception as e:
        raise RuntimeError(f"Error reading or subsetting star file {refinement_star_file}: {e}")

    # --- Convert latent colors to numpy arrays ---
    latent_colors_np = {}
    latent_colors_h1_np = {}
    latent_colors_h2_np = {}
    if half_set != 0:
        for k, v in latent_colors.items():
            latent_colors_np[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    else:
        for k, v in latent_colors_h1.items():
            latent_colors_h1_np[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        for k, v in latent_colors_h2.items():
            latent_colors_h2_np[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    # --- Package Results ---
    results = {
        "half_set": half_set,
        "z": embedded_latent_space_np if half_set != 0 else embedded_latent_space_h1_np,
        "latent_space": latent_space.cpu().numpy() if half_set != 0 else latent_space_h1.cpu().numpy(),
        "decoder": decoder if half_set != 0 else decoder_h1,
        "V0": V0_np if half_set != 0 else V0_h1_np,
        "cons_volume": cons_volume_np if half_set != 0 else cons_volume_h1_np,
        "latent_colors": latent_colors_np if half_set != 0 else latent_colors_h1_np,
        "latent_closest": latent_closest_np,
        "dataframe": dataframe_half,
        "ang_pix": ang_pix,
        "box_size": box_size,
        "latent_dim_original": latent_dim,
        "dimensionality_reduction_method": dimensionality_reduction_method,
        "V0_h2": V0_h2_np if half_set == 0 else None,
        "cons_volume_h2": cons_volume_h2_np if half_set == 0 else None,
    }
    logger.info("Data processing finished.")
    return results


def run_visualizer(
    z: Optional[np.ndarray],
    latent_space: Optional[np.ndarray],
    decoder: Any,
    V0: Optional[np.ndarray],
    cons_volume: Optional[np.ndarray],
    latent_colors: Dict[str, np.ndarray],
    latent_closest: Optional[np.ndarray],
    dataframe: Optional[pd.DataFrame],
    ang_pix: float,
    box_size: int,
    latent_dim_original: int,
    dimensionality_reduction_method: str,
    half_set: int,
    V0_h2: Optional[np.ndarray] = None,
    cons_volume_h2: Optional[np.ndarray] = None,
):
    """
    Streamlit app to visualize latent space and volumes.
    
    Displays a scatter plot of the latent space alongside reference and consensus volume slices.
    """
    st.title(f"Dynamight Latent Space Visualizer (Half Set {half_set})")

    if "current_color_choice" not in st.session_state or st.session_state.get("vis_half_set") != half_set:
        st.session_state.vis_half_set = half_set
        st.session_state.current_color_choice = list(latent_colors.keys())[0] if latent_colors else None

    with st.sidebar:
        st.header("Controls")
        available_colors = list(latent_colors.keys())
        if available_colors:
            selected_color_index = available_colors.index(st.session_state.current_color_choice) if st.session_state.current_color_choice in available_colors else 0
            st.session_state.current_color_choice = st.selectbox(
                "Color Latent Space By:", available_colors, index=selected_color_index, key="color_select"
            )
        else:
            st.warning("No coloring data provided.")
            st.session_state.current_color_choice = None

    st.subheader("Latent Space")
    scatter_placeholder = st.container()
    st.markdown("---")
    vol_col1, vol_col2 = st.columns(2)

    if z is None or not isinstance(z, np.ndarray) or z.ndim < 2:
        st.error("Invalid embedded latent space data ('z') provided.")
        return

    # Prepare scatter plot data using first two dimensions.
    plot_z = z[:, :2]
    plot_data = pd.DataFrame(plot_z, columns=["Dim1", "Dim2"])
    color_column_name = "color"
    current_colors_array = None

    if st.session_state.current_color_choice:
        current_colors_array = latent_colors.get(st.session_state.current_color_choice)
        if current_colors_array is not None and len(current_colors_array) == len(plot_data):
            plot_data[color_column_name] = current_colors_array
            if pd.api.types.is_integer_dtype(current_colors_array) or pd.api.types.is_object_dtype(current_colors_array):
                plot_data[color_column_name] = plot_data[color_column_name].astype(str)
        else:
            logger.warning(f"Color data '{st.session_state.current_color_choice}' mismatch or missing. Using default.")
            plot_data[color_column_name] = "default"
    else:
        plot_data[color_column_name] = "default"

    hover_indices = dataframe.index.values if dataframe is not None and len(dataframe) == len(z) else np.arange(len(z))
    plot_data["_hover_index"] = hover_indices

    fig_scatter = px.scatter(
        plot_data,
        x="Dim1",
        y="Dim2",
        color=color_column_name,
        color_discrete_sequence=px.colors.qualitative.Plotly if plot_data[color_column_name].dtype == "object" else None,
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={color_column_name: st.session_state.current_color_choice or ""},
        custom_data=["_hover_index"],
    )

    if latent_closest is not None and len(latent_closest) >= 2:
        fig_scatter.add_trace(
            go.Scatter(
                x=[latent_closest[0]],
                y=[latent_closest[1]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="x"),
                name="Reference Point",
            )
        )

    fig_scatter.update_layout(
        xaxis_title=f"Embedded Dim 1 ({dimensionality_reduction_method})",
        yaxis_title=f"Embedded Dim 2 ({dimensionality_reduction_method})",
        margin=dict(l=10, r=10, t=30, b=10),
        height=500,
    )
    fig_scatter.update_traces(hovertemplate="<b>Index:</b> %{customdata[0]}<br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<extra></extra>")

    with scatter_placeholder:
        st.plotly_chart(fig_scatter, use_container_width=True)

    with vol_col1:
        st.subheader("Reference Volume (V0)")
        fig_ref_vol = plot_volume_slices(V0, "Reference Volume", ang_pix)
        if fig_ref_vol:
            st.plotly_chart(fig_ref_vol, use_container_width=True)
        if V0_h2 is not None:
            st.subheader("Reference Volume (V0 H2)")
            fig_ref_vol_h2 = plot_volume_slices(V0_h2, "Reference Volume H2", ang_pix)
            if fig_ref_vol_h2:
                st.plotly_chart(fig_ref_vol_h2, use_container_width=True)

    with vol_col2:
        st.subheader("Consensus Volume")
        fig_cons_vol = plot_volume_slices(cons_volume, "Consensus Volume", ang_pix)
        if fig_cons_vol:
            st.plotly_chart(fig_cons_vol, use_container_width=True)
        if cons_volume_h2 is not None:
            st.subheader("Consensus Volume (H2)")
            fig_cons_vol_h2 = plot_volume_slices(cons_volume_h2, "Consensus Volume H2", ang_pix)
            if fig_cons_vol_h2:
                st.plotly_chart(fig_cons_vol_h2, use_container_width=True)

    st.markdown("---")
    st.subheader("Statistics & Info")
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("Particles Processed", len(z) if z is not None else "N/A")
        st.metric("Original Latent Dim", latent_dim_original)
        st.metric("Embedded Latent Dim", z.shape[1] if z is not None else "N/A")
    with stat_col2:
        st.write(f"**Box Size:** {box_size} px")
        st.write(f"**Pixel Spacing:** {ang_pix:.3f} Å/px")
        st.write(f"**Dim Reduction:** {dimensionality_reduction_method}")

    st.subheader("Particle Metadata (Subset)")
    if dataframe is not None and isinstance(dataframe, pd.DataFrame):
        st.dataframe(dataframe)
    else:
        st.info("Particle metadata not available.")


def plot_dynamight(folder: str, nodes: List[str]):
    """
    Main entry point to run the Dynamight Streamlit visualizer for a given job.
    
    Args:
        folder: The base directory containing the job folder.
        nodes: List of output node filenames from the RELION pipeline.
        **kwargs: Optional keyword arguments to override default computation parameters.
    """
    st.header("Dynamight Job Visualizer")
    try:
        if not nodes:
            raise ValueError("Node list is empty, cannot determine job path.")
        job_name = nodes[0].split('/')[:-2]
        logger.info(f"Job name {job_name}")

        job_path = os.path.join(folder, *job_name)
        logger.info(f"Attempting to plot job: {job_path}")

        checkpoint_files = glob.glob(os.path.join(job_path, '**', "checkpoint_final.pth"), recursive=True)
        if not checkpoint_files:
            raise FileNotFoundError(f"No 'checkpoint_final.pth' found within {job_path}")
        checkpoint_file = Path(checkpoint_files[0])
        logger.info(f"Using checkpoint: {checkpoint_file}")
    except Exception as e:
        st.error(f"Error determining job path or finding checkpoint: {e}")
        logger.error(f"Path/Checkpoint Error: {e}", exc_info=True)
        return

    half_set =1
    gpu_id = 0
    n_workers = 4
    reduce_by_def = False

    st.info(
        f"""
        **Running with:**
        - Job: `{job_path}`
        - Checkpoint: `{checkpoint_file}`
        - Half Set: `{half_set}`
        - GPU ID: `{gpu_id}`
        """
    )

    try:
        computed_data = _compute_dynamight_data(
            output_directory_str=job_path,
            refinement_star_file_str=None,
            checkpoint_file_str=checkpoint_file,
            half_set=half_set,
            mask_file_str=None,
            gpu_id=gpu_id,
            preload_images=True,
            n_workers=n_workers,
            dimensionality_reduction_method='PCA',
            reduce_by_deformation=reduce_by_def,
            
            particle_diameter = None,
            soft_edge_width=None,
            batch_size=100,
            inverse_deformation_str = None,
            atomic_model_str = None,
            cluster = False
            
        )

        if computed_data.get("error"):
            st.error(f"Data computation failed: {computed_data['error']}")
            logger.error(f"Data computation error: {computed_data['error']}")
            return

        run_visualizer(**computed_data)

    except FileNotFoundError as e:
        st.error(f"File not found during computation: {e}")
    except ValueError as e:
        st.error(f"Value error during computation: {e}")
    except RuntimeError as e:
        st.error(f"Runtime error (check logs/CUDA?): {e}")
    except ImportError as e:
        st.error(f"Import error during computation (Dynamight installed?): {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during computation: {e}")
        logger.error("Computation/Visualization error", exc_info=True)
