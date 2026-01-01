# modelangelo_job.py

import os
import re
import traceback
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import io
from concurrent.futures import ThreadPoolExecutor

from scipy.ndimage import map_coordinates
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import mrcfile

# Import shared utilities - ensure these are accessible
from lib.utils import get_note, report_error
# Import the specific volume plotting function
from lib.image_utils import plot_volume

logger = logging.getLogger("main_app")

# --- Constants ---
AA_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# --- Helper Functions ---
def _three_to_one(residue_code: str) -> str:
    return AA_THREE_TO_ONE.get(str(residue_code).upper(), "?")


def _extract_v_parameter(note_content: str) -> Optional[str]:
    pattern = r"-v\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))"
    match = re.search(pattern, note_content)
    if match:
        return next((g for g in match.groups() if g is not None), None)
    logger.warning("Could not find '-v <volume_path>' in note content.")
    return None


def calculate_residue_mean_density(
    atoms_df: pd.DataFrame,
    volume_data: np.ndarray,
    voxel_size: float,
    mode: str = "CA",
    radius: float = 1.0,
    scaled: bool = False,
    origin_angstrom: Optional[np.ndarray] = None
) -> Dict[str, pd.Series]:
    """
    Calculates the average map density for each residue per chain using trilinear interpolation
    at atom coordinates.

    Parameters:
      atoms_df: DataFrame with columns "x", "y", "z", "chain", "residue", "seq_id".
      volume_data: 3D numpy array representing the cryo-EM map in pixel coordinates.
      voxel_size: The voxel size (in angstrom per pixel).
      mode: "CA" to use only C-alpha atoms (if input has others) or "all".
      radius: Unused in this interpolation method (kept for API compatibility).
      scaled: Boolean flag indicating whether the coordinates in atoms_df are already scaled to pixel units.
      origin_angstrom: The origin of the map in Angstroms (x, y, z). Defaults to (0,0,0) if None.

    Returns:
      A dictionary mapping chain identifiers to a Pandas Series. In each series the index is the residue identifier (seq_id)
      and the value is the mean map density calculated from the volume.
    """
    # Work on a copy of the atoms DataFrame
    df = atoms_df.copy()

    ox, oy, oz = 0.0, 0.0, 0.0
    if origin_angstrom is not None:
         ox, oy, oz = origin_angstrom

    # If the coordinates are not already in pixel space, scale them.
    # We assume 'x', 'y', 'z' cols in atoms_df correspond to physical space (Angstroms) if not scaled.
    if not scaled:
        df["x"] = (df["x"] - ox) / voxel_size
        df["y"] = (df["y"] - oy) / voxel_size
        df["z"] = (df["z"] - oz) / voxel_size
    
    # If scaled=True, we assume the caller has already handled origin subtraction and division by voxel size.
    # (Existing callers might need updates if they didn't handle origin)

    # Group the DataFrame by chain and residue
    grouped = df.groupby(["chain", "seq_id"])
    
    shape = volume_data.shape # (Z, Y, X) typically for mrcfile data

    def compute_density_for_group(name_group: Tuple[Tuple[Any, Any], pd.DataFrame]):
        """
        For one residue group, interpolate map density at atom positions.
        """
        name, group = name_group
        
        # Coordinates for interpolation. 
        # Standard MRC data in numpy is usually (Z, Y, X) order.
        # We map DataFrame columns x -> X, y -> Y, z -> Z.
        # So we request interpolation at indices [z, y, x].
        
        zs = group["z"].values
        ys = group["y"].values
        xs = group["x"].values
        
        # map_coordinates input coordinates must be shape (ndim, n_points)
        coords = np.stack([zs, ys, xs])
        
        # Trilinear interpolation (order=1)
        # mode='nearest' handles atoms slightly outside the box by clamping to edge
        vals = map_coordinates(volume_data, coords, order=1, mode='nearest')

        if vals.size > 0:
            return name, np.mean(vals)
        else:
            return name, np.nan

    # Process each residue group using threads for speed.
    results = {}
    # Use ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        for name, mean_val in executor.map(compute_density_for_group, grouped):
            chain, seq_id = name
            if chain not in results:
                results[chain] = {}
            results[chain][seq_id] = mean_val

    # Convert each chain's dictionary into a sorted Pandas Series.
    density_series = {}
    for chain, d in results.items():
        s = pd.Series(d)
        s.sort_index(inplace=True)
        density_series[chain] = s

    return density_series

def plot_residue_density_ui(
    atoms_df: pd.DataFrame,
    volume_data: np.ndarray,
    voxel_size: float,
    mode: str = "CA",
    radius: float = 1.0,
    scaled: bool = False,
    origin_angstrom: Optional[np.ndarray] = None
) -> None:
    """
    Calculates the mean map density per residue and displays a Streamlit user interface.
    """
    st.subheader("Per-Residue Density Metrics")

    col_opt, col_go = st.columns([3, 1])
    with col_opt:
        calc_density = st.checkbox("Calculate & Plot Per-Residue Density", key="calc_res_dens")

    if calc_density:
        with st.spinner("Calculating residue densities..."):
            density_data = calculate_residue_mean_density(
                atoms_df, volume_data, voxel_size, mode=mode, radius=radius, scaled=scaled, origin_angstrom=origin_angstrom
            )

        if not density_data:
            st.warning("No density data calculated.")
            return

        # Streamlit UI
        col1, col2 = st.columns([1, 4])
        chain_options = list(density_data.keys())
        with col1:
            selected_chain = st.selectbox("Select chain", chain_options, key="res_dens_chain_sel")

        if selected_chain:
            series = density_data[selected_chain]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines+markers",
                    name="Mean Density",
                    line=dict(color='cyan')
                )
            )

            fig.update_layout(
                title=f"Density Fit for Chain {selected_chain}",
                xaxis_title="Residue ID",
                yaxis_title="Mean Density",
                height=500,
                template="plotly_dark",
                hovermode="x unified"
            )
            col2.plotly_chart(fig, use_container_width=True)

            # Show stats
            if st.checkbox("Show Density Stats", key="show_dens_stats"):
                st.write(series.describe())


def _read_cif_file(cif_path: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
    # --- Using the Pandas text parsing version ---
    col_names = []
    lines_to_skip = 0
    try:
        logger.debug(f"Attempting to parse CIF with Pandas: {cif_path}")
        with open(cif_path, 'r') as f:
            in_loop_header = False
            found_data = False
            for i, line in enumerate(f):
                current_line_num = i + 1
                line = line.strip()
                if not line or line.startswith('#') or line.startswith("data_"):
                    lines_to_skip = current_line_num
                    continue
                if line.startswith('loop_'):
                    in_loop_header = True
                    lines_to_skip = current_line_num
                    continue
                if in_loop_header and line.startswith('_atom_site.'):
                    col_names.append(line)
                    lines_to_skip = current_line_num
                    continue
                if in_loop_header and not line.startswith('_'):
                    in_loop_header = False
                    found_data = True
                    break
            if not found_data and not col_names:
                logger.error(f"No loop_/_atom_site found in {cif_path}")
                return None, None
            if not found_data and col_names:
                logger.error(f"Found _atom_site cols but no data in {cif_path}")
                return None, None
        if not col_names:
            logger.error(f"No _atom_site.* col names found in {cif_path}")
            return None, None
        with open(cif_path, 'r') as f:
            for _ in range(lines_to_skip):
                next(f)
            data_buffer = io.StringIO()
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    data_buffer.write(line)
            data_buffer.seek(0)
        if not data_buffer.getvalue():
            logger.warning(f"No data after header for {cif_path}")
            return pd.DataFrame(
                columns=["x", "y", "z", "chain", "residue", "seq_id"]
            ), {}
        df = pd.read_csv(
            data_buffer,
            delim_whitespace=True,
            header=None,
            names=col_names,
            on_bad_lines='warn',
            comment='#',
            skipinitialspace=True
        )
        if df.empty:
            logger.warning(f"Pandas read_csv empty DF for {cif_path}")
            return pd.DataFrame(
                columns=["x", "y", "z", "chain", "residue", "seq_id"]
            ), {}
        required_cols_map = {
            '_atom_site.group_PDB': 'g',
            '_atom_site.label_atom_id': 'a',
            '_atom_site.label_seq_id': 's',
            '_atom_site.Cartn_x': 'x',
            '_atom_site.Cartn_y': 'y',
            '_atom_site.Cartn_z': 'z',
            '_atom_site.label_asym_id': 'c',
            '_atom_site.label_comp_id': 'r'
        }
        if not all(col in df.columns for col in required_cols_map.keys()):
            missing = [k for k, v in required_cols_map.items() if k not in df.columns]
            logger.error(f"Pandas parsing missed columns: {missing}")
            return None, None
        try:
            atom_df = df.loc[df['_atom_site.group_PDB'] == 'ATOM'].copy()
        except KeyError:
            logger.warning("'_atom_site.group_PDB' missing. Assuming all atoms relevant.")
            atom_df = df.copy()
        if atom_df.empty and '_atom_site.group_PDB' in df.columns:
            logger.warning(f"No 'ATOM' records found in {cif_path}")
        final_sequences = {}
        seq_build_cols = [
            '_atom_site.label_asym_id',
            '_atom_site.label_seq_id',
            '_atom_site.label_comp_id'
        ]
        if all(c in df.columns for c in seq_build_cols):
            try:
                df['seq_id_int'] = pd.to_numeric(
                    df['_atom_site.label_seq_id'], errors='coerce'
                )
                df_seq = df.dropna(subset=['seq_id_int'])
                df_seq['seq_id_int'] = df_seq['seq_id_int'].astype(int)
                df_seq['chain_id_str'] = df_seq['_atom_site.label_asym_id'].astype(str)
                unique_residues = df_seq.drop_duplicates(subset=['chain_id_str', 'seq_id_int'])
                unique_residues_sorted = unique_residues.sort_values(
                    by=['chain_id_str', 'seq_id_int']
                )
                for chain_id, group in unique_residues_sorted.groupby('chain_id_str'):
                    final_sequences[chain_id] = "".join(
                        group['_atom_site.label_comp_id'].apply(_three_to_one).tolist()
                    )
            except Exception as seq_err:
                logger.error(f"Error building sequences: {seq_err}", exc_info=True)
        else:
            logger.error(
                f"Missing columns for sequence building: {[c for c in seq_build_cols if c not in df.columns]}"
            )
        ca_atoms_df_out = pd.DataFrame(columns=["x", "y", "z", "chain", "residue", "seq_id"])
        if not atom_df.empty and '_atom_site.label_atom_id' in atom_df.columns:
            ca_df = atom_df.loc[atom_df['_atom_site.label_atom_id'] == 'CA'].copy()
            if not ca_df.empty:
                try:
                    ca_source_cols = [
                        '_atom_site.Cartn_x',
                        '_atom_site.Cartn_y',
                        '_atom_site.Cartn_z',
                        '_atom_site.label_asym_id',
                        '_atom_site.label_comp_id',
                        '_atom_site.label_seq_id'
                    ]
                    if not all(col in ca_df.columns for col in ca_source_cols):
                        missing_ca = [c for c in ca_source_cols if c not in ca_df.columns]
                        logger.error(f"Missing CA cols: {missing_ca}")
                        return None, final_sequences
                    temp_ca_data = {}
                    temp_ca_data['x'] = pd.to_numeric(ca_df['_atom_site.Cartn_x'], errors='coerce')
                    temp_ca_data['y'] = pd.to_numeric(ca_df['_atom_site.Cartn_y'], errors='coerce')
                    temp_ca_data['z'] = pd.to_numeric(ca_df['_atom_site.Cartn_z'], errors='coerce')
                    temp_ca_data['chain'] = ca_df['_atom_site.label_asym_id'].astype(str)
                    temp_ca_data['residue'] = ca_df['_atom_site.label_comp_id'].astype(str)
                    temp_ca_data['seq_id'] = pd.to_numeric(ca_df['_atom_site.label_seq_id'], errors='coerce')
                    ca_atoms_df_out = pd.DataFrame(temp_ca_data)
                    ca_atoms_df_out.dropna(subset=['x', 'y', 'z', 'seq_id'], inplace=True)
                    if not ca_atoms_df_out.empty:
                        ca_atoms_df_out['seq_id'] = ca_atoms_df_out['seq_id'].astype(int)
                except Exception as conv_err:
                    logger.error(f"Error converting CA data: {conv_err}", exc_info=True)
                    return None, final_sequences
            else:
                logger.warning(f"No 'CA' atoms found in {cif_path}")
        logger.info(f"Processed {cif_path}. Found {len(ca_atoms_df_out)} CA atoms.")
        return ca_atoms_df_out, final_sequences if 'final_sequences' in locals() else None
    except FileNotFoundError:
        report_error(FileNotFoundError(f"CIF FNF: {cif_path}"), "FNF")
        logger.error(f"CIF FNF: {cif_path}")
        return None, None
    except pd.errors.EmptyDataError:
        logger.error(f"Pandas empty DF for {cif_path}")
        return pd.DataFrame(
            columns=["x", "y", "z", "chain", "residue", "seq_id"]
        ), {}
    except Exception as e:
        report_error(e, f"Failed pandas parse: {cif_path}")
        logger.error(f"Error pandas parse {cif_path}: {e}\n{traceback.format_exc()}")
        return None, None


def _load_mrc_volume(mrc_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    try:
        with mrcfile.mmap(mrc_path, permissive=True) as mrc:
            volume_data = mrc.data.copy()
            voxel_size = 1.0
            if mrc.voxel_size.x and mrc.voxel_size.x > 0:
                voxel_size = float(mrc.voxel_size.x)
            else:
                logger.warning(f"Invalid/missing voxel size in MRC {mrc_path}. Assuming 1.0 Å/px.")
            origin_px = np.array(
                [getattr(mrc.header, s, 0) for s in ['nxstart', 'nystart', 'nzstart']],
                dtype=float
            )
            origin_angstrom = origin_px * voxel_size
            logger.info(
                f"Loaded volume {mrc_path}, shape={volume_data.shape}, voxel={voxel_size:.3f} Å/px, origin={origin_angstrom} Å"
            )
            return volume_data, origin_angstrom, voxel_size
    except FileNotFoundError:
        report_error(FileNotFoundError(f"MRC file not found: {mrc_path}"), "FNF")
        logger.error(f"MRC FNF: {mrc_path}")
        return None, None, None
    except Exception as e:
        report_error(e, f"Failed to parse MRC: {mrc_path}")
        logger.error(f"MRC Error {mrc_path}: {e}\n{traceback.format_exc()}")
        return None, None, None

# --- Plotting Functions ---

def plot_ca_atoms_plotly(ca_atoms_df: pd.DataFrame, axis_unit: str = "Å") -> Optional[go.Figure]:
    """Plots C-alpha atoms with hidden axis details (labels, ticks, numbers, grid lines)."""
    if ca_atoms_df is None or ca_atoms_df.empty:
        return None

    required_plot_cols = ["x", "y", "z", "chain", "residue", "seq_id"]
    if not all(col in ca_atoms_df.columns for col in required_plot_cols):
        missing_cols = [c for c in required_plot_cols if c not in ca_atoms_df.columns]
        logger.error(f"C-alpha DF missing required columns for plotting: {missing_cols}")
        st.error(f"Cannot plot atoms. Data missing columns: {', '.join(missing_cols)}")
        return None

    fig = go.Figure()

    try:
        if 'seq_id' in ca_atoms_df.columns and pd.api.types.is_numeric_dtype(ca_atoms_df['seq_id']):
            ca_atoms_df = ca_atoms_df.sort_values(by=["chain", "seq_id"])
        else:
            ca_atoms_df = ca_atoms_df.sort_values(by=["chain"])
            logger.warning("seq_id column missing/non-numeric; sorting by chain only.")
    except Exception as sort_err:
        logger.warning(f"Error sorting atom DataFrame: {sort_err}")

    unique_chains = ca_atoms_df["chain"].unique()
    colors = px.colors.qualitative.Plotly

    logger.debug(f"Atom coordinate ranges ({axis_unit}):")
    try:
        logger.debug(f"  X: {ca_atoms_df['x'].min():.2f} - {ca_atoms_df['x'].max():.2f}")
        logger.debug(f"  Y: {ca_atoms_df['y'].min():.2f} - {ca_atoms_df['y'].max():.2f}")
        logger.debug(f"  Z: {ca_atoms_df['z'].min():.2f} - {ca_atoms_df['z'].max():.2f}")
    except Exception as e:
        logger.warning(f"Could not log atom coord ranges: {e}")

    for i, chain_id in enumerate(unique_chains):
        chain_data = ca_atoms_df[ca_atoms_df["chain"] == chain_id]
        if chain_data.empty:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter3d(
                x=chain_data["x"],
                y=chain_data["y"],
                z=chain_data["z"],
                mode="lines",
                marker=dict(size=3.5, opacity=0.8, color=color),
                line=dict(width=8, color=color),
                name=f"Chain {chain_id}",
                customdata=chain_data[['residue', 'seq_id']],
                hovertemplate=(
                    f"<b>Chain {chain_id}</b><br>"
                    f"Residue: %{{customdata[0]}}%{{customdata[1]}}<br>"
                    f"X: %{{x:.2f}} {axis_unit}<br>"
                    f"Y: %{{y:.2f}} {axis_unit}<br>"
                    f"Z: %{{z:.2f}} {axis_unit}<extra></extra>"
                )
            )
        )

    # Update layout to hide axis labels, ticks, numbers, and grid lines.
    fig.update_layout(
        title=f"C-alpha Backbone Trace ({axis_unit})",
        scene=dict(
            xaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            zaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            aspectmode='data'
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        height=600,
        hovermode='closest',
        legend_title_text="Chains",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        legend_title_font_color="white",
        legend_font_color="white",
        legend_bgcolor="black",
        legend_bordercolor="white",
        legend_borderwidth=1
    )

    # Ensure the aspect ratio mode remains set.
    fig.update_scenes(aspectmode='data')
    return fig


def overlay_plots_plotly(atom_fig: Optional[go.Figure], volume_fig: Optional[go.Figure], axis_unit: str = "px") -> go.Figure:
    """Overlays atom traces and volume trace into a NEW figure. Axis labels use the provided unit."""
    combined_fig = go.Figure()

    # Add volume trace first (if exists) - Assuming volume plot uses pixel units
    if volume_fig is not None and volume_fig.data:
        logger.debug(f"Adding {len(volume_fig.data)} volume trace(s) to combined figure.")
        for trace in volume_fig.data:
            combined_fig.add_trace(trace)
    else:
        logger.debug("No volume figure or trace provided for overlay.")

    # Add atom traces second (if exists) - Assuming atom plot uses pixel units here
    if atom_fig is not None and atom_fig.data:
        logger.debug(f"Adding {len(atom_fig.data)} atom trace(s) to combined figure.")
        for trace in atom_fig.data:
            combined_fig.add_trace(trace)
    else:
        logger.debug("No atom figure or traces provided for overlay.")

    # Apply final layout properties
    combined_fig.update_layout(
        title=f"C-alpha Trace Overlaid with Volume Density ({axis_unit})",
        scene=dict(
            xaxis_title=f"X ({axis_unit})",
            yaxis_title=f"Y ({axis_unit})",
            zaxis_title=f"Z ({axis_unit})",
            aspectmode='data',
            bgcolor="black"
        ),
        hovermode='closest',
        margin=dict(l=10, r=10, b=10, t=40),
        height=600,
        legend_title_text="Traces"
    )
    combined_fig.update_scenes(aspectmode='data')
    # Update layout to hide axis labels, ticks, numbers, and grid lines.
    combined_fig.update_layout(
        title=f"C-alpha Backbone Trace ({axis_unit})",
        scene=dict(
            xaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            zaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            aspectmode='data'
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        height=600,
        hovermode='closest',
        legend_title_text="Chains",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        legend_title_font_color="white",
        legend_font_color="white",
        legend_bgcolor="black",
        legend_bordercolor="white",
        legend_borderwidth=1
    )

    logger.debug(f"Combined figure created with {len(combined_fig.data)} total traces.")
    return combined_fig


# --- Main Function ---
def plot_modelangelo(folder: str, node_files: List[str]) -> None:
    """Main Streamlit function to plot ModelAngelo job outputs with correct scaling."""
    if not node_files:
        st.error("No node files provided...")
        logger.error("No node files.")
        return

    cif_rel_path = node_files[0]
    cif_abs_path = os.path.join(folder, cif_rel_path)
    if not os.path.exists(cif_abs_path):
        st.error(f"Input model FNF: {cif_abs_path}")
        logger.error(f"Model FNF: {cif_abs_path}")
        return

    job_dir_rel = os.path.dirname(cif_rel_path)
    job_path = os.path.join(folder, job_dir_rel)
    try:
        cif_mtime = os.path.getmtime(cif_abs_path)
    except OSError:
        cif_mtime = 0
    job_id_key = f"modelangelo_{job_path}_{cif_mtime}"
    # --- Use a single cache key for all job data ---
    job_data_cache_key = f"{job_id_key}_data"
    volume_data_cache_key_base = f"{job_id_key}_volume"  # Base key for volume

    logger.info(f"{datetime.now()}: plot_modelangelo called for job '{job_dir_rel}'")
    st.subheader(f"ModelAngelo Output: {job_dir_rel}")
    st.write(f"Displaying model: `{cif_rel_path}`")

    # --- Load/Retrieve Cached Job Data ---
    sequences = None
    ca_atoms_angstrom = None
    ca_atoms_pixels = None
    scaling_info = {"applied": False, "voxel_size": None}  # Store scaling state

    if job_data_cache_key in st.session_state:
        cached_data = st.session_state[job_data_cache_key]
        if isinstance(cached_data, dict):
            ca_atoms_angstrom = cached_data.get("atoms_angstrom")
            ca_atoms_pixels = cached_data.get("atoms_pixels")  # Might be None initially
            sequences = cached_data.get("sequences")
            scaling_info = cached_data.get("scaling_info", scaling_info)  # Get saved scaling state

            # Validate cached data types
            valid_cache = isinstance(ca_atoms_angstrom, pd.DataFrame) and isinstance(sequences, dict)
            if valid_cache:
                logger.debug(f"Using cached Job data for {job_dir_rel}. Scaling applied: {scaling_info['applied']}")
            else:
                logger.warning(f"Invalid structure in job cache {job_data_cache_key}. Reloading.")
                ca_atoms_angstrom = None
                sequences = None
                ca_atoms_pixels = None
                scaling_info = {"applied": False, "voxel_size": None}
        else:
            logger.warning(f"Invalid data type in job cache {job_data_cache_key}. Reloading.")
            # Force reload by ensuring ca_atoms_angstrom is None

    # If not cached or cache was invalid, load from file
    if ca_atoms_angstrom is None:
        logger.debug(f"Reading PDB/CIF file with Pandas: {cif_abs_path}")
        ca_atoms_angstrom, sequences = _read_cif_file(cif_abs_path)  # Use Pandas version
        if ca_atoms_angstrom is not None and sequences is not None:
            st.session_state[job_data_cache_key] = {
                "atoms_angstrom": ca_atoms_angstrom,
                "atoms_pixels": None,  # Initialize pixel coords to None
                "sequences": sequences,
                "scaling_info": {"applied": False, "voxel_size": None}  # Initialize scaling state
            }
            logger.debug(f"Cached initial Job data for {job_dir_rel}")
        else:
            st.error(f"Failed to read or parse model file: {cif_rel_path}")
            if not sequences:
                return

    # At this point, ca_atoms_angstrom and sequences should be populated if successful
    if ca_atoms_angstrom is None or ca_atoms_angstrom.empty:
        st.warning("Could not load C-alpha atom data. 3D plot unavailable.")
        if not sequences:
            return

    c1, c2 = st.columns([1, 4])
    # --- GUI Controls ---
    controls_col1, controls_col2 = st.columns([1, 2])
    with controls_col1:
        volume_enabled = ca_atoms_angstrom is not None and not ca_atoms_angstrom.empty
        show_volume = c1.checkbox(
            "Overlay Volume Density?",
            key=f"{job_id_key}_show_vol",
            disabled=not volume_enabled,
            help="Requires atom model to be loaded successfully." if not volume_enabled else None
        )
        volume_threshold = 0.5
        volume_opacity = 0.5
        if show_volume and volume_enabled:
            volume_threshold = c1.slider(
                "Volume Threshold",
                0.0,
                1.0,
                volume_threshold,
                0.05,
                key=f"{job_id_key}_vol_thresh"
            )
            volume_opacity = c1.slider(
                "Volume Opacity",
                0.0,
                1.0,
                volume_opacity,
                0.05,
                key=f"{job_id_key}_vol_opac"
            )

    # --- Load Volume Data ---
    volume_data, voxel_size, origin_angstrom = None, None, None
    if show_volume and volume_enabled:
        # [Volume loading logic - same as before]
        with st.spinner("Checking for volume data..."):
            note_path = os.path.join(job_path, "note.txt")
            if not os.path.exists(note_path):
                st.warning("note.txt not found.")
                show_volume = False
            else:
                note_content = get_note(note_path)
                volume_path_rel = _extract_v_parameter(note_content)
                if not volume_path_rel:
                    st.warning("No volume path in note.txt.")
                    show_volume = False
                else:
                    volume_path_abs = os.path.normpath(os.path.join(folder, volume_path_rel))
                    try:
                        vol_mtime = os.path.getmtime(volume_path_abs)
                    except OSError:
                        vol_mtime = 0
                    current_volume_cache_key = f"{volume_data_cache_key_base}_{vol_mtime}"  # Unique key for this volume version
                    if current_volume_cache_key in st.session_state:
                        cached_vol = st.session_state[current_volume_cache_key]
                        if isinstance(cached_vol, dict):
                            volume_data = cached_vol.get("data")
                            voxel_size = cached_vol.get("voxel_size")
                            origin_angstrom = cached_vol.get("origin")
                            if not (isinstance(volume_data, np.ndarray) and voxel_size and origin_angstrom is not None):
                                volume_data = None
                            else:
                                logger.debug("Using cached Volume data.")
                        else:
                            volume_data = None
                    if volume_data is None:
                        if not os.path.exists(volume_path_abs):
                            st.error(f"Volume file not found: {volume_path_abs}")
                            show_volume = False
                        else:
                            logger.debug(f"Reading Volume file: {volume_path_abs}")
                            with st.spinner(f"Loading volume: {os.path.basename(volume_path_rel)}..."):
                                volume_data, origin_angstrom, voxel_size = _load_mrc_volume(volume_path_abs)
                            if volume_data is not None and voxel_size and origin_angstrom is not None:
                                st.session_state[current_volume_cache_key] = {
                                    "data": volume_data,
                                    "voxel_size": voxel_size,
                                    "origin": origin_angstrom
                                }
                                logger.debug("Cached Volume data.")
                            else:
                                st.error(f"Failed to load volume file: {volume_path_rel}")
                                show_volume = False

    # --- Perform Scaling IF Needed ---
    if show_volume and volume_enabled and voxel_size and voxel_size > 0 and ca_atoms_angstrom is not None:
        if not scaling_info.get("applied") or scaling_info.get("voxel_size") != voxel_size:
            logger.info(f"Applying scaling to atom coordinates with voxel size: {voxel_size:.3f} Å/px")
            with st.spinner("Scaling coordinates..."):
                ca_atoms_pixels = ca_atoms_angstrom.copy()
                
                ox, oy, oz = 0.0, 0.0, 0.0
                if origin_angstrom is not None:
                     ox, oy, oz = origin_angstrom

                ca_atoms_pixels["x"] = (ca_atoms_pixels["x"] - ox) / voxel_size
                ca_atoms_pixels["y"] = (ca_atoms_pixels["y"] - oy) / voxel_size
                ca_atoms_pixels["z"] = (ca_atoms_pixels["z"] - oz) / voxel_size

                updated_cache = st.session_state.get(job_data_cache_key, {})
                updated_cache["atoms_pixels"] = ca_atoms_pixels
                updated_cache["scaling_info"] = {"applied": True, "voxel_size": voxel_size}
                st.session_state[job_data_cache_key] = updated_cache
                scaling_info = updated_cache["scaling_info"]
                logger.debug("Stored scaled pixel coordinates in session state.")
        else:
            ca_atoms_pixels = st.session_state[job_data_cache_key].get("atoms_pixels")
            if ca_atoms_pixels is None:
                logger.error("Scaling was marked applied, but pixel coordinates are missing! Re-applying.")
                scaling_info["applied"] = False

    if volume_enabled and volume_data is not None and ca_atoms_pixels is not None:
        with st.expander("Residue Density Analysis", expanded=False):
            # Use Angstrom coordinates for density calculation to ensure correct origin handling
            plot_residue_density_ui(ca_atoms_angstrom, volume_data, voxel_size, mode="CA", radius=1.5, scaled=False, origin_angstrom=origin_angstrom)

    # --- Plotting ---
    st.markdown("---")
    plot_placeholder = st.empty()
    final_figure = None
    atom_fig = None
    volume_fig = None

    try:
        if show_volume and volume_data is not None and ca_atoms_pixels is not None and not ca_atoms_pixels.empty:
            atoms_to_plot = ca_atoms_pixels
            axis_unit = "px"
            logger.info(f"Plotting atoms in PIXEL coordinates (Voxel Size: {voxel_size:.3f} Å/px)")
        elif ca_atoms_angstrom is not None and not ca_atoms_angstrom.empty:
            atoms_to_plot = ca_atoms_angstrom
            axis_unit = "Å"
            logger.info("Plotting atoms in ANGSTROM coordinates.")
        else:
            atoms_to_plot = None
            axis_unit = ""

        if atoms_to_plot is not None:
            with st.spinner(f"Rendering C-alpha trace ({axis_unit})..."):
                atom_fig = plot_ca_atoms_plotly(atoms_to_plot, axis_unit=axis_unit)
                if not atom_fig:
                    st.error("Failed to generate atom plot.")

        if show_volume and volume_data is not None:
            with st.spinner("Rendering volume (pixels)..."):
                volume_fig_maybe = plot_volume(volume_data, volume_threshold, max_size=volume_data.shape[0], opacity=volume_opacity)
                if isinstance(volume_fig_maybe, go.Figure):
                    volume_fig = volume_fig_maybe
                    volume_fig.update_scenes(aspectmode='data', bgcolor="black")
                    volume_fig.update_layout(
                        paper_bgcolor="black"
                    )
                else:
                    logger.warning("image_utils.plot_volume did not return a Plotly Figure.")
                    st.warning("Could not render volume data (invalid plot type).")

        if atom_fig:
            with st.spinner("Combining plots..."):
                final_figure = overlay_plots_plotly(atom_fig, volume_fig, axis_unit=axis_unit)

        if final_figure:
            logger.info(f"Displaying final plot ({axis_unit}).")
            c2.plotly_chart(final_figure, use_container_width=True)
        elif atom_fig:
            logger.info(f"Displaying atom-only plot ({axis_unit}).")
            c2.plotly_chart(atom_fig, use_container_width=True)
    except Exception as plot_err:
        logger.error(f"Plotting error: {report_error(plot_err)}")
        plot_placeholder.error(f"Plotting error: {plot_err}")

    # --- Sequences ---
    st.subheader("Sequences")
    if sequences:
        num_chains = len(sequences)
        seq_cols = st.columns(min(num_chains, 3))
        col_idx = 0
        for chain_id, seq in sorted(sequences.items()):
            with seq_cols[col_idx % len(seq_cols)]:
                st.text_area(
                    f"Chain {chain_id} ({len(seq)} AA):",
                    seq,
                    height=max(80, min(300, (len(seq) // 60 + 1) * 25)),
                    key=f"{job_id_key}_seq_{chain_id}"
                )
            col_idx += 1
    elif ca_atoms_angstrom is not None:
        st.info("No sequence information extracted.")
    elif not sequences:
        st.error("Could not load atom or sequence data.")

    logger.info(f"{datetime.now()}: plot_modelangelo for {job_dir_rel} done.")
