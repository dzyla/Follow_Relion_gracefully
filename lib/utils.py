#utils.py

# Standard Library Imports
import base64
import glob
import hmac
import logging
import os
import re
import tempfile
import traceback
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import gemmi.cif as cif
import numpy as np
import pandas as pd
import polars as pl
import starfile_rs
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde  # Used in interactive_scatter_plot
import datashader as ds
from datashader import transfer_functions as tf
import holoviews as hv
from holoviews.operation.datashader import datashade, dynspread

hv.extension('bokeh')

# Globals
ERROR_HANDLER: Optional[callable] = None
logger = logging.getLogger("main_app")  # Assumes main app created this logger


# --- Error Reporting ---

def report_error(exc: Exception, text: str='') -> None:
    """
    Reports an error using the global error handler or prints to console.

    Args:
        exc: The caught exception.
    """
    error_info = traceback.format_exc()
    if ERROR_HANDLER is not None:
        try:
            ERROR_HANDLER(text, exc, error_info)
        except Exception as handler_exc:
            # Fallback if the error handler itself fails
            print(f"{datetime.now()}: Error handler failed: {handler_exc}")
            print(f"{datetime.now()}: Original error:\n{error_info}")
    else:
        # Fallback if no error handler is set
        print(f"{datetime.now()}: An unexpected error occurred:\n{error_info}")


# --- STAR File Handling ---

def parse_star(file_path: str, lazy: bool = False) -> Dict[str, Union[pd.DataFrame, pl.LazyFrame]]:
    """
    Parses a STAR file into a dictionary of DataFrames using starfile-rs.

    Args:
        file_path: The path to the STAR file.
        lazy: If True, returns Polars LazyFrames. If False (default), returns Pandas DataFrames.

    Returns:
        A dictionary where keys are block names and values are DataFrames.
        Returns an empty dictionary if the file is not found or parsing fails.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {}
    try:
        star_data_dict = starfile_rs.read_star(file_path)
        result = {}
        for block_name, block in star_data_dict.items():
            try:
                lf = block.to_polars().lazy()
                # starfile-rs strips leading underscores from column names.
                # We restore them for backward compatibility with the rest of the codebase.
                new_columns = [
                    f"_{col}" if not col.startswith("_") and col.startswith("rln") else col
                    for col in lf.collect_schema().names()
                ]
                # Mapping of old names to new names
                rename_map = {old: new for old, new in zip(lf.collect_schema().names(), new_columns) if old != new}
                if rename_map:
                    lf = lf.rename(rename_map)

                if lazy:
                    result[block_name] = lf
                else:
                    result[block_name] = lf.collect().to_pandas()

            except AttributeError:
                logger.warning(f"Could not convert block '{block_name}' to DataFrame directly.")
        return result
    except Exception as exc:
        logger.error(f"Failed to parse STAR file: {file_path}")
        report_error(exc)
        return {}


def star_from_df(dicts_of_df: Dict[str, Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]]) -> str:
    """
    Converts a dictionary of DataFrames (Pandas or Polars) into a STAR file string.

    Args:
        dicts_of_df: Dictionary mapping block names (str) to DataFrames.

    Returns:
        A string representing the STAR file content.
    """
    output_parts = []

    # Header
    output_parts.append("\ndata_\n")

    for block_name, df in dicts_of_df.items():
        try:
            # Convert to Polars DataFrame if it's Pandas or LazyFrame
            if isinstance(df, pd.DataFrame):
                pl_df = pl.from_pandas(df)
            elif isinstance(df, pl.LazyFrame):
                pl_df = df.collect()
            elif isinstance(df, pl.DataFrame):
                pl_df = df
            else:
                 raise TypeError(f"Unsupported dataframe type for block '{block_name}': {type(df)}")

            # Use starfile_rs to convert block to string
            # starfile_rs.LoopDataBlock.from_polars(df=pl_df, name=block_name)
            # Create a LoopDataBlock
            loop_block = starfile_rs.LoopDataBlock.from_polars(df=pl_df, name=block_name)
            output_parts.append(loop_block.to_string())

        except Exception as exc:
            logger.error(f"Error converting block '{block_name}' to STAR format: {exc}")
            raise

    return "\n".join(output_parts)


# --- Filesystem Utilities ---

def get_folders(path: str) -> List[str]:
    """
    Returns a sorted list of folder names within the specified directory.

    Args:
        path: The directory path to scan.

    Returns:
        A sorted list of folder names, or an empty list if the path
        doesn't exist or an error occurs.
    """
    if not os.path.isdir(path):
        logger.warning(f"Directory not found or invalid: {path}")
        return []
    try:
        entries = os.listdir(path)
        folders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
        return sorted(folders, key=str.lower)
    except OSError as exc:
        logger.error(f"Error listing directory {path}: {exc}")
        report_error(exc)
        return []


def get_newest_change(folder: str, include_subfolders: bool = False) -> str:
    """
    Gets the timestamp of the most recently modified file in a folder.

    Args:
        folder: The path to the folder to scan.
        include_subfolders: If True, scans subfolders recursively.
                            If False, only scans files in the top-level folder.

    Returns:
        A string representing the modification time ("YYYY-MM-DD HH-MM"),
        "No files found", or "Error occurred".
    """
    newest_mod_time_secs = None
    try:
        # Use os.scandir for potentially better performance
        for entry in os.scandir(folder):
            if entry.is_file():
                mod_time = entry.stat().st_mtime
                if newest_mod_time_secs is None or mod_time > newest_mod_time_secs:
                    newest_mod_time_secs = mod_time
            elif entry.is_dir() and include_subfolders:
                # Recursively call for subfolders
                subfolder_time_str = get_newest_change(entry.path, include_subfolders=True)
                if subfolder_time_str.startswith("20"): # Basic check for valid date format
                    try:
                        subfolder_dt = datetime.strptime(subfolder_time_str, "%Y-%m-%d %H-%M")
                        subfolder_mod_time_secs = subfolder_dt.timestamp()
                        if newest_mod_time_secs is None or subfolder_mod_time_secs > newest_mod_time_secs:
                             newest_mod_time_secs = subfolder_mod_time_secs
                    except ValueError:
                         logger.warning(f"Could not parse timestamp from subfolder: {subfolder_time_str}")


        if newest_mod_time_secs is not None:
            return datetime.fromtimestamp(newest_mod_time_secs).strftime("%Y-%m-%d %H-%M")
        else:
            return "No files found in the folder"
    except FileNotFoundError:
        logger.warning(f"Folder not found for timestamp check: {folder}")
        return "Folder not found"
    except Exception as exc:
        logger.error(f"Error scanning folder {folder} for timestamp.")
        report_error(exc)
        return "Error occurred while scanning the folder."


def get_modification_time(file_path: str) -> Optional[datetime]:
    """
    Retrieves the modification time for a given file.

    Args:
        file_path: The absolute or relative path to the file.

    Returns:
        The modification time as a datetime object, or None if the file
        doesn't exist or the time cannot be read.
    """
    if os.path.exists(file_path):
        try:
            mod_time_secs = os.path.getmtime(file_path)
            return datetime.fromtimestamp(mod_time_secs)
        except OSError as exc:
            logger.error(f"Error reading modification timestamp for {file_path}")
            report_error(exc)
            return None
    else:
        logger.warning(f"File not found for timestamp check: {file_path}")
        return None


def get_git_commit_hash(short: bool = True) -> str:
    """
    Retrieves the current git commit hash.

    Args:
        short: If True, returns the short hash. If False, returns the full hash.

    Returns:
        The git commit hash as a string, or an empty string if retrieval fails.
    """
    try:
        args = ["git", "rev-parse", "HEAD"]
        if short:
            args.insert(2, "--short")

        # Run git command
        commit_hash = subprocess.check_output(args, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to retrieve git commit hash.")
        return ""
    except Exception as exc:
        logger.error(f"Unexpected error retrieving git commit hash: {exc}")
        return ""


def get_subfolders(parent_folder: str, job_type: str) -> List[str]:
    """
    Gets a list of subfolder names directly under a job type folder.

    Args:
        parent_folder: The root directory containing job type folders (e.g., Project/RELION/).
        job_type: The specific job type folder name (e.g., "Class2D").

    Returns:
        A list of subfolder names (e.g., ["job001", "job002"]).
    """
    folder_path = os.path.join(parent_folder, job_type)
    return get_folders(folder_path) # Reuse get_folders


def get_job_files(job_folder: str, file_suffix: str) -> List[str]:
    """
    Gets a list of files within a specific job folder ending with a given suffix.

    Args:
        job_folder: The path to the specific job folder (e.g., Project/RELION/Class2D/job001).
        file_suffix: The file suffix to match (e.g., "_data.star").

    Returns:
        A list of matching filenames. Returns an empty list on error.
    """
    files = []
    if not os.path.isdir(job_folder):
        logger.warning(f"Job folder not found: {job_folder}")
        return []
    try:
        for item in os.listdir(job_folder):
            item_path = os.path.join(job_folder, item)
            # Check if it's a file and ends with the specified suffix
            if os.path.isfile(item_path) and item.endswith(file_suffix):
                files.append(item)
        return sorted(files) # Return sorted list for consistency
    except OSError as exc:
        logger.error(f"Error reading job folder {job_folder}: {exc}")
        report_error(exc)
        return []


# --- RELION Specific Utilities ---

def get_relationships_df(pipeline_edges_df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]) -> pd.DataFrame:
    """
    Creates a DataFrame showing job relationships (parent/child) from pipeline edges.

    Args:
        pipeline_edges_df: DataFrame parsed from the 'pipeline_edges' block
                           of a RELION pipeline.star file. Requires columns
                           '_rlnPipeLineEdgeProcess' and '_rlnPipeLineEdgeFromNode'.

    Returns:
        A DataFrame with columns ['Job', 'Children', 'Parents'], detailing the
        workflow connections. Returns an empty DataFrame if input is invalid or
        an error occurs.
    """
    # Convert input to Pandas for graph processing (NetworkX usually used, or just dicts)
    # The existing logic uses pandas iterrows.
    if isinstance(pipeline_edges_df, pl.LazyFrame):
        pipeline_edges_df = pipeline_edges_df.collect().to_pandas()
    elif isinstance(pipeline_edges_df, pl.DataFrame):
        pipeline_edges_df = pipeline_edges_df.to_pandas()

    required_cols = ["_rlnPipeLineEdgeProcess", "_rlnPipeLineEdgeFromNode"]
    if not all(col in pipeline_edges_df.columns for col in required_cols):
        logger.error("Input DataFrame for relationships is missing required columns.")
        return pd.DataFrame(columns=["Job", "Children", "Parents"])

    # Helper to extract the job name (e.g., "Class2D/job001/")
    def extract_job_name(process_name: str) -> str:
        parts = process_name.split("/")
        # Handle cases like "External/job001/" vs just "Import/job001"
        return "/".join(parts[:-1]) + "/" if len(parts) > 1 and parts[-1] == '' else process_name

    children: Dict[str, List[str]] = {} # Key: Parent Job, Value: List of Child Jobs
    parents: Dict[str, List[str]] = {}  # Key: Child Job, Value: List of Parent Jobs

    try:
        for _, row in pipeline_edges_df.iterrows():
            # 'from_node' is the parent, 'to_node' is the child process
            from_node = extract_job_name(row["_rlnPipeLineEdgeFromNode"])
            to_node = extract_job_name(row["_rlnPipeLineEdgeProcess"])

            # Record parent -> child relationship
            parents.setdefault(to_node, []).append(from_node)
            # Record child -> parent relationship
            children.setdefault(from_node, []).append(to_node)

        # Create DataFrames from the dictionaries
        # Use list(set(...)) to ensure unique entries if needed, though RELION pipeline should be unique
        children_df = pd.DataFrame(
            [(job, list(set(child_list))) for job, child_list in children.items()],
            columns=["Job", "Children"]
        )
        parents_df = pd.DataFrame(
             [(job, list(set(parent_list))) for job, parent_list in parents.items()],
            columns=["Job", "Parents"]
        )

        # Merge parent and child info, keeping all jobs
        merged_df = pd.merge(children_df, parents_df, on="Job", how="outer")
        # Fill NaN values for jobs that have only parents or only children
        merged_df['Children'] = merged_df['Children'].apply(lambda x: x if isinstance(x, list) else [])
        merged_df['Parents'] = merged_df['Parents'].apply(lambda x: x if isinstance(x, list) else [])

        return merged_df.sort_values(by="Job").reset_index(drop=True)

    except Exception as exc:
        logger.error("Failed to process job relationships from pipeline edges.")
        report_error(exc)
        return pd.DataFrame(columns=["Job", "Children", "Parents"])


def get_note(note_file_path: str) -> str:
    """
    Reads the content of a RELION job's note file.

    Args:
        note_file_path: Path to the note file (e.g., job_directory/note.txt).

    Returns:
        The content of the note file as a string, or a default message
        if the file doesn't exist or cannot be read.
    """
    if not os.path.exists(note_file_path):
        return "Note file not found."

    try:
        with open(note_file_path, 'r', encoding='utf-8') as f:
            file_data = f.read()
        # Apply specific formatting replacements if needed (consider if these are always desired)
        # file_data = file_data.replace("++++", "\n").replace("`", "").replace("which", "\nwhich")
        return file_data
    except Exception as exc:
        logger.error(f"Failed to read note file: {note_file_path}")
        report_error(exc)
        return f"Error reading note file: {exc}"


def extract_source_job(note_content: str) -> str:
    """
    Extracts the primary input file path (STAR or MRC) from RELION note content.

    It looks for common RELION input flags like --i, --opt, --tilt-series-star-file.
    It replaces 'optimiser.star' with 'data.star' as per convention sometimes needed.

    Args:
        note_content: The string content of a note.txt file.

    Returns:
        The extracted source file path, or an empty string if no known
        input flag/path is found or an error occurs.
    """
    patterns = {
        # Standard input (often data.star or optimiser.star)
        "i": r"--i\s+([\w\d/\.\-\_\+]+(?:star|mrcs?))",
        # Optimiser input (often optimiser.star)
        "opt": r"--opt\s+([\w\d/\.\-\_\+]+\.star)",
         # AlignTiltSeries specific input
        "tilt": r"--tilt-series-star-file\s+([\w\d/\.\-\_\+]+\.star)",
    }

    source_path = ""
    try:
        if "--i" in note_content:
            match = re.search(patterns["i"], note_content)
            if match:
                source_path = match.group(1)
                # Convention: If optimiser.star is input, often data.star holds the relevant particles
                if "optimiser.star" in source_path:
                     # Check if a corresponding data.star exists before replacing
                     potential_data_star = source_path.replace("optimiser.star", "data.star")
                     # We need the directory context here, which isn't available.
                     # Assume replacement is generally desired based on original code's intent.
                     return potential_data_star
                elif source_path.endswith((".mrc", ".mrcs")):
                    # If input is MRC/MRCS (e.g., MaskCreate), return it directly
                    return source_path
                else:
                    # Assume it's data.star or another relevant star file
                    return source_path
            else:
                 logger.debug("Found '--i' flag but couldn't match path pattern.")

        elif "--opt" in note_content:
            match = re.search(patterns["opt"], note_content)
            if match:
                source_path = match.group(1)
                # Assume data.star is the relevant file derived from optimiser
                return source_path.replace("optimiser.star", "data.star")
            else:
                logger.debug("Found '--opt' flag but couldn't match path pattern.")

        elif "--tilt-series-star-file" in note_content:
            match = re.search(patterns["tilt"], note_content)
            if match:
                source_path = match.group(1)
                # Assume data.star is the relevant file derived from optimiser
                return source_path.replace("optimiser.star", "data.star")
            else:
                logger.debug("Found '--tilt-series-star-file' flag but couldn't match pattern.")

        if not source_path:
            logger.info("No known source file flag found in note content.")
        return source_path

    except Exception as exc:
        logger.error("Error occurred during source job extraction from note.")
        report_error(exc)
        return ""


def get_angles(job_path: str, limit: Optional[int] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extracts Euler angles (Rot, Tilt, Psi) from the latest _data.star file
    found within a RELION job directory.

    Args:
        job_path: The path to the specific RELION job directory.
        limit: Optional. If provided, randomly samples this many particles.

    Returns:
        A tuple containing three pandas Series: (rot_angles, tilt_angles, psi_angles).
        Returns empty Series if the file or columns are not found or an error occurs.
    """
    empty_series = pd.Series(dtype=float)
    try:
        # Find all files ending with _data.star in the job directory
        data_star_files = glob.glob(os.path.join(job_path, "*_data.star"))
        if not data_star_files:
            logger.warning(f"No '*_data.star' files found in {job_path}")
            return empty_series, empty_series, empty_series

        # Sort by modification time to get the latest one
        data_star_files.sort(key=os.path.getmtime)
        latest_data_star_path = data_star_files[-1]
        logger.info(f"Parsing angles from: {latest_data_star_path}")

        # Parse the STAR file
        star_data = parse_star(latest_data_star_path, lazy=True)

        # Expect angles in the 'particles' or 'micrographs' block usually
        data_block = None
        if "particles" in star_data:
            data_block = star_data["particles"]
        elif "micrographs" in star_data: # Some jobs might store orientations here
             data_block = star_data["micrographs"]

        if data_block is None:
             logger.warning(f"No 'particles' or 'micrographs' data found in {latest_data_star_path}")
             return empty_series, empty_series, empty_series

        # Apply sampling if requested
        if limit is not None:
             # Lazy evaluation allows us to not count rows first if we sample
             # But polars sample usually requires collect or known length for random sampling in some versions?
             # For LazyFrame, sample is supported.
             data_block = data_block.collect().sample(n=limit, with_replacement=False, seed=42).lazy()

        data_df = data_block.collect() # Collect to process columns

        # Extract angle columns
        required_cols = ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
        if not all(col in data_df.columns for col in required_cols):
            logger.error(f"Missing one or more angle columns in {latest_data_star_path}")
            return empty_series, empty_series, empty_series

        # Polars to Pandas Series for compatibility with return type
        rot_angles = data_df["_rlnAngleRot"].cast(pl.Float64, strict=False).fill_null(0.0).to_pandas()
        tilt_angles = data_df["_rlnAngleTilt"].cast(pl.Float64, strict=False).fill_null(0.0).to_pandas()
        psi_angles = data_df["_rlnAnglePsi"].cast(pl.Float64, strict=False).fill_null(0.0).to_pandas()

        return rot_angles, tilt_angles, psi_angles

    except Exception as exc:
        logger.error(f"Failed to get angles from {job_path}.")
        report_error(exc)
        return empty_series, empty_series, empty_series

def get_classes(
    job_path: str, model_star_files: List[str]
) -> Tuple[List[str], int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses RELION model.star files to extract class information over iterations.
    Refactored to use Polars.
    """
    empty_result = ([], 0, 0, np.array([]), np.array([]), np.array([]), np.array([]))
    if not model_star_files:
        logger.warning("No model.star files provided to get_classes.")
        return empty_result

    class_dist_per_iter: List[np.ndarray] = []
    class_res_per_iter: List[np.ndarray] = []
    fsc_vals_per_iter: List[np.ndarray] = []
    fsc_res_per_iter: List[np.ndarray] = []

    logger.debug(f"Parsing {len(model_star_files)} model files from job: {job_path}")

    valid_files_processed = 0
    for i, file_path in enumerate(model_star_files):
        # Skip half maps, process only the main model file per iteration usually named model_itXXX_optimiser.star or similar
        # Adjust this logic if your naming convention differs significantly
        if 'half1' in file_path or 'half2' in file_path or 'data.star' in file_path:
             logger.debug(f"Skipping potential half-map or data file: {os.path.basename(file_path)}")
             continue

        logger.debug(f"Parsing model file: {file_path}")
        star_data = parse_star(file_path, lazy=True)

        # --- Extract model_classes data ---
        if "model_classes" not in star_data:
            logger.warning(f"'model_classes' block missing in {file_path}. Skipping iteration {i}.")
            continue

        model_classes_lf = star_data["model_classes"]
        try:
             model_classes_df = model_classes_lf.collect()
        except Exception as e:
             logger.warning(f"Error collecting model_classes in {file_path}: {e}")
             continue

        if model_classes_df.is_empty():
             continue

        # Distribution (required)
        if "_rlnClassDistribution" not in model_classes_df.columns:
            logger.warning(f"'_rlnClassDistribution' missing in {file_path}. Skipping iteration {i}.")
            continue

        distribution = model_classes_df["_rlnClassDistribution"].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
        class_dist_per_iter.append(distribution)

        # Resolution (optional)
        if "_rlnEstimatedResolution" in model_classes_df.columns:
             resolution = model_classes_df["_rlnEstimatedResolution"].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
             class_res_per_iter.append(resolution)
        else:
             class_res_per_iter.append(np.zeros_like(distribution))


        # --- Extract FSC data (optional, assumes block names like 'model_class_1', 'model_class_2') ---
        # We only care about the FSC from the *first* class block if present ('model_class_1')
        fsc_block_name = "model_class_1"
        if fsc_block_name in star_data:
             fsc_lf = star_data[fsc_block_name]
             try:
                 fsc_df = fsc_lf.collect()
                 if "_rlnGoldStandardFsc" in fsc_df.columns and "_rlnAngstromResolution" in fsc_df.columns:
                     fsc_vals = fsc_df["_rlnGoldStandardFsc"].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
                     fsc_res_bins = fsc_df["_rlnAngstromResolution"].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()

                     min_len = min(len(fsc_vals), len(fsc_res_bins))
                     fsc_vals_per_iter.append(fsc_vals[:min_len])
                     fsc_res_per_iter.append(fsc_res_bins[:min_len])
                 else:
                     logger.debug(f"FSC columns missing in block '{fsc_block_name}' in {file_path}")
                     fsc_vals_per_iter.append(np.array([]))
                     fsc_res_per_iter.append(np.array([]))
             except Exception:
                  fsc_vals_per_iter.append(np.array([]))
                  fsc_res_per_iter.append(np.array([]))
        else:
             logger.debug(f"FSC block '{fsc_block_name}' not found in {file_path}")
             fsc_vals_per_iter.append(np.array([]))
             fsc_res_per_iter.append(np.array([]))

        valid_files_processed += 1

    # --- Post-processing and array building ---
    if not class_dist_per_iter:
        logger.warning(f"No valid class distribution data found in any model file for job {job_path}.")
        return empty_result

    try:
        arr_dist = np.stack(class_dist_per_iter, axis=0).transpose(1, 0)

        if class_res_per_iter and all(len(x) == arr_dist.shape[0] for x in class_res_per_iter):
            arr_res = np.stack(class_res_per_iter, axis=0).transpose(1, 0)
        else:
            logger.warning("Inconsistent or missing resolution data across iterations.")
            arr_res = np.array([])

        n_classes = arr_dist.shape[0]
        n_iterations = arr_dist.shape[1]

        fsc_res_final = fsc_res_per_iter[-1] if fsc_res_per_iter else np.array([])
        fsc_vals_final = fsc_vals_per_iter[-1] if fsc_vals_per_iter else np.array([])

        last_valid_model_path = model_star_files[valid_files_processed-1]
        last_star_data = parse_star(last_valid_model_path, lazy=True)
        class_path = []
        if "model_classes" in last_star_data:
             last_model_df = last_star_data["model_classes"].collect()
             if "_rlnReferenceImage" in last_model_df.columns:
                  class_files_relative = last_model_df["_rlnReferenceImage"].to_list()
                  for rel_path in class_files_relative:
                        mrc_part = rel_path.split('@')[-1]
                        abs_path = os.path.abspath(os.path.join(job_path, os.path.basename(mrc_part)))
                        if abs_path not in class_path:
                            class_path.append(abs_path)
             else:
                 logger.warning("'_rlnReferenceImage' column missing in last model file.")
        else:
             logger.warning("Could not find 'model_classes' in the last processed model file to extract MRC paths.")


        logger.info(f"Extracted {n_classes} classes over {n_iterations} iterations.")
        return (
            class_path,
            int(n_classes),
            int(n_iterations),
            arr_dist,
            arr_res,
            fsc_res_final,
            fsc_vals_final,
        )

    except ValueError as exc:
        logger.error("Error building final arrays from model file data.")
        report_error(exc)
        return empty_result
    except Exception as exc:
        logger.error("Unexpected error processing class data.")
        report_error(exc)
        return empty_result


# --- Streamlit UI Components ---

# ... [No changes to UI components till check_password] ...

def get_footer(show: bool = True) -> str:
    if not show:
        return ""
    footer_html = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;700&display=swap');
        a:link, a:visited {
            color: #007bff; background-color: transparent; text-decoration: none;
        }
        a:hover, a:active {
            color: #0056b3; background-color: transparent; text-decoration: underline;
        }
        .footer {
            position: relative;
            left: 0; bottom: 0;
            width: 100%; text-align: left;
            padding: 10px; margin-top: 20px;
            font-size: 14px;
            font-family: 'Segoe UI', sans-serif;
            border-top: 1px solid #e7e7e7;
            color: #6c757d;
        }
        .footer p { margin: 5px 0; }
        .footer a { font-weight: 500; }
        .bug-report-link { color: #336699 !important; }
    </style>
    <div class="footer">
        <p><b>Developed by <a href="www.dzyla.com" target="_blank" class="bug-report-link">
                Dawid Zyla
            </a></b></p>
        <p>
            <a href="https://github.com/dzyla/Follow_Relion_gracefully/issues" target="_blank" class="bug-report-link">
                Report bugs and suggestions
            </a>
        </p>
    </div>
    """
    return footer_html


def display_job_selection(job_type: str, file_suffix: str) -> Optional[str]:
    folder_root = st.session_state.get('default_job_folder')
    if not folder_root or not os.path.isdir(folder_root):
        st.warning("Project folder not set or invalid.")
        return None

    subfolders = get_subfolders(folder_root, job_type)
    if not subfolders:
        st.info(f"No subfolders found for job type '{job_type}'.")
        return None

    select_key = f"select_{job_type}_job"
    selected_job_folder_name = st.selectbox(
        f'Select {job_type} job:',
        options=subfolders,
        index=len(subfolders) - 1,
        key=select_key
        )

    if selected_job_folder_name:
        job_folder_path = os.path.join(folder_root, job_type, selected_job_folder_name)
        files = get_job_files(job_folder_path, file_suffix)

        if not files:
            st.warning(f"No files with suffix '{file_suffix}' found in {selected_job_folder_name}.")
            return None
        elif len(files) == 1:
             file_path = os.path.join(job_folder_path, files[0])
             return file_path
        else:
            file_select_key = f"select_{job_type}_file"
            selected_file = st.selectbox(
                 f"Select file in {selected_job_folder_name}:",
                 options=files,
                 index=len(files) -1,
                 key=file_select_key
            )
            if selected_file:
                return os.path.join(job_folder_path, selected_file)

    return None


def dynamic_folder_explorer(initial_root_path: str) -> str:
    # --- State Keys ---
    KEY_CURRENT_PATH = "folder_explorer_current_path_v2"
    KEY_CONFIRMED_PATH = "folder_explorer_confirmed_path_v2"
    KEY_SELECTBOX_SUBFOLDER = "folder_explorer_selectbox_v2"
    KEY_EXPANDER_STATE = "folder_explorer_expanded_v2"
    KEY_INITIALIZED = "folder_explorer_initialized_v2"

    # --- Helper for Path Validation ---
    def _normalize_validate_path(path_str: Optional[str]) -> Optional[str]:
        if not path_str:
            return None
        try:
            norm_path = os.path.abspath(os.path.expanduser(str(path_str)))
            return norm_path if os.path.isdir(norm_path) else None
        except Exception as e:
            logger.error(f"Error validating path '{path_str}': {e}")
            return None

    # --- Initialization ---
    if not st.session_state.get(KEY_INITIALIZED, False):
        validated_initial = _normalize_validate_path(initial_root_path)
        fallback_path = _normalize_validate_path("~") or "/"
        start_path = validated_initial or fallback_path

        st.session_state[KEY_CURRENT_PATH] = start_path
        st.session_state[KEY_CONFIRMED_PATH] = start_path
        st.session_state[KEY_EXPANDER_STATE] = True
        st.session_state[KEY_INITIALIZED] = True
        st.session_state["default_job_folder"] = start_path
        logger.info(f"Folder explorer initialized. Path: {start_path}")

    current_path_input = st.session_state.get(KEY_CURRENT_PATH, "/")
    confirmed_path = st.session_state.get(KEY_CONFIRMED_PATH, "/")
    current_path_validated = _normalize_validate_path(current_path_input)

    def handle_text_input_change():
        new_path = st.session_state[KEY_CURRENT_PATH]
        validated = _normalize_validate_path(new_path)
        if validated:
            st.session_state[KEY_CURRENT_PATH] = validated
        else:
            st.toast(f"Invalid path entered: {new_path}", icon="⚠️")

    def handle_go_up():
        validated = _normalize_validate_path(st.session_state[KEY_CURRENT_PATH])
        if validated:
            parent = os.path.dirname(validated)
            if parent and parent != validated:
                st.session_state[KEY_CURRENT_PATH] = parent

    def handle_selectbox_change():
        sel = st.session_state.get(KEY_SELECTBOX_SUBFOLDER)
        validated = _normalize_validate_path(st.session_state[KEY_CURRENT_PATH])
        if validated and sel and sel.startswith("📁 "):
            folder_name = sel.split("📁 ", 1)[1]
            target = os.path.join(validated, folder_name)
            validated_target = _normalize_validate_path(target)
            if validated_target:
                st.session_state[KEY_CURRENT_PATH] = validated_target

    def auto_confirm_logic():
        nonlocal confirmed_path
        if current_path_validated:
            pipeline_file = os.path.join(current_path_validated, "default_pipeline.star")
            has_pipeline = os.path.exists(pipeline_file)
            if current_path_validated != confirmed_path:
                st.session_state[KEY_CONFIRMED_PATH] = current_path_validated
                st.session_state["default_job_folder"] = current_path_validated
                if has_pipeline:
                    st.toast(f"Project folder selected: {os.path.basename(current_path_validated)}", icon="✅")
                    st.session_state[KEY_EXPANDER_STATE] = False
                else:
                    st.session_state[KEY_EXPANDER_STATE] = True
            elif has_pipeline and st.session_state.get(KEY_EXPANDER_STATE, True):
                st.session_state[KEY_EXPANDER_STATE] = False
            elif not has_pipeline and not st.session_state.get(KEY_EXPANDER_STATE, True):
                st.session_state[KEY_EXPANDER_STATE] = True
        else:
            st.session_state[KEY_EXPANDER_STATE] = True

    exp_state = st.session_state.get(KEY_EXPANDER_STATE, True)
    with st.sidebar.expander("**Project Location**", expanded=exp_state):
        if current_path_validated:
            pf = os.path.join(current_path_validated, "default_pipeline.star")
            if os.path.exists(pf):
                icon, text = "✅", "`default_pipeline.star` found"
            else:
                icon, text = "📁", "`default_pipeline.star` not found"
        else:
            icon, text = "❌", "Invalid folder path"
        st.markdown(f"**Status:** {icon} _{text}_")

        st.text_input(
            "Current Path",
            value=current_path_input,
            key=KEY_CURRENT_PATH,
            on_change=handle_text_input_change,
            help="Enter path and press Enter. Use ~ for home."
        )

        subfolders = []
        if current_path_validated:
            try:
                subfolders = [d for d in os.listdir(current_path_validated)
                              if os.path.isdir(os.path.join(current_path_validated, d))]
            except Exception:
                subfolders = []
        options = [""] + [f"📁 {d}" for d in subfolders]
        st.selectbox(
            "Subfolders:",
            options=options,
            key=KEY_SELECTBOX_SUBFOLDER,
            on_change=handle_selectbox_change,
            index=0,
            help="Select a subfolder to navigate into it."
        )

        st.button("⬆️ Go Up",
                  key="fe_go_up_button",
                  on_click=handle_go_up,
                  help="Navigate to parent directory")

        auto_confirm_logic()

    final = _normalize_validate_path(st.session_state.get(KEY_CONFIRMED_PATH))
    if final:
        if st.session_state.get("default_job_folder") != final:
            st.session_state["default_job_folder"] = final
        return final

    fallback = _normalize_validate_path("~") or "/"
    st.session_state[KEY_CONFIRMED_PATH] = fallback
    st.session_state["default_job_folder"] = fallback
    return fallback


def render_svg(svg_path: str) -> None:
    if not os.path.exists(svg_path):
        logger.error(f"SVG file not found: {svg_path}")
        st.sidebar.warning(f"SVG not found at {svg_path}")
        return
    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        b64_svg = base64.b64encode(svg_content.encode('utf-8')).decode("utf-8")
        html = f'<img src="data:image/svg+xml;base64,{b64_svg}" alt="SVG Image" style="max-width: 100%; height: auto;">'
        st.sidebar.markdown(html, unsafe_allow_html=True)
        st.sidebar.markdown('')
        st.sidebar.markdown('')
    except Exception as exc:
        logger.error(f"Failed to read or render SVG: {svg_path}")
        report_error(exc)
        st.sidebar.error("Could not display SVG image.")


def custom_css() -> str:
    css = """
    <style>
        section[data-testid="stSidebar"] {
            width: 450px !important;
        }
        div[data-testid="stSidebarUserContent"] {
            padding-top: 1rem;
        }
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stForm"] {
            border: 1px dashed #ccc;
            padding: 10px;
        }
         div[data-testid="block-container"] {
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
    </style>
    """
    return css


def check_password(password_arg: str) -> bool:
    PASSWORD_KEY = "password_input_field"
    PASSWORD_CORRECT_KEY = "password_correct_flag"
    PASSWORD_ARG_KEY = "password_correct_value_internal"

    if PASSWORD_ARG_KEY not in st.session_state:
         st.session_state[PASSWORD_ARG_KEY] = password_arg

    if st.session_state.get(PASSWORD_CORRECT_KEY, False):
        return True

    def password_entered_callback():
        entered_password = st.session_state.get(PASSWORD_KEY, "")
        correct_password = st.session_state.get(PASSWORD_ARG_KEY, "")

        if correct_password and hmac.compare_digest(entered_password, correct_password):
            st.session_state[PASSWORD_CORRECT_KEY] = True
            if PASSWORD_KEY in st.session_state:
                 del st.session_state[PASSWORD_KEY]
        else:
            st.session_state[PASSWORD_CORRECT_KEY] = False
            st.error("Incorrect password.")

    st.title('Follow Relion Gracefully :microscope:')
    st.text_input(
        "Password:",
        type="password",
        on_change=password_entered_callback,
        key=PASSWORD_KEY,
        value=""
    )

    return st.session_state.get(PASSWORD_CORRECT_KEY, False)


# --- General Dictionary/Data Utilities ---

def get_first_key(data_dict: Dict[Any, Any]) -> Optional[Any]:
    if not data_dict:
        return None
    try:
        return next(iter(data_dict))
    except StopIteration:
        return None


def get_values_from_first_key(data_dict: Dict[Any, Any]) -> Any:
    first_key = get_first_key(data_dict)
    if first_key is not None:
        return data_dict[first_key]
    return None


def get_unique_key(*args: Any) -> str:
    return "".join(map(str, args))


def convert_columns_to_float(df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], columns_to_convert: List[str]) -> pd.DataFrame:
    """
    Attempts to convert specified columns to numeric (float).
    Returns pandas DataFrame for backwards compatibility.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    for col_name in columns_to_convert:
        if col_name in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col_name]):
                original_type = df[col_name].dtype
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                if df[col_name].isnull().any():
                    if not pd.api.types.is_numeric_dtype(df[col_name]):
                         logger.warning(f"Column '{col_name}' (type {original_type}) could not be fully converted to numeric.")
                    else:
                         logger.debug(f"Column '{col_name}' converted to numeric; some values became NaN.")
        else:
             logger.warning(f"Column '{col_name}' not found in DataFrame for float conversion.")
    return df

# --- Interactive Plotting (Complex Function) ---

def interactive_scatter_plot(
    data_source: Union[str, Dict[str, Union[pd.DataFrame, pl.LazyFrame, pl.DataFrame]]],
    block_selector_options: Optional[List[str]] = None,
    default_block: Optional[str] = None,
    title_prefix: str = "Interactive Plot",
    max_rows_default: int = 50_000,
    allow_sampling: bool = True,
) -> None:
    """
    Interactive explorer for RELION-style STAR data using Polars and Datashader.

    Updated to handle Polars LazyFrames and use Datashader for large datasets.
    """

    COLOR_SCALES: Dict[str, List[str]] = {
        "Viridis": px.colors.sequential.Viridis,
        "Plasma": px.colors.sequential.Plasma,
        "Blues": px.colors.sequential.Blues,
        "Cividis": px.colors.sequential.Cividis,
        "Turbo": px.colors.sequential.Turbo,
        "Plotly (Qualitative)": px.colors.qualitative.Plotly,
        "Bold (Qualitative)": px.colors.qualitative.Bold,
    }

    # 1. LOAD DATA
    star_data = {}
    source_name = "Loaded Data"

    if isinstance(data_source, str):
        source_name = os.path.basename(data_source)
        try:
            star_data = parse_star(data_source, lazy=True)
        except Exception as exc:
            st.error(f"Failed to parse STAR file '{source_name}': {exc}")
            return
        if not star_data:
            st.error(f"No data blocks found in '{source_name}'.")
            return
        if default_block is None:
            default_block = next(
                (b for b in ["particles", "micrographs", "movies",
                             "model_classes", "helices", "global"]
                 if b in star_data),
                get_first_key(star_data),
            )
    elif isinstance(data_source, dict):
        star_data = data_source
        if not star_data:
            st.error("Provided data dictionary is empty.")
            return
        if default_block is None:
            default_block = get_first_key(star_data)
    else:
        st.error("`data_source` must be a path or dict of DataFrames/LazyFrames.")
        return

    # 2. BLOCK SELECTION
    blocks = block_selector_options or list(star_data.keys())
    if not blocks:
        st.error("No blocks available to plot.")
        return

    selected_block = (
        st.selectbox(
            f"Select Data Block ({source_name}):",
            options=blocks,
            index=blocks.index(default_block) if default_block in blocks else 0,
            key=f"{title_prefix}_block",
        )
        if len(blocks) > 1
        else blocks[0]
    )

    if selected_block not in star_data:
        st.error(f"Block '{selected_block}' not present.")
        return

    lf = star_data[selected_block]
    if isinstance(lf, pd.DataFrame):
        lf = pl.from_pandas(lf).lazy()
    elif isinstance(lf, pl.DataFrame):
        lf = lf.lazy()
    elif not isinstance(lf, pl.LazyFrame):
         st.error(f"Block '{selected_block}' is not a valid DataFrame type ({type(lf)}).")
         return

    # Count rows efficiently
    try:
        # Fetch schema to get column names without collecting
        schema = lf.collect_schema()
        columns_list = [""] + schema.names()

        # approximate count or exact count
        n_rows = lf.select(pl.len()).collect().item()

    except Exception as exc:
        st.error(f"Error inspecting data: {exc}")
        return

    if n_rows == 0:
        st.info(f"Block '{selected_block}' is empty.")
        return

    st.write(f"Original data: {n_rows:,} rows × {len(schema)} columns")

    # 3. CONFIG UI
    st.subheader("Plot Configuration")
    cfg_cols = st.columns([1, 1, 2])

    with cfg_cols[0]:
        plot_type = st.selectbox(
            "Plot Type",
            ["2D Scatter", "2D Histogram", "3D Scatter"],
            key=f"{title_prefix}_plot_type",
        )
    with cfg_cols[1]:
        coord_system = st.selectbox(
            "Coordinate System",
            ["Cartesian", "Polar"],
            disabled=(plot_type != "2D Scatter"),
            key=f"{title_prefix}_coord",
        )
        if plot_type != "2D Scatter":
            coord_system = "Cartesian"

    if len(columns_list) == 1:
        st.error("No columns found in block.")
        return

    st.markdown("---")
    axis_cols = st.columns(5 if plot_type == "3D Scatter" else 4)

    def _suggest(suffixes: List[str]) -> Optional[str]:
        for sfx in suffixes:
            match = next((c for c in schema.names()
                          if isinstance(c, str)
                          and c.lower().endswith(sfx.lower())), None)
            if match:
                return match
        return None

    x_default = _suggest(["X", "Rot"]) or schema.names()[0]
    y_default = _suggest(["Y", "Tilt"]) or schema.names()[min(1, len(schema)-1)]
    z_default = _suggest(["Z", "Psi"]) or schema.names()[min(2, len(schema)-1)]

    x_sel = axis_cols[0].selectbox(
        "Theta (θ)" if coord_system == "Polar" else "X-axis",
        columns_list,
        index=columns_list.index(x_default),
        key=f"{title_prefix}_x",
    )
    y_sel = axis_cols[1].selectbox(
        "Radius (r)" if coord_system == "Polar" else "Y-axis",
        columns_list,
        index=columns_list.index(y_default),
        key=f"{title_prefix}_y",
    )
    z_sel = None
    if plot_type == "3D Scatter":
        z_sel = axis_cols[2].selectbox(
            "Z-axis",
            columns_list,
            index=columns_list.index(z_default),
            key=f"{title_prefix}_z",
        )

    clr_col_idx = 3 if plot_type == "3D Scatter" else 2
    scheme_idx = 4 if plot_type == "3D Scatter" else 3
    colour_options = ["None"] + schema.names()
    colour_sel = axis_cols[clr_col_idx].selectbox(
        "Color by",
        colour_options,
        key=f"{title_prefix}_color",
    )
    colour_scheme = axis_cols[scheme_idx].selectbox(
        "Color Scheme",
        list(COLOR_SCALES.keys()),
        key=f"{title_prefix}_scheme",
    )

    # 4. DATA PREP (Datashader vs Plotly decision)
    use_datashader = n_rows > 50000 and plot_type == "2D Scatter" and coord_system == "Cartesian"

    # Determine if we should use WebGL (go.Scattergl)
    # Datashader is preferred for huge datasets (>50k by default or user preference)
    # Scattergl is preferred for intermediate datasets (e.g. 10k - 50k/100k)
    # Scatter is for small datasets (<10k)

    use_webgl = False
    if not use_datashader and n_rows > 10000:
        use_webgl = True
        st.info("Using WebGL for improved performance (10k-50k points).")

    if use_datashader:
        st.info("Large dataset detected (>50k rows). Using Datashader for high-performance rendering.")

    # Select columns to fetch
    cols_to_fetch = {x_sel, y_sel}
    if z_sel: cols_to_fetch.add(z_sel)
    if colour_sel != "None": cols_to_fetch.add(colour_sel)

    # Add index if needed for selection mapping?
    # For now, let's keep it simple. Datashader generates an image.

    # 5. FETCH & CAST
    try:
        # Cast to float for plotting
        lf_cast = lf.with_columns([
            pl.col(c).cast(pl.Float64, strict=False) for c in cols_to_fetch if c
        ])

        # For datashader we can stay in polars/arrow/pandas, but holoviews supports pandas best or dask.
        # Let's collect to Pandas for now as intermediate step, it's efficient enough for 1M rows in RAM.
        # But we wanted to avoid full load if possible. Datashader can work on Dask.
        # Here we will collect to Pandas as it is compatible with both Plotly and Holoviews/Datashader.
        # (Polars -> Pandas conversion is zero-copy for arrow-backed types mostly)

        if allow_sampling and not use_datashader and n_rows > max_rows_default:
             st.info(f"Plotting random sample of {max_rows_default} points.")
             df = lf_cast.select(list(cols_to_fetch)).collect().sample(n=max_rows_default, seed=42).to_pandas()
        else:
             # For datashader, we load full data (it handles millions easily)
             df = lf_cast.select(list(cols_to_fetch)).collect().to_pandas()

    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return

    # 6. PLOTTING

    if use_datashader:
        # Use Holoviews + Datashader
        try:
            points = hv.Points(df, kdims=[x_sel, y_sel])

            # Datashader cmap
            cmap = COLOR_SCALES[colour_scheme]

            if colour_sel != "None":
                 # Use mean aggregator for value if numeric
                 rasterized = datashade(points, aggregator=ds.mean(colour_sel), cmap=cmap)
            else:
                 rasterized = datashade(points, cmap=cmap)

            # Make it interactive with spread
            rasterized = dynspread(rasterized, threshold=0.5, max_px=4)

            rasterized = rasterized.opts(width=800, height=600, title=f"{title_prefix}: {selected_block} (Datashader)")

            st.bokeh_chart(hv.render(rasterized, backend='bokeh'), use_container_width=True)

            st.warning("High performance mode enabled. Lasso selection is disabled.")
            if st.button("Switch to Standard Interactive Plot (Slower, allows selection)"):
                # Force reload/re-render without datashader logic implies strictly needing a rerun with a flag,
                # but function is stateless. We can hint user to sample down.
                st.info("To select points, please reduce 'Max points' using the slider above to under 50,000, or use the sampling option.")

        except Exception as e:
            st.error(f"Datashader plotting failed: {e}")
            report_error(e)

    else:
        # FULL RESTORED PLOTLY IMPLEMENTATION
        temp_cols_to_drop = []
        perf_mode = st.checkbox("Slider performance mode", value=False, key=f"{title_prefix}_perf_mode")

        try:
            # --- 7. LOG + SAMPLING UI ---
            # Logic handled below within/without form
            pass

            # --- 8. DENSITY (if requested) ---
            dens_col = "_calculated_density"
            # Need to decide if density calc happens before or after form.
            # It depends on df which might be sampled.
            # We'll define a function to update data/density

            def calc_density(d, p_type, c_sys):
                try:
                    if p_type == "3D Scatter":
                        coords = d[[x_sel, y_sel, z_sel]].dropna()
                    elif p_type == "2D Scatter":
                        coords = d[[x_sel, y_sel]].dropna()
                    else:
                        coords = pd.DataFrame()

                    if len(coords) > 1:
                        # Use Pandas/Numpy for KDE
                        kde = gaussian_kde(coords.T)
                        d[dens_col] = np.nan
                        d.loc[coords.index, dens_col] = kde(coords.T)
                        return True
                    else:
                        st.warning("Need >1 point for density; falling back.")
                        return False
                except Exception as exc:
                    st.error(f"Density calculation failed: {exc}")
                    return False

            # Setup controls
            ctrl_cols_needed = 2 + (plot_type == "3D Scatter" and coord_system == "Cartesian")
            if allow_sampling: ctrl_cols_needed += 1

            # Variables to be set by controls
            log_x = False
            log_y = False
            log_z = False
            rows_to_plot = len(df)

            if perf_mode:
                with st.form(key=f"{title_prefix}_perf_form"):
                    ctrl = st.columns(ctrl_cols_needed)
                    log_x = ctrl[0].checkbox("Log X", key=f"{title_prefix}_logx", disabled=(coord_system == "Polar"))
                    log_y = ctrl[1].checkbox("Log Y", key=f"{title_prefix}_logy")
                    if plot_type == "3D Scatter" and coord_system == "Cartesian":
                        log_z = ctrl[2].checkbox("Log Z", key=f"{title_prefix}_logz")

                    if allow_sampling and len(df) > 1:
                        slider_col = ctrl[-1]
                        rows_to_plot = slider_col.slider(
                            f"Max points (total {len(df)})",
                            min_value=1 if len(df) <= 100 else 100,
                            max_value=len(df),
                            value=min(max_rows_default, len(df)),
                            key=f"{title_prefix}_sample_perf",
                        )
                    st.form_submit_button("Update Plot")
            else:
                ctrl = st.columns(ctrl_cols_needed)
                log_x = ctrl[0].checkbox("Log X", key=f"{title_prefix}_logx", disabled=(coord_system == "Polar"))
                log_y = ctrl[1].checkbox("Log Y", key=f"{title_prefix}_logy")
                if plot_type == "3D Scatter" and coord_system == "Cartesian":
                    log_z = ctrl[2].checkbox("Log Z", key=f"{title_prefix}_logz")

                if allow_sampling and len(df) > 1:
                    slider_col = ctrl[-1]
                    rows_to_plot = slider_col.slider(
                        f"Max points (total {len(df)})",
                        min_value=1 if len(df) <= 100 else 100,
                        max_value=len(df),
                        value=min(max_rows_default, len(df)),
                        key=f"{title_prefix}_sample",
                    )

            # Apply sampling
            if rows_to_plot < len(df):
                df = df.sample(n=rows_to_plot, seed=42)
                st.info(f"Plotting {rows_to_plot} points.")

            # Apply Density
            colour_col_to_plot = colour_sel
            if colour_sel == "Density":
                if calc_density(df, plot_type, coord_system):
                    colour_col_to_plot = dens_col
                    temp_cols_to_drop.append(dens_col)
                else:
                    colour_col_to_plot = None
            elif colour_sel == "None":
                colour_col_to_plot = None

            # --- 9. PLOT ARGS ---
            hover_data = {
                col: True for col in df.columns
                if col not in {x_sel, y_sel, z_sel, colour_sel}
            }
            plot_kwargs = {"hover_data": hover_data}

            scale = COLOR_SCALES[colour_scheme]
            if colour_col_to_plot and colour_col_to_plot in df.columns:
                if pd.api.types.is_numeric_dtype(df[colour_col_to_plot]):
                    plot_kwargs["color_continuous_scale"] = scale
                else:
                    plot_kwargs["color_discrete_sequence"] = scale
            elif colour_col_to_plot: # Should be None if not found, but safe check
                 colour_col_to_plot = None

            label_map = {x_sel: x_sel, y_sel: y_sel}
            if z_sel: label_map[z_sel] = z_sel
            if colour_col_to_plot == dens_col: label_map[colour_col_to_plot] = "Density"
            plot_kwargs["labels"] = label_map

            title = f"{title_prefix}: {selected_block}"
            fig = None

            # --- 10. PLOT CONSTRUCTION ---
            if plot_type == "2D Scatter":
                if coord_system == "Polar":
                    fig = px.scatter_polar(
                        df, theta=x_sel, r=y_sel, color=colour_col_to_plot,
                        title=title, **plot_kwargs
                    )
                    if log_y: fig.update_layout(polar_radialaxis_type="log")
                else:
                    if use_webgl:
                        # Use go.Scattergl manually since px.scatter doesn't always expose it easily via args
                        # Construct trace manually or use px and update traces
                        fig = px.scatter(
                            df, x=x_sel, y=y_sel, color=colour_col_to_plot,
                            title=title, **plot_kwargs
                        )
                        # Switch to Scattergl
                        fig.update_traces(mode='markers') # Ensure markers mode
                        for data in fig.data:
                            data.type = 'scattergl'
                    else:
                        fig = px.scatter(
                            df, x=x_sel, y=y_sel, color=colour_col_to_plot,
                            title=title, **plot_kwargs
                        )

                    if log_x: fig.update_xaxes(type="log")
                    if log_y: fig.update_yaxes(type="log")

                    keep_ratio = st.checkbox("Keep data aspect ratio?", value=True, key=f"{title_prefix}_aspect")
                    if keep_ratio:
                        fig.update_yaxes(scaleanchor="x", scaleratio=1)

            elif plot_type == "2D Histogram":
                fig = go.Figure(
                    go.Histogram2d(
                        x=df[x_sel], y=df[y_sel], colorscale=scale,
                    )
                )
                fig.update_layout(
                    title=title, xaxis_title=x_sel, yaxis_title=y_sel,
                    coloraxis_colorbar=dict(title="Count"),
                )
                if log_x: fig.update_xaxes(type="log")
                if log_y: fig.update_yaxes(type="log")

            else:  # "3D Scatter"
                fig = px.scatter_3d(
                    df, x=x_sel, y=y_sel, z=z_sel,
                    color=colour_col_to_plot,
                    title=title, **plot_kwargs
                )
                scene = {}
                if log_x: scene["xaxis_type"] = "log"
                if log_y: scene["yaxis_type"] = "log"
                if log_z: scene["zaxis_type"] = "log"
                if scene: fig.update_layout(scene=scene)

            fig.update_layout(height=700 if plot_type == "3D Scatter" else 600)
            if plot_type == "2D Scatter" and coord_system == "Cartesian":
                fig.update_layout(dragmode="select")

            # --- 11. DISPLAY & SELECTION ---
            plot_key = f"{title_prefix}_{selected_block}_plot"
            if "plotly_selection" not in st.session_state:
                st.session_state.plotly_selection = {}

            current_selection = st.session_state.plotly_selection.get(plot_key, {"points": []})

            event_data = st.plotly_chart(
                fig, use_container_width=True, key=plot_key, on_select="rerun"
            )

            if event_data and event_data.selection:
                st.session_state.plotly_selection[plot_key] = event_data.selection
                current_selection = event_data.selection
            elif event_data and event_data.selection is None:
                st.session_state.plotly_selection[plot_key] = {"points": []}
                current_selection = {"points": []}

            selected_indices = []
            if current_selection and current_selection.get("points"):
                try:
                    # Map plot points back to dataframe indices
                    sel_point_idx = [pt["point_index"] for pt in current_selection["points"]]
                    # Use iloc to get the corresponding rows in current view (df)
                    # Then get their indices (which should match original if not reset, but we did reset in some cases)
                    # For robust mapping, we rely on df.iloc[sel_point_idx].index
                    selected_indices = df.iloc[sel_point_idx].index.tolist()
                except Exception as exc:
                    st.error(f"Selection processing failed: {exc}")
                    report_error(exc)

            # --- 12. DOWNLOAD SELECTED SUBSET ---
            if selected_indices:
                st.markdown("---")
                st.subheader("Save Selection")

                safe_block = re.sub(r"\W+", "_", selected_block)
                file_name = f"selection_{safe_block}.star"

                try:
                    # We need to filter the original data.
                    # If we loaded from Polars, 'lf' is the LazyFrame.
                    # We can use the selected indices if we assume 'df' has same index as source.
                    # BUT 'df' might be a sample.
                    # If df is sample, indices might be preserved if we didn't reset_index drop=True.
                    # Polars to Pandas usually preserves index as RangeIndex 0..N.

                    # If we used sampling:
                    # st.info(f"Plotting random sample of {max_rows_default} points.")
                    # The selected indices refer to the SAMPLE.
                    # We can only save the selected SAMPLE.

                    subset_df = df.loc[selected_indices].copy()

                    # Convert subset back to Star format
                    # Create a dict with just this block (or others too? Original code kept others)
                    # Original code:
                    # out_dict = {k: v.copy() for k, v in star_data.items() if k != selected_block}
                    # out_dict[selected_block] = subset_df
                    # star_doc = star_from_df(out_dict)

                    # We only have 'star_data' which contains LazyFrames or DataFrames.
                    # Reconstructing the whole file might be expensive if lazy.
                    # Let's just save the selection block for now, or try to keep others if easy.

                    out_dict = {selected_block: subset_df}

                    star_content = star_from_df(out_dict)

                    st.download_button(
                        label=f"Download {len(selected_indices)} selected rows as **{file_name}**",
                        data=star_content,
                        file_name=file_name,
                        mime="text/plain",
                        key=f"{title_prefix}_dl",
                    )
                except Exception as exc:
                    st.error(f"Subset save failed: {exc}")
                    report_error(exc)

        except Exception as exc:
            st.error(f"Plot creation failed: {exc}")
            report_error(exc)
        finally:
            # Cleanup
            if temp_cols_to_drop:
                df.drop(columns=temp_cols_to_drop, errors="ignore", inplace=True)
