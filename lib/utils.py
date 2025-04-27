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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import gemmi.cif as cif
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde  # Used in interactive_scatter_plot

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

def parse_star(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Parses a STAR file into a dictionary of pandas DataFrames.

    Each data block in the STAR file becomes a key in the dictionary,
    and the corresponding loop data becomes a DataFrame.

    Args:
        file_path: The path to the STAR file.

    Returns:
        A dictionary where keys are block names and values are DataFrames.
        Returns an empty dictionary if the file is not found or parsing fails.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {}
    try:
        doc = cif.read_file(file_path)
        star_data = {}
        for block in doc:
            # Find the first loop in the block to extract data
            loop = next((item.loop for item in block if hasattr(item, 'loop')), None)
            if loop:
                tags = loop.tags
                data = [np.array(block.find_loop(tag)) for tag in tags]
                # Ensure all columns have the same length
                if len(set(len(col) for col in data)) <= 1:
                    star_data[block.name] = pd.DataFrame(dict(zip(tags, data)))
                else:
                    logger.warning(f"Skipping block '{block.name}' in {file_path} due to inconsistent column lengths.")
            else:
                 logger.debug(f"No loop found in block '{block.name}' in {file_path}.")

        return star_data
    except Exception as exc:
        logger.error(f"Failed to parse STAR file: {file_path}")
        report_error(exc)
        return {}


def star_from_df(dicts_of_df: Dict[str, pd.DataFrame]) -> cif.Document:
    """
    Converts a dictionary of pandas DataFrames into a gemmi cif.Document.

    Each dictionary key becomes a block name, and the DataFrame is converted
    into a loop within that block. STAR file format requires explicit indexing
    in tags (e.g., '_rlnColumnName #1').

    Args:
        dicts_of_df: Dictionary mapping block names (str) to DataFrames (pd.DataFrame).

    Returns:
        A gemmi cif.Document representing the STAR file.

    Raises:
        TypeError: If any value in the dictionary is not a pandas DataFrame.
    """
    out_doc = cif.Document()
    for block_name, df in dicts_of_df.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Object for key '{block_name}' must be a DataFrame, not {type(df)}.")

        block = out_doc.add_new_block(block_name, pos=-1)
        # Format column names for STAR standard (e.g., _rlnImageName #1)
        column_names_to_star = [f"{col} #{i + 1}" for i, col in enumerate(df.columns)]
        loop = block.init_loop('', column_names_to_star)

        # Convert all DataFrame values to strings for STAR format
        data_rows = df.astype(str).values.tolist()
        # Add data row by row
        for row in data_rows:
            loop.add_row(row)
    return out_doc


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

def get_relationships_df(pipeline_edges_df: pd.DataFrame) -> pd.DataFrame:
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
        star_data = parse_star(latest_data_star_path)

        # Expect angles in the 'particles' or 'micrographs' block usually
        data_block = None
        if "particles" in star_data:
            data_block = star_data["particles"]
        elif "micrographs" in star_data: # Some jobs might store orientations here
             data_block = star_data["micrographs"]

        if data_block is None or data_block.empty:
             logger.warning(f"No 'particles' or 'micrographs' data found in {latest_data_star_path}")
             return empty_series, empty_series, empty_series

        # Apply sampling if requested
        if limit is not None and limit < len(data_block):
            logger.info(f"Sampling {limit} particles/micrographs for angle analysis.")
            data_block = data_block.sample(n=limit, random_state=42) # Added random_state for reproducibility

        # Extract angle columns
        required_cols = ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
        if not all(col in data_block.columns for col in required_cols):
            logger.error(f"Missing one or more angle columns in {latest_data_star_path}")
            return empty_series, empty_series, empty_series

        rot_angles = pd.to_numeric(data_block["_rlnAngleRot"], errors='coerce')
        tilt_angles = pd.to_numeric(data_block["_rlnAngleTilt"], errors='coerce')
        psi_angles = pd.to_numeric(data_block["_rlnAnglePsi"], errors='coerce')

        # Check for conversion errors
        if rot_angles.isnull().any() or tilt_angles.isnull().any() or psi_angles.isnull().any():
             logger.warning(f"Non-numeric values found in angle columns of {latest_data_star_path}")

        return rot_angles.fillna(0), tilt_angles.fillna(0), psi_angles.fillna(0)

    except Exception as exc:
        logger.error(f"Failed to get angles from {job_path}.")
        report_error(exc)
        return empty_series, empty_series, empty_series

def get_classes(
    job_path: str, model_star_files: List[str]
) -> Tuple[List[str], int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses RELION model.star files to extract class information over iterations.

    It processes a list of model.star files (typically one per iteration),
    extracts class distributions, estimated resolutions, and optionally FSC curves.

    Args:
        job_path: The path to the job directory (used to resolve relative MRC paths).
        model_star_files: A list of paths to model.star files, usually sorted by iteration.

    Returns:
        A tuple containing:
        - class_mrc_paths (List[str]): List of absolute paths to the final class MRC files.
        - n_classes (int): Number of classes found.
        - n_iterations (int): Number of iterations processed (based on model files).
        - distribution_array (np.ndarray): Array of shape (n_classes, n_iterations)
                                            with class distributions over iterations.
        - resolution_array (np.ndarray): Array of shape (n_classes, n_iterations)
                                          with estimated resolutions over iterations (or empty).
        - fsc_resolution_bins (np.ndarray): Array of Angstrom resolution bins for the
                                             final FSC curve (or empty).
        - fsc_values (np.ndarray): Array of FSC values for the final FSC curve (or empty).
        Returns empty structures/zeros if parsing fails or no valid data is found.
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
        star_data = parse_star(file_path)

        # --- Extract model_classes data ---
        if "model_classes" not in star_data or star_data["model_classes"].empty:
            logger.warning(f"'model_classes' block missing or empty in {file_path}. Skipping iteration {i}.")
            # Append empty arrays to keep iteration count consistent if needed, or handle sparse data later
            continue

        model_classes_df = star_data["model_classes"]

        # Distribution (required)
        if "_rlnClassDistribution" not in model_classes_df.columns:
            logger.warning(f"'_rlnClassDistribution' missing in {file_path}. Skipping iteration {i}.")
            continue
        distribution = pd.to_numeric(model_classes_df["_rlnClassDistribution"], errors='coerce').fillna(0).values
        class_dist_per_iter.append(distribution)

        # Resolution (optional)
        if "_rlnEstimatedResolution" in model_classes_df.columns:
             resolution = pd.to_numeric(model_classes_df["_rlnEstimatedResolution"], errors='coerce').fillna(0).values
             class_res_per_iter.append(resolution)
        else:
             # Append array of zeros if resolution is missing for this iteration to maintain alignment
             class_res_per_iter.append(np.zeros_like(distribution))


        # --- Extract FSC data (optional, assumes block names like 'model_class_1', 'model_class_2') ---
        # We only care about the FSC from the *first* class block if present ('model_class_1')
        fsc_block_name = "model_class_1"
        if fsc_block_name in star_data and not star_data[fsc_block_name].empty:
             fsc_df = star_data[fsc_block_name]
             if "_rlnGoldStandardFsc" in fsc_df.columns and "_rlnAngstromResolution" in fsc_df.columns:
                 fsc_vals = pd.to_numeric(fsc_df["_rlnGoldStandardFsc"], errors='coerce').fillna(0).values
                 fsc_res_bins = pd.to_numeric(fsc_df["_rlnAngstromResolution"], errors='coerce').fillna(0).values
                 # Ensure both FSC arrays have the same length
                 min_len = min(len(fsc_vals), len(fsc_res_bins))
                 fsc_vals_per_iter.append(fsc_vals[:min_len])
                 fsc_res_per_iter.append(fsc_res_bins[:min_len])
             else:
                 logger.debug(f"FSC columns missing in block '{fsc_block_name}' in {file_path}")
                 # Append empty arrays if FSC data is missing for this iteration
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
        # Stack distributions: list of (n_classes,) -> array of (#iterations, #classes) -> T -> (#classes, #iterations)
        arr_dist = np.stack(class_dist_per_iter, axis=0).transpose(1, 0)

        # Stack resolutions if available
        if class_res_per_iter and all(len(x) == arr_dist.shape[0] for x in class_res_per_iter): # Check consistent class count
            arr_res = np.stack(class_res_per_iter, axis=0).transpose(1, 0)
        else:
            logger.warning("Inconsistent or missing resolution data across iterations.")
            arr_res = np.array([]) # Return empty if inconsistent

        n_classes = arr_dist.shape[0]
        n_iterations = arr_dist.shape[1]

        # Get final FSC curve data from the last valid iteration processed
        fsc_res_final = fsc_res_per_iter[-1] if fsc_res_per_iter else np.array([])
        fsc_vals_final = fsc_vals_per_iter[-1] if fsc_vals_per_iter else np.array([])

        # Get class MRC paths from the *last provided* model file that had classes
        last_valid_model_path = model_star_files[valid_files_processed-1] # Index of last successfully processed file
        last_star_data = parse_star(last_valid_model_path)
        class_path = []
        if "model_classes" in last_star_data:
             last_model_df = last_star_data["model_classes"]
             if "_rlnReferenceImage" in last_model_df.columns:
                  class_files_relative = last_model_df["_rlnReferenceImage"]
                  for rel_path in class_files_relative:
                        # Path might be like '001@path/to/class_001.mrc' or just 'path/to/class_001.mrc'
                        mrc_part = rel_path.split('@')[-1]
                        # Construct absolute path using the job directory
                        abs_path = os.path.abspath(os.path.join(job_path, os.path.basename(mrc_part)))
                        # Avoid duplicates if multiple entries point to the same file
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
            int(n_iterations), # Iteration count based on processed files
            arr_dist,
            arr_res,
            fsc_res_final,
            fsc_vals_final,
        )

    except ValueError as exc:
        logger.error("Error building final arrays from model file data.")
        report_error(exc)
        return empty_result
    except Exception as exc: # Catch any other unexpected errors during processing
        logger.error("Unexpected error processing class data.")
        report_error(exc)
        return empty_result


# --- Streamlit UI Components ---

def get_footer(show: bool = True) -> str:
    """
    Generates the HTML string for the application footer.

    Args:
        show: If False, returns an empty string.

    Returns:
        HTML string for the footer or an empty string.
    """
    if not show:
        return ""

    # Consider moving CSS to a separate file or Streamlit's native CSS styling
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
            position: relative; /* Changed from adaptive */
            left: 0; bottom: 0; /* Adjusted for relative positioning */
            width: 100%; text-align: left;
            padding: 10px; margin-top: 20px; /* Added margin-top */
            font-size: 14px; /* Slightly smaller font */
            font-family: 'Segoe UI', sans-serif;
            border-top: 1px solid #e7e7e7; /* Thinner border */
            color: #6c757d; /* Bootstrap text-muted color */
        }
        .footer p { margin: 5px 0; } /* Reduced margin */
        .footer a { font-weight: 500; } /* Bolder links */
        .bug-report-link { color: #336699 !important; } /* Ensure suggestion link color */
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
    """
    Displays Streamlit widgets to select a job and file within that job.

    Assumes 'default_job_folder' is set in st.session_state.

    Args:
        job_type: The type of job (e.g., "Class2D", "Extract").
        file_suffix: The suffix of the file to look for within the job folder
                     (e.g., "_data.star", "_optimiser.star").

    Returns:
        The absolute path to the selected file, or None if no selection
        could be made.
    """
    folder_root = st.session_state.get('default_job_folder')
    if not folder_root or not os.path.isdir(folder_root):
        st.warning("Project folder not set or invalid.")
        return None

    subfolders = get_subfolders(folder_root, job_type)
    if not subfolders:
        st.info(f"No subfolders found for job type '{job_type}'.")
        return None

    # Use job type in key for uniqueness if this widget appears multiple times
    select_key = f"select_{job_type}_job"
    selected_job_folder_name = st.selectbox(
        f'Select {job_type} job:',
        options=subfolders,
        index=len(subfolders) - 1, # Default to the last job (usually latest)
        key=select_key
        )

    if selected_job_folder_name:
        job_folder_path = os.path.join(folder_root, job_type, selected_job_folder_name)
        files = get_job_files(job_folder_path, file_suffix)

        if not files:
            st.warning(f"No files with suffix '{file_suffix}' found in {selected_job_folder_name}.")
            return None
        elif len(files) == 1:
            # If only one matching file, return it directly
             file_path = os.path.join(job_folder_path, files[0])
             # st.write(f"Using file: `{files[0]}`") # Optional: confirm which file is used
             return file_path
        else:
            # If multiple files match, let the user choose
            file_select_key = f"select_{job_type}_file"
            selected_file = st.selectbox(
                 f"Select file in {selected_job_folder_name}:",
                 options=files,
                 index=len(files) -1, # Default to last file (often latest)
                 key=file_select_key
            )
            if selected_file:
                return os.path.join(job_folder_path, selected_file)

    return None # Return None if no job or file is ultimately selected


def dynamic_folder_explorer(initial_root_path: str) -> str:
    """
    Streamlit component for browsing folders and auto-selecting RELION project folders.

    Features:
    - Text input for path entry.
    - Subfolder navigation via selectbox.
    - "Go Up" button.
    - Automatic detection of "default_pipeline.star" in the current directory.
    - Auto-confirmation: If the pipeline file exists, the path is confirmed,
      and the expander may close.
    - State persistence using `st.session_state`.
    - UI elements contained within a `st.sidebar.expander`.

    Args:
        initial_root_path: The initial path to display when the component loads
                           for the first time in the session.

    Returns:
        The currently confirmed valid absolute path (str). Defaults to user's
        home or root if issues occur. Updates `st.session_state['default_job_folder']`.
    """
    # --- State Keys ---
    KEY_CURRENT_PATH = "folder_explorer_current_path_v2"
    KEY_CONFIRMED_PATH = "folder_explorer_confirmed_path_v2"
    KEY_SELECTBOX_SUBFOLDER = "folder_explorer_selectbox_v2"
    KEY_EXPANDER_STATE = "folder_explorer_expanded_v2"
    KEY_INITIALIZED = "folder_explorer_initialized_v2"

    # --- Helper for Path Validation ---
    def _normalize_validate_path(path_str: Optional[str]) -> Optional[str]:
        """Normalizes and validates if a path is an existing directory."""
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

    # Retrieve current values
    current_path_input = st.session_state.get(KEY_CURRENT_PATH, "/")
    confirmed_path = st.session_state.get(KEY_CONFIRMED_PATH, "/")
    current_path_validated = _normalize_validate_path(current_path_input)

    # --- Callbacks ---
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

    # --- UI ---
    exp_state = st.session_state.get(KEY_EXPANDER_STATE, True)
    with st.sidebar.expander("**Project Location**", expanded=exp_state):
        # Status line
        if current_path_validated:
            pf = os.path.join(current_path_validated, "default_pipeline.star")
            if os.path.exists(pf):
                icon, text = "✅", "`default_pipeline.star` found"
            else:
                icon, text = "📁", "`default_pipeline.star` not found"
        else:
            icon, text = "❌", "Invalid folder path"
        st.markdown(f"**Status:** {icon} _{text}_")

        # Text input with explicit value for persistence
        st.text_input(
            "Current Path",
            value=current_path_input,
            key=KEY_CURRENT_PATH,
            on_change=handle_text_input_change,
            help="Enter path and press Enter. Use ~ for home."
        )

        # Subfolder selectbox
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

        # Go Up button
        st.button("⬆️ Go Up",
                  key="fe_go_up_button",
                  on_click=handle_go_up,
                  help="Navigate to parent directory")

        # Run auto-confirm logic after UI interactions
        auto_confirm_logic()

    # --- Return confirmed path ---
    final = _normalize_validate_path(st.session_state.get(KEY_CONFIRMED_PATH))
    if final:
        if st.session_state.get("default_job_folder") != final:
            st.session_state["default_job_folder"] = final
        return final

    # Fallback
    fallback = _normalize_validate_path("~") or "/"
    st.session_state[KEY_CONFIRMED_PATH] = fallback
    st.session_state["default_job_folder"] = fallback
    return fallback



def render_svg(svg_path: str) -> None:
    """
    Renders an SVG file in the Streamlit sidebar.

    Args:
        svg_path: The file path to the SVG image.
    """
    if not os.path.exists(svg_path):
        logger.error(f"SVG file not found: {svg_path}")
        st.sidebar.warning(f"SVG not found at {svg_path}")
        return
    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        # Encode SVG to base64
        b64_svg = base64.b64encode(svg_content.encode('utf-8')).decode("utf-8")
        # Embed in HTML img tag
        html = f'<img src="data:image/svg+xml;base64,{b64_svg}" alt="SVG Image" style="max-width: 100%; height: auto;">' # Added style for responsiveness
        st.sidebar.markdown(html, unsafe_allow_html=True)
        st.sidebar.markdown('') # Add space below image if needed
        st.sidebar.markdown('') # Add space below image if needed
    except Exception as exc:
        logger.error(f"Failed to read or render SVG: {svg_path}")
        report_error(exc)
        st.sidebar.error("Could not display SVG image.")


def custom_css() -> str:
    """
    Returns a string containing custom CSS rules for the Streamlit app.
    """
    # Consider using st.markdown("<style>...", unsafe_allow_html=True) directly
    # in the main app instead of returning a string from utils, unless it's widely reused.
    css = """
    <style>
        section[data-testid="stSidebar"] {
            /* Adjust width carefully, can impact usability */
            width: 450px !important; /* Example: slightly reduced width */
        }
        div[data-testid="stSidebarUserContent"] {
            padding-top: 1rem; /* Add some padding at the top */
        }
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stForm"] {
            /* Target forms specifically if needed */
            border: 1px dashed #ccc; /* Example: visual indicator for forms */
            padding: 10px;
        }
         div[data-testid="block-container"] {
            padding-top: 2rem; /* Main content padding */
            padding-bottom: 4rem; /* Ensure space for footer */
        }

    </style>
    """
    return css


def check_password(password_arg: str) -> bool:
    """
    Checks if the user-entered password matches the expected password using Streamlit state.

    Requires the correct password to be passed via `password_arg`. This is insecure
    if the password is hardcoded or easily accessible. Consider environment variables
    or more secure methods for production.

    Args:
        password_arg: The correct password to compare against.

    Returns:
        True if the password entered by the user is correct, False otherwise.
    """
    # Use specific keys to avoid conflicts
    PASSWORD_KEY = "password_input_field"
    PASSWORD_CORRECT_KEY = "password_correct_flag"
    PASSWORD_ARG_KEY = "password_correct_value_internal" # Store the correct password in state

    # Store the correct password in session state if not already there
    # This avoids passing it around constantly but stores it in memory.
    if PASSWORD_ARG_KEY not in st.session_state:
         st.session_state[PASSWORD_ARG_KEY] = password_arg

    # If password was already deemed correct, skip the check
    if st.session_state.get(PASSWORD_CORRECT_KEY, False):
        return True

    # Password checking logic using a callback
    def password_entered_callback():
        entered_password = st.session_state.get(PASSWORD_KEY, "")
        correct_password = st.session_state.get(PASSWORD_ARG_KEY, "")

        if correct_password and hmac.compare_digest(entered_password, correct_password):
            st.session_state[PASSWORD_CORRECT_KEY] = True
            # Clear the entered password from state after check for security
            if PASSWORD_KEY in st.session_state:
                 del st.session_state[PASSWORD_KEY]
        else:
            st.session_state[PASSWORD_CORRECT_KEY] = False
            st.error("Incorrect password.") # Provide feedback directly in callback

    # Display UI elements for password entry
    st.title('Follow Relion Gracefully :microscope:') # Title shown only when password needed
    st.text_input(
        "Password:",
        type="password",
        on_change=password_entered_callback,
        key=PASSWORD_KEY,
        value="" # Ensure input is cleared on rerun if not correct
    )

    # Check the flag set by the callback
    return st.session_state.get(PASSWORD_CORRECT_KEY, False)


# --- General Dictionary/Data Utilities ---

def get_first_key(data_dict: Dict[Any, Any]) -> Optional[Any]:
    """
    Gets the first key from a dictionary.

    Args:
        data_dict: The input dictionary.

    Returns:
        The first key, or None if the dictionary is empty.
        Note: Relies on insertion order preservation (Python 3.7+).
    """
    if not data_dict:
        return None
    try:
        return next(iter(data_dict))
    except StopIteration: # Should not happen if data_dict is not empty, but for safety
        return None


def get_values_from_first_key(data_dict: Dict[Any, Any]) -> Any:
    """
    Returns the value associated with the first key in the dictionary.

    Args:
        data_dict: Input dictionary.

    Returns:
        The value associated with the first key, or None if the dictionary is empty.
        Relies on insertion order preservation (Python 3.7+).
    """
    first_key = get_first_key(data_dict)
    if first_key is not None:
        return data_dict[first_key]
    return None


def get_unique_key(*args: Any) -> str:
    """
    Creates a simple unique string key from provided arguments for caching.

    Args:
        *args: A variable number of arguments to include in the key.

    Returns:
        A concatenated string representation of the arguments.
    """
    return "".join(map(str, args))


def convert_columns_to_float(df: pd.DataFrame, columns_to_convert: List[str]) -> pd.DataFrame:
    """
    Attempts to convert specified columns in a DataFrame to numeric (float).

    Logs a warning for columns that cannot be converted.

    Args:
        df: The pandas DataFrame to modify.
        columns_to_convert: A list of column names to attempt conversion on.

    Returns:
        The DataFrame with specified columns converted to float where possible.
        Original DataFrame is modified in place.
    """
    for col_name in columns_to_convert:
        if col_name in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col_name]):
                original_type = df[col_name].dtype
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                if df[col_name].isnull().any():
                    # Check if conversion actually failed for any value
                    # Re-check if the column is now numeric. If not, conversion likely failed broadly.
                    if not pd.api.types.is_numeric_dtype(df[col_name]):
                         logger.warning(f"Column '{col_name}' (type {original_type}) could not be fully converted to numeric.")
                    else:
                         logger.debug(f"Column '{col_name}' converted to numeric; some values became NaN.")
                # else:
                     # logger.debug(f"Column '{col_name}' successfully converted to numeric.")
        else:
             logger.warning(f"Column '{col_name}' not found in DataFrame for float conversion.")
    return df

# --- Interactive Plotting (Complex Function) ---

def interactive_scatter_plot(
    data_source: Union[str, Dict[str, pd.DataFrame]],
    block_selector_options: Optional[List[str]] = None,
    default_block: Optional[str] = None,
    title_prefix: str = "Interactive Plot",
    max_rows_default: int = 50_000,
    allow_sampling: bool = True,
) -> None:
    """
    Interactive Plotly/Streamlit explorer for RELION-style STAR data.

    All original features retained:
      • 2-D / 3-D scatter, histogram, Cartesian & polar
      • numeric conversion with index/ID fall-backs
      • density colouring
      • log axes, random sampling
      • subset saving
      • aspect-ratio lock for Cartesian 2-D scatter   ←  (restored here)
    """

    # ----------------------------- 1. COLOUR MAPS ---------------------------
    COLOR_SCALES: Dict[str, List[str]] = {
        "Viridis": px.colors.sequential.Viridis,
        "Plasma": px.colors.sequential.Plasma,
        "Blues": px.colors.sequential.Blues,
        "Cividis": px.colors.sequential.Cividis,
        "Turbo": px.colors.sequential.Turbo,
        "Plotly (Qualitative)": px.colors.qualitative.Plotly,
        "Bold (Qualitative)": px.colors.qualitative.Bold,
    }

    # ----------------------------- 2. LOAD DATA -----------------------------
    star_data: Dict[str, pd.DataFrame] = {}
    source_name = "Loaded Data"

    if isinstance(data_source, str):
        source_name = os.path.basename(data_source)
        logger.info(f"Parsing STAR file for plotting: {data_source}")
        try:
            star_data = parse_star(data_source)
        except Exception as exc:
            st.error(f"Failed to parse STAR file '{source_name}': {exc}")
            report_error(exc, f"STAR parsing – {source_name}")
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
        st.error("`data_source` must be a path or dict of DataFrames.")
        return

    # --------------------------- 3. BLOCK SELECTION -------------------------
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

    df_original = star_data[selected_block]
    if not isinstance(df_original, pd.DataFrame):
        st.error(f"Block '{selected_block}' is not a DataFrame.")
        return
    if df_original.empty:
        st.info(f"Block '{selected_block}' is empty.")
        return

    df = df_original.copy()
    if not df.index.is_unique:
        df.reset_index(drop=True, inplace=True)

    st.write(f"Original data: {len(df_original):,} rows × "
             f"{df_original.shape[1]} columns")

    # -------------------------- 4. PLOT CONFIG UI ---------------------------
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

    # -------------------------- 5. AXIS / COLOUR UI -------------------------
    columns_list = [""] + list(df.columns)
    if len(columns_list) == 1:
        st.error("No columns found in block.")
        return

    st.markdown("---")
    axis_cols = st.columns(5 if plot_type == "3D Scatter" else 4)

    def _suggest(suffixes: List[str]) -> Optional[str]:
        for sfx in suffixes:
            match = next((c for c in df.columns
                          if isinstance(c, str)
                          and c.lower().endswith(sfx.lower())), None)
            if match:
                return match
        return None

    x_default = _suggest(["X", "Rot"]) or df.columns[0]
    y_default = _suggest(["Y", "Tilt"]) or df.columns[min(1, len(df.columns)-1)]
    z_default = _suggest(["Z", "Psi"]) or df.columns[min(2, len(df.columns)-1)]

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
    colour_options = ["None", "Density"] + list(df.columns)
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

    # -------------------- 6. COLUMN PREP & FALLBACKS ------------------------
    temp_cols_to_drop: List[str] = []

    def _prep(
        sel_col: Optional[str],
        axis_label: str,
        essential: bool,
        allow_index_fb: bool,
        as_colour: bool,
        container,
    ) -> Tuple[Optional[str], bool]:
        if not sel_col or sel_col not in df.columns:
            if essential:
                st.error(f"Axis '{axis_label}' must be valid.")
                return None, False
            return None, True

        series = df[sel_col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            df[sel_col] = numeric
            return sel_col, True

        logger.info(f"Numeric conversion failed for '{sel_col}' "
                    f"(dtype {series.dtype}).")

        if as_colour:
            base = f"_plot_id_{sel_col}"
            tmp = base
            suffix = 1
            while tmp in df.columns:
                tmp = f"{base}_{suffix}"
                suffix += 1
            df[tmp] = pd.factorize(series)[0]
            temp_cols_to_drop.append(tmp)
            return tmp, True

        if essential and allow_index_fb:
            with container:
                choice = st.radio(
                    f"Plot '{sel_col}' by:",
                    ["Index", "Unique ID"],
                    horizontal=True,
                    key=f"{title_prefix}_{sel_col}_fb",
                )
            base = ("_plot_index_" if choice == "Index" else "_plot_id_") + sel_col
            tmp = base
            suffix = 1
            while tmp in df.columns:
                tmp = f"{base}_{suffix}"
                suffix += 1
            df[tmp] = (df.index
                       if choice == "Index"
                       else pd.factorize(series)[0])
            temp_cols_to_drop.append(tmp)
            return tmp, True

        if essential:
            st.error(f"Axis '{axis_label}' must be numeric for {plot_type}.")
            return None, False

        return None, True  # non-essential, ignore

    allow_axis_fb = (plot_type in {"2D Scatter", "3D Scatter"}
                     and coord_system == "Cartesian")

    x_col, ok = _prep(x_sel, "X", True, allow_axis_fb, False, axis_cols[0])
    if not ok:
        st.stop()
    y_col, ok = _prep(y_sel, "Y", True, allow_axis_fb, False, axis_cols[1])
    if not ok:
        st.stop()
    z_col = None
    if plot_type == "3D Scatter":
        z_col, ok = _prep(z_sel, "Z", True, allow_axis_fb, False, axis_cols[2])
        if not ok:
            st.stop()
    colour_col = colour_sel
    if colour_sel not in {"None", "Density"}:
        colour_col, _ = _prep(colour_sel, "Color", False, False, True,
                              axis_cols[clr_col_idx])

    # ---------------------- 7. LOG + SAMPLING UI ----------------------------
    ctrl_cols_needed = 2 + (plot_type == "3D Scatter"
                            and coord_system == "Cartesian")
    if allow_sampling and len(df_original) > 1:
        ctrl_cols_needed += 1
    ctrl = st.columns(ctrl_cols_needed)

    log_x = ctrl[0].checkbox(
        "Log X", key=f"{title_prefix}_logx",
        disabled=(coord_system == "Polar"),
    )
    log_y = ctrl[1].checkbox("Log Y", key=f"{title_prefix}_logy")
    log_z = False
    if plot_type == "3D Scatter" and coord_system == "Cartesian":
        log_z = ctrl[2].checkbox("Log Z", key=f"{title_prefix}_logz")

    if allow_sampling and len(df_original) > 1:
        slider_col = ctrl[-1]
        max_allowed = len(df_original)
        rows_to_plot = slider_col.slider(
            f"Max points (total {max_allowed})",
            min_value=1 if max_allowed <= 100 else 100,
            max_value=max_allowed,
            value=min(max_rows_default, max_allowed),
            key=f"{title_prefix}_sample",
        )
        if rows_to_plot < len(df):
            st.info(f"Plotting random sample of {rows_to_plot} points.")
            df = df.sample(rows_to_plot, random_state=42)

    # ------------------- 8. DENSITY (if requested) --------------------------
    dens_col = "_calculated_density"
    if colour_col == "Density":
        try:
            if plot_type == "3D Scatter":
                coords = df[[x_col, y_col, z_col]].dropna()
            elif plot_type == "2D Scatter":
                coords = df[[x_col, y_col]].dropna()
            else:
                coords = pd.DataFrame()

            if len(coords) > 1:
                kde = gaussian_kde(coords.T)
                df.loc[coords.index, dens_col] = kde(coords.T)
                colour_col = dens_col
                temp_cols_to_drop.append(dens_col)
            else:
                st.warning("Need >1 point for density; falling back.")
                colour_col = "None"
        except Exception as exc:
            st.error(f"Density calculation failed: {exc}")
            report_error(exc, "KDE failure")
            colour_col = "None"

    # -------------------- 9. HOVER DATA (safe) ------------------------------
    hover_data = {
        col: True for col in df_original.columns
        if col not in {x_sel, y_sel, z_sel, colour_sel}
    }

    plot_kwargs = {"hover_data": hover_data}

    # -------------------- 10. COLOUR SETTINGS -------------------------------
    scale = COLOR_SCALES[colour_scheme]
    if colour_col and colour_col != "None":
        if pd.api.types.is_numeric_dtype(df[colour_col]):
            plot_kwargs["color_continuous_scale"] = scale
        else:
            plot_kwargs["color_discrete_sequence"] = scale

    label_map = {x_col: x_sel or x_col, y_col: y_sel or y_col}
    if z_col:
        label_map[z_col] = z_sel or z_col
    if colour_col == dens_col:
        label_map[colour_col] = "Density"
    plot_kwargs["labels"] = label_map

    title = f"{title_prefix}: {selected_block}"
    fig = None

    # ---------------------- 11. PLOT CONSTRUCTION ---------------------------
    try:
        if plot_type == "2D Scatter":
            if coord_system == "Polar":
                fig = px.scatter_polar(
                    df, theta=x_col, r=y_col, color=None
                    if colour_col == "None" else colour_col,
                    title=title, **plot_kwargs
                )
                if log_y:
                    fig.update_layout(polar_radialaxis_type="log")
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=None
                    if colour_col == "None" else colour_col,
                    title=title, **plot_kwargs
                )
                if log_x:
                    fig.update_xaxes(type="log")
                if log_y:
                    fig.update_yaxes(type="log")

                # ---------- RESTORED: aspect-ratio checkbox ---------------
                keep_ratio = st.checkbox(
                    "Keep data aspect ratio?",
                    value=True,
                    key=f"{title_prefix}_aspect",
                )
                if keep_ratio:
                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                # ----------------------------------------------------------

        elif plot_type == "2D Histogram":
            fig = go.Figure(
                go.Histogram2d(
                    x=df[x_col], y=df[y_col], colorscale=scale,
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title=label_map[x_col],
                yaxis_title=label_map[y_col],
                coloraxis_colorbar=dict(title="Count"),
            )
            if log_x:
                fig.update_xaxes(type="log")
            if log_y:
                fig.update_yaxes(type="log")

        else:  # "3D Scatter"
            fig = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col,
                color=None if colour_col == "None" else colour_col,
                title=title, **plot_kwargs
            )
            scene = {}
            if log_x:
                scene["xaxis_type"] = "log"
            if log_y:
                scene["yaxis_type"] = "log"
            if log_z:
                scene["zaxis_type"] = "log"
            if scene:
                fig.update_layout(scene=scene)

        # hide legend if too many discrete categories
        if (colour_sel not in {"None", "Density"}
                and colour_sel in df_original.columns
                and df_original[colour_sel].nunique() > 100):
            fig.update_layout(showlegend=False)
            if (colour_col and not pd.api.types.is_numeric_dtype(df[colour_col])):
                fig.update_layout(coloraxis_showscale=False)

        fig.update_layout(height=700 if plot_type == "3D Scatter" else 600)

    except Exception as exc:
        st.error(f"Plot creation failed: {exc}")
        report_error(exc, "Plot build")
        df.drop(columns=temp_cols_to_drop, errors="ignore", inplace=True)
        return
    if plot_type == "2D Scatter" and coord_system == "Cartesian":
        fig.update_layout(dragmode="select")
    # -------------------------- 12. SHOW FIGURE -----------------------------
    st.plotly_chart(fig, use_container_width=True,
                    key=f"{title_prefix}_{selected_block}_plot")

    # -------------------------- 13. CLEAN-UP --------------------------------
    df.drop(columns=temp_cols_to_drop, errors="ignore", inplace=True)
    

def interactive_scatter_plot(
    data_source: Union[str, Dict[str, pd.DataFrame]],
    block_selector_options: Optional[List[str]] = None,
    default_block: Optional[str] = None,
    title_prefix: str = "Interactive Plot",
    max_rows_default: int = 50_000,
    allow_sampling: bool = True,
) -> None:
    """
    Fully featured interactive explorer for RELION-style STAR data.

    Features
    --------
    • 2-D Cartesian / polar scatter, 3-D scatter, 2-D histogram  
    • Numeric conversion with index / ID fall-backs for non-numeric axes  
    • Density colouring via Gaussian KDE  
    • Log scaling, random sampling, aspect-ratio lock (2-D)  
    • Lasso & rectangle selection with persistent session-state  
    • Save selected subset as a new STAR file
    """

    # ----------------------------- 1. COLOUR MAPS ---------------------------
    COLOR_SCALES: Dict[str, List[str]] = {
        "Viridis": px.colors.sequential.Viridis,
        "Plasma": px.colors.sequential.Plasma,
        "Blues": px.colors.sequential.Blues,
        "Cividis": px.colors.sequential.Cividis,
        "Turbo": px.colors.sequential.Turbo,
        "Plotly (Qualitative)": px.colors.qualitative.Plotly,
        "Bold (Qualitative)": px.colors.qualitative.Bold,
    }

    # ----------------------------- 2. LOAD DATA -----------------------------
    star_data: Dict[str, pd.DataFrame] = {}
    source_name = "Loaded Data"

    if isinstance(data_source, str):
        source_name = os.path.basename(data_source)
        logger.info(f"Parsing STAR file for plotting: {data_source}")
        try:
            star_data = parse_star(data_source)
        except Exception as exc:
            st.error(f"Failed to parse STAR file '{source_name}': {exc}")
            report_error(exc, f"STAR parsing – {source_name}")
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
        st.error("`data_source` must be a path or dict of DataFrames.")
        return

    # --------------------------- 3. BLOCK SELECTION -------------------------
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

    df_original = star_data[selected_block]
    if not isinstance(df_original, pd.DataFrame):
        st.error(f"Block '{selected_block}' is not a DataFrame.")
        return
    if df_original.empty:
        st.info(f"Block '{selected_block}' is empty.")
        return

    df = df_original.copy()
    if not df.index.is_unique:
        df.reset_index(drop=True, inplace=True)

    st.write(f"Original data: {len(df_original):,} rows × "
             f"{df_original.shape[1]} columns")

    # -------------------------- 4. PLOT CONFIG UI ---------------------------
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

    # -------------------------- 5. AXIS / COLOUR UI -------------------------
    columns_list = [""] + list(df.columns)
    if len(columns_list) == 1:
        st.error("No columns found in block.")
        return

    st.markdown("---")
    axis_cols = st.columns(5 if plot_type == "3D Scatter" else 4)

    def _suggest(suffixes: List[str]) -> Optional[str]:
        for sfx in suffixes:
            match = next((c for c in df.columns
                          if isinstance(c, str)
                          and c.lower().endswith(sfx.lower())), None)
            if match:
                return match
        return None

    x_default = _suggest(["X", "Rot"]) or df.columns[0]
    y_default = _suggest(["Y", "Tilt"]) or df.columns[min(1, len(df.columns)-1)]
    z_default = _suggest(["Z", "Psi"]) or df.columns[min(2, len(df.columns)-1)]

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
    colour_options = ["None", "Density"] + list(df.columns)
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

    # -------------------- 6. COLUMN PREP & FALLBACKS ------------------------
    temp_cols_to_drop: List[str] = []

    def _prep(
        sel_col: Optional[str],
        axis_label: str,
        essential: bool,
        allow_index_fb: bool,
        as_colour: bool,
        container,
    ) -> Tuple[Optional[str], bool]:
        if not sel_col or sel_col not in df.columns:
            if essential:
                st.error(f"Axis '{axis_label}' must be valid.")
                return None, False
            return None, True

        series = df[sel_col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            df[sel_col] = numeric
            return sel_col, True

        logger.info(f"Numeric conversion failed for '{sel_col}' "
                    f"(dtype {series.dtype}).")

        if as_colour:
            base = f"_plot_id_{sel_col}"
            tmp = base
            suffix = 1
            while tmp in df.columns:
                tmp = f"{base}_{suffix}"
                suffix += 1
            df[tmp] = pd.factorize(series)[0]
            temp_cols_to_drop.append(tmp)
            return tmp, True

        if essential and allow_index_fb:
            with container:
                choice = st.radio(
                    f"Plot '{sel_col}' by:",
                    ["Index", "Unique ID"],
                    horizontal=True,
                    key=f"{title_prefix}_{sel_col}_fb",
                )
            base = ("_plot_index_" if choice == "Index" else "_plot_id_") + sel_col
            tmp = base
            suffix = 1
            while tmp in df.columns:
                tmp = f"{base}_{suffix}"
                suffix += 1
            df[tmp] = (df.index
                       if choice == "Index"
                       else pd.factorize(series)[0])
            temp_cols_to_drop.append(tmp)
            return tmp, True

        if essential:
            st.error(f"Axis '{axis_label}' must be numeric for {plot_type}.")
            return None, False

        return None, True  # non-essential, ignore

    allow_axis_fb = (plot_type in {"2D Scatter", "3D Scatter"}
                     and coord_system == "Cartesian")

    x_col, ok = _prep(x_sel, "X", True, allow_axis_fb, False, axis_cols[0])
    if not ok:
        st.stop()
    y_col, ok = _prep(y_sel, "Y", True, allow_axis_fb, False, axis_cols[1])
    if not ok:
        st.stop()
    z_col = None
    if plot_type == "3D Scatter":
        z_col, ok = _prep(z_sel, "Z", True, allow_axis_fb, False, axis_cols[2])
        if not ok:
            st.stop()
    colour_col = colour_sel
    if colour_sel not in {"None", "Density"}:
        colour_col, _ = _prep(colour_sel, "Color", False, False, True,
                              axis_cols[clr_col_idx])

    # ---------------------- 7. LOG + SAMPLING UI ----------------------------
    ctrl_cols_needed = 2 + (plot_type == "3D Scatter"
                            and coord_system == "Cartesian")
    if allow_sampling and len(df_original) > 1:
        ctrl_cols_needed += 1
    ctrl = st.columns(ctrl_cols_needed)

    log_x = ctrl[0].checkbox(
        "Log X", key=f"{title_prefix}_logx",
        disabled=(coord_system == "Polar"),
    )
    log_y = ctrl[1].checkbox("Log Y", key=f"{title_prefix}_logy")
    log_z = False
    if plot_type == "3D Scatter" and coord_system == "Cartesian":
        log_z = ctrl[2].checkbox("Log Z", key=f"{title_prefix}_logz")

    if allow_sampling and len(df_original) > 1:
        slider_col = ctrl[-1]
        max_allowed = len(df_original)
        rows_to_plot = slider_col.slider(
            f"Max points (total {max_allowed})",
            min_value=1 if max_allowed <= 100 else 100,
            max_value=max_allowed,
            value=min(max_rows_default, max_allowed),
            key=f"{title_prefix}_sample",
        )
        if rows_to_plot < len(df):
            st.info(f"Plotting random sample of {rows_to_plot} points.")
            df = df.sample(rows_to_plot, random_state=42)

    # ------------------- 8. DENSITY (if requested) --------------------------
    dens_col = "_calculated_density"
    if colour_col == "Density":
        try:
            if plot_type == "3D Scatter":
                coords = df[[x_col, y_col, z_col]].dropna()
            elif plot_type == "2D Scatter":
                coords = df[[x_col, y_col]].dropna()
            else:
                coords = pd.DataFrame()

            if len(coords) > 1:
                kde = gaussian_kde(coords.T)
                df.loc[coords.index, dens_col] = kde(coords.T)
                colour_col = dens_col
                temp_cols_to_drop.append(dens_col)
            else:
                st.warning("Need >1 point for density; falling back.")
                colour_col = "None"
        except Exception as exc:
            st.error(f"Density calculation failed: {exc}")
            report_error(exc, "KDE failure")
            colour_col = "None"

    # -------------------- 9. HOVER DATA (safe) ------------------------------
    hover_data = {
        col: True for col in df_original.columns
        if col not in {x_sel, y_sel, z_sel, colour_sel}
    }

    plot_kwargs = {"hover_data": hover_data}

    # -------------------- 10. COLOUR SETTINGS -------------------------------
    scale = COLOR_SCALES[colour_scheme]
    if colour_col and colour_col != "None":
        if pd.api.types.is_numeric_dtype(df[colour_col]):
            plot_kwargs["color_continuous_scale"] = scale
        else:
            plot_kwargs["color_discrete_sequence"] = scale

    label_map = {x_col: x_sel or x_col, y_col: y_sel or y_col}
    if z_col:
        label_map[z_col] = z_sel or z_col
    if colour_col == dens_col:
        label_map[colour_col] = "Density"
    plot_kwargs["labels"] = label_map

    title = f"{title_prefix}: {selected_block}"
    fig = None

    # ---------------------- 11. PLOT CONSTRUCTION ---------------------------
    try:
        if plot_type == "2D Scatter":
            if coord_system == "Polar":
                fig = px.scatter_polar(
                    df, theta=x_col, r=y_col, color=None
                    if colour_col == "None" else colour_col,
                    title=title, **plot_kwargs
                )
                if log_y:
                    fig.update_layout(polar_radialaxis_type="log")
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=None
                    if colour_col == "None" else colour_col,
                    title=title, **plot_kwargs
                )
                if log_x:
                    fig.update_xaxes(type="log")
                if log_y:
                    fig.update_yaxes(type="log")

                # ---------- aspect-ratio checkbox ---------------
                keep_ratio = st.checkbox(
                    "Keep data aspect ratio?",
                    value=True,
                    key=f"{title_prefix}_aspect",
                )
                if keep_ratio:
                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                # ------------------------------------------------

        elif plot_type == "2D Histogram":
            fig = go.Figure(
                go.Histogram2d(
                    x=df[x_col], y=df[y_col], colorscale=scale,
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title=label_map[x_col],
                yaxis_title=label_map[y_col],
                coloraxis_colorbar=dict(title="Count"),
            )
            if log_x:
                fig.update_xaxes(type="log")
            if log_y:
                fig.update_yaxes(type="log")

        else:  # "3D Scatter"
            fig = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col,
                color=None if colour_col == "None" else colour_col,
                title=title, **plot_kwargs
            )
            scene = {}
            if log_x:
                scene["xaxis_type"] = "log"
            if log_y:
                scene["yaxis_type"] = "log"
            if log_z:
                scene["zaxis_type"] = "log"
            if scene:
                fig.update_layout(scene=scene)

        # hide legend if too many discrete categories
        if (colour_sel not in {"None", "Density"}
                and colour_sel in df_original.columns
                and df_original[colour_sel].nunique() > 100):
            fig.update_layout(showlegend=False)
            if (colour_col and not pd.api.types.is_numeric_dtype(df[colour_col])):
                fig.update_layout(coloraxis_showscale=False)

        fig.update_layout(height=700 if plot_type == "3D Scatter" else 600)

    except Exception as exc:
        st.error(f"Plot creation failed: {exc}")
        report_error(exc, "Plot build")
        df.drop(columns=temp_cols_to_drop, errors="ignore", inplace=True)
        return

    # -------------------- 12. DISPLAY & SELECTION ---------------------------
    plot_key = f"{title_prefix}_{selected_block}_plot"

    if "plotly_selection" not in st.session_state:
        st.session_state.plotly_selection = {}

    # pre-populate current_selection from session-state
    current_selection = st.session_state.plotly_selection.get(
        plot_key,
        {"points": []},
    )

    event_data = st.plotly_chart(
        fig,
        use_container_width=True,
        key=plot_key,
        on_select="rerun",      # ← lasso / rectangle restored
    )

    # capture selection
    if event_data and event_data.selection:
        st.session_state.plotly_selection[plot_key] = event_data.selection
        current_selection = event_data.selection
    elif event_data and event_data.selection is None:  # deselect
        st.session_state.plotly_selection[plot_key] = {"points": []}
        current_selection = {"points": []}

    selected_indices: List[int] = []
    if current_selection and current_selection.get("points"):
        try:
            sel_point_idx = [pt["point_index"] for pt in current_selection["points"]]
            selected_original_idx = df.iloc[sel_point_idx].index.tolist()
            selected_indices = [
                idx for idx in selected_original_idx if idx in df_original.index
            ]
        except Exception as exc:
            st.error(f"Selection processing failed: {exc}")
            report_error(exc, "Selection mapping")
            selected_indices = []

    # -------------------- 13. DOWNLOAD SELECTED SUBSET ----------------------
    if selected_indices:
        st.markdown("---")
        st.subheader("Save Selection")

        try:
            job_str = ""
            if isinstance(data_source, str):
                m = re.search(r"job\d+", data_source, re.IGNORECASE)
                job_str = m.group(0) + "_" if m else ""
            base_name = (
                os.path.basename(source_name).replace(".star", "")
                if source_name != "Loaded Data" else selected_block
            )
            safe_block = re.sub(r"\W+", "_", selected_block)
            file_name = f"{job_str}{base_name}_{safe_block}_selection.star"
            file_name = re.sub(r"_+", "_", file_name).strip("_")
        except Exception as exc:
            logger.error(f"Filename generation failed: {exc}")
            file_name = f"selected_subset_{selected_block}.star"

        try:
            subset_df = df_original.loc[selected_indices].copy()
            out_dict = {k: v.copy() for k, v in star_data.items()
                        if k != selected_block}
            out_dict[selected_block] = subset_df

            star_doc = star_from_df(out_dict)

            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=".star",
                                             mode="w+",
                                             encoding="utf-8") as tmp:
                tmp_path = tmp.name
                star_doc.write_file(tmp_path)

            with open(tmp_path, "rb") as f:
                binary_star = f.read()

            st.download_button(
                label=f"Download {len(selected_indices)} selected rows "
                      f"as **{file_name}**",
                data=binary_star,
                file_name=file_name,
                mime="application/octet-stream",
                key=f"{title_prefix}_dl",
            )
            os.remove(tmp_path)
        except Exception as exc:
            st.error(f"Subset save failed: {exc}")
            report_error(exc, "STAR subset save")

    # -------------------------- 14. CLEAN-UP --------------------------------
    df.drop(columns=temp_cols_to_drop, errors="ignore", inplace=True)