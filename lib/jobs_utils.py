# Standard Library Imports
import logging
import os
import re
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Third-Party Imports
import pandas as pd
import streamlit as st

# Local Imports
# Import utility functions ONLY from lib.utils
from lib.utils import (
    get_newest_change,
    get_note,
    get_relationships_df,
    report_error,
)
# Import job-specific plotting/processing functions
from relion_jobs.class2d_job import plot_class2d
from relion_jobs.class3d_job import plot_class3d
from relion_jobs.ctffind_job import plot_ctf_stats
from relion_jobs.ctfrefine_job import plot_ctf_refine
from relion_jobs.excludetilt_job import plot_exclude_tilt
from relion_jobs.extract_job import process_extract
from relion_jobs.import_job import plot_import
from relion_jobs.localres_job import plot_locres
from relion_jobs.mask_job import plot_mask
from relion_jobs.modelangelo_job import plot_modelangelo
from relion_jobs.motioncorr_job import plot_motioncorr
from relion_jobs.picking_job import plot_picks
from relion_jobs.polish_job import plot_polish
from relion_jobs.postprocess_job import plot_postprocess
from relion_jobs.select_job import plot_selection
from relion_jobs.tomo_pick_job import plot_pick_tomo
from relion_jobs.tomograms_job import plot_tomographs

# --- Globals ---
ERROR_HANDLER: Optional[Callable[[Exception, str], None]] = None
logger = logging.getLogger("main_app")  # Assumes main app created this logger


# --- Error Handling Setup ---

def set_error_handler(handler: Callable[[Exception, str], None]) -> None:
    """
    Sets the global error handler function for this module.

    Args:
        handler: A callable that accepts an Exception and a formatted
                 traceback string.
    """
    global ERROR_HANDLER
    ERROR_HANDLER = handler


# --- Utility Functions ---

def create_download_button(file_path: str, label: str, file_name: str) -> None:
    """
    Creates a Streamlit button that prepares and offers a file for download.

    Initially shows a 'Prepare' button. When clicked, it reads the file
    and replaces the button with a 'Download' button.

    Args:
        file_path: The absolute path to the file to be downloaded.
        label: The text label for the final download button.
        file_name: The suggested filename for the download.
    """
    # Use a unique key based on file_name to avoid state conflicts
    button_key = f"prepare_download_{file_name}"
    download_key = f"download_{file_name}"
    placeholder = st.empty()

    if placeholder.button(f"Prepare {label}", key=button_key):
        if not os.path.exists(file_path):
             st.error(f"File not found: {file_name}")
             logger.error(f"Download preparation failed: File not found at {file_path}")
             return
        try:
            # Read the file content when the 'Prepare' button is clicked
            with open(file_path, "rb") as fp:
                file_data = fp.read()

            # Replace the 'Prepare' button with the actual download button
            placeholder.download_button(
                label=label,
                data=file_data,
                file_name=file_name,
                mime="application/octet-stream", # Generic binary mime type
                key=download_key,
                type="primary"
            )
            # Optional: Add a success message or clear the button after some time
            # st.success(f"{file_name} ready for download.")
        except OSError as io_exc:
            report_error(io_exc, f"IO Error preparing download for {file_name}")
            st.error(f"Error reading file {file_name}. Check permissions.")
        except Exception as exc:
            report_error(exc, f"Error preparing download for {file_name}")
            st.error(f"Could not prepare {file_name} for download.")


def get_last_lines(text: str, number_of_lines: int = 20) -> str:
    """
    Extracts the last N lines from a multi-line string.

    Args:
        text: The input string.
        number_of_lines: The maximum number of lines to return from the end.

    Returns:
        A string containing the last N lines, joined by newlines.
    """
    lines = text.splitlines()
    # Handle cases where text has fewer lines than requested
    start_index = max(0, len(lines) - number_of_lines)
    return "\n".join(lines[start_index:])


def format_display_name(name: str, prefix: str = "relion.") -> str:
    """
    Formats internal RELION names into more readable display names for the UI.

    Removes a prefix, splits by separators, applies capitalization rules,
    and handles special cases like 'Class2D', 'CtfFind', etc.

    Args:
        name: The internal name string (e.g., "relion.class2dauto").
        prefix: The prefix string to remove (default: "relion.").

    Returns:
        A formatted display name string (e.g., "Class2D Auto").
    """
    if not isinstance(name, str):
         return str(name) # Return non-strings as is

    # Remove the prefix if present.
    if name.startswith(prefix):
        name = name[len(prefix):]

    # Define special formatting rules (case-insensitive matching)
    # Exact matches take precedence
    special_exact = {
        "em": "EM", "class3d": "Class3D", "refine3d": "Refine3D",
        "initialmodel": "InitialModel", "modelangelo": "ModelAngelo",
        "interactiveplot": "Interactive Plot", "class2dauto": "Class2D Auto",
        "vdam": "VDAM", "dynamight": "DynaMight", "postprocess": "PostProcess",
        "motioncorr": "MotionCorr", "excludetilts": "ExcludeTilts",
        "aligntiltseries": "AlignTiltSeries",
        "reconstructtomograms": "Reconstruct Tomograms", "denoisetomo": "Denoise Tomo",
        "picktomo": "Pick Tomo", "pseudosubtomo": "PseudoSub Tomo",
        "reconstructparticletomo": "Reconstruct Particle Tomo",
        "importtomo": "Import Tomo", "ctfrefinetomo": "CTF Refine Tomo",
    }
    # Prefix matches (applied if no exact match)
    special_prefixes = {
        "ctffind": "CtfFind", "ctfrefine": "CTFrefine", "class2d": "Class2D",
        # Add more prefix matches as needed
    }

    # Split by likely separators (e.g., '.', '_', ' ')
    parts = re.split(r'[._\s]+', name)
    formatted_parts = []

    for part in parts:
        if not part: continue # Skip empty parts resulting from multiple separators
        lower_part = part.lower()

        # 1. Check for exact special match
        if lower_part in special_exact:
            formatted_parts.append(special_exact[lower_part])
            continue

        # 2. Check for special prefix match
        matched_prefix = False
        for prefix_key, replacement in special_prefixes.items():
            if lower_part.startswith(prefix_key):
                # Append replacement and the rest of the original part (maintaining case)
                remainder = part[len(prefix_key):]
                formatted_parts.append(replacement + remainder)
                matched_prefix = True
                break
        if matched_prefix:
            continue

        # 3. Default formatting: Capitalize, handle 2D/3D
        default_part = part.capitalize()
        default_part = default_part.replace("2d", "2D").replace("3d", "3D")
        formatted_parts.append(default_part)

    return " ".join(formatted_parts)


# --- Job Execution and Display ---

def execute_relion_job(selected_job: str, folder: str, node_files: List[str]) -> bool:
    """
    Executes the appropriate plotting or processing function for a RELION job type.

    Args:
        selected_job: The job name string (e.g., "Class2D/job001").
        folder: The base directory of the RELION project.
        node_files: List of node file paths associated with the job.

    Returns:
        True if a matching job action was found and called, False otherwise.
    """
    
    
    # Map job type keywords (expected prefixes) to their handler functions
    # Use lambdas to defer function execution until a match is found
    job_actions = {
        "Import": lambda: plot_import(folder, node_files),
        "MotionCorr": lambda: plot_motioncorr(folder, node_files[0]), 
        "CtfFind": lambda: plot_ctf_stats(folder, node_files[0]),    
        "AutoPick": lambda: plot_picks(folder, selected_job),        # Uses job path directly
        "ManualPick": lambda: plot_picks(folder, selected_job),      # Uses job path directly
        "Extract": lambda: process_extract(folder, node_files),
        "Subtract": lambda: process_extract(folder, node_files),
        "Select": lambda: plot_selection(node_files, folder, selected_job),
        "Class2D": lambda: plot_class2d(folder, node_files),
        "InitialModel": lambda: plot_class3d(folder, node_files),   # Shares 3D plotting
        "Class3D": lambda: plot_class3d(folder, node_files),
        "Refine3D": lambda: plot_class3d(folder, node_files),
        "MaskCreate": lambda: plot_mask(folder, node_files),
        "PostProcess": lambda: plot_postprocess(folder, node_files),
        "CtfRefine": lambda: plot_ctf_refine(folder, node_files), # Handle Tomo separately if needed
        "Polish": lambda: plot_polish(folder, node_files),
        "LocalRes": lambda: plot_locres(node_files, folder, selected_job),
        "ModelAngelo": lambda: plot_modelangelo(folder, node_files),
        "JoinStar": lambda: plot_selection(node_files, folder, selected_job),
        
        # Tomo Jobs
        "ReconstructParticleTomo": lambda: plot_class3d(folder, node_files), 
        "ExcludeTiltImages": lambda: plot_exclude_tilt(folder, node_files[0]),
        "AlignTiltSeries": lambda: plot_exclude_tilt(folder, node_files[0]), 
        "ReconstructTomograms": lambda: plot_tomographs(folder, node_files[0]), 
        "Tomograms": lambda: plot_tomographs(folder, node_files[0]),
        "Denoise": lambda: plot_tomographs(folder, node_files[0]), 
        "Picks": lambda: plot_pick_tomo(folder, node_files), 
        "PseudoSubtomo": lambda: process_extract(folder, node_files), 
        "Reconstruct": lambda: plot_class3d(folder, node_files), 
        "SubtractImages": lambda: process_extract(folder, node_files), 
        # "DynaMight": lambda: plot_dynamight(folder, node_files), # to be implemented
    }

    job_type_found = False
    for job_type, action in job_actions.items():
        # Check if the job_type keyword is present in the selected_job path string
        # Use regex word boundary \b ? or simple 'in'? Simple 'in' might match substrings wrongly.
        # Let's check if selected_job *starts* with the job_type prefix for better matching.
        # e.g., "Class2D/job001" starts with "Class2D"
        job_prefix = selected_job.split('/')[0] # Get the part before the first '/'
        if job_prefix == job_type:
            logger.info(f"Executing action for job type: {job_type} (Job: {selected_job})")
            try:
                action() # Call the lambda function
                job_type_found = True
                break # Stop after first match
            except IndexError as idx_err:
                 if not node_files:
                     st.error(f"Error executing '{job_type}': No node files provided.")
                     logger.error(f"IndexError for job {selected_job}: No node files provided. {idx_err}")
                 else:
                     report_error(idx_err, f"IndexError executing action for job type {job_type}")
                     st.error(f"An processing error occurred for '{job_type}'. See logs.")
                 job_type_found = True # Mark as found but failed
                 break
            except FileNotFoundError as fnf_err:
                  st.error(f"Error executing '{job_type}': Required file not found: {fnf_err.filename}")
                  logger.error(f"FileNotFoundError for job {selected_job}: {fnf_err}")
                  job_type_found = True
                  break
            except Exception as e:
                report_error(e, f"Error executing action for job type {job_type}")
                st.error(f"An error occurred while processing job '{selected_job}'. See logs for details.")
                job_type_found = True # Mark as found but failed
                break # Stop after error

    if not job_type_found:
        st.info(f"Job is not supported (yet) '{selected_job}'.")
        logger.info(f"No matching action found for job: {selected_job}")
        return False

    return job_type_found # Return True if found (even if failed), False otherwise


def display_job_info(
    selected_job: str,
    folder: str,
    pipeline_processes: pd.DataFrame,
    pipeline_star: Dict[str, pd.DataFrame],
) -> None:
    """
    Displays detailed information and controls for a selected RELION job in Streamlit.

    Args:
        selected_job: The identifier of the selected job (e.g., "Class2D/job001").
        folder: The root folder of the RELION project.
        pipeline_processes: DataFrame containing job process metadata (_rlnPipeLineProcess* columns).
        pipeline_star: Dictionary parsed from pipeline.star, expecting keys
                       'pipeline_nodes' and 'pipeline_input_edges'.
    """
    logger.info(f"Displaying info for job: {selected_job}")
    job_path = os.path.join(folder, selected_job)

    if not os.path.isdir(job_path):
         st.error(f"Job directory not found: {job_path}")
         return

    try:
        # --- Extract Basic Job Info ---
        job_process_info = pipeline_processes[
            pipeline_processes["_rlnPipeLineProcessName"] == selected_job
        ].iloc[0] # Assume unique job name, get first row

        alias = job_process_info.get("_rlnPipeLineProcessAlias", "N/A")
        status = job_process_info.get("_rlnPipeLineProcessStatusLabel", "Unknown")

        st.title(selected_job)
        if alias not in ["N/A", "None", None, ""]:
            st.write(f"**Alias:** _{alias}_")
        st.divider()

        # --- Display Status & Metadata ---
        icon_status = {
            "Succeeded": ":white_check_mark:", "Failed": ":heavy_exclamation_mark:",
            "Aborted": ":x:", "Running": ":recycle:"
        }.get(status, ":question:") # Default icon

        newest_change_ts = get_newest_change(job_path, include_subfolders=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Status:** {status} {icon_status}")
            st.markdown(f"**Last Change:** {newest_change_ts}")
        with col2:
             # Display Parent/Child info
             if "pipeline_input_edges" in pipeline_star:
                  # Use utility function for relationships
                  job_relations = get_relationships_df(pipeline_star["pipeline_input_edges"])
                  job_row = job_relations[job_relations["Job"] == selected_job]
                  if not job_row.empty:
                       children = job_row.iloc[0]['Children']
                       parents = job_row.iloc[0]['Parents']
                       children_text = ", ".join(children) if children else "None"
                       parents_text = ", ".join(parents) if parents else "None"
                  else: children_text, parents_text = "N/A", "N/A"
             else: children_text, parents_text = "N/A (missing edges)", "N/A (missing edges)"

             # Use format_display_name for readability? Depends if names are long/complex
             st.markdown(f"**Input From (Parents):** {parents_text}")
             st.markdown(f"**Output To (Children):** {children_text}")


        # --- Associated Files & Refresh ---
        if "pipeline_nodes" in pipeline_star:
             process_nodes = pipeline_star["pipeline_nodes"]["_rlnPipeLineNodeName"]
             # Filter nodes that belong to the current job directory
             node_files = [
                 node for node in process_nodes if node.startswith(selected_job)
             ]
        else:
             node_files = []
             logger.warning("Missing 'pipeline_nodes' data for listing related files.")


        with col1: # Place expander and refresh in columns
             with st.expander("Associated Files", expanded=False):
                 if node_files:
                     # Show only basenames for brevity
                     st.code("\n".join([os.path.basename(nf) for nf in node_files]), language=None)
                 else:
                     st.markdown("_(No specific node files listed in pipeline)_")

        with col2:
            if st.button("🔄 Refresh Job Data", help="Clear caches and reload job info", key=f"refresh_{selected_job}"):
                 st.cache_data.clear()
                 # Potentially clear specific session state related to this job if needed
                 st.rerun()
        st.divider()

        
        # --- Execute Job-Specific Action ---
        job_executed = execute_relion_job(selected_job, folder, node_files)
        if job_executed:
             logger.info(f"Job action executed for {selected_job}.")
        else:
             logger.warning(f"No action was executed for job {selected_job}.")


        # --- Download Buttons ---
        st.subheader("Download Associated Files")
        buttons_done = set()
        # Focus on STAR, MRC(S), PDF, CIF files for download
        downloadable_extensions = {".star", ".mrc", ".mrcs", ".pdf", ".cif"}
        valid_node_files_to_download = [
            node for node in node_files
            if os.path.splitext(node)[1].lower() in downloadable_extensions
        ]

        if valid_node_files_to_download:
            num_buttons = len(valid_node_files_to_download)
            # Add potential half2 maps if half1 exists
            half_files_to_add = []
            for node in valid_node_files_to_download:
                 if "half1" in node:
                      half2_node = node.replace("half1", "half2")
                      if os.path.exists(os.path.join(folder, half2_node)):
                           if half2_node not in valid_node_files_to_download:
                                half_files_to_add.append(half2_node)
            valid_node_files_to_download.extend(half_files_to_add)
            num_buttons = len(valid_node_files_to_download)


            cols_count = min(num_buttons, 4) # Max 4 download buttons per row
            cols = st.columns(cols_count)
            col_idx = 0
            job_prefix = selected_job.split('/')[1] if '/' in selected_job else selected_job

            for node in sorted(valid_node_files_to_download): # Sort for consistent order
                base_name = os.path.basename(node)
                file_name_download = f"{job_prefix}_{base_name}"
                if file_name_download not in buttons_done:
                    file_path_full = os.path.join(folder, node)
                    if os.path.exists(file_path_full):
                         with cols[col_idx % cols_count]:
                             create_download_button(
                                 file_path_full, f"Download {base_name}", file_name_download
                             )
                             buttons_done.add(file_name_download)
                             col_idx += 1
                    else:
                         logger.warning(f"Node file listed but not found: {file_path_full}")

        else:
             st.caption("_No downloadable files (STAR, MRC(S), PDF, CIF) associated with this job node found._")

        st.divider()

        # --- Parameters and Logs ---
        st.subheader("Job Details")
        note_path = os.path.join(job_path, "note.txt")
        note_content = get_note(note_path) # Use utility

        runlog_path = os.path.join(job_path, "run.out")
        runlog_content = get_note(runlog_path) # Use utility

        log_col1, log_col2 = st.columns(2)
        with log_col1.expander("RELION Parameters (from note.txt)", expanded=False):
            st.code(note_content, language='bash') # Display as code block
        with log_col2.expander("Run Log Tail (run.out)", expanded=False):
             st.text(get_last_lines(runlog_content, 50)) # Show more lines

    except IndexError:
         report_error(IndexError(f"Could not find job '{selected_job}' in pipeline_processes DataFrame."),
                      f"Error displaying job info for {selected_job}")
         st.error(f"Job '{selected_job}' not found or has incomplete data in the pipeline DataFrame.")
    except KeyError as e:
         report_error(e, f"Missing expected data key while displaying job info for {selected_job}")
         st.error(f"Data inconsistency: Missing key '{e}' needed to display job info.")
    except Exception as exc:
        report_error(exc, f"Failed to display job info for {selected_job}")
        st.error(f"An unexpected error occurred while displaying job info: {exc}")


# def create_network(
#     pipeline_star: Dict[str, pd.DataFrame], orientation: str = "top-bottom"
# ) -> Optional[str]:
#     """
#     Creates a Graphviz DOT language string for visualizing the job pipeline.

#     Focuses on job-to-job connections with simplified node names and styling.

#     Args:
#         pipeline_star: Dictionary from parsed pipeline.star, expecting
#                        'pipeline_input_edges'.
#         orientation: Layout direction ("top-bottom" or "left-right").

#     Returns:
#         A string in Graphviz DOT format, or None if input is invalid or error occurs.
#     """
#     logger.info(f"Creating network graph with orientation: {orientation}")
#     # --- Input Validation ---
#     if (
#         "pipeline_input_edges" not in pipeline_star
#         or not isinstance(pipeline_star["pipeline_input_edges"], pd.DataFrame)
#         or pipeline_star["pipeline_input_edges"].empty
#     ):
#         logger.error("Graph Creation Error: Missing or empty 'pipeline_input_edges' DataFrame.")
#         # Optionally return a minimal DOT string indicating error?
#         # return "digraph G { error [label=\"Invalid Input\"]; }"
#         return None

#     try:
#         job_edges_df = pipeline_star["pipeline_input_edges"]

#         # --- Node Name Simplification ---
#         def simplify_node_name(name):
#             if isinstance(name, str):
#                 # Keep only the first two parts (e.g., "Import/job001")
#                 parts = name.split("/")
#                 return "/".join(parts[:2]) if len(parts) >= 2 else name
#             return str(name) # Handle non-string names gracefully

#         # Create simplified 'from' and 'to' columns
#         edges = pd.DataFrame({
#             'from_node': job_edges_df["_rlnPipeLineEdgeFromNode"].apply(simplify_node_name),
#             'to_node': job_edges_df["_rlnPipeLineEdgeProcess"].apply(simplify_node_name)
#         })

#         # --- Edge Filtering (Focus on Job-to-Job connections) ---
#         # Regex to match typical job format like "Word/word###"
#         job_pattern = re.compile(r"^[A-Za-z0-9_]+/[a-zA-Z0-9_]+job\d+$")
#         filtered_edges = []
#         valid_job_nodes = set()

#         for _, row in edges.iterrows():
#             src, dest = row["from_node"], row["to_node"]
#             # Check if *both* source and destination look like job names
#             # Adapt pattern if job naming scheme differs significantly
#             if job_pattern.match(src) and job_pattern.match(dest):
#                 filtered_edges.append((src, dest))
#                 valid_job_nodes.add(src)
#                 valid_job_nodes.add(dest)

#         if not filtered_edges:
#             logger.warning("No valid job-to-job edges found to create network graph.")
#             return None # Return None if no edges to draw

#         # --- Styling Dictionary (Graphviz attributes) ---
#         # Using a slightly more subdued palette
#         palette = {
#             "red": "#F4B9B8", "orange": "#FAD5A5", "yellow": "#FDEDC4",
#             "green": "#C8E6C9", "teal": "#B2DFDB", "cyan": "#B2EBF2",
#             "blue": "#BBDEFB", "indigo": "#C5CAE9", "purple": "#D1C4E9",
#             "pink": "#F8BBD0", "brown": "#D7CCC8", "grey": "#E0E0E0"
#         }
#         # Style definitions per job type
#         job_type_styles_gv = {
#             "Import": {"shape": "diamond", "fillcolor": palette["red"]},
#             "MotionCorr": {"shape": "ellipse", "fillcolor": palette["orange"]},
#             "CtfFind": {"shape": "ellipse", "fillcolor": palette["yellow"], "fontcolor": "#333"},
#             "AutoPick": {"shape": "hexagon", "fillcolor": palette["cyan"], "fontcolor": "#333"},
#             "ManualPick": {"shape": "hexagon", "fillcolor": palette["teal"], "fontcolor": "#333"},
#             "Extract": {"shape": "invhouse", "fillcolor": palette["blue"]},
#             "Select": {"shape": "ellipse", "fillcolor": palette["green"], "fontcolor": "#333"},
#             "Class2D": {"shape": "box", "fillcolor": palette["grey"], "fontcolor": "#333"},
#             "InitialModel": {"shape": "doublecircle", "fillcolor": palette["indigo"]},
#             "Class3D": {"shape": "ellipse", "fillcolor": palette["purple"]},
#             "Refine3D": {"shape": "ellipse", "fillcolor": palette["indigo"]},
#             "MaskCreate": {"shape": "pentagon", "fillcolor": palette["pink"], "fontcolor": "#333"},
#             "PostProcess": {"shape": "note", "fillcolor": palette["green"]},
#             "CtfRefine": {"shape": "octagon", "fillcolor": palette["orange"]},
#             "Polish": {"shape": "parallelogram", "fillcolor": palette["blue"]},
#             "LocalRes": {"shape": "trapezium", "fillcolor": palette["purple"]},
#             "ModelAngelo": {"shape": "ellipse", "fillcolor": palette["brown"]},
#             "DynaMight": {"shape": "diamond", "fillcolor": palette["cyan"]},
#              # Tomo additions (example styles)
#             "ImportTomo": {"shape": "diamond", "fillcolor": palette["indigo"]},
#             "AlignTiltSeries": {"shape": "parallelogram", "fillcolor": palette["purple"]},
#             "ReconstructTomograms": {"shape": "hexagon", "fillcolor": palette["green"]},
#             "Denoise": {"shape": "ellipse", "fillcolor": palette["teal"]},
#             "Picks": {"shape": "hexagon", "fillcolor": palette["cyan"]},
#             "PseudoSubtomo": {"shape": "invhouse", "fillcolor": palette["blue"]},
#             "ReconstructParticleTomo": {"shape": "ellipse", "fillcolor": palette["purple"]},
#             "CtfRefineTomo": {"shape": "octagon", "fillcolor": palette["orange"]},
#             # Fallback style
#             "default": {"shape": "box", "fillcolor": "#E8E8E8", "fontcolor": "#555"}
#         }
#         # Add default font color if missing
#         for style in job_type_styles_gv.values():
#             style.setdefault("fontcolor", "black") # Default to black font
#             style.setdefault("color", style["fillcolor"]) # Border same as fill

#         # --- Build DOT String ---
#         rankdir = "TB" if orientation == "top-bottom" else "LR"
#         nodesep = 0.1 if rankdir == "LR" else 0.05
#         ranksep = 0.5 # Consistent rank separation

#         dot = [
#             "digraph RelionPipeline {",
#             "    bgcolor=transparent;",
#             f"    rankdir={rankdir};",
#             "    splines=ortho;      // Use orthogonal edges",
#             "    overlap=false;      // Prevent node overlap",
#             f"    nodesep={nodesep:.2f};",
#             f"    ranksep={ranksep:.2f};",
#             "    outputorder=edgesfirst;",
#             "    node [style=\"filled\", fontname=\"Helvetica\", fontsize=9, margin=\"0.05,0.04\"];",
#             "    edge [arrowsize=0.8, color=\"#888888\", penwidth=1.2];",
#             "" # Newline before nodes
#         ]

#         # Define Nodes
#         for node_name in sorted(list(valid_job_nodes)):
#             job_type = node_name.split("/")[0]
#             style = job_type_styles_gv.get(job_type, job_type_styles_gv["default"])
#             # Label uses job number on new line
#             label = node_name.replace("/", "\\n") # Use Graphviz newline
#             node_id = f'"{node_name}"' # Quote node names
#             tooltip = f"Job: {node_name}\\nType: {job_type}" # Tooltip for interactivity

#             dot.append(
#                 f'    {node_id} [label="{label}", shape={style["shape"]}, '
#                 f'fillcolor="{style["fillcolor"]}", color="{style["color"]}", '
#                 f'fontcolor="{style["fontcolor"]}", tooltip="{tooltip}"];'
#             )

#         # Define Edges
#         dot.append("\n    // Edges")
#         for src, dest in filtered_edges:
#             dot.append(f'    "{src}" -> "{dest}";')

#         dot.append("}")
#         return "\n".join(dot)

#     except KeyError as e:
#         report_error(KeyError(f"DOT Generation Error: Missing expected column: {e}"))
#         return None
#     except Exception as e:
#         report_error(e, "Error generating DOT network graph.")
#         return None
    


def create_network(
    pipeline_star: Dict[str, pd.DataFrame], orientation: str = "top-bottom"
) -> Optional[str]:
    """
    Creates a Graphviz DOT language string for visualizing the job pipeline.

    Includes all nodes and edges found in 'pipeline_input_edges', applying
    styling based on node type (job vs. other).

    Args:
        pipeline_star: Dictionary from parsed pipeline.star, expecting
                       'pipeline_input_edges'.
        orientation: Layout direction ("top-bottom" or "left-right").

    Returns:
        A string in Graphviz DOT format, or a minimal DOT graph if no
        valid edges are found, or None on critical error.
    """
    logger.info(f"Creating network graph with orientation: {orientation}")
    # --- Input Validation ---
    if (
        "pipeline_input_edges" not in pipeline_star
        or not isinstance(pipeline_star["pipeline_input_edges"], pd.DataFrame)
        # Allow empty dataframe, will result in empty graph string
    ):
        logger.warning("Graph Creation Warning: Missing or invalid 'pipeline_input_edges' DataFrame.")
        # Return an empty graph instead of None
        return "digraph RelionPipeline { rankdir=TB; label=\"Pipeline data missing\"; }"

    try:
        job_edges_df = pipeline_star["pipeline_input_edges"]
        if job_edges_df.empty:
            logger.warning("Graph Creation Warning: 'pipeline_input_edges' DataFrame is empty.")
            return "digraph RelionPipeline { rankdir=TB; label=\"No pipeline edges found\"; }"


        # --- Node Name Simplification ---
        def simplify_node_name(name):
            # Handles "Type/JobName/MaybeMore" -> "Type/JobName"
            # Handles "Type/JobName" -> "Type/JobName"
            # Handles "Type" -> "Type"
            if isinstance(name, str):
                parts = name.split("/")
                return "/".join(parts[:2]) if len(parts) >= 2 else name
            return str(name)

        # Process ALL edges and collect unique nodes and edges
        all_edges = set()
        all_nodes = set()
        for _, row in job_edges_df.iterrows():
            # Ensure columns exist before accessing
            if "_rlnPipeLineEdgeFromNode" not in row or "_rlnPipeLineEdgeProcess" not in row:
                 logger.warning(f"Skipping edge due to missing column(s): {row}")
                 continue
            src_simple = simplify_node_name(row["_rlnPipeLineEdgeFromNode"])
            dest_simple = simplify_node_name(row["_rlnPipeLineEdgeProcess"])
            if src_simple and dest_simple: # Avoid edges with empty nodes
                all_edges.add((src_simple, dest_simple))
                all_nodes.add(src_simple)
                all_nodes.add(dest_simple)

        if not all_nodes:
            logger.warning("No valid nodes found after processing edges.")
            return "digraph RelionPipeline { rankdir=TB; label=\"No valid nodes found\"; }"

        # --- Styling ---
        # Pattern to identify likely job nodes (Type/Identifier) for styling
        style_job_pattern = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_]+$")
        # Simplified palette
        palette = {
            "red": "#F4C7C3", "orange": "#FAD9A1", "yellow": "#FFFACD",
            "green": "#C1E1C1", "teal": "#A0D2DB", "cyan": "#B4E1E7",
            "blue": "#AEC6CF", "indigo": "#C9CBE0", "purple": "#D8BFD8",
            "pink": "#F8C8DC", "brown": "#E0D8C0", "grey": "#E0E0E0"
        }
        job_type_styles = {
            "Import": {"shape": "diamond", "fillcolor": palette["red"]},
            "MotionCorr": {"shape": "ellipse", "fillcolor": palette["orange"]},
            "CtfFind": {"shape": "ellipse", "fillcolor": palette["yellow"], "fontcolor": "#333"},
            "AutoPick": {"shape": "hexagon", "fillcolor": palette["cyan"], "fontcolor": "#333"},
            "ManualPick": {"shape": "hexagon", "fillcolor": palette["teal"], "fontcolor": "#333"},
            "Extract": {"shape": "invhouse", "fillcolor": palette["blue"]},
            "Subtract": {"shape": "invhouse", "fillcolor": palette["blue"]},
            "Select": {"shape": "ellipse", "fillcolor": palette["green"], "fontcolor": "#333"},
            "Class2D": {"shape": "box", "fillcolor": palette["grey"], "fontcolor": "#333"},
            "InitialModel": {"shape": "doublecircle", "fillcolor": palette["indigo"]},
            "Class3D": {"shape": "ellipse", "fillcolor": palette["purple"]},
            "Refine3D": {"shape": "ellipse", "fillcolor": palette["indigo"]},
            "MaskCreate": {"shape": "pentagon", "fillcolor": palette["pink"], "fontcolor": "#333"},
            "PostProcess": {"shape": "note", "fillcolor": palette["green"]},
            "CtfRefine": {"shape": "octagon", "fillcolor": palette["orange"]},
            "Polish": {"shape": "parallelogram", "fillcolor": palette["blue"]},
            "LocalRes": {"shape": "trapezium", "fillcolor": palette["purple"]},
            "ModelAngelo": {"shape": "ellipse", "fillcolor": palette["brown"]},
            "JoinStar": {"shape": "ellipse", "fillcolor": palette["grey"]},
            # Tomo
            "AlignTiltSeries": {"shape": "parallelogram", "fillcolor": palette["purple"]},
            "ReconstructTomograms": {"shape": "hexagon", "fillcolor": palette["green"]},
             # Default for recognized job types not explicitly listed
            "job_default": {"shape": "ellipse", "fillcolor": palette["grey"], "fontcolor": "#333"},
            # Default for non-job nodes (e.g., files, other steps)
            "other_default": {"shape": "box", "style": "filled,dashed", "fillcolor": "#FAFAFA", "color": "#BBBBBB", "fontcolor": "#777", "fontsize": 8},
        }
        # Ensure base attributes
        for style in job_type_styles.values():
            style.setdefault("fontcolor", "black")
            style.setdefault("color", "#555555") # Default border color
            style.setdefault("style", "filled")
            style.setdefault("fontsize", 9)

        # --- Build DOT String ---
        rankdir = "TB" if orientation == "top-bottom" else "LR"
        nodesep = 0.1 if rankdir == "LR" else 0.05
        ranksep = 0.5

        dot = [
            "digraph RelionPipeline {",
            "    bgcolor=transparent;", f"    rankdir={rankdir};", "    splines=ortho;",
            "    overlap=scale; concentrate=true;", # Try concentrate for edge merging
            f"    nodesep={nodesep:.2f};", f"    ranksep={ranksep:.2f};",
            "    outputorder=edgesfirst;",
            "    node [fontname=\"Helvetica\", margin=\"0.08,0.05\"];", # Slightly larger margin
            "    edge [arrowsize=0.7, color=\"#999999\", penwidth=1.0];", ""
        ]

        # Define Nodes
        for node_name in sorted(list(all_nodes)):
            job_type = node_name.split("/")[0]
            # Determine style: check specific type, then if it looks like a job, else other_default
            if job_type in job_type_styles:
                 style = job_type_styles[job_type]
            elif style_job_pattern.match(node_name):
                 style = job_type_styles["job_default"]
            else:
                 style = job_type_styles["other_default"]

            # Format label: Use job number/second part on new line if possible
            parts = node_name.split("/")
            label = "\\n".join(parts) if len(parts) > 1 else node_name
            node_id = f'"{node_name}"' # Quote node names
            tooltip = node_name # Simple tooltip

            # Build attribute string
            attrs = [f'label="{label}"', f'shape={style["shape"]}',
                     f'style="{style["style"]}"', f'fillcolor="{style["fillcolor"]}"',
                     f'color="{style["color"]}"', f'fontcolor="{style["fontcolor"]}"',
                     f'fontsize={style["fontsize"]}', f'tooltip="{tooltip}"']
            dot.append(f'    {node_id} [{", ".join(attrs)}];')

        # Define Edges
        dot.append("\n    // Edges")
        for src, dest in sorted(list(all_edges)):
            dot.append(f'    "{src}" -> "{dest}";')

        dot.append("}")
        return "\n".join(dot)

    except KeyError as e:
        report_error(KeyError(f"DOT Generation Error: Missing expected column: {e}"))
        return None
    except Exception as e:
        report_error(e, "Error generating DOT network graph.")
        return None