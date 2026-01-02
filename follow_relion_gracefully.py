#
#  ____        ___    ___                           ____            ___                            ____                                    ___          ___    ___
# /\  __\     /\_ \  /\_ \                         /\  _`\         /\_ \    __                    /\  _`\                                /'___\        /\_ \  /\_ \
# \ \ \_    __\//\ \ \//\ \     ___   __  __  __   \ \ \L\ \     __\//\ \  /\_\    ___     ___    \ \ \L\_\  _ __    __      ___     __ /\ \__/  __  __\//\ \ \//\ \    __  __
#  \ \  _\/ __`\\ \ \  \ \ \   / __`\/\ \/\ \/\ \   \ \ ,  /   /'__`\\ \ \ \/\ \  / __`\ /' _ `\   \ \ \L_L /\`'__\/'__`\   /'___\ /'__`\ \ ,__\/\ \/\ \ \ \ \  \ \ \  /\ \/\ \
#   \ \ \/\ \L\ \\_\ \_ \_\ \_/\ \L\ \ \ \_/ \_/ \   \ \ \\ \ /\  __/ \_\ \_\ \ \/\ \L\ \/\ \/\ \   \ \ \/, \ \ \//\ \L\.\_/\ \__//\  __/\ \ \_/\ \ \_\ \ \_\ \_ \_\ \_\ \ \_\ \
#    \ \_\ \____//\____\/\____\ \____/\ \___x___/'    \ \_\ \_\ \____\/\____\\ \_\ \____/\ \_\ \_\   \ \____/\ \_\\ \__/.\_\ \____\ \____\\ \_\  \ \____/ /\____\/\____\\/`____ \
#     \/_/\/___/ \/____/\/____/\/___/  \/__//__/       \/_/\/ /\/____/\/____/ \/_/\/___/  \/_/\/_/    \/___/  \/_/ \/__/\/_/\/____/\/____/ \/_/   \/___/  \/____/\/____/ `/___/> \
#                                                                                                                                                                           \\___/
# Follow Relion Gracefully (v6)
# Developed by Dawid Zyla, La Jolla Institute for Immunology
# Non-Profit Open Software License 3.0

# update v6 (2025-04-26)

# ## Main Changes
# -> Better integration with Streamlit platform (https://streamlit.io/)
# -> Added support for all (most) Relion jobs covering all cryo-ET and SPA jobs (except for DynaMight)
# -> Temporary removed live and in-browser job execution
# -> General QOL fixes and improvements
#
# ## New Features
# -> Support for all cryo-ET jobs with job previews
# -> Optimized visualizations for most of the Relion jobs
# -> Divided the code into smaller modules for better readability and maintainability
# -> Overhauled visualization of Local Resolution, picking, micrograph previews, and other jobs
# -> Increased performance and reduced loading times (in most cases)
# -> Added better logging and error handling
#
#
# ## To Do
# -> Add own DynaMight job preview (currently not supported)
# -> Add support for cryoSPARC cs files and export to Relion (most likely via pyem)
# -> Further speed optimization and code cleanup

# Standard Library Imports
import argparse
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

# Third-Party Imports
import pandas as pd
import streamlit as st

# Local Imports
from lib.jobs_utils import (  # Assuming these are correctly defined in jobs_utils
    create_network,
    create_network_agraph_data,
    display_job_info,
    format_display_name,
)
from streamlit_agraph import agraph, Node, Edge, Config
from lib.utils import (  # Assuming these are correctly defined in utils
    check_password,
    custom_css,
    dynamic_folder_explorer,
    get_footer,
    # get_newest_change,  # Not directly used here, used within display_job_info
    # get_note,          # Not directly used here, used within display_job_info
    # get_relationships_df, # Not directly used here, used within display_job_info
    interactive_scatter_plot,
    parse_star,
    render_svg,
    report_error,  # Use the central report_error
)
from lib.state import StateManager
from lib.constants import *

# Logger configuration
logging_level = logging.DEBUG

# =============================================================================
# Logger and Global Error Handling
# =============================================================================
logger = logging.getLogger("main_app")
if not logger.handlers:
    logger.setLevel(logging_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

ERROR_HANDLER: Optional[Callable[[Exception, str], None]] = None


def local_report_error_handler(exc: Exception, error_info: str):
    """Local fallback error handler that logs the error."""
    logger.error("An unexpected error occurred:\n%s", error_info)


def set_error_handler(handler: Callable[[Exception, str], None]) -> None:
    """Sets the global error handler for this module."""
    global ERROR_HANDLER
    ERROR_HANDLER = handler


# =============================================================================
# UI Setup
# =============================================================================
def set_style() -> None:
    """Configures Streamlit page settings and applies custom CSS."""
    try:
        st.set_page_config(
            page_title="Follow Relion Gracefully",
            page_icon=":microscope:",
            initial_sidebar_state="expanded",
            layout="wide",
        )
        st.markdown(custom_css(), unsafe_allow_html=True)
        st.markdown(
            """
        <style>
            header[data-testid="stHeader"] { background-color: transparent; }
            div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
        </style>""",
            unsafe_allow_html=True,
        )
    except Exception as e:
        logger.error(f"Failed to set page style: {e}")


# =============================================================================
# Utility Functions
# =============================================================================
def parse_help_text(help_text: str) -> Dict[str, str]:
    """Parses --parameter (description) format from help text."""
    pattern = (
        r"--(\w+)(?: \(([^)]+)\))?"  # Non-capturing group for optional description
    )
    matches = re.findall(pattern, help_text)
    return {f"--{match[0]}": match[1].strip() if match[1] else "" for match in matches}


def create_temp_directory(temp_dir: str = TEMP_DIR_PATH) -> bool:
    """Creates a temporary directory if it doesn't exist."""
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir)
            logger.info("Created temporary directory: %s", temp_dir)
            return True
        except OSError as e:
            # Use imported report_error
            report_error(e, f"Failed to create temporary directory {temp_dir}")
            st.error(f"Failed to create required directory '{temp_dir}': {e}")
            return False
    return True


# =============================================================================
# Caching Heavy Computations
# =============================================================================
@st.cache_data(ttl=3600)  # Cache STAR file parsing for 1 hour
def load_pipeline_star(folder: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Loads and parses the default_pipeline.star file from the given folder."""
    default_pipeline_path = os.path.join(folder, "default_pipeline.star")
    if not os.path.exists(default_pipeline_path):
        logger.warning("File not found: %s", default_pipeline_path)
        return None
    try:
        logger.info("Loading and parsing STAR file: %s", default_pipeline_path)
        pipeline_star = parse_star(default_pipeline_path)
        if not pipeline_star:
            logger.warning(
                "Parsing STAR file returned empty: %s", default_pipeline_path
            )
            return None
        logger.info("Successfully parsed STAR file: %s", default_pipeline_path)
        return pipeline_star
    except Exception as exc:
        report_error(exc, f"Failed to parse STAR file: {default_pipeline_path}")
        st.error(
            f"Failed to parse STAR file: {os.path.basename(default_pipeline_path)}. Error: {exc}"
        )
        return None


# =============================================================================
# Data Preparation Functions
# =============================================================================
def get_pipeline_df(pipeline_star: Optional[Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Extracts and prepares the 'pipeline_processes' DataFrame."""
    if not pipeline_star:
        return pd.DataFrame()
    pipeline_processes = pipeline_star.get(PIPELINE_PROCESSES_KEY)
    if not isinstance(pipeline_processes, pd.DataFrame) or pipeline_processes.empty:
        logger.warning("'%s' missing or empty in STAR data.", PIPELINE_PROCESSES_KEY)
        return pd.DataFrame()

    df = pipeline_processes.copy()
    required_cols = {
        RLN_PROCESS_TYPE_LABEL: "",
        RLN_PROCESS_NAME: "",
        RLN_STATUS_LABEL: "Unknown",
        RLN_PROCESS_ALIAS: "None",
    }
    for col, default in required_cols.items():
        if col not in df.columns:
            logger.warning(
                "Column '%s' missing, adding with default '%s'.", col, default
            )
            df[col] = default
    df[RLN_PROCESS_ALIAS] = df[RLN_PROCESS_ALIAS].fillna("None").astype(str)
    df[RLN_STATUS_LABEL] = df[RLN_STATUS_LABEL].fillna("Unknown").astype(str)
    return df


@st.cache_data
def get_jobs_for_process(
    selected_process: str, df: pd.DataFrame
) -> Tuple[List[str], Dict[str, str]]:
    """
    Filters jobs for a selected process type and creates a display key mapping.
    Jobs are returned in the order they appear in the DataFrame.

    Args:
        selected_process: Original process name (e.g., "relion.Class2D").
        df: Prepared pipeline_processes DataFrame.

    Returns:
        Tuple: (list of display keys in original order, mapping display_key -> job_name).
    """
    if df.empty or RLN_PROCESS_TYPE_LABEL not in df.columns:
        logger.debug(
            "Cannot get jobs: DataFrame empty or missing '%s'.", RLN_PROCESS_TYPE_LABEL
        )
        return [], {}

    jobs_df = df.loc[df[RLN_PROCESS_TYPE_LABEL] == selected_process]
    if jobs_df.empty:
        logger.debug("No jobs found for process type '%s'.", selected_process)
        return [], {}

    jobs_dict: Dict[str, str] = {}
    jobs_display_keys: List[str] = []  # Maintain order
    display_key_counts: Dict[str, int] = {}

    for _, row in jobs_df.iterrows():  # Iterate in DataFrame order
        alias = row.get(RLN_PROCESS_ALIAS, "None")
        job_name = row.get(RLN_PROCESS_NAME, "")
        if not job_name:
            continue

        base_display_key = (
            alias if alias != "None" and alias.strip() != "" else job_name
        )
        current_count = display_key_counts.get(base_display_key, 0)
        display_key = base_display_key
        if current_count > 0:
            display_key = (
                f"{base_display_key} ({current_count})"  # Append count for uniqueness
            )
        display_key_counts[base_display_key] = current_count + 1

        if display_key not in jobs_dict:
            jobs_dict[display_key] = job_name
            jobs_display_keys.append(display_key)  # Add to ordered list
        # Do not sort jobs_display_keys

    logger.debug(
        "Found %d jobs for process '%s'.", len(jobs_display_keys), selected_process
    )
    return jobs_display_keys, jobs_dict


# =============================================================================
# State Management Callbacks
# =============================================================================
def handle_folder_change():
    """Resets relevant session state variables when the folder changes."""
    logger.info("Resetting process/job state due to folder change.")
    keys_to_reset = [
        STATE_SELECTED_PROCESS,
        STATE_CURRENT_JOB,
        STATE_JOB_PARAMS,
        STATE_PROCESS_RADIO_KEY,
        STATE_JOB_RADIO_KEY,
        STATE_JOBS_DICT,
        STATE_DISPLAY_TO_ORIGINAL_PROCESS,
    ]
    for key in keys_to_reset:
        StateManager.set(key, None) # Reset to None or appropriate default

    StateManager.set(STATE_JOBS_DICT, {})  # Ensure these are empty dicts
    StateManager.set(STATE_JOB_PARAMS, {})
    StateManager.set(STATE_DISPLAY_TO_ORIGINAL_PROCESS, {})

    # Update query params to persist folder across refreshes
    current_folder = StateManager.get_current_folder()
    if current_folder:
        st.query_params["folder"] = current_folder


def handle_process_change():
    """Resets job state when the selected process type changes via the radio button."""
    selected_display_name = StateManager.get(STATE_PROCESS_RADIO_KEY)
    display_to_original = StateManager.get(STATE_DISPLAY_TO_ORIGINAL_PROCESS, {})
    newly_selected_process = display_to_original.get(selected_display_name)

    # Check if the *actual underlying process name* has changed
    if newly_selected_process != StateManager.get_selected_process():
        logger.info("Process selection changed to: '%s'", newly_selected_process)
        StateManager.set_selected_process(newly_selected_process)
        # Reset only job-related state
        StateManager.set_current_job(None)
        StateManager.set_job_params({})
        StateManager.set(STATE_JOB_RADIO_KEY, None)
        StateManager.set(STATE_JOBS_DICT, {})
        logger.debug("Job state reset due to process change.")


def handle_job_change():
    """Updates the current job based on the job radio button selection."""
    selected_display_key = StateManager.get(STATE_JOB_RADIO_KEY)
    jobs_dict = StateManager.get(STATE_JOBS_DICT, {})
    newly_selected_job_name = jobs_dict.get(selected_display_key)

    # Check if the *actual underlying job name* has changed
    if newly_selected_job_name != StateManager.get_current_job():
        logger.info("Job selection changed to: '%s'", newly_selected_job_name)
        StateManager.set_current_job(newly_selected_job_name)
        StateManager.set_job_params({
            "folder": StateManager.get_current_folder(),
            "process": StateManager.get_selected_process(),
        })


# =============================================================================
# Main Application Logic
# =============================================================================
def main() -> None:
    """Main function to run the Streamlit application."""
    try:
        # --- Argument Parsing ---
        parser = argparse.ArgumentParser(description="Follow Relion Gracefully Viewer")
        parser.add_argument(
            "-i",
            "--folder",
            type=str,
            help="Initial Relion project folder",
            default="~",
        )
        # parser.add_argument("--relion-path", type=str, help="Path to RELION installation (bin)", default="/")
        parser.add_argument(
            "-p",
            "--password",
            type=str,
            help="Password to protect the instance",
            default="",
        )
        parser.add_argument(
            "--watchdog-interval",
            type=int,
            help="Interval in seconds for the watchdog polling loop",
            default=2,
        )
        args, _ = parser.parse_known_args()

        # --- Session State Initialization ---
        # Initialize folder state using query params if available, else command line argument
        if not StateManager.get(STATE_DEFAULT_FOLDER):
            query_params = st.query_params
            if "folder" in query_params:
                initial_folder = query_params["folder"]
                logger.info("Initialized state: Default folder set from query param to %s", initial_folder)
            else:
                initial_folder = os.path.abspath(os.path.expanduser(args.folder))
                logger.info("Initialized state: Default folder set from arg to %s", initial_folder)

            StateManager.set_default_folder(initial_folder)
            StateManager.set_current_folder(initial_folder)

        # Ensure other states have default values if not set previously
        StateManager.initialize(STATE_CURRENT_FOLDER, StateManager.get_default_folder())
        StateManager.initialize(STATE_PASSWORD_ARGS, args.password)
        StateManager.initialize(STATE_SELECTED_PROCESS, None)
        StateManager.initialize(STATE_CURRENT_JOB, None)
        StateManager.initialize(STATE_JOB_PARAMS, {})
        StateManager.initialize(STATE_PROCESS_RADIO_KEY, None)
        StateManager.initialize(STATE_JOB_RADIO_KEY, None)
        StateManager.initialize(STATE_DISPLAY_TO_ORIGINAL_PROCESS, {})
        StateManager.initialize(STATE_JOBS_DICT, {})

        # --- Password Check ---
        if StateManager.get(STATE_PASSWORD_ARGS):
            if not check_password(
                StateManager.get(STATE_PASSWORD_ARGS)
            ):  # Pass correct arg
                st.warning("Password required.")
                st.stop()

        # --- UI Rendering & Folder Selection ---
        render_svg("./static/frg.svg")
        footer = get_footer()
        create_temp_directory()

        # dynamic_folder_explorer updates STATE_DEFAULT_FOLDER internally now
        # It uses STATE_DEFAULT_FOLDER as its starting point if uninitialized.
        # It returns the confirmed path, which we use to update STATE_CURRENT_FOLDER if needed.
        selected_folder = dynamic_folder_explorer(
            StateManager.get_default_folder()
        )

        # Detect if the folder confirmed by the explorer differs from the current viewing folder
        if selected_folder != StateManager.get_current_folder():
            logger.info(
                "Folder changed via explorer: '%s' -> '%s'",
                StateManager.get_current_folder(),
                selected_folder,
            )
            StateManager.set_current_folder(selected_folder)
            # Crucially, update the DEFAULT folder as well if the user explicitly selected it
            StateManager.set_default_folder(selected_folder)
            handle_folder_change()  # Reset dependent states
            StateManager.clear_cache()  # Clear data cache on folder change
            st.rerun()

        current_folder = StateManager.get_current_folder()

        # --- Load Data for Current Folder ---
        pipeline_star = load_pipeline_star(current_folder)  # Uses cache
        df = get_pipeline_df(pipeline_star)  # Process potentially cached data

        # --- Sidebar: Process Selection ---
        st.sidebar.title("Process Types")
        process_types = []
        if df.empty:
            st.sidebar.warning("No pipeline data found.")
        elif RLN_PROCESS_TYPE_LABEL not in df.columns:
            st.sidebar.error(f"Missing column: {RLN_PROCESS_TYPE_LABEL}")
        else:
            # Get unique process types IN THE ORDER THEY APPEAR in the DataFrame
            process_types = list(df[RLN_PROCESS_TYPE_LABEL].unique())  # No sorting here

        special_processes = [FLOWCHART_PROCESS, INTERACTIVE_PLOT_PROCESS]
        all_processes = (
            process_types + special_processes
        )  # Add special views at the end

        if not all_processes:
            st.sidebar.info("No processes found.")
            st.info("Select a valid RELION project folder.")
        else:
            # Map original names to display names
            original_to_display = {p: format_display_name(p) for p in all_processes}
            display_to_original = {d: p for p, d in original_to_display.items()}
            StateManager.set(STATE_DISPLAY_TO_ORIGINAL_PROCESS, display_to_original)
            display_options = list(original_to_display.values())

            # Determine current selection index for the radio button
            current_display = StateManager.get(STATE_PROCESS_RADIO_KEY)
            if current_display not in display_options:
                current_display = display_options[0] if display_options else None
                # If defaulting, update the underlying selected process state
                if current_display:
                    StateManager.set(STATE_PROCESS_RADIO_KEY, current_display)
                    StateManager.set_selected_process(display_to_original.get(current_display))
                    handle_process_change()  # Reset job state implicitly

            current_index = (
                display_options.index(current_display)
                if current_display in display_options
                else 0
            )

            # Process Radio Button
            st.sidebar.radio(
                "Select View:",
                options=display_options,
                index=current_index,
                key=STATE_PROCESS_RADIO_KEY,
                on_change=handle_process_change,
            )

            # --- Sidebar: Job Selection ---
            selected_process = StateManager.get_selected_process()
            if selected_process and selected_process not in special_processes:
                # Get jobs IN ORIGINAL ORDER
                jobs_display_keys, jobs_dict = get_jobs_for_process(
                    selected_process, df
                )  # Uses cache
                StateManager.set(STATE_JOBS_DICT, jobs_dict)

                if jobs_display_keys:
                    process_display_name = format_display_name(selected_process)
                    st.sidebar.title(f"{process_display_name} Jobs")

                    current_job_key = StateManager.get(STATE_JOB_RADIO_KEY)
                    # Check if current selection is valid, default to LAST job if not
                    if current_job_key not in jobs_display_keys:
                        current_job_key = jobs_display_keys[
                            -1
                        ]  # Default to last job in the list
                        StateManager.set(STATE_JOB_RADIO_KEY, current_job_key)
                        StateManager.set_current_job(jobs_dict.get(current_job_key))
                        StateManager.set_job_params({
                            "folder": current_folder,
                            "process": selected_process,
                        })

                    job_index = jobs_display_keys.index(current_job_key)

                    # Job Radio Button
                    st.sidebar.radio(
                        "Select Job:",
                        options=jobs_display_keys,
                        index=job_index,
                        key=STATE_JOB_RADIO_KEY,
                        on_change=handle_job_change,
                    )
                else:
                    st.sidebar.caption("No jobs of this type found.")
                    if StateManager.get_current_job() is not None:  # Clear state if no jobs
                        StateManager.set_current_job(None)
                        StateManager.set_job_params({})
                        StateManager.set(STATE_JOB_RADIO_KEY, None)

            # --- Main Area Display ---
            st.markdown("---")
            if selected_process == FLOWCHART_PROCESS:
                st.title("Pipeline Flowchart")

                chart_type = st.radio("Flowchart Type", ["Dynamic (Interactive)", "Static (Graphviz)"], horizontal=True)

                if pipeline_star:
                    if chart_type == "Dynamic (Interactive)":
                        nodes, edges = create_network_agraph_data(pipeline_star)

                        if nodes and edges:
                            config = Config(width=800,
                                            height=600,
                                            directed=True,
                                            nodeHighlightBehavior=True,
                                            highlightColor="#F7A7A6", # or "blue"
                                            collapsible=False,
                                            node={'labelProperty': 'label'},
                                            link={'labelProperty': 'label', 'renderLabel': False},
                                            hierarchical=True # Use hierarchical layout
                                            )

                            return_value = agraph(nodes=nodes,
                                                  edges=edges,
                                                  config=config)

                            if return_value:
                                # Assuming the return_value is the node ID
                                selected_node_id = return_value
                                logger.info(f"Node clicked in flowchart: {selected_node_id}")

                                # Check if the clicked node is a job
                                # The node ID format is expected to be "Type/JobName" or similar
                                # We need to map this back to our selection logic

                                # Try to find the job in our known processes or jobs
                                # Ideally, we should set the process and job.

                                parts = selected_node_id.split('/')
                                if len(parts) >= 2:
                                    potential_job_type = parts[0]

                                    # Check if it is a valid process type
                                    if potential_job_type in process_types:
                                        StateManager.set_selected_process(potential_job_type)
                                        StateManager.set_current_job(selected_node_id) # "Type/JobName" matches our job naming convention usually

                                        # Update radio buttons if possible, or just let the state drive the UI on rerun
                                        # We need to update params too
                                        StateManager.set_job_params({
                                            "folder": current_folder,
                                            "process": potential_job_type,
                                        })

                                        # Force radio updates by setting keys?
                                        # Radio buttons use 'index' based on options.
                                        # We updated the underlying state variables (SELECTED_PROCESS, CURRENT_JOB).
                                        # The main loop logic "Determine current selection index..." should pick this up
                                        # if we set the PROCESS_RADIO_KEY and JOB_RADIO_KEY.

                                        # Find display name for process
                                        display_map = StateManager.get(STATE_DISPLAY_TO_ORIGINAL_PROCESS)
                                        # Invert map: original -> display
                                        original_to_display = {v: k for k, v in display_map.items()} if display_map else {}

                                        process_display = original_to_display.get(potential_job_type)
                                        if process_display:
                                            StateManager.set(STATE_PROCESS_RADIO_KEY, process_display)

                                        # Job radio key is usually the display name of the job (alias or name)
                                        # We need to re-fetch the jobs list for this process to find the correct display key for this job ID
                                        # This is a bit circular because we are inside the rendering loop.
                                        # A rerun will handle the logic at the top of the script.

                                        # However, to set the job radio correctly, we might need to pre-calculate.
                                        # For now, setting selected_process and current_job is the core requirement.

                                        st.rerun()
                                    else:
                                        st.warning(f"Selected node '{selected_node_id}' does not match a known process type.")
                        else:
                            st.warning("Could not generate flowchart data.")
                    else: # Static Graphviz
                        dot_source = create_network(pipeline_star, orientation="top-bottom")
                        if dot_source:
                            st.graphviz_chart(dot_source)
                        else:
                            st.warning("Could not generate static flowchart.")

                else:
                    st.warning("Load a project first.")

            elif selected_process == INTERACTIVE_PLOT_PROCESS:
                st.title("Interactive STAR File Plot")
                col1, col2 = st.columns([1, 2])
                star_file_data = None
                plot_file_name = "Data"
                viewer_prefix = "iplot_"
                with col1.expander("Select Data", expanded=True):
                    star_path = st.text_input(
                        "STAR file path:", key=f"{viewer_prefix}path"
                    )
                    up_file = st.file_uploader(
                        "Or upload STAR:", type=["star"], key=f"{viewer_prefix}up"
                    )
                    if up_file:
                        tmp_path = os.path.join(TEMP_DIR_PATH, up_file.name)
                        try:
                            with open(tmp_path, "wb") as f:
                                f.write(up_file.getbuffer())
                            star_file_data = parse_star(tmp_path)
                            plot_file_name = up_file.name
                            os.remove(tmp_path)
                        except Exception as e:
                            st.error(f"Error processing upload: {e}")
                    elif star_path and os.path.exists(star_path):
                        star_file_data = parse_star(star_path)
                        plot_file_name = os.path.basename(star_path)
                    elif star_path:
                        st.warning("Path does not exist.")

                if star_file_data:
                    blocks = list(star_file_data.keys())
                    if blocks:
                        sel_block = col1.selectbox(
                            "Select Block:",
                            blocks,
                            index=len(blocks) - 1,
                            key=f"{viewer_prefix}block",
                        )
                        if sel_block:
                            with col2:
                                interactive_scatter_plot(
                                    data_source=star_file_data,  # Pass data dict
                                    block_selector_options=blocks,
                                    default_block=sel_block,
                                    title_prefix=plot_file_name,
                                )
                    else:
                        col1.warning("No data blocks found.")
                else:
                    col1.info("Upload or provide path to a STAR file.")

            elif selected_process:  # Regular RELION job type
                selected_job = StateManager.get_current_job()
                if selected_job:
                    display_job_info(
                        selected_job, current_folder, df, pipeline_star or {}
                    )
                else:
                    if selected_process in process_types:
                        st.info(
                            f"Select a job for '{format_display_name(selected_process)}' from the sidebar."
                        )

            else:  # No process selected
                st.info("Select a process type or view from the sidebar.")

        def on_refresh_click():
            StateManager.clear_cache()
            handle_folder_change()

        st.sidebar.button(
            "Refresh",
            key="refresh_button",
            help="Refresh the page to reload data and clear cache.",
            on_click=on_refresh_click,  # Clear cache and reset state
        )

        # Watchdog logic
        enable_watchdog = st.sidebar.checkbox("Live Watchdog", value=False, help="Automatically refresh when file system changes are detected.")

        if enable_watchdog:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            import time

            class RelionHandler(FileSystemEventHandler):
                def on_any_event(self, event):
                    # Trigger a rerun by updating a state variable if needed,
                    # but since this runs in a separate thread, we need to handle it carefully.
                    # Streamlit reruns usually happen on interaction or st.rerun().
                    # We can use st.empty() to poll or check modification times.
                    # For simplicity in Streamlit, a polling loop inside the script with st.rerun is often safer/easier
                    # than integrating the threaded observer callback directly with the main thread loop.
                    pass

            # Since threading/watchdog callbacks don't easily trigger st.rerun() in the main thread,
            # we'll use a simple polling mechanism which is often robust enough for this use case.

            last_mod_time = st.session_state.get("last_mod_time", 0)

            # Check modification time of the pipeline file as a proxy for significant changes
            pipeline_path = os.path.join(current_folder, "default_pipeline.star")
            if os.path.exists(pipeline_path):
                current_mod_time = os.path.getmtime(pipeline_path)
                if last_mod_time > 0 and current_mod_time > last_mod_time:
                    st.session_state["last_mod_time"] = current_mod_time
                    StateManager.clear_cache()
                    st.rerun()
                st.session_state["last_mod_time"] = current_mod_time

            # Poll every few seconds using st.empty() (which effectively sleeps/waits if we loop)
            # OR just rely on the script rerun cycle if we had a loop.
            # But main() runs once. Streamlit handles re-runs.
            # We can use st.fragment or simply a sleep loop if we want "live" updates without interaction.
            # However, blocking the script with a loop prevents other interactions.
            # A common pattern is `time.sleep(2); st.rerun()` but only if we are in a "live" mode.

            interval = args.watchdog_interval
            time.sleep(interval)
            st.rerun()
        # --- Footer ---
        st.sidebar.markdown("---")
        st.sidebar.markdown(footer, unsafe_allow_html=True)

    except Exception as exc:
        report_error(exc, "An unexpected error occurred in the main application.")
        st.error("An unexpected error occurred. Check logs for details.")


if __name__ == "__main__":
    set_style()
    # Set local fallback error handler for logging
    set_error_handler(local_report_error_handler)
    main()
