import streamlit as st
from typing import Any, Dict, List, Optional
from lib.constants import *

class StateManager:
    """
    Centralized manager for Streamlit session state.
    """

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        st.session_state[key] = value

    @staticmethod
    def initialize(key: str, default: Any) -> None:
        if key not in st.session_state:
            st.session_state[key] = default

    @staticmethod
    def get_selected_process() -> Optional[str]:
        return st.session_state.get(STATE_SELECTED_PROCESS)

    @staticmethod
    def set_selected_process(process: Optional[str]) -> None:
        st.session_state[STATE_SELECTED_PROCESS] = process

    @staticmethod
    def get_current_job() -> Optional[str]:
        return st.session_state.get(STATE_CURRENT_JOB)

    @staticmethod
    def set_current_job(job: Optional[str]) -> None:
        st.session_state[STATE_CURRENT_JOB] = job

    @staticmethod
    def get_current_folder() -> str:
        return st.session_state.get(STATE_CURRENT_FOLDER, "~")

    @staticmethod
    def set_current_folder(folder: str) -> None:
        st.session_state[STATE_CURRENT_FOLDER] = folder

    @staticmethod
    def get_default_folder() -> str:
        return st.session_state.get(STATE_DEFAULT_FOLDER, "~")

    @staticmethod
    def set_default_folder(folder: str) -> None:
        st.session_state[STATE_DEFAULT_FOLDER] = folder

    @staticmethod
    def get_job_params() -> Dict:
        return st.session_state.get(STATE_JOB_PARAMS, {})

    @staticmethod
    def set_job_params(params: Dict) -> None:
        st.session_state[STATE_JOB_PARAMS] = params

    @staticmethod
    def reset_job_state() -> None:
        st.session_state[STATE_CURRENT_JOB] = None
        st.session_state[STATE_JOB_PARAMS] = {}
        st.session_state[STATE_JOB_RADIO_KEY] = None
        st.session_state[STATE_JOBS_DICT] = {}

    @staticmethod
    def clear_cache() -> None:
        st.cache_data.clear()
