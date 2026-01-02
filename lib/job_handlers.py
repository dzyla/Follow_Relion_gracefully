from abc import ABC, abstractmethod
import logging
from typing import List
import streamlit as st
import os

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
from lib.utils import report_error

logger = logging.getLogger("main_app")

class RelionJob(ABC):
    @abstractmethod
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        pass

class ImportJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_import(folder, node_files)

class MotionCorrJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_motioncorr(folder, node_files[0])

class CtfFindJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_ctf_stats(folder, node_files[0])

class AutoPickJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_picks(folder, selected_job)

class ManualPickJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_picks(folder, selected_job)

class ExtractJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        process_extract(folder, node_files)

class SubtractJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        process_extract(folder, node_files)

class SelectJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_selection(node_files, folder, selected_job)

class Class2DJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_class2d(folder, node_files)

class InitialModelJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_class3d(folder, node_files)

class Class3DJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_class3d(folder, node_files)

class Refine3DJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_class3d(folder, node_files)

class MaskCreateJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_mask(folder, node_files)

class PostProcessJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_postprocess(folder, node_files)

class CtfRefineJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_ctf_refine(folder, node_files)

class PolishJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_polish(folder, node_files)

class LocalResJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_locres(node_files, folder, selected_job)

class ModelAngeloJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_modelangelo(folder, node_files)

class JoinStarJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_selection(node_files, folder, selected_job)

# Tomo Jobs
class ReconstructParticleTomoJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_class3d(folder, node_files)

class ExcludeTiltImagesJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_exclude_tilt(folder, node_files[0])

class AlignTiltSeriesJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_exclude_tilt(folder, node_files[0])

class ReconstructTomogramsJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_tomographs(folder, node_files[0])

class TomogramsJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_tomographs(folder, node_files[0])

class DenoiseJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_tomographs(folder, node_files[0])

class PicksJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_pick_tomo(folder, node_files)

class PseudoSubtomoJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        process_extract(folder, node_files)

class ReconstructJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        plot_class3d(folder, node_files)

class SubtractImagesJob(RelionJob):
    def execute(self, folder: str, node_files: List[str], selected_job: str = ""):
        process_extract(folder, node_files)

def get_job_handler(job_type: str) -> RelionJob:
    handlers = {
        "Import": ImportJob(),
        "MotionCorr": MotionCorrJob(),
        "CtfFind": CtfFindJob(),
        "AutoPick": AutoPickJob(),
        "ManualPick": ManualPickJob(),
        "Extract": ExtractJob(),
        "Subtract": SubtractJob(),
        "Select": SelectJob(),
        "Class2D": Class2DJob(),
        "InitialModel": InitialModelJob(),
        "Class3D": Class3DJob(),
        "Refine3D": Refine3DJob(),
        "MaskCreate": MaskCreateJob(),
        "PostProcess": PostProcessJob(),
        "CtfRefine": CtfRefineJob(),
        "Polish": PolishJob(),
        "LocalRes": LocalResJob(),
        "ModelAngelo": ModelAngeloJob(),
        "JoinStar": JoinStarJob(),
        "ReconstructParticleTomo": ReconstructParticleTomoJob(),
        "ExcludeTiltImages": ExcludeTiltImagesJob(),
        "AlignTiltSeries": AlignTiltSeriesJob(),
        "ReconstructTomograms": ReconstructTomogramsJob(),
        "Tomograms": TomogramsJob(),
        "Denoise": DenoiseJob(),
        "Picks": PicksJob(),
        "PseudoSubtomo": PseudoSubtomoJob(),
        "Reconstruct": ReconstructJob(),
        "SubtractImages": SubtractImagesJob(),
    }
    return handlers.get(job_type)
