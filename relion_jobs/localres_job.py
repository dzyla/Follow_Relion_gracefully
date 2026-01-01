# localres_job.py

from __future__ import annotations

import os
import re
import traceback
from typing import List, Tuple, Optional

import mrcfile
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from skimage.transform import resize

from lib.utils import report_error
import logging

logger = logging.getLogger("main_app")

# ── pymcubes ────────────────────────────────────────────────────────────────
try:
    import mcubes
except ImportError:
    mcubes = None

# ─────────────────────────────────────────────────────────────────────────────
#                               helper funcs
# ─────────────────────────────────────────────────────────────────────────────
def _load_mrc(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        return mrc.data.copy()


def _assign_maps(nodes: List[str], folder: str) -> Tuple[str, str]:
    raw, loc = "", ""
    for f in nodes:
        if not f.lower().endswith(".mrc"):
            continue
        base = os.path.basename(f).lower()
        full = os.path.join(folder, f)
        if "locres_filtered" in base:
            raw = full
        elif "locres" in base and "filtered" not in base:
            loc = full
        if raw and loc:
            break
    if not raw or not loc:
        raise ValueError("Missing relion_locres_filtered.mrc or relion_locres.mrc")
    return raw, loc


def _orthogonal_slices(vol: np.ndarray, idx: int) -> tuple[np.ndarray, ...]:
    i = int(np.clip(idx, 0, min(vol.shape) - 1))
    return vol[i], vol[:, i, :], vol[:, :, i]


def _sample_vertex_values(vol: np.ndarray, verts: np.ndarray) -> np.ndarray:
    x = np.clip(np.round(verts[:, 0]).astype(int), 0, vol.shape[0] - 1)
    y = np.clip(np.round(verts[:, 1]).astype(int), 0, vol.shape[1] - 1)
    z = np.clip(np.round(verts[:, 2]).astype(int), 0, vol.shape[2] - 1)
    return vol[x, y, z]

# ───────────────────────────── 3-D isosurface ───────────────────────────────
def _plot_isosurface(
    raw_vol: np.ndarray,
    loc_vol: np.ndarray,
    rel_thr: float,
    *,
    max_size: int,
    colourscale: str,
) -> Optional[go.Figure]:
    if mcubes is None:
        st.error("Install *pymcubes* for 3-D rendering.")
        return None
    if raw_vol.shape != loc_vol.shape:
        st.warning("Shape mismatch.")
        return None

    # down-sample for performance
    if max(raw_vol.shape) > max_size:
        fac = max_size / max(raw_vol.shape)
        new_shape = tuple(int(round(d * fac)) for d in raw_vol.shape)
        raw_vol = resize(raw_vol, new_shape, preserve_range=True, anti_aliasing=False)
        loc_vol = resize(loc_vol, new_shape, preserve_range=True, anti_aliasing=False)

    level = raw_vol.min() + rel_thr * (raw_vol.max() - raw_vol.min())
    verts, faces = mcubes.marching_cubes(np.ascontiguousarray(raw_vol, np.float32), level)
    if verts.size == 0:
        return None

    intens = _sample_vertex_values(loc_vol, verts)
    cmin, cmax = float(loc_vol.min()), float(loc_vol.max())
    if cmin == cmax:
        cmax = cmin + 1e-6

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=intens,
                colorscale=colourscale,
                cmin=cmin, cmax=cmax,
                showscale=True,
                opacity=1.0,
                lighting=dict(ambient=0.3, diffuse=0.5, specular=0.2),
                lightposition=dict(x=0, y=0, z=0),
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor="black", dragmode="orbit"
        ),
        paper_bgcolor="black",
        margin=dict(l=10, r=10, t=30, b=10),
        height=600,
        title=f"Isosurface (thr ={rel_thr:.2f}) – clipped above ceiling",
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#                                main entry
# ─────────────────────────────────────────────────────────────────────────────
def plot_locres(nodes: List[str], folder: str, job: str) -> None:
    """
    Streamlit visualiser with:
      • view radio  (3-D / 2-D)
      • threshold slider
      • max-dim slider
      • colour-scale select
      • *new* resolution-ceiling slider
    """
    try:
        raw_p, loc_p = _assign_maps(nodes, folder)

        key_raw = f"raw_{job}_{os.path.getmtime(raw_p)}"
        key_loc = f"loc_{job}_{os.path.getmtime(loc_p)}"
        if key_raw not in st.session_state:
            with st.spinner("Loading raw map…"):
                st.session_state[key_raw] = _load_mrc(raw_p)
        if key_loc not in st.session_state:
            with st.spinner("Loading resolution map…"):
                st.session_state[key_loc] = _load_mrc(loc_p)

        raw = st.session_state[key_raw]
        loc = st.session_state[key_loc]

        # optional mask
        note = os.path.join(folder, job, "note.txt")
        if os.path.isfile(note):
            txt = open(note, encoding="utf-8").read()
            m = re.search(r"--mask\s+([\w\d/\\\.-]+\.mrc)", txt, re.I)
            if m:
                mpath = os.path.join(folder, m.group(1))
                if os.path.isfile(mpath):
                    mask = _load_mrc(mpath) > 0
                    raw = np.where(mask, raw, 0)
                    loc = np.where(mask, loc, 0)
                    #st.info(f"Mask applied: {os.path.basename(mpath)}")

        # convert zeros to NaN
        loc_float = loc.astype(float)
        loc_float[loc == 0] = np.nan

        # ── UI controls ───────────────────────────────────────────────────
        view = st.radio("View mode:", ("3-D isosurface", "2-D slices"), horizontal=True)

        # common resolution-ceiling slider
        finite_all = loc_float[np.isfinite(loc_float)]
        res_min, res_max = float(finite_all.min()), float(finite_all.max())
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            res_clip = st.slider(
                "Max resolution (Å)",
                min_value=round(res_min, 1),
                max_value=round(res_max, 1),
                value=round(res_max, 1),
                step=0.1,
                help="Values above this are clipped (set equal to the ceiling).",
            )

        # apply clipping for visualisation
        loc_clip = np.minimum(loc_float, res_clip)

        if view == "2-D slices":
            idx = col2.slider(
                "Slice index", 0, min(loc.shape) - 1, (min(loc.shape) - 1) // 2, key=f"{job}_idx"
            )
            slices = _orthogonal_slices(loc_clip, idx)
            vmin, vmax = float(loc_clip.min()), float(loc_clip.max())
            if vmin == vmax:
                vmax = vmin + 1e-6

            fig_s = make_subplots(rows=1, cols=3, subplot_titles=("XY", "XZ", "YZ"))
            for c, slc in enumerate(slices, 1):
                fig_s.add_trace(
                    go.Heatmap(
                        z=slc.T,
                        colorscale="RdYlBu_r",
                        zmin=vmin,
                        zmax=vmax,
                        showscale=(c == 3),
                        colorbar=dict(title="Å") if c == 3 else None,
                        hoverinfo="skip",
                    ),
                    row=1, col=c,
                )
                fig_s.update_xaxes(visible=False, row=1, col=c)
                fig_s.update_yaxes(visible=False, row=1, col=c)
            fig_s.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=400)
            fig_s.update_layout(
                scene=dict(
                    xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                    aspectmode="cube"
                ),
                margin=dict(l=10, r=10, t=30, b=10),
                height=600,
            )

            st.plotly_chart(fig_s, use_container_width=True)

        else:  # 3-D
            with col2:
                thr = st.slider("Relative threshold", 0.05, 0.95, 0.5, 0.01, key=f"{job}_thr")
            with col3:
                dim = st.slider("Max dimension (px)", 80, 256, 200, 16, key=f"{job}_dim")
            with col4:
                colours = st.selectbox(
                    "Colour scale",
                    ("Turbo", "Viridis", "Cividis", "Plasma", "Inferno", "Haline", "RdYlBu", "RdBu", "Jet", "Rainbow"),
                    index=0,
                    key=f"{job}_cmap",
                )

            white_bg = st.checkbox("White Background", key=f"{job}_bg")

            with st.spinner("Rendering 3-D view…"):
                fig_iso = _plot_isosurface(raw, loc_clip, thr, max_size=dim, colourscale=colours)
            if fig_iso:
                if white_bg:
                    fig_iso.update_layout(
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        scene=dict(bgcolor="white")
                    )
                    # Update text colors if needed for visibility
                    fig_iso.update_layout(font=dict(color="black"))
                st.plotly_chart(fig_iso, use_container_width=True)

        # ── histogram / KDE (use clipped values) ───────────────────────────
        st.subheader("Local-resolution distribution")
        finite_vals = loc_clip[np.isfinite(loc_clip)]
        if finite_vals.size:
            kde = gaussian_kde(finite_vals)
            xs = np.linspace(finite_vals.min(), finite_vals.max(), 200)
            p10, p25, p50, p75, p90 = np.percentile(finite_vals, [10, 25, 50, 75, 90])

            fig_h = go.Figure()
            fig_h.add_trace(
                go.Histogram(
                    x=finite_vals,
                    histnorm="probability density",
                    marker_color="#9cc2cb",
                    opacity=0.6,
                    name="Histogram",
                )
            )
            fig_h.add_trace(
                go.Scatter(x=xs, y=kde(xs), mode="lines", name="KDE", line=dict(color="#417996", width=3))
            )
            for val, lbl, colr in [
                (p10, "10 %", "grey"),
                (p25, "25 %", "orange"),
                (p50, "50 %", "red"),
                (p75, "75 %", "orange"),
                (p90, "90 %", "grey"),
            ]:
                fig_h.add_vline(x=val, line=dict(color=colr, dash="dash"), annotation_text=lbl, annotation_position="top")

            fig_h.update_layout(
                xaxis_title="Resolution (Å)",
                yaxis_title="Density",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_h, use_container_width=True)
            st.caption(
                f"10 %: {p10:.2f} Å   25 %: {p25:.2f} Å   Median: {p50:.2f} Å   "
                f"75 %: {p75:.2f} Å   90 %: {p90:.2f} Å   (clipped at {res_clip:.2f} Å)"
            )
        else:
            st.warning("No finite resolution values to plot.")

    except Exception as exc:
        st.error(f"plot_locres failed: {exc}")
        report_error(exc)
        logger.error(f"plot_locres error\n{traceback.format_exc()}")
