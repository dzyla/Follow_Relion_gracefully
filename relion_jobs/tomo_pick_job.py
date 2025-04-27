# tomo_pick_job.py

import os
import glob
import logging
import traceback
from typing import List, Any

import numpy as np
import pandas as pd
import streamlit as st
import mrcfile
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from morphosamplers import Sphere, sphere_samplers

from lib.utils import parse_star
from lib.image_utils import process_micrograph 

logger = logging.getLogger("main_app")


def report_error(exc: Exception) -> None:
    """
    Log a full traceback for errors.
    """
    error_info = traceback.format_exc()
    logger.error("An unexpected error occurred:\n%s", error_info)


def convert_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns of DataFrame to float where possible.
    """
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    return df


def reduce_resolution(data: np.ndarray, n: int) -> np.ndarray:
    """
    Reduces the resolution of the data by sampling every nth pixel in X and Y.
    """
    if n <= 0:
        raise ValueError("The interval n must be a positive integer.")
    return data[:, ::n, ::n]


def circle_diameter_at_z(z_centers, radii, z_value):
    """
    Computes 2D circle diameters where a plane z=z_value intersects multiple spheres.

    Returns an array of diameters (one per sphere),
    with zeros for spheres not intersecting the z-slice.
    """
    z_centers = np.array(z_centers, dtype=float)
    radii = np.array(radii, dtype=float)

    # Distance of each center from the plane z=z_value
    z_distances = np.abs(z_centers - z_value)

    # Identify which spheres intersect this z-slice
    intersects = z_distances <= radii

    # Compute the circle radius at intersection
    circle_radii = np.sqrt(np.maximum(0, radii**2 - z_distances**2))

    # Build an array of diameters, zero if no intersection
    circle_diameters = np.zeros_like(z_centers)
    circle_diameters[intersects] = 2 * circle_radii[intersects]
    return circle_diameters


def plot_micrograph(
    micrograph: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    img_resize_fac: float = 1.0,
    s: Any = 120,
    opacity: float = 0.8,
) -> None:
    """
    2D Micrograph Plot with picks overlay using matplotlib inline in Streamlit.

    If 's' is a scalar, it will be multiplied by 40 for better visibility.
    If 's' is an array, it will be scaled by 40 so each pick's size is visible.
    """
    fig = plt.figure(dpi=200, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(micrograph, cmap="gray")

    if len(coords_x) > 0 and len(coords_y) > 0:
        if isinstance(s, (int, float)):
            logger.info(f"Using scalar size for markers: {s}")
            marker_size = s * 40
        else:
            # Scale the entire array by 40 so it is easier to see in the plot
            marker_size = np.array(s) * 40.0

        ax.scatter(
            coords_x * img_resize_fac,
            coords_y * img_resize_fac,
            edgecolor="limegreen",
            facecolors="none",
            s=marker_size,
            linewidth=1,
            alpha=opacity,
        )

    ax.axis("off")
    st.pyplot(fig, use_container_width=False)


def generate_sphere_mesh(cx, cy, cz, r, steps=10):
    """
    Generates x, y, z coordinates plus triangular indices (i, j, k)
    for a sphere of radius r centered at (cx, cy, cz).
    steps controls the mesh resolution.
    """
    phi = np.linspace(0, np.pi, steps)
    theta = np.linspace(0, 2 * np.pi, steps)

    x_ = []
    y_ = []
    z_ = []

    for p in phi:
        for t in theta:
            x_.append(cx + r * np.sin(p) * np.cos(t))
            y_.append(cy + r * np.sin(p) * np.sin(t))
            z_.append(cz + r * np.cos(p))

    x_ = np.array(x_)
    y_ = np.array(y_)
    z_ = np.array(z_)

    i_ = []
    j_ = []
    k_ = []
    for ip in range(steps - 1):
        for it in range(steps - 1):
            idx = ip * steps + it
            idx_right = ip * steps + it + 1
            idx_down = (ip + 1) * steps + it
            idx_down_right = (ip + 1) * steps + it + 1

            # Two triangles per quad
            i_.append(idx)
            j_.append(idx_right)
            k_.append(idx_down)

            i_.append(idx_right)
            j_.append(idx_down_right)
            k_.append(idx_down)

    return x_, y_, z_, i_, j_, k_


def plot_micrograph_3D(
    micrograph: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    img_resize_fac: float,
    image_z: int,
    color: any = None,
    s: float = 5,
    opacity: float = 0.5,
    use_mesh: bool = False,
    mesh_resolution: int = 20
) -> None:
    """
    Renders a 3D volume slice in Plotly with scatter points in 3D or sphere meshes.
    """
    fig = go.Figure()

    # Display the 2D micrograph as a 'surface' at plane Z = image_z/2
    fig.add_trace(
        go.Surface(
            z=np.ones(micrograph.shape) * (image_z / 2.0),
            surfacecolor=micrograph,
            colorscale="gray",
            showscale=False,
            opacity=1.0,
        )
    )

    if len(coords_x) > 0 and len(coords_y) > 0 and len(coords_z) > 0:
        if not use_mesh:
            marker_size = s
            if isinstance(s, np.ndarray):
                marker_size = float(np.mean(s))

            fig.add_trace(
                go.Scatter3d(
                    x=coords_x * img_resize_fac,
                    y=coords_y * img_resize_fac,
                    z=coords_z * img_resize_fac,
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=(color if isinstance(color, (list, np.ndarray)) else "limegreen"),
                        opacity=opacity,
                    ),
                )
            )
        else:
            if isinstance(s, (int, float)):
                sphere_radii = np.full_like(coords_x, float(s), dtype=float)
            else:
                sphere_radii = np.array(s, dtype=float)

            if not isinstance(color, (list, np.ndarray)):
                color = [color if color else "limegreen"] * len(coords_x)
            else:
                color = np.array(color, dtype=object)

            for idx, (cx, cy, cz, r) in enumerate(zip(coords_x, coords_y, coords_z, sphere_radii)):
                x_sphere, y_sphere, z_sphere, i_sphere, j_sphere, k_sphere = generate_sphere_mesh(
                    cx * img_resize_fac, 
                    cy * img_resize_fac, 
                    cz * img_resize_fac, 
                    r * img_resize_fac, 
                    steps=mesh_resolution
                )
                c_sphere = color[idx] if idx < len(color) else "limegreen"

                fig.add_trace(
                    go.Mesh3d(
                        x=x_sphere,
                        y=y_sphere,
                        z=z_sphere,
                        i=i_sphere,
                        j=j_sphere,
                        k=k_sphere,
                        color=c_sphere,
                        opacity=opacity,
                        flatshading=True,
                    )
                )

    fig.update_layout(
        scene=dict(aspectmode="data"),
        height=600,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_3d_picks(df: pd.DataFrame) -> None:
    """
    Simple 3D scatter using Plotly Express for user picks, colored by a chosen column if it exists.
    """
    df = convert_to_float(df)
    color_col = "_rlnCenteredCoordinateZAngst"
    if "_rlnTomoSubtomogramTilt" in df.columns:
        color_col = "_rlnTomoSubtomogramTilt"

    fig = px.scatter_3d(
        df,
        x="_rlnCenteredCoordinateXAngst",
        y="_rlnCenteredCoordinateYAngst",
        z="_rlnCenteredCoordinateZAngst",
        color=color_col,
        size_max=2,
        title=f"Particle picks colored by {color_col}",
    )
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_scenes(aspectmode="data")
    st.plotly_chart(fig, use_container_width=True)


def plot_pick_tomo(folder: str, node_files: List[str]) -> None:
    """
    Main function to pick tomogram slices or volumes, overlay annotations or particles,
    and display them in Streamlit. The UI is organized so that the left column (ratio 1)
    has all the controls, while the right column (ratio 4) shows the final image or 3D view.
    """
    logger.info(f"Working on tomogram pick job... folder: {folder}, node_files: {node_files}")

    # 1) Parse the main star file with particles
    particles_star_path = os.path.join(folder, node_files[0])
    try:
        star_dict = parse_star(particles_star_path)
    except Exception as exc:
        report_error(exc)
        st.error(f"Failed to parse the star file: {particles_star_path}")
        return

    if "particles" not in star_dict:
        st.error("Could not find 'particles' in the star file.")
        return

    # 2) Parse the optimisation_set star for advanced volume data
    optimisation_star_path = os.path.join(folder, node_files[1])
    try:
        star_dict_opt = parse_star(optimisation_star_path)
    except Exception as exc:
        report_error(exc)
        st.error(f"Failed to parse the star file: {optimisation_star_path}")
        return

    if "optimisation_set" not in star_dict_opt:
        st.error("Could not find 'optimisation_set' in the star file.")
        return

    particles_df = star_dict["particles"]
    unique_tomo = np.unique(particles_df["_rlnTomoName"])

    col_controls, col_display = st.columns([1, 4])

    with col_controls:
        st.subheader("Tomogram & Z-Slice")

        if len(unique_tomo) > 1:
            idx_tomo = st.slider("Select tomogram", 0, len(unique_tomo) - 1, 0)
        else:
            idx_tomo = 0

        selected_tomo = unique_tomo[idx_tomo]
        st.write(f"**Selected tomogram**: {selected_tomo}")

        selected_particles = particles_df[particles_df["_rlnTomoName"] == selected_tomo].copy()
        selected_particles = convert_to_float(selected_particles)

        # Look for an 'annotations' folder
        node_dir = os.path.dirname(node_files[0])  # e.g., "some_node"
        annotation_path_pattern = os.path.join(folder, node_dir, "*")
        matched_paths = glob.glob(annotation_path_pattern)

        annotation_found = False
        annotation_files = []
        for path_item in matched_paths:
            if "annotations" in path_item.lower():
                annotation_found = True
                annotation_files = sorted(glob.glob(os.path.join(path_item, "*.star")))
                break
        
        
        # Default UI states
        show_spacing_slider = True
        show_circle_diam_slider = True

        z_slice = 0
        if annotation_found:
            opt_set = star_dict_opt["optimisation_set"]
            tomo_star_file = opt_set["_rlnTomoTomogramsFile"].values[0]
            tomo_star_path = os.path.join(folder, tomo_star_file)

            try:
                tomo_star = parse_star(tomo_star_path)["global"]
            except Exception as exc:
                report_error(exc)
                st.error(f"Failed to parse tomogram star: {tomo_star_path}")
                return

            # Decide if we have a denoised or half volume
            if "_rlnTomoReconstructedTomogramDenoised" in tomo_star.columns:
                volumes = tomo_star["_rlnTomoReconstructedTomogramDenoised"].values
            else:
                volumes = tomo_star["_rlnTomoReconstructedTomogramHalf1"].values

            if idx_tomo >= len(volumes):
                st.error("Tomogram index out of range for available volumes.")
                return

            selected_tomo_file = volumes[idx_tomo]
            tomo_path = os.path.join(folder, selected_tomo_file)

            if idx_tomo >= len(annotation_files):
                st.warning("Index out of range for annotation files; using 0.")
                anno_idx = 0
            else:
                anno_idx = idx_tomo
            annotation_star_path = annotation_files[anno_idx]

            # Load MRC and reduce resolution
            downsample_xy_factor = 2
            try:
                with mrcfile.mmap(tomo_path, permissive=True) as mrc:
                    data_3d = mrc.data
                volume_reduced = reduce_resolution(data_3d, downsample_xy_factor)
            except Exception as exc:
                report_error(exc)
                st.error(f"Could not load MRC or reduce resolution: {tomo_path}")
                return

            max_z = volume_reduced.shape[0] - 1
            z_slice = st.slider("Z slice", 0, max_z, int(max_z / 2), step=1)
            # Select plot type (2D or 3D)
            plot_type = st.radio("Plot Type", ["2D", "3D"], index=1)

        with st.expander("Advanced Visualization"):
            gaussian_blur = st.slider("Gaussian blur (std dev)", 0.0, 5.0, 0.1, 0.1)
            intensity_rescale = st.slider(
                "Intensity rescale (percentiles)",
                min_value=0.0,
                max_value=100.0,
                value=(1.0, 99.0),
                step=0.1,
            )

            select_view_type = "Particles"
            picks_opacity = 0.8
            radius_in_annotation = False
            if annotation_found:
                select_view_type = st.radio("Show Annotations or Particles?", ["Annotations", "Particles"])
                # Quickly check if radius is in the annotation table
                try:
                    tmp_anno_dict = parse_star(annotation_star_path)
                    tmp_anno_key = list(tmp_anno_dict.keys())[0]
                    tmp_anno_df = convert_to_float(tmp_anno_dict[tmp_anno_key])
                    if "_rlnSphereRadius" in tmp_anno_df.columns:
                        radius_in_annotation = True
                except:
                    pass

            # Hide or show sliders depending on scenario:
            if plot_type == "2D":
                if radius_in_annotation and select_view_type == "Annotations":
                    # Scenario 1 => fixed annotation size => hide both sliders
                    show_spacing_slider = False
                    show_circle_diam_slider = False
                else:
                    # Scenario 2 or 3 => we allow circle diameter if the user wants
                    # (But if there's no radius, we have just user picks => circle slider remains)
                    pass

            if show_spacing_slider:
                spacing_ang = st.slider("Particle spacing (Å)", 5, 200, 100, step=1)
            else:
                spacing_ang = 10  # default

            if show_circle_diam_slider:
                circle_scale_factor = st.slider("Circle diameter scale factor (2D mode)", 0.05, 10.0, 1.0, 0.05)
            else:
                circle_scale_factor = 1.0

            picks_opacity = st.slider("Picks opacity", 0.1, 1.0, 0.6, 0.1)

    with col_display:
        st.subheader("Tomogram Viewer")

        if not annotation_found:
            st.write("No annotation folder found. Displaying available picks in 3D.")
            if len(selected_particles) == 0:
                st.warning("No picks found for this tomogram.")
                return
            plot_3d_picks(selected_particles)
            return

        # Parse the annotation star
        try:
            anno_star_dict = parse_star(annotation_star_path)
            anno_key = list(anno_star_dict.keys())[0]
            annotation_star = anno_star_dict[anno_key]
            annotation_star = convert_to_float(annotation_star)
        except Exception as exc:
            report_error(exc)
            st.error(f"Failed to parse annotation file: {annotation_star_path}")
            return

        # Process micrograph slice
        slice_2d = volume_reduced[z_slice].copy()
        try:
            slice_2d, _ = process_micrograph(
                slice_2d,
                #img_resize_fac=1,
                gaussian_blur_stdev=gaussian_blur,
                low_percent=intensity_rescale[0],
                high_percent=intensity_rescale[1],
            )
        except Exception as exc:
            report_error(exc)
            st.error("Failed processing micrograph slice.")
            return

        z_lower, z_upper = z_slice - 5, z_slice + 5

        # -------------------- 2D Mode --------------------
        if plot_type == "2D":
            if select_view_type == "Annotations":
                if "_rlnSphereRadius" in annotation_star.columns:
                    diam_array = circle_diameter_at_z(
                        annotation_star["_rlnCoordinateZ"],
                        annotation_star["_rlnSphereRadius"],
                        z_slice,
                    )
                    # scenario 1 => user scale factor or forced to 1.0
                    diam_array_scaled = diam_array * circle_scale_factor
                    plot_micrograph(
                        slice_2d,
                        annotation_star["_rlnCoordinateX"],
                        annotation_star["_rlnCoordinateY"],
                        1 / downsample_xy_factor,
                        s=diam_array_scaled,
                        opacity=picks_opacity,
                    )
                else:
                    # scenario 2 => radius missing => just show picks near this slice
                    anno_subset = annotation_star[
                        (annotation_star["_rlnCoordinateZ"] >= z_lower)
                        & (annotation_star["_rlnCoordinateZ"] <= z_upper)
                    ]
                    # Use circle_scale_factor to scale the point sizes if you want them adjustable
                    # For example:
                    diam_array_scaled = 10 * circle_scale_factor
                    plot_micrograph(
                        slice_2d,
                        anno_subset["_rlnCoordinateX"],
                        anno_subset["_rlnCoordinateY"],
                        1 / downsample_xy_factor,
                        s=diam_array_scaled,
                        opacity=picks_opacity,
                    )
            else:
                # "Particles" => scenario 2 or 3
                if "_rlnSphereRadius" not in annotation_star.columns:
                    # no radius => just user picks near the slice, but we want the circle slider to work
                    anno_subset = annotation_star[
                        (annotation_star["_rlnCoordinateZ"] >= z_lower)
                        & (annotation_star["_rlnCoordinateZ"] <= z_upper)
                    ]
                    diam_array_scaled = 10 * circle_scale_factor
                    plot_micrograph(
                        slice_2d,
                        anno_subset["_rlnCoordinateX"],
                        anno_subset["_rlnCoordinateY"],
                        1 / downsample_xy_factor,
                        s=diam_array_scaled,
                        opacity=picks_opacity,
                    )
                else:
                    # radius present => scenario 3 => user can pick spacing, scale factor, etc.
                    bin_factor = float(tomo_star["_rlnTomoTomogramBinning"].values[0])
                    px_size = float(tomo_star["_rlnMicrographOriginalPixelSize"].values[0])
                    spacing_angstroms = spacing_ang / bin_factor / px_size

                    centers = annotation_star[
                        ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]
                    ].to_numpy()
                    radii = annotation_star["_rlnSphereRadius"].to_numpy()

                    dfs = []
                    for c, r in zip(centers, radii):
                        sph = Sphere(center=c, radius=r)
                        sampler = sphere_samplers.PoseSampler(spacing=spacing_angstroms)
                        poses = sampler.sample(sph)
                        dfp = pd.DataFrame({
                            "_rlnCoordinateX": poses.positions[:, 0],
                            "_rlnCoordinateY": poses.positions[:, 1],
                            "_rlnCoordinateZ": poses.positions[:, 2],
                        })
                        dfs.append(dfp)
                    part_df = pd.concat(dfs, ignore_index=True)
                    st.write(f"**Total Particles**: {len(part_df)}")

                    part_df_slice = part_df[
                        (part_df["_rlnCoordinateZ"] >= z_lower) & (part_df["_rlnCoordinateZ"] <= z_upper)
                    ]
                    diam_array_scaled = 5 * circle_scale_factor
                    plot_micrograph(
                        slice_2d,
                        part_df_slice["_rlnCoordinateX"],
                        part_df_slice["_rlnCoordinateY"],
                        1 / downsample_xy_factor,
                        s=diam_array_scaled,
                        opacity=picks_opacity,
                    )

        # -------------------- 3D Mode --------------------
        else:
            bin_factor = float(tomo_star["_rlnTomoTomogramBinning"].values[0])
            px_size = float(tomo_star["_rlnMicrographOriginalPixelSize"].values[0])
            spacing_angstroms = spacing_ang / bin_factor / px_size

            if select_view_type == "Annotations":
                if "_rlnSphereRadius" in annotation_star.columns:
                    plot_micrograph_3D(
                        slice_2d,
                        annotation_star["_rlnCoordinateX"],
                        annotation_star["_rlnCoordinateY"],
                        annotation_star["_rlnCoordinateZ"],
                        1 / downsample_xy_factor,
                        z_slice,
                        s=np.array(annotation_star["_rlnSphereRadius"]),
                        opacity=picks_opacity,
                        use_mesh=True,
                    )
                else:
                    plot_micrograph_3D(
                        slice_2d,
                        annotation_star["_rlnCoordinateX"],
                        annotation_star["_rlnCoordinateY"],
                        annotation_star["_rlnCoordinateZ"],
                        1 / downsample_xy_factor,
                        z_slice,
                        opacity=picks_opacity,
                    )
            else:
                # "Particles"
                if "_rlnSphereRadius" not in annotation_star.columns:
                    logger.info("No '_rlnSphereRadius' in annotation star. Plotting existing picks without sampling.")
                    plot_micrograph_3D(
                        slice_2d,
                        annotation_star["_rlnCoordinateX"],
                        annotation_star["_rlnCoordinateY"],
                        annotation_star["_rlnCoordinateZ"],
                        1 / downsample_xy_factor,
                        z_slice,
                        opacity=picks_opacity,
                    )
                else:
                    centers = annotation_star[
                        ["_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ"]
                    ].to_numpy()
                    radii = annotation_star["_rlnSphereRadius"].to_numpy()

                    df_list = []
                    for c, r in zip(centers, radii):
                        sph = Sphere(center=c, radius=r)
                        sampler = sphere_samplers.PoseSampler(spacing=spacing_angstroms)
                        poses = sampler.sample(sph)
                        dfp = pd.DataFrame({
                            "_rlnCoordinateX": poses.positions[:, 0],
                            "_rlnCoordinateY": poses.positions[:, 1],
                            "_rlnCoordinateZ": poses.positions[:, 2],
                        })
                        df_list.append(dfp)
                    all_parts_df = pd.concat(df_list, ignore_index=True)
                    st.write(f"**Total 3D Particles**: {len(all_parts_df)}")

                    plot_micrograph_3D(
                        slice_2d,
                        all_parts_df["_rlnCoordinateX"],
                        all_parts_df["_rlnCoordinateY"],
                        all_parts_df["_rlnCoordinateZ"],
                        1 / downsample_xy_factor,
                        z_slice,
                        opacity=picks_opacity,
                    )


'''
import os
import glob
import logging
import traceback
from typing import List, Any

import numpy as np
import pandas as pd
import streamlit as st
import mrcfile
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from utils import parse_star
from morphosamplers import Sphere, sphere_samplers  # same as before
from image_utils import process_micrograph 

logger = logging.getLogger("main_app")

def report_error(exc: Exception) -> None:
    error_info = traceback.format_exc()
    logger.error("An unexpected error occurred:\n%s", error_info)


def convert_to_float(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    return df


def reduce_resolution(data: np.ndarray, n: int) -> np.ndarray:
    """
    Only downsample in X and Y by factor n, leaving Z fully sampled.
    data.shape = (Z, Y, X).
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    return data[:, ::n, ::n]


def plot_micrograph(
    micrograph: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    img_resize_fac: float = 1.0,
    s: Any = 120,
    opacity: float = 0.8,
) -> None:
    """
    2D Micrograph Plot with picks overlay using matplotlib inline in Streamlit.
    Reverted to original scaling for XY. 
    """
    fig = plt.figure(dpi=200, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(micrograph, cmap="gray")

    if len(coords_x) and len(coords_y):
        if isinstance(s, (int, float)):
            marker_size = s * 40
        else:
            marker_size = np.array(s) * 40

        ax.scatter(
            coords_x * img_resize_fac,
            coords_y * img_resize_fac,
            edgecolor="limegreen",
            facecolors="none",
            s=marker_size,
            linewidth=1,
            alpha=opacity,
        )

    ax.axis("off")
    st.pyplot(fig, use_container_width=False)


def generate_plane_mesh_3d(
    slice_2d: np.ndarray,
    slice_idx: float,
    projection: str,
    downsample_xy_factor: int,
    px_size_z: float = 1.0,
    px_size_xy: float = 1.0,
) -> go.Surface:
    """
    Build a 3D surface for slice_2d (shape HxW) oriented correctly 
    for XY, XZ, or YZ. 
    We assume each pixel in X/Y is px_size_xy in world units, 
    and each step in Z is px_size_z in world units.
    The 'downsample_xy_factor' indicates how much X/Y were reduced.
    """
    H, W = slice_2d.shape
    # Build a coordinate mesh
    # row in [0, H), col in [0, W)
    yy, xx = np.mgrid[0:H, 0:W]  # shape => (H, W) each

    # We'll multiply X or Y by (px_size_xy * downsample_xy_factor if needed).
    # The user wants the final plane to match original picks if the picks are in "true" XY units.
    # For a simpler approach: if we skip every nth pixel, 
    # the "actual" dimension is W*n in X. So we do e.g. x = col * px_size_xy * n
    # so that the plane lines up in 3D with the original coordinate system.
    # The same for Y => row * px_size_xy * n.
    x_world = xx * px_size_xy * downsample_xy_factor
    y_world = yy * px_size_xy * downsample_xy_factor

    # We'll set up the plane:
    if projection == "XY":
        # The plane is Z=slice_idx, X=x_world, Y=y_world
        xSurf = x_world
        ySurf = y_world
        zSurf = np.ones_like(x_world) * (slice_idx * px_size_z)
    elif projection == "XZ":
        # The plane is Y=slice_idx => y= slice_idx*px_size_xy
        # so row=Z => z= row*px_size_z
        # col=X => x= col*px_size_xy*n
        xSurf = x_world
        zSurf = y_world * px_size_z  # interpret 'y_world' as "rows" => z dimension
        ySurf = np.ones_like(x_world) * (slice_idx * px_size_xy * downsample_xy_factor)
    else:  # "YZ"
        # plane X=slice_idx => x= slice_idx*px_size_xy
        # row=Z => z= row*px_size_z
        # col=Y => y= col*px_size_xy*n
        ySurf = x_world
        zSurf = y_world * px_size_z
        xSurf = np.ones_like(x_world) * (slice_idx * px_size_xy * downsample_xy_factor)

    surface = go.Surface(
        x=xSurf,
        y=ySurf,
        z=zSurf,
        surfacecolor=slice_2d,
        colorscale="gray",
        showscale=False,
        opacity=1.0,
    )
    return surface


def plot_picks_3d(
    fig: go.Figure,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    color: Any = None,
    s: Any = 5,
    opacity: float = 0.5,
    use_mesh: bool = False,
    mesh_resolution: int = 20,
) -> None:
    """
    Add picks or spheres to an existing 3D figure in "true" coordinates.
    (No additional XY downsample factor here, since picks are in original coords.)
    """
    if len(coords_x) == 0:
        return

    if not use_mesh:
        # fallback to average if array
        marker_size = s
        if isinstance(s, np.ndarray):
            marker_size = float(np.mean(s))
        fig.add_trace(
            go.Scatter3d(
                x=coords_x,
                y=coords_y,
                z=coords_z,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=(color if isinstance(color, (list, np.ndarray)) else "limegreen"),
                    opacity=opacity,
                ),
            )
        )
    else:
        # for each sphere
        if isinstance(s, (int, float)):
            sphere_radii = np.full_like(coords_x, float(s))
        else:
            sphere_radii = np.array(s, dtype=float)

        if not isinstance(color, (list, np.ndarray)):
            color = [color if color else "limegreen"] * len(coords_x)
        else:
            color = np.array(color, dtype=object)

        for i, (cx, cy, cz, rr) in enumerate(zip(coords_x, coords_y, coords_z, sphere_radii)):
            xs, ys, zs, i_, j_, k_ = generate_sphere_mesh(cx, cy, cz, rr, steps=mesh_resolution)
            ccol = color[i] if i < len(color) else "limegreen"
            fig.add_trace(
                go.Mesh3d(
                    x=xs, y=ys, z=zs,
                    i=i_, j=j_, k=k_,
                    color=ccol,
                    opacity=opacity,
                    flatshading=True,
                )
            )


def plot_3d_picks(df: pd.DataFrame) -> None:
    df = convert_to_float(df)
    color_col = "_rlnCenteredCoordinateZAngst"
    if "_rlnTomoSubtomogramTilt" in df.columns:
        color_col = "_rlnTomoSubtomogramTilt"

    fig = px.scatter_3d(
        df,
        x="_rlnCenteredCoordinateXAngst",
        y="_rlnCenteredCoordinateYAngst",
        z="_rlnCenteredCoordinateZAngst",
        color=color_col,
        size_max=2,
        title=f"Particle picks colored by {color_col}",
    )
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_scenes(aspectmode="data")
    st.plotly_chart(fig, use_container_width=True)


def plot_pick_tomo(folder: str, node_files: List[str]) -> None:
    logger.info(f"Working on tomogram pick job... folder={folder}, node_files={node_files}")

    # 1) Parse main star
    particles_star_path = os.path.join(folder, node_files[0])
    try:
        star_dict = parse_star(particles_star_path)
    except Exception as exc:
        report_error(exc)
        st.error(f"Failed to parse the star file: {particles_star_path}")
        return

    if "particles" not in star_dict:
        st.error("Could not find 'particles' in star file.")
        return

    # 2) Parse the optimisation_set star
    opt_star_path = os.path.join(folder, node_files[1])
    try:
        star_dict_opt = parse_star(opt_star_path)
    except Exception as exc:
        report_error(exc)
        st.error(f"Failed to parse star file: {opt_star_path}")
        return

    if "optimisation_set" not in star_dict_opt:
        st.error("Could not find 'optimisation_set' in the star file.")
        return

    particles_df = star_dict["particles"]
    unique_tomo = np.unique(particles_df["_rlnTomoName"])

    col_controls, col_display = st.columns([1, 4])
    with col_controls:
        st.subheader("Tomogram & Projection")

        if len(unique_tomo) > 1:
            idx_tomo = st.slider("Select tomogram", 0, len(unique_tomo)-1, 0)
        else:
            idx_tomo = 0

        selected_tomo = unique_tomo[idx_tomo]
        st.write(f"**Selected tomogram**: {selected_tomo}")

        selected_particles = particles_df[particles_df["_rlnTomoName"]==selected_tomo].copy()
        selected_particles = convert_to_float(selected_particles)

        node_dir = os.path.dirname(node_files[0])
        ann_pattern = os.path.join(folder, node_dir, "*")
        matched_paths = glob.glob(ann_pattern)

        annotation_found = False
        annotation_files = []
        for pth in matched_paths:
            if "annotations" in pth.lower():
                annotation_found = True
                annotation_files = sorted(glob.glob(os.path.join(pth, "*.star")))
                break

        # load volume if found
        volume_3d = None
        downsample_xy_factor = 2  # this is your chosen factor for X/Y
        if annotation_found:
            opt_set = star_dict_opt["optimisation_set"]
            tomo_star_file = opt_set["_rlnTomoTomogramsFile"].values[0]
            tomo_star_path = os.path.join(folder, tomo_star_file)
            try:
                tomo_star = parse_star(tomo_star_path)["global"]
            except Exception as exc:
                report_error(exc)
                st.error(f"Failed to parse tomo star: {tomo_star_path}")
                return

            # pick volume
            if "_rlnTomoReconstructedTomogramDenoised" in tomo_star.columns:
                volumes = tomo_star["_rlnTomoReconstructedTomogramDenoised"].values
            else:
                volumes = tomo_star["_rlnTomoReconstructedTomogramHalf1"].values

            if idx_tomo >= len(volumes):
                st.error("Tomo index out of range for volumes.")
                return

            selected_vol = volumes[idx_tomo]
            tomo_path = os.path.join(folder, selected_vol)

            if idx_tomo >= len(annotation_files):
                st.warning("Index out of range for annotation files; using 0.")
                anno_idx = 0
            else:
                anno_idx = idx_tomo

            annotation_star_path = annotation_files[anno_idx]

            # load MRC
            try:
                with mrcfile.mmap(tomo_path, permissive=True) as mrc:
                    data_3d = mrc.data  # shape (Z, Y, X)
                volume_3d = reduce_resolution(data_3d, downsample_xy_factor)
            except Exception as exc:
                report_error(exc)
                st.error("Could not load MRC or reduce resolution.")
                return

        # radio 2D or 3D
        plot_type = st.radio("Plot Type", ["2D", "3D"])
        # projection
        proj_type = st.radio("Projection", ["XY","XZ","YZ"])

        # if volume_3d is loaded, let user pick slice
        slice_idx = 0
        if volume_3d is not None:
            zdim, ydim, xdim = volume_3d.shape
            if proj_type == "XY":
                max_slice = zdim-1
                slice_label = "Z slice"
            elif proj_type == "XZ":
                max_slice = ydim-1
                slice_label = "Y slice"
            else:
                max_slice = xdim-1
                slice_label = "X slice"

            slice_idx = st.slider(slice_label, 0, max_slice, 0)

        with st.expander("Advanced Visualization"):
            gaussian_blur = st.slider("Gaussian blur (std dev)", 0.0, 5.0, 0.1, 0.1)
            intensity_rescale = st.slider(
                "Intensity rescale (percentiles)",
                0.0, 100.0, (1.0,99.0), 0.1
            )
            select_view_type = "Particles"
            picks_opacity = 0.8
            radius_in_annotation = False

            if annotation_found:
                select_view_type = st.radio("Show Annotations or Particles?", ["Annotations","Particles"])
                try:
                    anno_test = parse_star(annotation_star_path)
                    ak = list(anno_test.keys())[0]
                    df_anno_test = convert_to_float(anno_test[ak])
                    if "_rlnSphereRadius" in df_anno_test.columns:
                        radius_in_annotation = True
                except:
                    pass

            # show circle or spacing
            show_spacing_slider = True
            show_circle_slider = True
            if plot_type == "2D":
                if radius_in_annotation and select_view_type=="Annotations":
                    show_spacing_slider=False
                    show_circle_slider=False

            if show_spacing_slider:
                spacing_ang = st.slider("Particle spacing (Å)", 5, 200, 100, step=1)
            else:
                spacing_ang=10

            if show_circle_slider:
                circle_scale_factor = st.slider("Circle diameter scale factor (2D mode)", 0.05,10.0,1.0,0.05)
            else:
                circle_scale_factor=1.0

            picks_opacity = st.slider("Picks opacity",0.1,1.0,0.6,0.1)

    # Right column
    with col_display:
        st.subheader("Tomogram Viewer")

        if not annotation_found or volume_3d is None:
            st.write("No annotation folder or no volume loaded. Show picks in 3D.")
            if len(selected_particles)==0:
                st.warning("No picks found.")
                return
            plot_3d_picks(selected_particles)
            return

        # parse annotation star
        try:
            anno_star_dict = parse_star(annotation_star_path)
            anno_key = list(anno_star_dict.keys())[0]
            annotation_star = anno_star_dict[anno_key]
            annotation_star = convert_to_float(annotation_star)
        except Exception as exc:
            report_error(exc)
            st.error(f"Failed to parse annotation file: {annotation_star_path}")
            return

        # build 2D slice
        Z, Y, X = volume_3d.shape
        if proj_type=="XY":
            slice_2d = volume_3d[slice_idx,:,:]
            axis_out = 2
        elif proj_type=="XZ":
            slice_2d = volume_3d[:, slice_idx, :]
            axis_out = 1
        else:
            slice_2d = volume_3d[:, :, slice_idx]
            axis_out = 0

        # process
        try:
            slice_2d_proc, _ = process_micrograph(
                slice_2d, 
                img_resize_fac=1,
                gaussian_blur_stdev=gaussian_blur,
                low_percent=intensity_rescale[0],
                high_percent=intensity_rescale[1],
            )
        except Exception as exc:
            report_error(exc)
            st.error("Failed to process micrograph slice.")
            return

        # gather picks
        coords_arr = annotation_star[["_rlnCoordinateX","_rlnCoordinateY","_rlnCoordinateZ"]].to_numpy()

        if plot_type=="2D":
            # revert XY scaling changes => no forced resizing beyond circle_scale_factor
            if proj_type=="XY":
                inplane_1, inplane_2 = (0,1)
            elif proj_type=="XZ":
                inplane_1, inplane_2 = (0,2)
            else:
                inplane_1, inplane_2 = (1,2)

            lb, ub = slice_idx-5, slice_idx+5
            mask_slc = (coords_arr[:, axis_out]>=lb)&(coords_arr[:, axis_out]<=ub)
            picks_subset = coords_arr[mask_slc]

            def plot_2d_slice(img_2d, picks2d, diam):
                plot_micrograph(img_2d, picks2d[:,0], picks2d[:,1], s=diam, opacity=picks_opacity)

            if select_view_type=="Annotations":
                if "_rlnSphereRadius" in annotation_star.columns:
                    from . import circle_diameter_along_axis as cdaa  # or inline
                    diam_array = cdaa(coords_arr, annotation_star["_rlnSphereRadius"], slice_idx, axis_out)
                    diam_array *= circle_scale_factor

                    diam_subset = diam_array[mask_slc]
                    picks2d = picks_subset[:, [inplane_1, inplane_2]]
                    plot_2d_slice(slice_2d_proc, picks2d, diam_subset)
                else:
                    picks2d = picks_subset[:, [inplane_1, inplane_2]]
                    diam_array=10*circle_scale_factor
                    plot_2d_slice(slice_2d_proc, picks2d, diam_array)
            else:
                # particles
                if "_rlnSphereRadius" not in annotation_star.columns:
                    picks2d = picks_subset[:, [inplane_1, inplane_2]]
                    diam=10*circle_scale_factor
                    plot_2d_slice(slice_2d_proc, picks2d, diam)
                else:
                    # sample sphere
                    bin_factor = float(star_dict_opt["optimisation_set"]["_rlnTomoTomogramBinning"].values[0])
                    px_size = float(star_dict_opt["optimisation_set"]["_rlnMicrographOriginalPixelSize"].values[0])
                    spacing_angstroms = spacing_ang / bin_factor / px_size
                    centers = coords_arr
                    radii = annotation_star["_rlnSphereRadius"].to_numpy()

                    df_list=[]
                    for c_, r_ in zip(centers,radii):
                        sph=Sphere(center=c_,radius=r_)
                        sampler=sphere_samplers.PoseSampler(spacing=spacing_angstroms)
                        poses=sampler.sample(sph)
                        dfp=pd.DataFrame({
                            "_rlnCoordinateX":poses.positions[:,0],
                            "_rlnCoordinateY":poses.positions[:,1],
                            "_rlnCoordinateZ":poses.positions[:,2],
                        })
                        df_list.append(dfp)
                    all_parts_df = pd.concat(df_list,ignore_index=True)
                    st.write(f"**Total Particles**: {len(all_parts_df)}")

                    arrp = all_parts_df[["_rlnCoordinateX","_rlnCoordinateY","_rlnCoordinateZ"]].to_numpy()
                    mask2=(arrp[:, axis_out]>=lb)&(arrp[:, axis_out]<=ub)
                    sub2 = arrp[mask2]
                    picks2d = sub2[:, [inplane_1,inplane_2]]
                    diam=5*circle_scale_factor
                    plot_2d_slice(slice_2d_proc, picks2d, diam)

        else:
            # 3D
            fig = go.Figure()

            # we place the plane in the correct orientation 
            # for each projection, factoring in the skip in X,Y
            # Example: if your tomogram is in Å, px_size_xy could be the original pixel size 
            # but you might not have that handy. We'll assume 1 for demonstration.

            px_size_xy = 1.0  # or your real-world size in X/Y 
            px_size_z  = 1.0  # or real-world size in Z

            surf = generate_plane_mesh_3d(
                slice_2d_proc,
                slice_idx=slice_idx,
                projection=proj_type,
                downsample_xy_factor=downsample_xy_factor,
                px_size_z=px_size_z,
                px_size_xy=px_size_xy
            )
            fig.add_trace(surf)

            # now add picks in original coords
            if select_view_type=="Annotations":
                if "_rlnSphereRadius" in annotation_star.columns:
                    plot_picks_3d(
                        fig,
                        coords_arr[:,0], coords_arr[:,1], coords_arr[:,2],
                        s=np.array(annotation_star["_rlnSphereRadius"]),
                        opacity=picks_opacity,
                        use_mesh=True
                    )
                else:
                    plot_picks_3d(
                        fig,
                        coords_arr[:,0], coords_arr[:,1], coords_arr[:,2],
                        s=5, opacity=picks_opacity
                    )
            else:
                # "Particles"
                if "_rlnSphereRadius" not in annotation_star.columns:
                    logger.info("No radius in annotation. Just picks.")
                    plot_picks_3d(
                        fig,
                        coords_arr[:,0], coords_arr[:,1], coords_arr[:,2],
                        opacity=picks_opacity
                    )
                else:
                    bin_factor = float(star_dict_opt["optimisation_set"]["_rlnTomoTomogramBinning"].values[0])
                    px_size = float(star_dict_opt["optimisation_set"]["_rlnMicrographOriginalPixelSize"].values[0])
                    spacing_angstroms = spacing_ang / bin_factor / px_size
                    centers = coords_arr
                    radii = annotation_star["_rlnSphereRadius"].to_numpy()

                    df_list=[]
                    for c_,r_ in zip(centers,radii):
                        sph=Sphere(center=c_, radius=r_)
                        sampler=sphere_samplers.PoseSampler(spacing=spacing_angstroms)
                        poses = sampler.sample(sph)
                        dfp=pd.DataFrame({
                            "_rlnCoordinateX": poses.positions[:,0],
                            "_rlnCoordinateY": poses.positions[:,1],
                            "_rlnCoordinateZ": poses.positions[:,2],
                        })
                        df_list.append(dfp)
                    all_parts_df = pd.concat(df_list, ignore_index=True)
                    st.write(f"**Total 3D Particles**: {len(all_parts_df)}")

                    arrp = all_parts_df[["_rlnCoordinateX","_rlnCoordinateY","_rlnCoordinateZ"]].to_numpy()
                    plot_picks_3d(
                        fig,
                        arrp[:,0],arrp[:,1],arrp[:,2],
                        opacity=picks_opacity
                    )

            fig.update_layout(
                scene=dict(aspectmode="data"),
                height=600,
                margin=dict(l=0,r=0,t=20,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
'''