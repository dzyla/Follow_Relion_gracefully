import argparse
import glob
import os
import pathlib
import webbrowser
import time

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from gemmi import cif
from skimage import measure

# Disable numpy warning
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

'''
Relion 2D and 3D live preview

requires mrcfile, gemmi and matplotlib, skimage and plotly. Best to use in own environment:

python3 -m venv env
source ./env/bin/activate
pip3 install numpy matplotlib gemmi mrcfile plotly
python3 relion-2D-and-3D-live-preview .py --i /path/to/the/classification/ --w 300

Written and maintained by Dawid Zyla, La Jolla Institute for Immunology

v0.1 --> moved plot display to plotly and web browser. Added angular distribution plots.
v0.11 --> removed plotting histogram in classification 2D that was for the test purposes. Ups.

Under Non-Profit Open Software License 3.0 (NPOSL-3.0)

'''


def parse_star_model(file_path, loop_name):
    doc = cif.read_file(file_path)

    # block 1 is the per class information
    loop = doc[1].find_loop(loop_name)
    class_data = np.array(loop)

    return class_data


def parse_star_data(file_path, loop_name):
    do_again = True
    while do_again:
        try:
            doc = cif.read_file(file_path)

            if len(doc) == 2:
                particles_block = 1
            else:
                particles_block = 0

            # block 1 is the per class information
            loop = doc[particles_block].find_loop(loop_name)
            class_data = np.array(loop)

            do_again = False
            return class_data

        except RuntimeError:
            print('*star file is busy')
            time.sleep(5)


def get_classes(path_, model_star_files):
    class_dist_per_run = []
    class_res_per_run = []

    for iter, file in enumerate(model_star_files):

        # for the refinement, plot only half2 stats
        if not 'half1' in file:
            class_dist_per_run.append(parse_star_model(file, '_rlnClassDistribution'))
            class_res_per_run.append(parse_star_model(file, '_rlnEstimatedResolution'))

    # stack all data together
    class_dist_per_run = np.stack(class_dist_per_run)
    class_res_per_run = np.stack(class_res_per_run)

    # rotate matrix so the there is class(iteration) not iteration(class) and starting from iter 0 --> iter n
    class_dist_per_run = np.flip(np.rot90(class_dist_per_run), axis=0)
    class_res_per_run = np.flip(np.rot90(class_res_per_run), axis=0)

    # Find the class images (3D) or stack (2D)
    class_files = parse_star_model(model_star_files[-1], '_rlnReferenceImage')

    class_path = []
    for class_name in class_files:
        class_name = os.path.join(path_, os.path.basename(class_name))

        # Insert only new classes, in 2D only single file
        if class_name not in class_path:
            class_path.append(class_name)

    n_classes = class_dist_per_run.shape[0]
    iter_ = class_dist_per_run.shape[1] - 1

    return class_path, n_classes, iter_, class_dist_per_run, class_res_per_run


def get_2dclass_stack(path_):
    return mrcfile.open(path_[0]).data


def plot_2dclasses(path_, classnumber):
    # open mrcs stack
    classes = mrcfile.open(path_[0]).data

    z, x, y = classes.shape
    empty_class = np.zeros((x, y))
    line = []

    # Ratio to display classes best is 5x3
    x_axis = int(np.sqrt(classnumber) * 1.66)

    if z % x_axis == 0:
        y_axis = int(z / x_axis)
    elif z % x_axis != 0:
        y_axis = int(z / x_axis) + 1

    add_extra = int(x_axis * y_axis - z)

    for n, class_ in enumerate(classes):

        if np.average(class_) != 0:
            try:
                class_ = (class_ - np.min(class_)) / (np.max(class_) - np.min(class_))

            except:
                pass

        if n == 0:
            row = class_
        else:
            if len(row) == 0:
                row = class_
            else:
                row = np.concatenate((row, class_), axis=1)
        if (n + 1) % x_axis == 0:
            line.append(row)
            row = []

    # Fill the rectangle with empty classes
    if add_extra != 0:
        for i in range(0, add_extra):
            row = np.concatenate((row, empty_class), axis=1)
        line.append(row)

    # put lines of images together so the whole rectangle is finished (as a picture)
    w = 0
    for i in line:
        if w == 0:
            final = i
            w = 1
        else:
            final = np.concatenate((final, i), axis=0)

    return final


def plot_3dclasses(files):
    class_averages = []

    for n, class_ in enumerate(files):

        with mrcfile.open(class_) as mrc_stack:
            mrcs_file = mrc_stack.data
            z, x, y = mrc_stack.data.shape

            average_top = np.zeros((z, y))

            # only a slice of the volume is plotted
            # for i in range(int(0.45 * z), int(0.55 * z)):

            for i in range(0, z):
                img = mrcs_file[i, :, :]
                average_top += img
            try:
                average_top = (average_top - np.min(average_top)) / (np.max(average_top) - np.min(average_top))
            except:
                pass

            average_front = np.zeros((z, y))

            for i in range(0, z):
                img = mrcs_file[:, i, :]
                average_front += img

            try:
                average_front = (average_front - np.min(average_front)) / (
                        np.max(average_front) - np.min(average_front))
            except:
                pass

            average_side = np.zeros((z, y))

            for i in range(0, z):
                img = mrcs_file[:, :, i]
                average_side += img

            try:
                average_side = (average_side - np.min(average_side)) / (np.max(average_side) - np.min(average_side))
            except:
                pass

            average_class = np.concatenate((average_top, average_front, average_side), axis=0)
            class_averages.append(average_class)

    final_average = np.concatenate(class_averages, axis=1)

    return final_average


def get_angles(path_):
    '''
    Euler angles: (rot,tilt,psi) = (?,?,?). Positive rotations of object are clockwise. Projection direction is
    defined by (rot,tilt). Psi is in-plane rotation for tilted image. For untilted rot=psi=in-plane rotation.
    Angles in a STAR file rotate the reference into observations (i.e. particle image), while translations shift
    observations into the reference projection.

    :param path_:
    :return:
    '''

    data_star = glob.glob(path_ + '/*data.star')
    data_star.sort(key=os.path.getmtime)

    last_data_star = data_star[-1]

    rot_angles = parse_star_data(last_data_star, '_rlnAngleRot').astype(float)
    tilt_angles = parse_star_data(last_data_star, '_rlnAngleTilt').astype(float)
    psi_angles = parse_star_data(last_data_star, '_rlnAnglePsi').astype(float)

    return rot_angles, tilt_angles, psi_angles


def plot_new_stats_plotly(data_, fig_, position_c, position_r):
    for n, class_ in enumerate(data_):
        class_ = np.float16(class_)
        x = np.arange(0, data_.shape[1])

        fig_.add_scatter(x=x, y=class_, row=position_r,
                         col=position_c, name='class {}'.format(n + 1), showlegend=False)


def define_plot():
    fig_ = make_subplots(rows=2, cols=3, specs=[
        [{}, {"rowspan": 1, "colspan": 2}, None],
        [{}, {}, {}],
    ],
                         print_grid=False,
                         subplot_titles=("Class distribution", "Class Preview [class vs view]", "Class Resolution",
                                         "Particle orientation psi/rotation",
                                         "Particle orientation psi/tilt")
                         )

    # Update title
    work_folder_ = pathlib.PurePath(path)
    fig_.update_layout(
        title_text="<b>Relion preview of <i>{}</i>, iteration {}</b>".format(
            work_folder_.name, iter_))

    # Update xaxis properties
    fig_.update_xaxes(title_text="Iteration", row=1, col=1)
    fig_.update_xaxes(title_text="Iteration", row=2, col=1)
    fig_.update_xaxes(title_text="Psi [deg]", row=2, col=2)
    fig_.update_xaxes(title_text="Psi [deg]", row=2, col=3)

    # Update yaxis properties
    fig_.update_yaxes(title_text="Class distribution", row=1, col=1)
    fig_.update_yaxes(title_text="Class Resolution [A]", row=2, col=1)
    fig_.update_yaxes(title_text="Rotation [deg]", row=2, col=2)
    fig_.update_yaxes(title_text="Tilt [deg]", row=2, col=3)

    return fig_


def define_plot_2d():
    fig_ = make_subplots(rows=4, cols=3, specs=[
        [{"rowspan": 2, "colspan": 1}, {"rowspan": 3, "colspan": 2}, None],
        [None, None, None],
        [{"rowspan": 2, "colspan": 1}, None, None],
        [None, {"rowspan": 1, "colspan": 2}, None],

    ],
                         print_grid=False,
                         subplot_titles=("Class distribution", "Class Preview", "Class Resolution",
                                         "Particle angular distribution")
                         )

    # Update title
    work_folder = pathlib.PurePath(path)
    fig_.update_layout(
        title_text="<b>Relion preview of <i>{}</i>, iteration {}</b>".format(
            work_folder.name, iter_))

    # Update xaxis properties
    fig_.update_xaxes(title_text="Iteration", row=1, col=1)
    fig_.update_xaxes(title_text="Iteration", row=3, col=1)
    fig_.update_xaxes(title_text="Psi [deg]", row=4, col=2)

    # Update yaxis properties
    fig_.update_yaxes(title_text="Class distribution", row=1, col=1)
    fig_.update_yaxes(title_text="Class Resolution [A]", row=3, col=1)
    fig_.update_yaxes(title_text="Distribution", row=4, col=2)

    return fig_


def define_plot_refine():
    fig_ = make_subplots(rows=2, cols=3,
                         specs=[
                             [{}, {"rowspan": 1, "colspan": 2, "type": "scene"}, None],
                             [{}, {}, {}],
                         ],
                         print_grid=False,
                         subplot_titles=("Class distribution",
                                         "Volume Preview [Downscaled to 100px]",
                                         "Class Resolution",
                                         "Particle orientation psi/rotation",
                                         "Particle orientation psi/tilt")
                         )

    # Update title
    work_folder_ = pathlib.PurePath(path)
    fig_.update_layout(
        title_text="<b>Relion preview of <i>{}</i>, iteration {}</b>".format(
            work_folder_.name, iter_))

    # Update xaxis properties
    fig_.update_xaxes(title_text="Iteration", row=1, col=1)
    fig_.update_xaxes(title_text="Iteration", row=2, col=1)
    fig_.update_xaxes(title_text="Psi [deg]", row=2, col=2)
    fig_.update_xaxes(title_text="Psi [deg]", row=2, col=3)

    # Update yaxis properties
    fig_.update_yaxes(title_text="Class distribution", row=1, col=1)
    fig_.update_yaxes(title_text="Class Resolution [A]", row=2, col=1)
    fig_.update_yaxes(title_text="Rotation [deg]", row=2, col=2)
    fig_.update_yaxes(title_text="Tilt [deg]", row=2, col=3)

    return fig_


def add_refresh_timer(html_file_path, wait_time):
    import fileinput

    # Would it work with different coding?
    with fileinput.FileInput(html_file_path, inplace=True) as file:
        for n, line in enumerate(file):
            print(line.replace('<head><meta charset="utf-8" /></head>',
                               '<head><meta charset="utf-8" /><meta http-equiv="refresh" content="{}" ></head>'.format(
                                   wait_time)), end='')


def is_refinement_job(class_paths_):
    for path_ in class_paths_:
        if 'half' in path_:
            return True


def get_volume(volume_path):
    volume_ = mrcfile.open(volume_path).data.copy()

    return volume_


def resize_3d(volume_, new_size=100):
    # Otherwise is read only
    volume_ = volume_.copy()

    if new_size % 2 != 0:
        print('Box size has to be even!')
        quit()

    original_size = volume_.shape

    # Skip if volume is less than 100
    if original_size[0] <= 100:
        return volume_.copy()

    fft = np.fft.fftn(volume_)
    fft_shift = np.fft.fftshift(fft)

    # crop this part of the fft
    x1, x2 = int((volume_.shape[0] - new_size) / 2), volume_.shape[0] - int((volume_.shape[0] - new_size) / 2)

    fft_shift_new = fft_shift[x1:x2, x1:x2, x1:x2]

    # Apply spherical mask
    lx, ly, lz = fft_shift_new.shape
    X, Y, Z = np.ogrid[0:lx, 0:ly, 0:lz]
    dist_from_center = np.sqrt((X - lx / 2) ** 2 + (Y - ly / 2) ** 2 + (Z - lz / 2) ** 2)
    mask = dist_from_center <= lx / 2
    fft_shift_new[~mask] = 0

    fft_new = np.fft.ifftshift(fft_shift_new)
    new = np.fft.ifftn(fft_new)

    # Return only real part
    return new.real


def plot_volume(fig_, volume_):
    # Resize volume if too big
    volume_ = resize_3d(volume_, new_size=100)

    # Here one could adjust the volume threshold if want to by adding level=level_value to marching_cubes
    verts, faces, normals, values = measure.marching_cubes(volume_)

    # Set the color of the surface based on the faces order. Here you can provide your own colouring
    color = np.zeros(len(faces))
    color[0] = 1  # because there has to be a colour range, 1st element is 1

    # create a plotly trisurf figure
    fig_volume = ff.create_trisurf(x=verts[:, 2],
                                   y=verts[:, 1],
                                   z=verts[:, 0],
                                   plot_edges=False,
                                   colormap=['rgb(170,170,170)'],
                                   simplices=faces,
                                   showbackground=False,
                                   show_colorbar=False
                                   )

    fig_.add_trace(fig_volume['data'][0], col=2, row=1)
    return fig_


'''------------------------------------------------------------------------------------------------------------------'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Real-time preview Relion classification output from Class2D, Class3D and Refine3D jobs, '
                    'including volume projections, class distributions and estimated resolution plots')
    parser.add_argument('--i', type=str, help='Classification folder path')
    parser.add_argument('--w', type=int, default=60,
                        help='Wait time in seconds between refreshes. Adjust accordingly to the expected time of the iteration. '
                             'If the time is up and there no new classification results plots will freeze')

    args = parser.parse_args()

    # Start with empty list of old model files
    old_model_files = []

    # HTML file name
    # For multiple runs
    # html_name = 'rln_preview_{}.html'.format(np.random.randint(0, 1000, 1)[0])

    # For single run in the folder
    html_name = 'rln_preview.html'

    # If the loop started already
    run = False

    while True:

        try:
            # run from the CLI
            path = args.i
            model_files = glob.glob(path + "*model.star")
            model_files.sort(key=os.path.getmtime)


        # Here for the run without the command line, mostly for debugging
        except:

            print('Script should be run with --i /path/to/classification/folder flag')
            quit()

        print('Found {} *model.star files'.format(len(model_files)))

        # Go further only if there are new files in the folder
        if len(old_model_files) != len(model_files):

            # Parse the model star files
            (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(path, model_files)

            # Check in it is refinement job by checking the "half" in path
            is_refinement = is_refinement_job(model_files)

            # 2D classes are in a single file so number of files is different than number of classes
            class2d = False

            # Get the image of all 2D classes
            if len(class_paths) != n_classes_:
                class_image = plot_2dclasses(class_paths, n_classes_)
                class2d = True

            # If not 2D classification and not refinement plot x,y,z projections of the classes
            elif len(class_paths) == n_classes_ and not is_refinement:
                class_image = plot_3dclasses(class_paths)

            # Prepare plot for 3D refinement
            elif is_refinement:
                # get the last volume
                print(class_paths[-1])
                volume = get_volume(class_paths[-1])

            # Plot different stats depending on class type
            if not class2d and not is_refinement:
                fig = define_plot()
            elif class2d:
                fig = define_plot_2d()
            elif is_refinement:
                fig = define_plot_refine()

            # Plot stats and save a HTML file that is opened in the browser
            work_folder = pathlib.PurePath(path)

            # update plot depending on job type
            fig.update_layout(
                title_text="<b>Relion classification preview of <i>{}</i>, iteration {} \n </b>".format(
                    work_folder.name, iter_))

            # Plot class resolution and distribution

            # for 3D classification
            if not class2d and not is_refinement:
                for data in np.array([[class_dist_, 1, 1], [class_res_, 1, 2]]):
                    plot_new_stats_plotly(data[0], fig, data[1], data[2])

                # Plot angular distribution
                rot, tilt, psi = get_angles(path)
                fig.add_histogram2d(x=psi, y=tilt, row=2, col=2, showlegend=False)
                fig.add_histogram2d(x=psi, y=rot, row=2, col=3, showlegend=False)

                # Show classes preview
                fig_img = px.imshow(class_image, binary_string=True)
                fig_img = fig_img['data'][0]
                fig.add_trace(fig_img, row=1, col=2)

            # for 2D classification
            elif class2d:
                for data in np.array([[class_dist_, 1, 1], [class_res_, 1, 3]]):
                    plot_new_stats_plotly(data[0], fig, data[1], data[2])

                    # Plot angular distribution
                rot, tilt, psi = get_angles(path)
                fig.add_histogram(x=psi, row=4, col=2, nbinsx=50)

                # Show classes preview
                fig_img = px.imshow(class_image, binary_string=True)
                fig_img = fig_img['data'][0]
                fig.add_trace(fig_img, row=1, col=2)

            # for 3D refinement
            elif is_refinement:
                for data in np.array([[class_dist_, 1, 1], [class_res_, 1, 2]]):
                    plot_new_stats_plotly(data[0], fig, data[1], data[2])

                rot, tilt, psi = get_angles(path)
                fig.add_histogram2d(x=psi, y=tilt, row=2, col=2, showlegend=False)
                fig.add_histogram2d(x=psi, y=rot, row=2, col=3, showlegend=False)

                fig = plot_volume(fig, volume)

            # Save HTML file and then open it in browser. If run already, just save a new file.
            if not run:
                fig.write_html(html_name)

                # add a short script that refreshes website after wait time. Should be done in Dash but I don't
                # speak Dash.
                add_refresh_timer(html_name, args.w)

                webbrowser.open('file://' + os.path.realpath(html_name))

                run = True

            else:
                fig.write_html(html_name)
                add_refresh_timer(html_name, args.w)

            # update the old model files list
            old_model_files = model_files

        # If no new files found, wait
        else:
            print('Waiting for iteration to finish...')
            time.sleep(args.w)
