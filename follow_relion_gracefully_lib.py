import datetime
import glob
import os
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from gemmi import cif

from skimage import measure, exposure
from skimage.transform import rescale

PATH_CHARACTER = '/' if os.name != 'nt' else '\\'
PLOT_HEIGHT = 500
MAX_MICS = 30

def write_config(hostname):
    config = open('config.toml', 'w')
    print('''
baseurl = "{}"
title = "Follow Relion Gracefully"
languageCode = "en-us"

# Pagination
paginate = 10
paginatePath = "page"

# Copyright
copyright = "2021 / Follow Relion Gracefully / Dawid Zyla"

# Highlighting
pygmentsCodefences = true
pygmentsCodeFencesGuessSyntax = true
pygmentsoptions = "linenos=table"
pygmentsStyle = "vs"

# Taxonomies
[taxonomies]
  tag = "tags"
  category = "categories"

# Menu
[menu]
  # Header

  # Footer
  [[menu.footer]]
    url = "https://github.com/dzyla/Follow_Relion_gracefully"
    name = "GitHub"
    weight = 1

  [[menu.footer]]
    url = "mailto:dzyla@lji.org"
    name = "Contact"
    weight = 2


# Links format
[permalinks]
  posts = "/posts/:year/:month/:title/"

[params]
  # Date format (default: Jan 2, 2006)
  datefmt  = "Jan 2, 2006"
  plotly= true'''.format(hostname), file=config)

def parse_star_model(file_path, loop_name):
    doc = cif.read_file(file_path)

    # block 1 is the per class information
    loop = doc[1].find_loop(loop_name)
    class_data = np.array(loop)

    return class_data


def adjust_contrast(img, p1=2, p2=98):
    p1, p2 = np.percentile(img, (p1, p2))
    img = exposure.rescale_intensity(img, in_range=(p1, p2))
    return img


def get_class_list(path_):
    class_list = []
    classes = mrcfile.open(path_).data
    for cls in classes:
        cls = adjust_contrast(cls, 5, 100)
        class_list.append(cls)

    return class_list


def plot_2dclasses_(path_, classnumber):

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


def parse_star_whole(file_path):
    doc = cif.read_file(file_path)

    star_data = {}

    for data in doc:
        try:

            dataframe = pd.DataFrame()
            for item in data:

                for metadata in item.loop.tags:
                    value = data.find_loop(metadata)
                    dataframe[metadata] = np.array(value)
                star_data[data.name] = dataframe

        except AttributeError as e:
            pass
            # print(e)

    return star_data


def get_classes(path_, model_star_files):
    class_dist_per_run = []
    class_res_per_run = []

    for iter, file in enumerate(model_star_files):

        # for the refinement, plot only half2 stats
        if not 'half1' in file:
            class_dist_per_run.append(parse_star_model(file, '_rlnClassDistribution'))
            class_res_per_run.append(parse_star_model(file, '_rlnEstimatedResolution'))

    # stack all data together
    try:
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

    except ValueError:
        class_path, n_classes, iter_, class_dist_per_run, class_res_per_run = [], [], [], [], []

    return class_path, n_classes, iter_, class_dist_per_run, class_res_per_run


def plot_2d_ply(path_data, HUGO_FOLDER, job_name):
    import plotly.express as px
    import plotly.graph_objects as go

    cls2d_shortcodes = []

    model_files = glob.glob(path_data + "*model.star")
    model_files.sort(key=os.path.getmtime)

    n_iter = len(model_files)

    (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(path_data, model_files)

    '''------ 2D images ---------'''

    class_list = get_class_list(class_paths[0])

    cols = 5 if n_classes_ < 30 else 10

    fig = px.imshow(np.array(class_list), facet_col=0, binary_string=True, facet_col_wrap=cols, facet_row_spacing=0.1,
                    facet_col_spacing=0.0, width=800)
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)

    for i, cls_ in enumerate(range(0, n_classes_)):
        fig.layout.annotations[i]['text'] = 'Class {} <br> {}%'.format(cls_ + 1,
                                                                       round(np.float(class_dist_[:, -1][i]) * 100, 3))

    fig.update_yaxes(automargin=True)

    json_name = 'cls2d_'
    cls2d_height = 600 if n_classes_ < 30 else 800
    plotly_string = write_plot_get_shortcode(fig, json_name, job_name, HUGO_FOLDER, fig_height=cls2d_height)

    cls2d_shortcodes.append(plotly_string)



    '''Save 2D classes as images with a slider'''

    job = job_name.split('/')[1].replace('/', PATH_CHARACTER)

    for n, cls in enumerate(class_list):
        # plt.figure(figsize=(5,5), dpi=100)
        # plt.imshow(particle, cmap='gray')
        plt.imsave(HUGO_FOLDER + job + PATH_CHARACTER + '{}.jpg'.format(n + 1), cls, cmap='gray')
        last_n = n + 1

    js_code = '''

    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="XXX" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="250">
    </div>

    <script>

        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + 1 + ".jpg";
            function showVal(newVal){
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }
    </script>
    <br>
    '''.replace('XXX', str(last_n))

    js_string = "((<rawhtml >)) {} ((< /rawhtml >))".format(js_code)
    js_string = js_string.replace('((', '{{').replace('))', '}}')

    cls2d_shortcodes.append(js_string)



    '''Class distribution plot'''

    fig_ = go.Figure()

    for n, class_ in enumerate(class_dist_):
        class_ = np.float16(class_)
        x = np.arange(0, class_dist_.shape[1])

        fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

    fig_.update_xaxes(title_text="Iteration")
    fig_.update_yaxes(title_text="Class distribution")

    fig_.update_layout(
        title="Class distribution"
    )

    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    json_name = 'cls2d_dist_'
    plotly_string = write_plot_get_shortcode(fig_, json_name, job_name, HUGO_FOLDER)
    cls2d_shortcodes.append(plotly_string)



    '''Class resolution plot'''

    fig_ = go.Figure()

    for n, class_ in enumerate(class_res_):
        class_ = np.float16(class_)
        x = np.arange(0, class_res_.shape[1])

        fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

    fig_.update_xaxes(title_text="Iteration")
    fig_.update_yaxes(title_text="Class Resolution [A]")

    fig_.update_layout(
        title="Class Resolution [A]"
    )

    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    json_name = 'cls2d_res_'
    plotly_string = write_plot_get_shortcode(fig_, json_name, job_name, HUGO_FOLDER)
    cls2d_shortcodes.append(plotly_string)


    return cls2d_shortcodes


def parse_star(file_path):
    # import tqdm

    doc = cif.read_file(file_path)

    optics_data = {}

    # 3.1 star files have two data blocks Optics and particles
    _new_star_ = True if len(doc) == 2 else False

    if _new_star_:
        # print('Found Relion 3.1+ star file.')

        optics = doc[0]
        particles = doc[1]

        for item in optics:
            for optics_metadata in item.loop.tags:
                value = optics.find_loop(optics_metadata)
                optics_data[optics_metadata] = np.array(value)[0]

    else:
        # print('Found Relion 3.0 star file.')
        particles = doc[0]

    particles_data = pd.DataFrame()

    # print('Reading star file:')
    for item in particles:
        for particle_metadata in item.loop.tags:
            # If don't want to use tqdm uncomment bottom line and remove 'import tqdm'
            # for particle_metadata in item.loop.tags:
            loop = particles.find_loop(particle_metadata)
            particles_data[particle_metadata] = np.array(loop)

    return optics_data, particles_data


def create_circular_mask(h, w, radius=None):
    center = [int(w / 2), int(h / 2)]

    # use the smallest distance between the center and image walls
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def mask_particle(particle, radius=None):
    if radius == 1:
        return particle

    h, w = particle.shape[:2]
    radius = np.min([h, w]) / 2 * radius
    mask = create_circular_mask(h, w, radius)
    masked_img = particle.copy()
    masked_img[~mask] = 0

    return masked_img


def reverse_FFT(fft_image):
    fft_img_mod = np.fft.ifftshift(fft_image)

    img_mod = np.fft.ifft2(fft_img_mod)
    return img_mod.real


def do_fft_image(img_data):
    img_fft = np.fft.fft2(img_data)

    fft_img_shift = np.fft.fftshift(img_fft)

    # real = fft_img_shift.real
    # phases = fft_img_shift.imag

    return fft_img_shift


def mask_in_fft(img, radius):
    fft = do_fft_image(img)
    masked_fft = mask_particle(fft, radius)
    img_masked = reverse_FFT(masked_fft)

    return img_masked


def show_random_particles(star_path, project_path, random_size=100, r=1, adj_contrast=False):
    star = parse_star(star_path)[1]
    data_shape = star.shape[0]
    random_int = np.random.randint(0, data_shape, random_size)
    selected = star.iloc[random_int]

    particle_array = []
    for element in selected['_rlnImageName']:
        particle_data = element.split('@')
        img_path = project_path + particle_data[1]
        # print(img_path)
        try:
            particle = mrcfile.mmap(img_path).data[int(particle_data[0])]
            if r != 0:
                particle = mask_in_fft(particle, r)

            if adj_contrast:
                particle = adjust_contrast(particle)

            particle_array.append(particle)

        except IndexError:
            pass
        # except ValueError:
        #     pass

    return particle_array


def plot_extract_js(particle_list, HUGO_FOLDER, job_name):
    job_name = job_name.split('/')[1].replace('/', PATH_CHARACTER)

    for n, particle in enumerate(particle_list):
        # plt.figure(figsize=(5,5), dpi=100)
        # plt.imshow(particle, cmap='gray')
        plt.imsave(HUGO_FOLDER + job_name + PATH_CHARACTER + '{}.jpg'.format(n), particle, cmap='gray')
        last_n = n

    js_code = '''
   
<div class="center">
<p>Preview of random extracted particles:<p>
<input id="valR" type="range" min="0" max="XXX" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
<span id="range">0</span>
<img id="img" width="200">
</div>

<script>
    
    var val = document.getElementById("valR").value;
        document.getElementById("range").innerHTML=val;
        document.getElementById("img").src = val + ".jpg";
        function showVal(newVal){
          document.getElementById("range").innerHTML=newVal;
          document.getElementById("img").src = newVal+ ".jpg";
        }
</script>
<br>
'''.replace('XXX', str(last_n))

    js_string = "((<rawhtml >)) {} ((< /rawhtml >))".format(js_code)
    js_string = js_string.replace('((', '{{').replace('))', '}}')

    return js_string


def plot_ctf_stats(starctf, HUGO_FOLDER, job_name):
    import plotly.express as px

    shortcodes = []

    reduced_ctfcorrected = starctf[1]

    '''Defocus / index'''
    fig = px.line(x=range(0, reduced_ctfcorrected.shape[0]), y=reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                  title='_rlnDefocusV')

    fig.update_xaxes(title_text="Index")
    fig.update_yaxes(title_text="_rlnDefocusV")

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    ctf_name_json = 'ctffind_'

    shortcodes.append(write_plot_get_shortcode(fig, ctf_name_json, job_name, HUGO_FOLDER, fig_height=500))

    '''Max Res / index'''
    fig = px.line(x=range(0, reduced_ctfcorrected.shape[0]),
                  y=reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float),
                  title='_rlnCtfMaxResolution')

    fig.update_xaxes(title_text="Index")
    fig.update_yaxes(title_text="_rlnCtfMaxResolution")

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    ctf_name_json = 'ctffind1_'

    shortcodes.append(write_plot_get_shortcode(fig, ctf_name_json, job_name, HUGO_FOLDER, fig_height=500))

    '''Defocus hist'''

    fig = px.histogram(x=reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                       title='_rlnDefocusV', nbins=50)
    fig.update_xaxes(title_text="_rlnDefocusV")
    fig.update_yaxes(title_text="Number")

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    ctf_name_json = 'ctffind2_'

    shortcodes.append(write_plot_get_shortcode(fig, ctf_name_json, job_name, HUGO_FOLDER, fig_height=500))

    '''Astigmatism hist'''

    fig = px.histogram(x=reduced_ctfcorrected['_rlnCtfAstigmatism'].astype(float),
                       title='_rlnCtfAstigmatism', nbins=50)
    fig.update_xaxes(title_text="_rlnCtfAstigmatism")
    fig.update_yaxes(title_text="Number")

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    ctf_name_json = 'ctffind3_'

    shortcodes.append(write_plot_get_shortcode(fig, ctf_name_json, job_name, HUGO_FOLDER, fig_height=500))

    '''MaxRes hist'''

    fig = px.histogram(x=reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float),
                       title='_rlnCtfMaxResolution', nbins=50)
    fig.update_xaxes(title_text="_rlnCtfMaxResolution")
    fig.update_yaxes(title_text="Number")

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    ctf_name_json = 'ctffind4_'

    shortcodes.append(write_plot_get_shortcode(fig, ctf_name_json, job_name, HUGO_FOLDER, fig_height=500))

    '''Defocus / Max res'''
    fig = px.density_heatmap(x=reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                             y=reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float),
                             title='_rlnCtfMaxResolution', nbinsx=50, nbinsy=50, color_continuous_scale="blues")
    fig.update_xaxes(title_text="_rlnDefocusV")
    fig.update_yaxes(title_text="_rlnCtfMaxResolution")

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    ctf_name_json = 'ctffind5_'

    shortcodes.append(write_plot_get_shortcode(fig, ctf_name_json, job_name, HUGO_FOLDER, fig_height=500))

    '''Defocus / Max res'''
    fig = px.density_heatmap(x=reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                             y=reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float),
                             title='_rlnCtfFigureOfMerit', nbinsx=50, nbinsy=50, color_continuous_scale="blues")
    fig.update_xaxes(title_text="_rlnDefocusV")
    fig.update_yaxes(title_text="_rlnCtfFigureOfMerit")

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    ctf_name_json = 'ctffind6_'

    shortcodes.append(write_plot_get_shortcode(fig, ctf_name_json, job_name, HUGO_FOLDER, fig_height=500))

    return shortcodes


def plot_motioncorr_stats(star, HUGO_FOLDER, job_name):
    import plotly.graph_objects as go

    shortcodes = []

    star_data = star[1]

    '''Motion per index'''

    fig_ = go.Figure()

    for n, meta in enumerate(['_rlnAccumMotionTotal', '_rlnAccumMotionEarly', '_rlnAccumMotionLate']):
        fig_.add_scatter(x=np.arange(0, star_data.shape[0]), y=star_data[meta].astype(float), name=meta)

    fig_.update_xaxes(title_text="Index")
    fig_.update_yaxes(title_text="Motion")

    fig_.update_layout(
        title="MotionCorr statistics"
    )
    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    motioncorr_name_json = 'motioncorr_'
    shortcodes.append(write_plot_get_shortcode(fig_, motioncorr_name_json, job_name, HUGO_FOLDER,
                                                         fig_height=500))

    '''Motion histograms'''

    plotting_data = []
    meta_names = ['_rlnAccumMotionTotal', '_rlnAccumMotionEarly', '_rlnAccumMotionLate']

    # data = pd.DataFrame()
    data_list = []

    for n, meta in enumerate(meta_names):
        data_list = star_data[meta].astype(float)
        plotting_data.append(data_list)

    fig_ = ff.create_distplot(plotting_data, meta_names, bin_size=.2, show_rug=False)

    fig_.update_xaxes(title_text="Motion")
    fig_.update_yaxes(title_text="Number")

    fig_.update_layout(
        title="MotionCorr statistics"
    )
    fig_.update_traces(opacity=0.75)

    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    motioncorr_name_json = 'motioncorr1_'
    shortcodes.append(write_plot_get_shortcode(fig_, motioncorr_name_json, job_name, HUGO_FOLDER,
                                                         fig_height=500))

    return shortcodes


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
                if (np.max(average_side) - np.min(average_side)) != 0:
                    average_side = (average_side - np.min(average_side)) / (np.max(average_side) - np.min(average_side))
            except RuntimeWarning:
                pass

            average_class = np.concatenate((average_top, average_front, average_side), axis=0)
            class_averages.append(average_class)

    try:
        final_average = np.concatenate(class_averages, axis=1)

    except ValueError:
        final_average = []

    return final_average


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
    if volume_.shape[0] > 100:
        volume_ = resize_3d(volume_, new_size=100)
    else:
        volume_ = volume_.copy()

    # Here one could adjust the volume threshold if want to by adding level=level_value to marching_cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(volume_)

    except RuntimeError:
        return None

    # Set the color of the surface based on the faces order. Here you can provide your own colouring
    color = np.zeros(len(faces))
    color[0] = 1  # because there has to be a colour range, 1st element is 1

    # create a plotly trisurf figure
    fig_volume = ff.create_trisurf(x=verts[:, 2],
                                   y=verts[:, 1],
                                   z=verts[:, 0],
                                   plot_edges=False,
                                   colormap=['rgb(150,150,150)'],
                                   simplices=faces,
                                   showbackground=False,
                                   show_colorbar=False
                                   )

    fig_.add_trace(fig_volume['data'][0])
    return fig_


def write_plot_get_shortcode(fig, json_name, job_name, HUGO_FOLDER, fig_height=600):
    from follow_relion_gracefully_run import HOSTNAME
    name_json = json_name + job_name.replace('/', '_')
    plotly_file = HUGO_FOLDER + job_name.split('/')[1].replace('/', PATH_CHARACTER) + PATH_CHARACTER + "{}.json".format(
        name_json)
    fig.write_json(plotly_file)

    plotly_string = '((< plotly json="{}jobs/{}/{}" height="{}px" >))'.format(HOSTNAME,job_name.split('/')[1],
                                                                             "{}.json".format(
                                                                                 name_json), fig_height)
    shortcode = plotly_string.replace('((', '{{').replace('))', '}}')

    return shortcode


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

        except RuntimeError as e:
            # print('*star file is busy')
            # time.sleep(5)
            print(e)
            do_again = False


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


def plot_cls3d_stats(path_data, HUGO_FOLDER, job_name):
    import plotly.express as px
    import plotly.graph_objects as go

    model_files = glob.glob(path_data + "*model.star")
    model_files.sort(key=os.path.getmtime)

    n_inter = len(model_files)

    if n_inter != 0:

        (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(path_data, model_files)

        """Plot Classes in 3D"""

        cls3d_shortcodes = []
        for n, cls_path in enumerate(class_paths):
            fig_ = go.Figure()
            mrc_cls_data = mrcfile.open(cls_path, permissive=True).data
            fig_ = plot_volume(fig_, mrc_cls_data)

            if fig_ != None:
                fig_.update_layout(
                    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

                shortcode = write_plot_get_shortcode(fig_, 'cls3d_model{}_'.format(n), job_name, HUGO_FOLDER)
                cls3d_shortcodes.append('#### Class {}:'.format(n))
                cls3d_shortcodes.append(shortcode)

        """Plot Class projections"""

        cls3d_projections = plot_3dclasses(class_paths)

        fig = px.imshow(cls3d_projections, binary_string=True,
                        labels=dict(x="Class", y="Projection axis"))
        fig.update_xaxes(side="top")

        labels_positions_x = np.linspace(1 / len(class_paths) * cls3d_projections.shape[1], cls3d_projections.shape[1],
                                         len(class_paths)) - 0.5 * 1 / len(class_paths) * cls3d_projections.shape[1]
        labels_x = ["Class {}<br>{}%".format(x, round(np.float(class_dist_[:, -1][x]) * 100, 2))
                    for x, cls in enumerate(class_paths)]

        labels_positions_y = np.linspace(1 / 3 * cls3d_projections.shape[0], cls3d_projections.shape[0],
                                         3) - 0.5 * 1 / 3 * cls3d_projections.shape[0]
        labels_y = ["<b>Z </b>", "<b>X </b>", "<b>Y </b>"]

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=labels_positions_x,
                ticktext=labels_x
            )
        )

        fig.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=labels_positions_y,
                ticktext=labels_y
            )
        )

        shortcode = write_plot_get_shortcode(fig, 'cls3d_projection_', job_name, HUGO_FOLDER, fig_height=800)
        cls3d_shortcodes.append('#### Class Projections:')
        cls3d_shortcodes.append(shortcode)

        '''Class distribution plot'''

        fig_ = go.Figure()

        for n, class_ in enumerate(class_dist_):
            class_ = np.float16(class_)
            x = np.arange(0, class_dist_.shape[1])

            fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

        fig_.update_xaxes(title_text="Iteration")
        fig_.update_yaxes(title_text="Class distribution")

        fig_.update_layout(
            title="Class distribution"
        )

        shortcode = write_plot_get_shortcode(fig_, 'cls3d_dist_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)
        cls3d_shortcodes.append(shortcode)

        '''Class resolution plot'''

        fig_ = go.Figure()

        for n, class_ in enumerate(class_res_):
            class_ = np.float16(class_)
            x = np.arange(0, class_res_.shape[1])

            fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

        fig_.update_xaxes(title_text="Iteration")
        fig_.update_yaxes(title_text="Class Resolution [A]")

        fig_.update_layout(
            title="Class Resolution [A]"
        )

        shortcode = write_plot_get_shortcode(fig_, 'cls3d_res_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)
        cls3d_shortcodes.append(shortcode)

        '''Angular dist plot'''

        fig_ = go.Figure()

        rot, tilt, psi = get_angles(path_data)
        fig_.add_histogram2d(x=psi, y=tilt, showlegend=False)

        fig_.update_xaxes(title_text="Psi [deg]")
        fig_.update_yaxes(title_text="Rotation [deg]")

        fig_.update_layout(
            title="Psi (Rotation)"
        )

        shortcode = write_plot_get_shortcode(fig_, 'cls3d_psi_rot_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)
        cls3d_shortcodes.append(shortcode)

        fig_.add_histogram2d(x=psi, y=rot, showlegend=False)
        fig_.update_xaxes(title_text="Psi [deg]")
        fig_.update_yaxes(title_text="Tilt [deg]")

        fig_.update_layout(
            title="Psi (Tilt)"
        )

        shortcode = write_plot_get_shortcode(fig_, 'cls3d_psi_tilt_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)
        cls3d_shortcodes.append(shortcode)

    else:
        cls3d_shortcodes = ['No *_model.star found in the folder']

    return cls3d_shortcodes


def get_note(path_):
    file = open(path_)
    file_data = str(file.read())
    file_data = file_data.replace('++++', '\n')
    file_data = file_data.replace("`", '')

    file_data = file_data.replace('which', '\nwhich')

    file.close()
    return file_data


def plot_mask(path_data, HUGO_FOLDER, job_name):
    """Plot Class projections"""

    mask_projections = plot_3dclasses([path_data])
    shortcodes = []

    fig = px.imshow(mask_projections, binary_string=True,
                    labels=dict(x="Binary Mask", y="Projection axis"))
    fig.update_xaxes(side="top")

    labels_positions_x = [0.5 * mask_projections.shape[1]]
    labels_x = ["Mask"]

    labels_positions_y = np.linspace(1 / 3 * mask_projections.shape[0], mask_projections.shape[0],
                                     3) - 0.5 * 1 / 3 * mask_projections.shape[0]
    labels_y = ["<b>Z </b>", "<b>X </b>", "<b>Y </b>"]

    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=labels_positions_y,
            ticktext=labels_y
        )
    )

    shortcode = write_plot_get_shortcode(fig, 'mask_projection_', job_name, HUGO_FOLDER, fig_height=800)
    shortcodes.append(shortcode)

    return shortcodes


def rotate_volume(volume, angle, rotaxes_):
    from scipy import ndimage

    volume = np.array(volume)
    volume = ndimage.interpolation.rotate(volume, angle, reshape=False, axes=rotaxes_)
    return volume


def plot_refine3d(path_data, HUGO_FOLDER, job_name):
    """Plot Class projections"""



    model_files = glob.glob(path_data + "*model.star")
    model_files.sort(key=os.path.getmtime)

    n_iter = len(model_files)

    if n_iter != 0:
        ref3d_shortcodes = []
        (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(path_data, model_files)

        """Plot Refine3D in 3D"""

        cls_path = class_paths[-1]
        volume_data = mrcfile.open(cls_path, permissive=True).data
        volume_100px = resize_3d(volume_data, new_size=100)
        fig_ = go.Figure()
        fig_ = plot_volume(fig_, volume_100px)
        fig_['layout'].update(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

        shortcode = write_plot_get_shortcode(fig_, 'refine3d_model_', job_name, HUGO_FOLDER, fig_height=800)
        ref3d_shortcodes.append(shortcode)

        """Plot Refine3D in 2D projections"""

        job = job_name.split('/')[1].replace('/', PATH_CHARACTER)

        for n, angle in enumerate(np.arange(0, 360, 10)):
            projection = np.mean(rotate_volume(volume_100px, angle, [1, 2]), axis=2)
            plt.imsave(HUGO_FOLDER + job + PATH_CHARACTER + '{}.jpg'.format(n), projection, cmap='gray')
            last_n = n

        js_code = '''

        <div class="center">
        <p>Volume projections preview:<p>
        <input id="valR" type="range" min="0" max="XXX" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
        <span id="range">0</span>
        <img id="img" width="250">
        </div>

        <script>

            var val = document.getElementById("valR").value;
                document.getElementById("range").innerHTML=val;
                document.getElementById("img").src = val + ".jpg";
                function showVal(newVal){
                  document.getElementById("range").innerHTML=newVal;
                  document.getElementById("img").src = newVal+ ".jpg";
                }
        </script>
        <br>
        '''.replace('XXX', str(last_n))

        js_string = "((<rawhtml >)) {} ((< /rawhtml >))".format(js_code)
        js_string = js_string.replace('((', '{{').replace('))', '}}')

        ref3d_shortcodes.append(js_string)

        """Plot Class projections"""

        ref3d_projections = plot_3dclasses(class_paths)

        fig = px.imshow(ref3d_projections, binary_string=True,
                        labels=dict(x="Class", y="Projection axis"))
        fig.update_xaxes(side="top")

        labels_positions_x = np.linspace(1 / len(class_paths) * ref3d_projections.shape[1], ref3d_projections.shape[1],
                                         len(class_paths)) - 0.5 * 1 / len(class_paths) * ref3d_projections.shape[1]
        labels_x = ["Class {}<br>{}%".format(x, round(np.float(class_dist_[:, -1][x]) * 100, 2))
                    for x, cls in enumerate(class_paths)]

        labels_positions_y = np.linspace(1 / 3 * ref3d_projections.shape[0], ref3d_projections.shape[0],
                                         3) - 0.5 * 1 / 3 * ref3d_projections.shape[0]
        labels_y = ["<b>Z </b>", "<b>X </b>", "<b>Y </b>"]

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=labels_positions_x,
                ticktext=labels_x
            )
        )

        fig.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=labels_positions_y,
                ticktext=labels_y
            )
        )

        shortcode = write_plot_get_shortcode(fig, 'cls3d_projection_', job_name, HUGO_FOLDER, fig_height=800)
        ref3d_shortcodes.append('#### Class Projections:')
        ref3d_shortcodes.append(shortcode)

        '''Class resolution plot'''

        fig_ = go.Figure()

        for n, class_ in enumerate(class_res_):
            class_ = np.float16(class_)
            x = np.arange(0, class_res_.shape[1])

            fig_.add_scatter(x=x, y=class_, name='class {}'.format(n + 1), showlegend=False)

        fig_.update_xaxes(title_text="Iteration")
        fig_.update_yaxes(title_text="Class Resolution [A]")

        fig_.update_layout(
            title="Class Resolution [A]"
        )

        shortcode = write_plot_get_shortcode(fig_, 'cls3d_res_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)
        ref3d_shortcodes.append(shortcode)

        '''Angular dist plot'''

        fig_ = go.Figure()

        rot, tilt, psi = get_angles(path_data)
        fig_.add_histogram2d(x=psi, y=tilt, showlegend=False, nbinsx=50, nbinsy=50)

        fig_.update_xaxes(title_text="Psi [deg]")
        fig_.update_yaxes(title_text="Rotation [deg]")

        fig_.update_layout(
            title="Psi (Rotation)"
        )

        shortcode = write_plot_get_shortcode(fig_, 'cls3d_psi_rot_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)
        ref3d_shortcodes.append(shortcode)

        fig_.add_histogram2d(x=psi, y=rot, showlegend=False, nbinsx=50, nbinsy=50)
        fig_.update_xaxes(title_text="Psi [deg]")
        fig_.update_yaxes(title_text="Tilt [deg]")

        fig_.update_layout(
            title="Psi (Tilt)"
        )

        shortcode = write_plot_get_shortcode(fig_, 'cls3d_psi_tilt_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)
        ref3d_shortcodes.append(shortcode)

    else:
        ref3d_shortcodes = ['No *model.star found in the folder']

    return ref3d_shortcodes


# def save_bokeh(json_name, job_name, HUGO_FOLDER, fig):
#     from bokeh.embed import json_item
#     import json
#
#     name_json = json_name + job_name.replace('/', '_')
#     bokeh_file = HUGO_FOLDER + job_name.split('/')[1].replace('/', PATH_CHARACTER) + PATH_CHARACTER + "{}.json".format(
#         name_json)
#
#     with open(bokeh_file, "w") as json_file:
#         json.dump(json_item(fig, "myplot"), json_file)
#
#     plotly_string = '((< bokeh "{}" >))'.format(job_name.split('/')[1],
#                                                 "{}.json".format(
#                                                     name_json))
#     shortcode = plotly_string.replace('((', '{{').replace('))', '}}')
#
#     return shortcode


def plot_picks_plotly(rln_folder, path_data, HUGO_FOLDER, job_name):
    rln_folder = rln_folder.replace('/', PATH_CHARACTER)
    path_data = path_data.replace('/', PATH_CHARACTER)

    autopick_shorcode = []
    img_resize_fac = 0.2

    coordinate_files = glob.glob(path_data + "*.star")
    coordinate_files.sort(key=os.path.getmtime)

    # Relion 4 has much easier coordinate handling
    if os.path.exists(path_data + 'autopick.star') and os.path.getsize(path_data + 'autopick.star') > 0:
        pick_old = False

        autopick_star = parse_star_whole(path_data + 'autopick.star')['coordinate_files']
        mics_paths = autopick_star['_rlnMicrographName']
        coord_paths = autopick_star['_rlnMicrographCoordinates']
        if coord_paths.shape[0] != 1:
            coord_paths = np.squeeze(coord_paths.to_numpy())

    elif os.path.exists(path_data + 'manualpick.star') and os.path.getsize(path_data + 'manualpick.star') > 0:
        pick_old = False

        manpick_star = parse_star_whole(path_data + 'manualpick.star')['coordinate_files']
        mics_paths = manpick_star['_rlnMicrographName']
        coord_paths = manpick_star['_rlnMicrographCoordinates']
        if coord_paths.shape[0] != 1:
            coord_paths = np.squeeze(coord_paths.to_numpy())

    elif glob.glob(path_data + 'coords_suffix_*') != []:
        # Get the coordinates from subfolders
        pick_old = True

        # get suffix firsts

        suffix_file = glob.glob(path_data + 'coords_suffix_*')[0]
        suffix = os.path.basename(suffix_file).replace('coords_suffix_', '').replace('.star', '')

        # Get the folder with micrographs
        mics_data_path = open(glob.glob(path_data + 'coords_suffix_*')[0]).readlines()[0].replace('\n', '')

        all_mics_paths = parse_star_data(rln_folder + mics_data_path, '_rlnMicrographName')

        mics_paths = []
        for name in all_mics_paths:
            mics_paths.append(rln_folder + name)

    elif glob.glob(path_data + 'model_training.txt') != []:

        topaz_training_txt = glob.glob(path_data + 'model_training.txt')[0]

        data = pd.read_csv(topaz_training_txt, delimiter='\t')
        data_test = data[data['split'] == 'test']

        x = data_test['epoch']
        data_test = data_test.drop(['iter', 'split', 'ge_penalty'], axis=1)

        fig_ = go.Figure()

        for n, column in enumerate(data_test.columns):
            if column != 'epoch':
                y = data_test[column]
                fig_.add_scatter(x=x, y=y, name='{}'.format(column))

        fig_.update_xaxes(title_text="Epoch")
        fig_.update_yaxes(title_text="Statistics")

        fig_.update_layout(
            title="Topaz training stats. Best model: {}".format(data_test[data_test['auprc'].astype(float) ==
                                                    np.max(data_test['auprc'].astype(float))]['epoch'].values)
        )

        fig_.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))

        shortcode = write_plot_get_shortcode(fig_, 'topaz_train_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)

        return shortcode

    else:
        return 'Something went wrong'

    indices = np.arange(0, np.array(mics_paths).shape[0])
    np.random.shuffle(indices)

    processed = -1
    proceed = False

    for n, file in enumerate(np.array(mics_paths)[indices]):

        if pick_old:

            # Create a picking star file name based on the mic name
            pick_star_name = os.path.basename(file).replace('.mrc', '_{}.star'.format(suffix))

            # Find the file location if there
            pick_star_name_path = glob.glob(path_data + '/**/' + pick_star_name)

            # if the file is there go, if not it will be empty list
            if pick_star_name_path != []:
                # Open micrograph and corresponding star file
                micrograph = mrcfile.open(file, permissive=True).data
                coords_file = parse_star(pick_star_name_path[0])[1]
                coords_x = coords_file['_rlnCoordinateX']
                coords_y = coords_file['_rlnCoordinateY']
                autopick_fom = coords_file['_rlnAutopickFigureOfMerit']

                proceed = True

        # If relion 4 and has nicely placed all files together in autopick.star
        else:

            micrograph = mrcfile.open(rln_folder + file, permissive=True).data

            coords_file = parse_star(rln_folder + coord_paths[indices[n]])[1]
            coords_x = coords_file['_rlnCoordinateX']
            coords_y = coords_file['_rlnCoordinateY']

            # If one want to plot the score
            # autopick_fom = coords_file['_rlnAutopickFigureOfMerit']

            proceed = True

        if processed == MAX_MICS:
            break

        if proceed:
            processed += 1

            # Make plots
            mic_red = rescale(micrograph.astype(np.float), img_resize_fac)
            p1, p2 = np.percentile(mic_red, (0.1, 99.8))
            mic_red = np.array(exposure.rescale_intensity(mic_red, in_range=(p1, p2)))

            plt.imshow(mic_red, cmap='gray')
            plt.axis('off')

            plt.scatter(coords_x.astype(float) * img_resize_fac, coords_y.astype(float) * img_resize_fac,
                        s=250, facecolors='none', edgecolor="green", linewidth=1)

            plt.savefig(HUGO_FOLDER + job_name.split('/')[1].replace('/', PATH_CHARACTER) + PATH_CHARACTER + str(
                processed) + '.jpg', bbox_inches='tight', pad_inches = 0)
            plt.cla()

            proceed = False

    js_code = '''


    <div class="center">
    <p>Preview of random extracted particles:<p>
    <input id="valR" type="range" min="0" max="XXX" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="800">
    </div>

    <script>

        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + ".jpg";
            function showVal(newVal){
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }
    </script>
    <br>
    '''.replace('XXX', str(processed))

    js_string = "((<rawhtml >)) {} ((< /rawhtml >))".format(js_code)
    js_string = js_string.replace('((', '{{').replace('))', '}}')

    return js_string


def plot_ctf_refine(path_data, HUGO_FOLDER, job_name):
    from PIL import Image
    eps_files = glob.glob(path_data + '*.eps')

    TARGET_BOUNDS = (1024, 1024)
    shortcodes = []

    try:
        if eps_files != []:
            for eps in eps_files:
                img = Image.open(eps)
                img.load(scale=5)

                # Ensure scaling can anti-alias by converting 1-bit or paletted images
                if img.mode in ('P', '1'):
                    img = img.convert("RGB")

                # Calculate the new size, preserving the aspect ratio
                ratio = min(TARGET_BOUNDS[0] / img.size[0],
                            TARGET_BOUNDS[1] / img.size[1])
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))

                img = img.resize(new_size, Image.ANTIALIAS)

                filename = os.path.basename(eps).replace('eps', 'jpg')
                img.save(HUGO_FOLDER + job_name.split('/')[1] + PATH_CHARACTER + filename)

                shortcodes.append('![XXX}](YYY)'.replace('XXX', filename.replace('.jpg', '')).replace('YYY', filename))
                shortcodes.append('\n\n')

    except OSError:
        shortcodes.append('Ghostscript not found')

    return shortcodes


def bayesian_polishing_plot():
    '''
        find eps files
        for file in eps files:
        open eps, save as jpg

        for eps file in eps files:
        generate JS to show them

        return js

        :return:
        '''

    pass

def plot_import(rln_folder, star_data, HUGO_FOLDER, job_name):
    import plotly.graph_objects as go

    shortcode = []

    star_data = star_data[1]


    try:
        file_names = star_data['_rlnMicrographMovieName']

    except KeyError:
        try:
            file_names = star_data['_rlnMicrographName']

        except KeyError:
            return ['No Micrographs / Movies found']

    '''Import by time'''

    file_mod_times = []
    for file in file_names:
        file_mod_times.append(datetime.datetime.fromtimestamp(os.path.getmtime(rln_folder+file)))


    fig_ = go.Figure()

    fig_.add_scatter(x=np.arange(0, len(file_mod_times)), y=file_mod_times, name='Time stamp')

    fig_.update_xaxes(title_text="Index")
    fig_.update_yaxes(title_text="Time stamp")

    fig_.update_layout(
        title="Imported micrographs timeline"
    )
    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    name_json = 'import_'
    import_string = write_plot_get_shortcode(fig_, name_json, job_name, HUGO_FOLDER,
                                                         fig_height=500)
    shortcode.append(import_string)

    return shortcode

def plot_locres(path_data, HUGO_FOLDER, job_name):

    import plotly.graph_objects as go
    shortcodes = []

    # Well, it procudes huge files. If want to keep the interactive plot need to drop the file size.
    locres_data = mrcfile.open(path_data).data.flatten()

    h = np.histogram(locres_data, bins=np.arange(0.1, locres_data.max(), step=0.1))

    labels = ['{} - {}, {}'.format(round(n, 3), round(h[1][i+1], 3), h[0][i]) for i, n in enumerate(h[1][:-1])]

    fig = go.Figure(
        go.Bar(
            customdata=labels,
            x=h[1],
            y=h[0],
            hovertemplate="(%{customdata})<extra></extra>",
        )
    ).update_layout(bargap=0)

    shortcode = write_plot_get_shortcode(fig, 'locres_', job_name, HUGO_FOLDER, fig_height=600)
    shortcodes.append(shortcode)

    return shortcodes


def plot_postprocess(rln_folder, nodes, HUGO_FOLDER, job_name):

    shortcodes = []

    '''Plot FSC curves from postprocess.star'''

    postprocess_star_path = rln_folder+nodes[3]
    postprocess_star_data = parse_star_whole(postprocess_star_path)

    fsc_data = postprocess_star_data['fsc']
    guinier_data = postprocess_star_data['guinier']

    fsc_x = fsc_data['_rlnAngstromResolution'].astype(float)

    fsc_to_plot = ['_rlnFourierShellCorrelationCorrected','_rlnFourierShellCorrelationUnmaskedMaps', '_rlnFourierShellCorrelationMaskedMaps', '_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps']

    fig_ = go.Figure()
    for meta in fsc_to_plot:
        # Limit the range of the data. Nobody needs 999A res. Start from ~50A?
        fig_.add_scatter(x=fsc_x[8:], y=fsc_data[meta][8:].astype(float), name=meta)

    fig_.update_xaxes(title_text="Resolution, A")
    fig_.update_yaxes(title_text="FSC")

    fig_.update_layout(
        title="FSC"
    )
    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig_.update_xaxes(type="log")

    fig_.update_layout(
        xaxis=dict(autorange="reversed"))
    fig_.update_layout(xaxis_range=[30, 0])
    fig_.add_hline(y=0.143)
    fig_.update_layout(yaxis_range=[-0.1, 1.1])


    name_json = 'postprocess_'
    postprocess_string = write_plot_get_shortcode(fig_, name_json, job_name, HUGO_FOLDER,
                                             fig_height=500)
    shortcodes.append(postprocess_string)


    guiner_x = guinier_data['_rlnResolutionSquared'].astype(float)

    guinier_to_plot = ['_rlnLogAmplitudesOriginal', '_rlnLogAmplitudesMTFCorrected',
                   '_rlnLogAmplitudesWeighted',
                   '_rlnLogAmplitudesSharpened', '_rlnLogAmplitudesIntercept']


    '''Guinier plot'''
    fig_ = go.Figure()
    for meta in guinier_to_plot:
        try:
            y_data = guinier_data[meta].astype(float)

            #remove some points
            y_data[y_data == -99] = 'Nan'

            fig_.add_scatter(x=guiner_x, y=y_data, name=meta)
        except:
            #if MTF was not there?
            pass

    fig_.update_xaxes(title_text="Resolution Squared, [1/A^2]")
    fig_.update_yaxes(title_text="Ln(Amplitutes)")

    fig_.update_layout(
        title="Guinier plot")

    fig_.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    #fig_.update_yaxes(range=[-16, -8])
    # fig_.update_layout(
    #     yaxis=dict(autorange="reversed"))


    name_json = 'postprocess1_'
    postprocess_string = write_plot_get_shortcode(fig_, name_json, job_name, HUGO_FOLDER,
                                                  fig_height=500)
    shortcodes.append(postprocess_string)


    """Plot masked postprocess in 2D slices"""

    job = job_name.split('/')[1].replace('/', PATH_CHARACTER)

    volume_path = rln_folder+nodes[1]
    volume_data = mrcfile.open(volume_path).data

    for n, slice in enumerate(volume_data):
        #slice_img =
        plt.imsave(HUGO_FOLDER + job + PATH_CHARACTER + '{}.jpg'.format(n), slice, cmap='gray')
        last_n = n

    js_code = '''

            <div class="center">
            <p>Masked Volume slices preview:<p>
            <input id="valR" type="range" min="0" max="XXX" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
            <span id="range">0</span>
            <img id="img" width="350">
            </div>

            <script>

                var val = document.getElementById("valR").value;
                    document.getElementById("range").innerHTML=val;
                    document.getElementById("img").src = val + ".jpg";
                    function showVal(newVal){
                      document.getElementById("range").innerHTML=newVal;
                      document.getElementById("img").src = newVal+ ".jpg";
                    }
            </script>
            <br>
            '''.replace('XXX', str(last_n))

    js_string = "((<rawhtml >)) {} ((< /rawhtml >))".format(js_code)
    js_string = js_string.replace('((', '{{').replace('))', '}}')

    shortcodes.append(js_string)
    return shortcodes