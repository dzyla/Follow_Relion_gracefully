import random
from pathlib import Path

import mrcfile
import time
import glob
import os
from gemmi import cif
import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.transform import rescale


def save_star(dataframe_, filename='out.star', block_name='particles'):
    out_doc = cif.Document()
    out_particles = out_doc.add_new_block(block_name, pos=-1)

    # Row number is required for the column names to save the STAR file e.g. _rlnNrOfSignificantSamples #33
    column_names = dataframe_.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    loop = out_particles.init_loop('', column_names_to_star)
    data_rows = dataframe_.to_numpy().astype(str).tolist()

    for row in data_rows:
        loop.add_row(row)

    out_doc.write_file(filename)
    #print('File "{}" saved.'.format(filename))


def save_star_31(dataframe_optics, dataframe_particles, filename='out.star'):
    # For now only Relion star 3.1+ can be saved as 3.1 star. Adding optics will be implemented later.

    out_doc = cif.Document()
    out_particles = out_doc.add_new_block('optics', pos=-1)

    # Row number is required for the column names to save the STAR file e.g. _rlnNrOfSignificantSamples #33
    dataframe_optics = pd.DataFrame(dataframe_optics, index=[0])
    column_names = dataframe_optics.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    loop = out_particles.init_loop('', column_names_to_star)
    data_rows = dataframe_optics.to_numpy().astype(str).tolist()

    # save optics loop
    for row in data_rows:
        loop.add_row(row)

    out_particles = out_doc.add_new_block('particles', pos=-1)

    column_names = dataframe_particles.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    loop = out_particles.init_loop('', column_names_to_star)
    data_rows = dataframe_particles.to_numpy().astype(str).tolist()

    # save particles loop
    for row in data_rows:
        loop.add_row(row)

    out_doc.write_file(filename)
    print('File "{}" saved.'.format(filename))


def convert_optics(optics_data_):
    # used for saving Relion 3.1 files with optics groups.
    # Changes the dict so values are list now.

    for key in optics_data_.keys():
        optics_data_[key] = [optics_data_[key]]

    return optics_data_


def convert_new_to_old(dataframe_, optics_group, filename, magnification='100000'):
    if optics_group == {}:
        print('File is already in Relion 3.0 format. No conversion needed!')
        quit()

    # change the Origin from Angstoms to pixels
    dataframe_['_rlnOriginXAngst'] = dataframe_['_rlnOriginXAngst'].astype(float) / optics_group[
        '_rlnImagePixelSize'].astype(float)
    dataframe_['_rlnOriginYAngst'] = dataframe_['_rlnOriginYAngst'].astype(float) / optics_group[
        '_rlnImagePixelSize'].astype(float)
    dataframe_ = dataframe_.rename(columns={'_rlnOriginXAngst': '_rlnOriginX', '_rlnOriginYAngst': '_rlnOriginY'})

    # add columns which are in the optics group
    dataframe_['_rlnVoltage'] = np.zeros(dataframe_.shape[0]) + optics_group['_rlnVoltage'].astype(float)
    dataframe_['_rlnSphericalAberration'] = np.zeros(dataframe_.shape[0]) + optics_group[
        '_rlnSphericalAberration'].astype(float)
    dataframe_['_rlnDetectorPixelSize'] = np.zeros(dataframe_.shape[0]) + optics_group['_rlnImagePixelSize'].astype(
        float)
    dataframe_['_rlnMagnification'] = np.zeros(dataframe_.shape[0]) + int(magnification)
    dataframe_['_rlnSphericalAberration'] = np.zeros(dataframe_.shape[0]) + optics_group[
        '_rlnSphericalAberration'].astype(float)

    # remove non used columns
    for tag in ['_rlnOpticsGroup', '_rlnHelicalTrackLengthAngst']:
        try:
            dataframe_ = dataframe_.drop(columns=[tag])
        except:
            pass

    # Row number is required for the column names
    column_names = dataframe_.columns
    column_names_to_star = []
    for n, name in enumerate(column_names):
        column_names_to_star.append(name + ' #{}'.format(n + 1))

    out_doc = cif.Document()
    out_particles = out_doc.add_new_block('', pos=-1)

    loop = out_particles.init_loop('', column_names_to_star)

    # to save cif all list values must be str
    data_rows = dataframe_.to_numpy().astype(str).tolist()

    for row in data_rows:
        loop.add_row(row)

    out_name = filename.replace('.star', '_v30.star')

    out_doc.write_file(out_name)
    print('File "{}" saved.'.format(out_name))


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


def parse_star_selected_columns(file_path, *kargv):
    doc = cif.read_file(file_path)

    optics_data = {}

    # 3.1 star files have two data blocks Optics and particles
    _new_star_ = True if len(doc) == 2 else False

    if _new_star_:
        #print('Found Relion 3.1+ star file.')

        optics = doc[0]
        particles = doc[1]

        for item in optics:
            for optics_metadata in item.loop.tags:
                value = optics.find_loop(optics_metadata)
                optics_data[optics_metadata] = np.array(value)[0]

    else:
        #print('Found Relion 3.0 star file.')
        particles = doc[0]

    particles_data = pd.DataFrame()

    for particle_metadata in kargv:
        loop = particles.find_loop(particle_metadata)
        particles_data[particle_metadata] = np.array(loop)

    return optics_data, particles_data


def plot_columns(particles_data, col1_name, col2_name, plot_type='hist'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(8, 8), dpi=100)

    print('Plotting {} on x and {} on y'.format(col1_name, col2_name))

    if col1_name != 'index' and col2_name != 'index':
        x_data = np.array(particles_data[col1_name].astype(float))
        y_data = np.array(particles_data[col2_name].astype(float))

    elif col1_name == 'index':
        x_data = np.arange(1, particles_data.shape[0] + 1, 1)
        y_data = np.array(particles_data[col2_name].astype(float))

    elif col2_name == 'index':
        y_data = np.arange(0, particles_data.shape[0], 1)
        x_data = np.array(particles_data[col1_name].astype(float))

    if plot_type == 'hist':
        plt.hist2d(x_data, y_data, cmap='Blues', bins=50, norm=LogNorm())
        clb = plt.colorbar()
        clb.set_label('Number of particles')

    elif plot_type == 'line':
        plt.plot(x_data, y_data)

    elif plot_type == 'scat':
        plt.scatter(x_data, y_data, cmap='Blues')

    plt.xlabel(col1_name)
    plt.ylabel(col2_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_2dclasses_cs(path_, x_axis = None):
    # open mrcs stack
    if type(path_) == type(str()):
        classes = mrcfile.open(path_).data

    elif type(path_) == type(list()):
        classes = np.stack(path_)

    z, x, y = classes.shape
    classnumber = z
    empty_class = np.zeros((x, y))
    line = []

    # Ratio to display classes best is 5x3
    if x_axis == None:
        x_axis = int(np.sqrt(classnumber) * 1.66)
    else:
        x_axis = int(x_axis)

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


def plot_2dclasses(path_, classnumber):
    # open mrcs stack
    if type(path_) == type(str()):
        classes = mrcfile.open(path_).data

    elif type(path_) == type(list()):
        classes = np.stack(path_)

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


def get_class_list(path_):
    class_list = []
    classes = mrcfile.open(path_).data
    for cls in classes:
        class_list.append(cls)

    return class_list


def parse_star_model(file_path, loop_name):
    doc = cif.read_file(file_path)

    # block 1 is the per class information
    loop = doc[1].find_loop(loop_name)
    class_data = np.array(loop)

    return class_data


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


def get_angles(path_):
    '''
    Euler angles: (rot,tilt,psi) = (φ,θ,ψ). Positive rotations of object are clockwise. Projection direction is
    defined by (rot,tilt). Psi is in-plane rotation for tilted image. For untilted rot=psi=in-plane rotation.
    Angles in a STAR file rotate the reference into observations (i.e. particle image), while translations shift
    observations into the reference projection.
    '''

    data_star = glob.glob(path_ + '/*data.star')
    data_star.sort(key=os.path.getmtime)

    last_data_star = data_star[-1]

    rot_angles = parse_star_data(last_data_star, '_rlnAngleRot').astype(float)
    tilt_angles = parse_star_data(last_data_star, '_rlnAngleTilt').astype(float)
    psi_angles = parse_star_data(last_data_star, '_rlnAnglePsi').astype(float)

    return rot_angles, tilt_angles, psi_angles


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


def plot_picks_cryosparc(cs_path, data_path, n, score):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.15

    picking_cs = np.load(cs_path)
    mics = picking_cs['location/micrograph_path']
    unique_mics = np.unique(mics)
    mic_to_load = unique_mics[n]

    img = mrcfile.open(data_path + mic_to_load.decode("utf-8")).data
    img = rescale(img, ratio, anti_aliasing=True)
    # img = wiener(img, np.ones((5, 5)) / 2, 1)
    print(mic_to_load)

    try:
        data_per_mic = picking_cs[picking_cs['location/micrograph_path'] == mic_to_load]
        data_per_mic = data_per_mic[data_per_mic['pick_stats/ncc_score'] > score]
        x_data = data_per_mic['location/center_x_frac'] * img.shape[0]
        y_data = data_per_mic['location/center_y_frac'] * img.shape[1]
        score_ncc = data_per_mic['pick_stats/ncc_score']
        # score_ncc = score_ncc[score_ncc > score]

        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, c=score_ncc, alpha=0.4, s=100, facecolors='none')
        plt.colorbar()
        plt.title(str('Number of picks {}'.format(score_ncc.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def plot_picks_cryosparc_imported(cs_path, mic_path, n, score):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.2

    picking_cs = np.load(cs_path)
    mics = picking_cs['location/micrograph_path']
    unique_mics = np.unique(mics)
    mic_to_load = unique_mics[n]
    print(os.path.basename(mic_to_load.decode("utf-8")))

    #img = mrcfile.open(mic_path + os.path.basename(mic_to_load.decode("utf-8"))).data
    img = mrcfile.open(mic_path +mic_to_load.decode("utf-8")).data
    img = rescale(img, ratio, anti_aliasing=True)

    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    # img = wiener(img, np.ones((5, 5)) / 2, 1)

    try:
        data_per_mic = picking_cs[picking_cs['location/micrograph_path'] == mic_to_load]
        data_per_mic = data_per_mic[data_per_mic['pick_stats/ncc_score'] > score]
        x_data = data_per_mic['location/center_x_frac'] * img.shape[0]
        y_data = data_per_mic['location/center_y_frac'] * img.shape[1]
        score_ncc = data_per_mic['pick_stats/ncc_score']
        # score_ncc = score_ncc[score_ncc > score]

        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, c=score_ncc, alpha=0.4, s=100, facecolors='none')
        plt.colorbar()
        plt.title(str('Number of picks {}'.format(score_ncc.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()

def plot_picks(img_path, n, score, picking_folder, picking_file_end):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.15
    img_path = glob.glob('{}/*.mrc'.format(img_path))
    path = img_path[n]
    img = mrcfile.open(path).data
    img = rescale(img, ratio, anti_aliasing=True)
    # img = wiener(img, np.ones((5, 5)) / 2, 1)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    # picking_file_end = "_autopick.star"

    try:
        picked = \
        list(parse_star('{}/{}'.format(picking_folder, os.path.basename(path).replace('.mrc', picking_file_end))))[1]

        if np.average(picked['_rlnAutopickFigureOfMerit'].astype(float)) != -999:
            picked = picked[picked['_rlnAutopickFigureOfMerit'].astype(float) >= score]
        else:
            score = 0
        x_data = picked['_rlnCoordinateX'].values.astype(float) * ratio
        y_data = picked['_rlnCoordinateY'].values.astype(float) * ratio
        score = picked['_rlnAutopickFigureOfMerit'].values.astype(float)
        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, c=score, alpha=0.4, s=100, facecolors='none')
        plt.colorbar()
        plt.title(str('Number of picks {}'.format(score.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


class class2d_run:
    def __init__(self, data_folder, job_n):
        self.folder = data_folder
        self.path = self.folder + '/Class2D/{}/'.format(job_n)
        self.angles()
        self.cls_stats()
        # self.run_params()

    def angles(self):
        self.rot, self.tilt, self.psi = get_angles(self.path)

    def cls_stats(self):
        model_star = glob.glob(self.path + '/*model.star')
        model_star.sort(key=os.path.getmtime)
        self.model_star = model_star
        self.class_path, self.n_classes, self.iter_, self.class_dist_per_run, self.class_res_per_run = get_classes(
            self.path, self.model_star)

    def plot_stats_dist(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_dist_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Class distribution')
        plt.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)

        plt.show()

    def plot_stats_res(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_res_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Resolution, A')
        plt.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
        plt.show()

    def plot_all_classes(self):
        self.cls_stats()

        plt.figure(figsize=(10, 8), dpi=100)
        plt.imshow(plot_2dclasses(self.class_path[-1], len(self.class_path[-1])), cmap='gray')
        plt.tight_layout()
        plt.axis('off')
        plt.show()

    def get_2dcls(self):
        self.cls_stats()
        self.cls_list = get_class_list(self.class_path[-1])
        return self.cls_list

    def run_params(self):
        with open(self.path + 'note.txt') as file:
            print(file.read())


def show_class(cls, cls_list):
    img = cls_list[cls]
    plt.imshow(img, cmap='gray')
    plt.show()


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


def show_random_particles(star_path, project_path, random_size=100, r=1, adj_contrast=False):
    star = parse_star(star_path)[1]
    data_shape = star.shape[0]
    print(data_shape)
    random_int = np.random.randint(0, data_shape, random_size)
    selected = star.iloc[random_int]

    particle_array = []
    for element in selected['_rlnImageName']:
        particle_data = element.split('@')
        img_path = project_path + particle_data[1]
        # print(img_path)
        try:
            particle = mrcfile.mmap(img_path).data[int(particle_data[0])]
            particle = mask_in_fft(particle, r)
            if adj_contrast:
                particle = adjust_contrast(particle)

            particle_array.append(particle)
        except IndexError:
            pass

    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(plot_2dclasses(particle_array, len(particle_array)), cmap='gray')
    plt.axis('off')
    plt.show()


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


def show_mrc(n, r, datapath):
    plt.figure(figsize=(10, 10), dpi=100)
    img = datapath[n]
    img = mrcfile.open(img).data
    img = rescale(img, 0.2)

    img = mask_in_fft(img, r)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(img, cmap='gray')
    plt.show()

def load_topaz_curve(file):
    import pandas as pd
    data = pd.read_csv(file, delimiter='\t')
    data_test = data[data['split'] == 'test']

    print('Best model: {}'.format(data_test[data_test['auprc'] == np.max(data_test['auprc'].astype(float))]['epoch']))

    plt.plot(data_test['epoch'].astype(float), data_test['auprc'].astype(float)/np.max(data_test['auprc'].astype(float)), label='auprc')
    plt.plot(data_test['epoch'].astype(float), data_test['fpr'].astype(float)/np.max(data_test['fpr'].astype(float)), label='fpr')
    plt.plot(data_test['epoch'].astype(float), data_test['tpr'].astype(float)/np.max(data_test['tpr'].astype(float)), label='tpr')
    plt.plot(data_test['epoch'].astype(float), data_test['precision'].astype(float)/np.max(data_test['precision'].astype(float)), label='precision')
    plt.legend()
    plt.show()


def plot_picks_particle(img_paths, n, particles_star):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.15
    #img_path = glob.glob('{}/*.mrc'.format(img_path))
    path = img_paths[n]
    img = mrcfile.open(path).data
    img = rescale(img, ratio, anti_aliasing=True)
    # img = wiener(img, np.ones((5, 5)) / 2, 1)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    try:
        optics, picking_data = parse_star_selected_columns(particles_star, '_rlnMicrographName', '_rlnCoordinateY', '_rlnCoordinateX')
        print(picking_data.shape)

        picking_data = picking_data[picking_data['_rlnMicrographName'] == 'MotionCorr/job069//mnt/staging/hcallaway/HMC_8-27-21_Cryo_83-122/'+os.path.basename(path)]

        print(picking_data.shape)


        x_data = picking_data['_rlnCoordinateX'].values.astype(float) * ratio
        y_data = picking_data['_rlnCoordinateY'].values.astype(float) * ratio
        # score = picked['_rlnAutopickFigureOfMerit'].values.astype(float)
        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, alpha=0.4, s=100, facecolors='none')
        # plt.colorbar()
        plt.title(str('Number of picks {}'.format(x_data.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def plot_picks_cryosparc_imported_mod(cs_path, mic_path, n, score, particles_star, show_relion):
    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.2

    picking_cs = np.load(cs_path)
    mics = picking_cs['location/micrograph_path']
    unique_mics = np.unique(mics)
    mic_to_load = unique_mics[n]
    print(os.path.basename(mic_to_load.decode("utf-8")))

    optics, picking_data = parse_star_selected_columns(particles_star, '_rlnMicrographName', '_rlnCoordinateY',
                                                       '_rlnCoordinateX')

    micrograph_picks = picking_data[picking_data[
                                        '_rlnMicrographName'] == 'MotionCorr/job069//mnt/staging/hcallaway/HMC_8-27-21_Cryo_83-122/' + os.path.basename(
        mic_to_load.decode("utf-8")).replace('_patch_aligned_doseweighted', '')[21:]]

    # print(micrograph_picks)
    # img = mrcfile.open(mic_path + os.path.basename(mic_to_load.decode("utf-8"))).data
    img = mrcfile.open(mic_path + mic_to_load.decode("utf-8")).data
    img = rescale(img, ratio, anti_aliasing=True)

    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    sizex, sizey = img.shape
    # img = wiener(img, np.ones((5, 5)) / 2, 1)

    try:
        data_per_mic = picking_cs[picking_cs['location/micrograph_path'] == mic_to_load]
        data_per_mic = data_per_mic[data_per_mic['pick_stats/ncc_score'] > score]
        y_data = data_per_mic['location/center_x_frac'] * img.shape[0]
        x_data = data_per_mic['location/center_y_frac'] * img.shape[1]
        score_ncc = data_per_mic['pick_stats/ncc_score']
        # score_ncc = score_ncc[score_ncc > score]

        plt.imshow(img, cmap='gray')
        plt.scatter(x_data, y_data, c='red', alpha=0.4, s=100, facecolors='none')

        # Plot relion
        if show_relion:
            plt.scatter(micrograph_picks['_rlnCoordinateY'].astype(float) * ratio,
                        micrograph_picks['_rlnCoordinateX'].astype(float) * ratio, c='blue', alpha=0.2, s=100,
                        facecolors='none')

        plt.colorbar()
        plt.title(str('Number of picks {}'.format(score_ncc.shape[0])))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def adjust_contrast(img, p1=2, p2=98):
    p1, p2 = np.percentile(img, (p1, p2))
    img = exposure.rescale_intensity(img, in_range=(p1, p2))
    return img


def show_mrc_fft(n, r, datapath, fft=False):
    plt.figure(figsize=(10, 10), dpi=150)
    img = datapath[n]
    img = mrcfile.open(img).data  # [300:1200,300:1200]

    if not fft:
        img = rescale(img, 0.2)
        img = mask_in_fft(img, r)
        p2, p98 = np.percentile(img, (2, 98))
    else:
        img = np.log(do_fft_image(img).real ** 2)
        p2, p98 = np.percentile(img, (2, 100))
        img = rescale(img, 0.2)

    img = exposure.rescale_intensity(img, in_range=(p2, p98))

    plt.figure(figsize=(10, 10), dpi=80)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def plot_picks_new(img_path, n, score, picking_folder, picking_file_end, r, show_picks, use_img_paths=False):
    from matplotlib import cm

    plt.figure(figsize=(10, 10), dpi=100)
    ratio = 0.15
    if not use_img_paths:
        img_path = glob.glob('{}/*.mrc'.format(img_path))
    else:
        img_path = img_path

    path = img_path[n]
    img = mrcfile.open(path).data
    img = rescale(img, ratio, anti_aliasing=True)
    img = mask_in_fft(img, r)
    # img = wiener(img, np.ones((5, 5)) / 2, 1)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    # picking_file_end = "_autopick.star"

    try:
        picked = \
            list(parse_star('{}/{}'.format(picking_folder, os.path.basename(path).replace('.mrc', picking_file_end))))[
                1]

        if np.average(picked['_rlnAutopickFigureOfMerit'].astype(float)) != -999:

            colormap = cm.get_cmap('plasma', len(picked))

            score_min = picked['_rlnAutopickFigureOfMerit'].astype(float).min()
            score_max = picked['_rlnAutopickFigureOfMerit'].astype(float).max()

            selection_low = score[0] * (score_max - score_min) + score_min
            selection_high = score[1] * (score_max - score_min)

            picked = picked[picked['_rlnAutopickFigureOfMerit'].astype(float) <= selection_high]
            picked = picked[picked['_rlnAutopickFigureOfMerit'].astype(float) >= selection_low]

            print(picked.shape)
        else:
            colormap = cm.get_cmap('plasma', len(picked))
            score = 0

        x_data = picked['_rlnCoordinateX'].values.astype(float) * ratio
        y_data = picked['_rlnCoordinateY'].values.astype(float) * ratio
        score = picked['_rlnAutopickFigureOfMerit'].values.astype(float)
        plt.imshow(img, cmap='gray')

        if show_picks:
            plt.scatter(x_data, y_data, s=350, facecolors='none', edgecolor=colormap.colors, linewidth=1.5)

            # plt.colorbar()
        plt.title(str('Number of picks {}\nselection values: low={}, high={}'.format(score.shape[0], selection_low,
                                                                                     selection_high)))
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(e)
        plt.imshow(img, cmap='gray')
        plt.tight_layout()
        plt.show()


def project_last_volume(path, volume_string, n_volumes):
    volume_files = glob.glob(path + "*{}.mrc".format(volume_string))
    volume_files.sort(key=os.path.getmtime)
    volume_files = volume_files[::-1]

    selected_volumes = volume_files[:n_volumes]

    print(selected_volumes)

    projection_holder = []

    for volume in selected_volumes:
        v = mrcfile.open(volume).data

        projection = np.concatenate([np.mean(v, axis=axis_) for axis_ in [0, 1, 2]], axis=1)
        p2, p98 = np.percentile(projection, (0, 99.9))
        projection = exposure.rescale_intensity(projection, in_range=(p2, p98))

        projection_holder.append(projection)

    projections = np.concatenate([arr for arr in projection_holder], axis=0)

    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(projections, cmap='gray')
    plt.show()


def picks_from_particles(path_star, folder_particles='./Particle_picks/'):
    # Select the particle star file that will be split into coordinates
    particles_star = path_star

    # Saving folder name
    saving_folder = folder_particles

    # Create the folder if not present
    Path(saving_folder).mkdir(parents=True, exist_ok=True)

    # Read the star file into optics and particles sheets
    optics, data = parse_star_selected_columns(particles_star, '_rlnMicrographName', '_rlnCoordinateX',
                                                   '_rlnCoordinateY')

    # Find the unique micrograph names and iterate
    for micrograph in np.unique(data['_rlnMicrographName']):

        # Select rows where micrograph name is the current one
        data = data[data['_rlnMicrographName'] == micrograph]

        # remove the column with the column name from data for saving
        data = data.drop(columns='_rlnMicrographName')

        # Get the name of the star file from the Micrograph path
        pick_star_name = os.path.basename(micrograph).replace('mrc', 'star')

        # Save the new star file per micrograph with X and Y coordinates
        save_star(data, '{}/{}'.format(saving_folder, pick_star_name), block_name='')



def cs_to_relion_from_star(cs_file, starfile_in, replace_cs, replace_rln, star_out='Rln_selected_cs.star'):
    cs_data = np.load(cs_file)
    _, star_file_imported = parse_star(starfile_in)

    selected_bound_df = pd.DataFrame()
    selected_bound_df['blob/path'] = cs_data['blob/path']
    selected_bound_df['blob/idx'] = cs_data['blob/idx']
    selected_bound_df['blob/path'] = selected_bound_df['blob/path'].str.decode("utf-8").str.replace(replace_cs,
                                                                                                    replace_rln)

    selected_bound_df['blob/idx'] = selected_bound_df['blob/idx'].astype(str).str.zfill(6)

    selected_bound_df['_rlnImageName'] = selected_bound_df['blob/idx'].astype(str) + selected_bound_df['blob/path']

    selected_names = selected_bound_df['_rlnImageName']

    selected_from_cs = star_file_imported[star_file_imported['_rlnImageName'].isin(selected_names)]
    save_star_31(_, selected_from_cs, filename=star_out)


def plot_ctf_stats(starctf, index, defocus, max_res, fom, save_star=False):
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    reduced_ctfcorrected = starctf[0]

    index_low, index_high = np.percentile(range(reduced_ctfcorrected.shape[0]), (index[0], index[1]))

    defocus_low, defocus_high = np.percentile(np.linspace(reduced_ctfcorrected['_rlnDefocusV'].astype(float).min(),
                                                          reduced_ctfcorrected['_rlnDefocusV'].astype(float).max()),
                                              (defocus[0], defocus[1]))

    max_res_low, max_res_high = np.percentile(
        np.linspace(reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float).min(),
                    reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float).max()),
        (max_res[0], max_res[1]))
    fom_low, fom_high = np.percentile(np.linspace(reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float).min(),
                                                  reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float).max()),
                                      (fom[0], fom[1]))

    reduced_ctfcorrected = reduced_ctfcorrected[int(index_low):int(index_high)]

    # selection
    reduced_ctfcorrected = reduced_ctfcorrected[(defocus_low <= reduced_ctfcorrected['_rlnDefocusV'].astype(float)) &
                                                (reduced_ctfcorrected['_rlnDefocusV'].astype(float) <= defocus_high)]

    reduced_ctfcorrected = reduced_ctfcorrected[
        (max_res_low <= reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float)) &
        (reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float) <= max_res_high)]

    reduced_ctfcorrected = reduced_ctfcorrected[
        (fom_low <= reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float)) &
        (reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float) <= fom_high)]

    print('Selected files: {}, Selected defocus: {}-{}, Selected Ctf resolution: {}-{}, Selected FOM: {}-{}'.format(
        reduced_ctfcorrected.shape, defocus_low, defocus_high, max_res_low, max_res_high, fom_low, fom_high))

    axs[0, 0].plot(range(0, reduced_ctfcorrected.shape[0]),
                   reduced_ctfcorrected['_rlnDefocusV'].astype(float))
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('_rlnDefocusV')

    axs[0, 1].plot(range(0, reduced_ctfcorrected.shape[0]),
                   reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float))
    axs[0, 1].set_xlabel('Index')
    axs[0, 1].set_ylabel('_rlnCtfMaxResolution')

    axs[1, 0].hist2d(reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                     reduced_ctfcorrected['_rlnCtfMaxResolution'].astype(float),
                     bins=50, norm=LogNorm(), cmap='Blues')
    axs[1, 0].set_xlabel('_rlnDefocusV')
    axs[1, 0].set_ylabel('_rlnCtfMaxResolution')

    axs[1, 1].hist2d(reduced_ctfcorrected['_rlnDefocusV'].astype(float),
                     reduced_ctfcorrected['_rlnCtfFigureOfMerit'].astype(float),
                     bins=50, norm=LogNorm(), cmap='Blues')
    axs[1, 1].set_xlabel('_rlnDefocusV')
    axs[1, 1].set_ylabel('_rlnCtfFigureOfMerit')
    plt.show()

    if save_star:
        save_star_31(optics, reduced_ctfcorrected, '{}/micrographs_ctf_selected.star'.format(data_folder))
        save_star = False

    return reduced_ctfcorrected


class class3d_run:
    def __init__(self, data_folder, job_n, n_cls):
        self.folder = data_folder
        self.path = self.folder + '/Class3D/{}/'.format(job_n)
        self.angles()
        self.cls_stats()
        self.n_cls = n_cls
        # self.run_params()

    def angles(self):
        self.rot, self.tilt, self.psi = get_angles(self.path)

    def cls_stats(self):
        model_star = glob.glob(self.path + '/*model.star')
        model_star.sort(key=os.path.getmtime)
        self.model_star = model_star
        self.class_path, self.n_classes, self.iter_, self.class_dist_per_run, self.class_res_per_run = get_classes(
            self.path, self.model_star)

    def plot_stats_dist(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_dist_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Class distribution')
        plt.legend()

        plt.show()

    def plot_stats_res(self):
        self.cls_stats()
        plt.figure(figsize=(8, 4), dpi=100)

        for N, cls_ in enumerate(self.class_res_per_run):
            plt.plot(range(0, len(cls_)), cls_.astype(float), label='{}'.format(N))

        plt.xlabel('Iter')
        plt.ylabel('Resolution, A')
        plt.legend()
        plt.show()

    def plot_all_classes(self):
        self.cls_stats()
        project_last_volume(self.path, '', self.n_cls)


    def run_params(self):
        with open(self.path + 'note.txt') as file:
            print(file.read())