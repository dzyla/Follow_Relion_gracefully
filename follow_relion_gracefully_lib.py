# Standard library imports
import os
import sys
import re
import glob
import math
import time
import shutil
import zipfile
import tarfile
import logging
import datetime
import textwrap
import platform
import fnmatch
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import socket

# Data manipulation and computation
import numpy as np
import pandas as pd

# System and process management
import psutil

# Reading and working with MRC files
import mrcfile

# Image processing and filters
import skimage.filters
from skimage import measure, exposure
from skimage.transform import rescale

# Plotting and visualization
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.lines import Line2D
from datetime import datetime

# HTTP requests
import requests

# Plotly library for interactive plots
import plotly.subplots as sp
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Image manipulation
from PIL import Image

# Working with CIF/Star files
from gemmi import cif

# Set the matplotlib backend
matplotlib.use('agg')

PATH_CHARACTER = os.path.sep
PLOT_HEIGHT = 500
MAX_MICS = 30
DEBUG = False

def set_debug(debug_level):
    global DEBUG
    DEBUG = debug_level

# Folder observer
class FolderMonitor:
    def __init__(self, directory, depth=1, log=None):
        t1 = time.time()
        self.directory = directory
        self.depth = depth
        self.files = []
        self.latest_modified_file, self.latest_modification_time, self.depth_files = self.get_last_modified_file()
        self.log = log

        if self.log is not None:
            self.log.debug(
                f"FolderMonitor initialized in {time.time() - t1} seconds")

    def get_last_modified_file(self):
        path_pattern = os.path.join(self.directory, *['*'] * self.depth)

        # Get all files at the given depth
        depth_files = glob.glob(path_pattern)
        latest_modified_file = None
        latest_modification_time = 0

        for file in depth_files:
            if os.path.isfile(file):
                mtime = os.path.getmtime(file)
                if mtime > latest_modification_time:
                    latest_modification_time = mtime
                    latest_modified_file = file

        return latest_modified_file, latest_modification_time, depth_files

    def check_changes(self):
        new_latest_modified_file, new_latest_modification_time, _ = self.get_last_modified_file()

        if new_latest_modification_time > self.latest_modification_time:
            self.latest_modified_file = new_latest_modified_file
            self.latest_modification_time = new_latest_modification_time
            return True
        else:
            return False

def get_ip_address():
    try:
        # Create a temporary UDP socket to connect to an external address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as temp_socket:
            # Connect to an arbitrary address and port, without sending any data
            temp_socket.connect(("8.8.8.8", 80))
            # Get the IP address of the workstation
            ip_address = temp_socket.getsockname()[0]
        return ip_address
    except Exception as e:
        print(f"Error occurred while retrieving IP address: {e}")
        return None
    
    
    
def find_project_paths():
    # 1. Read the project_paths.txt file and load each line as a list element
    with open("resources/project_paths.txt", "r") as f:
        project_paths = f.readlines()

    # Remove newline characters and any leading/trailing whitespace
    project_paths = [path.strip() for path in project_paths]

    # 2. List the directories inside the content/ directory
    content_dir = "content"
    content_directories = [d for d in os.listdir(content_dir) if os.path.isdir(os.path.join(content_dir, d))]

    # 3. Check if the last directory in the path corresponds to a directory in content/
    valid_project_paths = []
    for path in project_paths:
        last_directory = os.path.basename(os.path.normpath(path))
        if last_directory in content_directories:
            if path not in valid_project_paths:
                valid_project_paths.append(path)

    # 4. Save the valid project paths to a new file called content/projects.txt
    with open("content/projects.txt", "w") as f:
        for path in valid_project_paths:
            f.write(path + "\n")

def monitor_folder(idx, project_folder, log):
    log = initialize_logger(DEBUG)
    
    folder_watcher = FolderMonitor(project_folder, depth=3, log=log)
    log.debug(f'folder_watcher.modified_file: {folder_watcher.latest_modified_file}')
    time_mod = datetime.fromtimestamp(folder_watcher.latest_modification_time).strftime('%Y-%m-%d %H:%M:%S')
    log.debug(f'folder_watcher.modified_file_time: {time_mod}')
    return folder_watcher

def initialize_logger(debug_mode=False):
    # Create a logger object to write logs to console and/or file
    log_file = "FollowRelion.log"
    log = logging.getLogger("LOG")

    # Set the log level based on debug mode value
    log.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Add a file handler for writing logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create a formatter for the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Remove any existing handlers
    while log.handlers:
        log.removeHandler(log.handlers[0])

    # Add the file handler to the logger
    log.addHandler(file_handler)
    log.debug(f'debug_mode: {debug_mode}')

    # Return the logger
    return log


def clean_hostname(hostname):
    if hostname.startswith("http://"):
        hostname = hostname[len("http://"):]

    if hostname.endswith("/"):
        hostname = hostname[:-1]

    return hostname


def new_project():
    log = initialize_logger(DEBUG)
    content_folder = "./content/"

    for dir_name in os.listdir(content_folder):
        dir_path = os.path.join(content_folder, dir_name)
        try:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                log.info(f'REMOVED dir_path : {dir_path}')

        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")


def write_config(hostname):
    if hostname == '/':
        hostname = 'localhost'

    hostname = clean_hostname(hostname)

    config = open('config.toml', 'w')
    print(f'''
baseurl = "http://{hostname}/"
title = "Follow Relion Gracefully"
languageCode = "en-us"

# Pagination
paginate = 10
paginatePath = "page"

# Copyright
copyright = "Dawid Zyla 2023 / Follow Relion Gracefully"

# Highlighting
pygmentsUseClasses = true
pygmentsCodefences = true
pygmentsStyle = 'vs'

# Taxonomies
[taxonomies]
  tag = "tags"
  jobtype = "jobtype"
  project = "project"

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
  plotly= true
  theme_style = "modern"


[markup]
  [markup.highlight]
    anchorLineNos = false
    codeFences = true
    guessSyntax = false
    hl_Lines = ''
    hl_inline = false
    lineAnchors = ''
    lineNoStart = 1
    lineNos = true
    lineNumbersInTable = true
    noClasses = true
    noHl = false
    tabWidth = 4

  ''', file=config)


def download_hugo():
    log = initialize_logger(DEBUG)

    # Get the latest release tag from GitHub API
    response = requests.get(
        "https://api.github.com/repos/gohugoio/hugo/releases/latest")
    latest_release = response.json()["tag_name"]

    # Detect the operating system and architecture
    os_name = platform.system().lower()
    arch = platform.machine().lower()

    log.debug(f"os_name {os_name}, arch {arch}")

    # Set the binary extension and archive format
    bin_ext = ""
    archive_ext = ".tar.gz"
    if os_name == "windows":
        bin_ext = ".exe"
        archive_ext = ".zip"

    if arch == "x86_64" or arch == "amd64":
        arch = "amd64"
    elif arch == "arm64":
        arch = "arm64"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Check if the system is running CentOS 7
    is_centos_7 = False
    if os_name == "linux":
        is_centos_7 = check_centos_7()

    # Download the appropriate binary for the system
    #binary_type = "extended" if not is_centos_7 else ""
    binary_type = "" if not is_centos_7 else ""
    url = f"https://github.com/gohugoio/hugo/releases/download/{latest_release}/hugo_{binary_type}{latest_release[1:]}_{os_name}-{arch}{archive_ext}"
    log.info(f"Downloading Hugo {binary_type} version from: {url}")
    response = requests.get(url)

    # Save the downloaded archive
    archive_file = f"hugo_{binary_type}{archive_ext}"
    with open(archive_file, "wb") as f:
        f.write(response.content)

    # Create Hugo directory if not exists
    hugo_dir = Path("Hugo")
    hugo_dir.mkdir(parents=True, exist_ok=True)

    # Extract the binary to Hugo directory
    if os_name == "windows":
        with zipfile.ZipFile(archive_file, "r") as zip_ref:
            zip_ref.extractall(hugo_dir)
    else:
        with tarfile.open(archive_file, "r:gz") as tar_ref:
            tar_ref.extractall(hugo_dir)

    # Remove the downloaded archive
    Path(archive_file).unlink()

    # Make the binary executable
    hugo_bin = hugo_dir / f"hugo{bin_ext}"
    os.chmod(hugo_bin, hugo_bin.stat().st_mode | 0o111)

    print(
        f"Hugo {binary_type.strip('_')} version downloaded successfully! Execute with {hugo_bin}")

def check_centos_7():
    with open("/etc/os-release", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("PRETTY_NAME"):
            os_pretty_name = line.split("=")[1].strip().strip('"')
            if os_pretty_name == "CentOS Linux 7 (Core)":
                return True
    return False

def check_kill_process_by_name(process_name, report=False):
    log = initialize_logger(DEBUG)

    is_windows = sys.platform == "win32"
    process_pattern = f"{process_name}*"

    for proc in psutil.process_iter(['pid', 'name']):
        # Check if process_name matches the process pattern (process_name or process_name.exe)
        if fnmatch.fnmatch(proc.info['name'], process_pattern):

            if report:
                return True

            # If report=False, attempt to kill the process
            else:
                try:
                    if not is_windows:
                        os.kill(proc.info['pid'], 9)  # 9 is the SIGKILL signal
                    else:
                        os.system(f"taskkill /F /PID {proc.info['pid']}")

                    log.info(
                        f"Killed process '{proc.info['name']}' with PID {proc.info['pid']}")
                    print('Killed process:',
                          proc.info['name'], 'PID:', proc.info['pid'])

                except psutil.NoSuchProcess:
                    log.info(
                        f"No such process: {proc.info['name']} (PID {proc.info['pid']})")

    # If the process is not found or was killed, return False
    return False


def start_detached_Hugo_server(Hugo_server):
    is_windows = sys.platform == "win32"

    if is_windows:
        # Use the START command with /B and /MIN flags on Windows to start the process minimized and detached
        detached_command = f"start /B /MIN {Hugo_server}"
        process = subprocess.Popen(detached_command, shell=True)
    else:
        # Use the original method to start the detached process on Unix
        process = subprocess.Popen(Hugo_server, shell=True, stdin=None,
                                   stdout=None, stderr=None, close_fds=True, start_new_session=True)
    return process


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

    classes = mrcfile.mmap(path_).data
    for cls in classes:
        cls = adjust_contrast(cls, 5, 100)
        class_list.append(cls)

    return class_list


def plot_2dclasses_(path_, classnumber):
    classes = mrcfile.mmap(path_[0]).data
    z, x, y = classes.shape
    empty_class = np.zeros((x, y))
    line = []

    x_axis = int(np.sqrt(classnumber) * 1.66)

    y_axis = z // x_axis + int(z % x_axis != 0)
    add_extra = x_axis * y_axis - z

    def normalize(class_):
        if np.average(class_) != 0:
            try:
                return (class_ - np.min(class_)) / (np.max(class_) - np.min(class_))
            except:
                pass
        return class_

    classes = [normalize(class_) for class_ in classes]
    rows = []

    for i in range(0, z, x_axis):
        rows.append(np.concatenate(classes[i:i + x_axis], axis=1))

    if add_extra:
        row = np.concatenate(
            [*classes[-add_extra:], empty_class] * add_extra, axis=1)
        rows.append(row)

    final = np.concatenate(rows, axis=0)

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
            class_dist_per_run.append(
                parse_star_model(file, '_rlnClassDistribution'))
            class_res_per_run.append(parse_star_model(
                file, '_rlnEstimatedResolution'))

    # stack all data together
    try:
        class_dist_per_run = np.stack(class_dist_per_run)
        class_res_per_run = np.stack(class_res_per_run)

        # rotate matrix so the there is class(iteration) not iteration(class) and starting from iter 0 --> iter n
        class_dist_per_run = np.flip(np.rot90(class_dist_per_run), axis=0)
        class_res_per_run = np.flip(np.rot90(class_res_per_run), axis=0)

        # Find the class images (3D) or stack (2D)
        class_files = parse_star_model(
            model_star_files[-1], '_rlnReferenceImage')

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


def save_class_images(class_list, job, HUGO_FOLDER):
    for n, cls in enumerate(class_list):
        plt.imsave(os.path.join(HUGO_FOLDER,
                   f"{n + 1}.jpg"), cls, cmap='gray')


def generate_slider_html(last_n):
    js_code = f'''
    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="{last_n}" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="250">
    </div>

    <script>

        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + 1 + ".jpg";
            function showVal(newVal){{
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }}
    </script>
    <br>
    '''
    return "{{{{<rawhtml >}}}} {} {{{{< /rawhtml >}}}}".format(js_code)

def display_2dclasses(
        images,
        HUGO_FOLDER, job_name, model_star=[],
        label_wrap_length=10, label_font_size=8, sort=True, label=True, dpi=150):

    if model_star:
        # for select 2D
        model_star_df = parse_star_whole(model_star)
        if 'model_classes' in model_star_df.keys():
            labels = []
            cls_dist = []
            for n, dist in enumerate(model_star_df['model_classes']['_rlnClassDistribution']):
                labels.append('Class {} {}%'.format(
                    n + 1, round(float(dist) * 100, 2)))
                cls_dist.append(float(dist))
        else:
            labels = []
            cls_dist = []
            for n, dist in enumerate(model_star_df['#']['_rlnClassDistribution']):
                labels.append(f'Class {n + 1} {round(float(dist) * 100, 2)}%')
                cls_dist.append(float(dist))

    max_images = len(images)

    if sort and model_star:
        sort_matrix = np.argsort(cls_dist)[::-1]
        images = np.array(images)[sort_matrix]
        labels = np.array(labels)[sort_matrix]

    # Calculate optimal number of columns and rows
    num_classes = len(images)
    columns = 6
    rows = math.ceil(num_classes / columns)

    # Calculate figure width and height
    width_per_image = 2
    height_per_image = 2
    width = columns * width_per_image
    height = rows * height_per_image

    plt.figure(figsize=(width, height), dpi=dpi)
    for i, image in enumerate(images):

        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, cmap='gray')

        if label:
            if model_star:
                title = labels[i]
                title = textwrap.wrap(title, label_wrap_length)
                title = "\n".join(title)
                plt.title(title, fontsize=label_font_size)
        plt.axis('off')
        plt.tight_layout()

    job = job_name.split('/')[1]
    job_type = job_name.split('/')[0]

    save_class_images(images, job, HUGO_FOLDER)

    fig_filename = f'{job_type}_{job}_all.png'
    fig_path = os.path.join(os.getcwd(), HUGO_FOLDER,
                            f'{job_type}_{job}_all.png')
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    shortcode = f'{{{{< img src="{fig_filename}" style="width: 100%;">}}}}'
    return shortcode

# def display_2dclasses(
#         images,
#         HUGO_FOLDER, job_name, model_star=[],
#         columns=6, width=10, height=2,
#         label_wrap_length=10, label_font_size=8, sort=True, label=True, dpi=150):

#     if model_star:
#         # for select 2D
#         model_star_df = parse_star_whole(model_star)
#         if 'model_classes' in model_star_df.keys():
#             labels = []
#             cls_dist = []
#             for n, dist in enumerate(model_star_df['model_classes']['_rlnClassDistribution']):
#                 labels.append('Class {} {}%'.format(
#                     n + 1, round(float(dist) * 100, 2)))
#                 cls_dist.append(float(dist))
#         else:
#             labels = []
#             cls_dist = []
#             for n, dist in enumerate(model_star_df['#']['_rlnClassDistribution']):
#                 labels.append(f'Class {n + 1} {round(float(dist) * 100, 2)}%')
#                 cls_dist.append(float(dist))

#     max_images = len(images)

#     if sort and model_star:
#         sort_matrix = np.argsort(cls_dist)[::-1]
#         images = np.array(images)[sort_matrix]
#         labels = np.array(labels)[sort_matrix]

#     height = max(height, int(len(images) / columns) * height)
#     plt.figure(figsize=(width, height), dpi=dpi)
#     for i, image in enumerate(images):

#         plt.subplot(int(len(images) / columns + 1), columns, i + 1)
#         plt.imshow(image, cmap='gray')

#         if label:
#             if model_star:
#                 title = labels[i]
#                 title = textwrap.wrap(title, label_wrap_length)
#                 title = "\n".join(title)
#                 plt.title(title, fontsize=label_font_size)
#         plt.axis('off')
#         plt.tight_layout()

#     job = job_name.split('/')[1]
#     job_type = job_name.split('/')[0]

#     save_class_images(images, job, HUGO_FOLDER)

#     fig_filename = f'{job_type}_{job}_all.png'
#     fig_path = os.path.join(os.getcwd(), HUGO_FOLDER,
#                             f'{job_type}_{job}_all.png')
#     plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

#     shortcode = f'{{{{< img src="{fig_filename}" >}}}}'
#     return shortcode


def plot_2d_pyplot(rln_folder, HUGO_FOLDER, job_name):

    # Initialize empty list to store shortcodes
    cls2d_shortcodes = []
    last_n = 0

    # Get all model files in path_data
    model_files = glob.glob(os.path.join(rln_folder,'', job_name, "*model.star"))

    # Sort model files by time modified
    model_files.sort(key=os.path.getmtime)

    # Get number of iterations
    n_iter = len(model_files)

    # Get classes from model files
    (class_paths, n_classes_, iter_, class_dist_,
     class_res_) = get_classes(os.path.join(rln_folder,'', job_name), model_files)

    '''------ 2D images ---------'''

    # Get class list from first class path
    images = get_class_list(class_paths[0])

    # Display 2D classes
    shortcode = display_2dclasses(
        images, HUGO_FOLDER, job_name, model_star=model_files[-1])

    # Append pyplot string to cls2d_shortcodes list
    cls2d_shortcodes.append(shortcode)

    # Generate and append slider HTML
    slider_html = generate_slider_html(len(images))
    cls2d_shortcodes.append(slider_html)

    # Generate and append class distribution plot
    class_dist_plot = plot_class_distribution_pyplot(class_dist_)
    # Write plot and get shortcode for pyplot
    class_dist_plotly_string = write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, class_dist_plot, 'dist')
    # Append class distribution plot to cls2d_shortcodes list
    cls2d_shortcodes.append(class_dist_plotly_string)

    # Generate and append class resolution plot
    class_res_plot = plot_class_resolution_pyplot(class_res_)
    # Write plot and get shortcode for pyplot
    class_res_plotly_string = write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, class_res_plot, 'res')
    # Append class resolution plot to cls2d_shortcodes list
    cls2d_shortcodes.append(class_res_plotly_string)

    # Return cls2d_shortcodes list
    return cls2d_shortcodes


def plot_class_distribution_pyplot(class_dist):
    # Set Seaborn style for a modern look
    sns.set(style="white")

    class_dist = class_dist.astype(np.float16) * 100

    fig, ax = plt.subplots(figsize=(15, 5))

    x = np.arange(0, class_dist.shape[1])
    # Create a list of class distributions for stackplot
    class_dist_list = [class_dist[i] for i in range(class_dist.shape[0])]

    # Use stackplot instead of plot
    colors = sns.cubehelix_palette(n_colors=class_dist.shape[0])

    ax.stackplot(x, *class_dist_list, labels=[f'class {i + 1}' for i in range(
        class_dist.shape[0])], colors=colors)

    ax.set_xlabel("Iteration", fontweight='bold')
    ax.set_ylabel("Class distribution", fontweight='bold')
    ax.set_title("Class distribution", fontweight='bold')

    plt.ylim(0, 100)
    if class_dist.shape[1]-1 > 0:
        plt.xlim(0, class_dist.shape[1]-1)

    # Create a custom legend divided into columns
    n_col = 2  # Number of columns for the legend

    # Create legend_elements based on the number of classes
    if class_dist.shape[0] > 50:
        legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=f'class {i + 1}')
                           for i in range(class_dist.shape[0]) if (i + 1) % 5 == 0]
    else:
        legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=f'class {i + 1}')
                           for i in range(class_dist.shape[0])]

    legend_per_col = int(np.ceil(len(legend_elements) / n_col))

    # Position the legend outside the plot and divide it into columns
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 0.5),
              loc='center left', borderaxespad=0., ncol=n_col, fontsize='8')

    return fig


def plot_class_resolution_pyplot(class_res):
    sns.set(style="white")

    fig, ax = plt.subplots(figsize=(15, 5))

    colors = sns.cubehelix_palette(n_colors=class_res.shape[0])

    for n, class_ in enumerate(class_res):
        class_ = np.float16(class_)
        x = np.arange(0, class_res.shape[1])

        ax.plot(x, class_, label=f'class {n + 1}', color=colors[n])

    ax.set_xlabel("Iteration", fontweight='bold')
    ax.set_ylabel("Class Resolution [A]", fontweight='bold')
    ax.set_title("Class Resolution [A]", fontweight='bold')

    n_col = 2  # Number of columns for the legend
    legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=f'class {i + 1}') for i in range(class_res.shape[0])]

    # If there are more than 50 classes, show every 5th element in the legend
    if len(legend_elements) > 50:
        legend_elements = [legend_elements[i] for i in range(0, len(legend_elements), 5)]

    legend_per_col = int(np.ceil(len(legend_elements) / n_col))
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 0.5),
              loc='center left', borderaxespad=0., ncol=n_col, fontsize='8')

    return fig


def write_plot_get_shortcode_pyplot(job_name, HUGO_FOLDER, fig=None, extra_name=''):
    log = initialize_logger(DEBUG)

    job = job_name.split('/')[1]
    log.debug(f'job: {job}')

    job_type = job_name.split('/')[0]
    fig_filename = f"{job_type}_{job}_{extra_name}.png"
    log.debug(f'fig_filename: {fig_filename}')

    plt.tight_layout()

    # Save the figure and generate the shortcode
    if fig:
        fig.savefig(os.path.join(os.getcwd(),
                    HUGO_FOLDER,
                    fig_filename), dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

    shortcode = f'{{{{< img src="{fig_filename}" >}}}}'
    return shortcode


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


def process_extract(node_files, FOLDER, HUGO_FOLDER, job_name):
    log = initialize_logger(DEBUG)
    
    if len(node_files) == 1 or len(node_files) > 1:
        file_index = 1 if len(node_files) > 1 else 0
        if os.path.exists(os.path.join(FOLDER,'', node_files[file_index])):
            star_path = os.path.join(FOLDER,'', node_files[file_index])
            log.debug(f'Extract star_path: {star_path}')
            rnd_particles, num_particles = show_random_particles(
                star_path, FOLDER, random_size=100, r=1, adj_contrast=False)
            return plot_extract_js(rnd_particles, HUGO_FOLDER, job_name, num_particles)
    
    return ['No particles found']


def show_random_particles(star_path, project_path, random_size=100, r=1, adj_contrast=False):
    star = parse_star(star_path)[1]
    data_shape = star.shape[0]
    random_int = np.random.randint(0, data_shape, random_size)
    selected = star.iloc[random_int]

    particle_array = []
    for element in selected['_rlnImageName']:
        particle_data = element.split('@')
        img_path = os.path.join(project_path,'', particle_data[1])
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

    return particle_array, data_shape


def plot_extract_js(particle_list, HUGO_FOLDER, job_name, num_particles):
    shortcodes = []

    for n, particle in enumerate(particle_list):
        # plt.figure(figsize=(5,5), dpi=100)
        # plt.imshow(particle, cmap='gray')
        plt.imsave(os.path.join(
            HUGO_FOLDER, f'{n}.jpg'), particle, cmap='gray')
        last_n = n

    js_code = f'''
   
<div class="center">
<p>Preview of random extracted particles:<p>
<input id="valR" type="range" min="0" max="{last_n}" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
<span id="range">0</span>
<img id="img" width="200">
</div>

<script>
    
    var val = document.getElementById("valR").value;
        document.getElementById("range").innerHTML=val;
        document.getElementById("img").src = val + ".jpg";
        function showVal(newVal){{
          document.getElementById("range").innerHTML=newVal;
          document.getElementById("img").src = newVal+ ".jpg";
        }}
</script>
<br>
'''

    shortcodes.append(f'### Extracted {num_particles} particles')
    shortcodes.append(f"{{{{< rawhtml >}}}} {js_code} {{{{< /rawhtml >}}}}")

    return shortcodes


def plot_ctf_stats(starctf, HUGO_FOLDER, job_name):

    # Set Seaborn style
    sns.set(style='whitegrid')

    # Initialize shortcodes
    shortcodes = []

    # Define column names for clarity
    defocus_v = '_rlnDefocusV'
    ctf_max_res = '_rlnCtfMaxResolution'
    ctf_astigmatism = '_rlnCtfAstigmatism'
    ctf_figure_of_merit = '_rlnCtfFigureOfMerit'

    # Get relevant data
    reduced_ctfcorrected = starctf[1][[
        defocus_v, ctf_max_res, ctf_astigmatism, ctf_figure_of_merit]].astype(float)

    # Defocus / index
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(0, reduced_ctfcorrected.shape[0]),
            reduced_ctfcorrected[defocus_v].astype(float))
    ax.set_xlabel("Index")
    ax.set_ylabel(defocus_v)
    ax.set_title(defocus_v)
    plt.legend([defocus_v.replace('_rln', '')], loc='lower right')
    shortcodes.append('### Defocus per micrograph')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='index'))

    # Max Res / index
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(0, reduced_ctfcorrected.shape[0]),
            reduced_ctfcorrected[ctf_max_res].astype(float))
    ax.set_xlabel("Index")
    ax.set_ylabel(ctf_max_res)
    ax.set_title(ctf_max_res)
    plt.legend([ctf_max_res.replace('_rln', '')], loc='lower right')
    shortcodes.append('### Max resolution per micrograph')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='res'))

    # Defocus hist
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=reduced_ctfcorrected,
                 x=defocus_v, kde=True, ax=ax, bins=50)
    ax.set_xlabel(defocus_v.replace('_rln', ''))
    ax.set_ylabel("Number")
    ax.set_title(defocus_v)
    shortcodes.append('### Defocus histogram')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='def_hist'))

    # Astigmatism hist
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=reduced_ctfcorrected,
                 x=ctf_astigmatism, kde=True, ax=ax, bins=50)
    ax.set_xlabel(ctf_astigmatism.replace('_rln', ''))
    ax.set_ylabel("Number")
    ax.set_title(ctf_astigmatism)
    shortcodes.append('### Astigmatism histogram')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='ast_hist'))

    # MaxRes hist
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=reduced_ctfcorrected,
                 x=ctf_max_res, kde=True, ax=ax, bins=50)
    ax.set_xlabel(ctf_max_res.replace('_rln', ''))
    ax.set_ylabel("Number")
    ax.set_title(ctf_max_res)
    shortcodes.append('### Max resolution histogram')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='res_hist'))

    # Defocus / Max res
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(data=reduced_ctfcorrected, x=defocus_v,
                 y=ctf_max_res, ax=ax, cmap='crest', bins=50)
    ax.set_xlabel(defocus_v.replace('_rln', ''))
    ax.set_ylabel(ctf_max_res)
    ax.set_title(f"{defocus_v} / {ctf_max_res}")
    shortcodes.append('### Defocus / Max resolution histogram')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='def_res'))

    # Defocus / Figure of merit
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(data=reduced_ctfcorrected, x=defocus_v,
                 y=ctf_figure_of_merit, ax=ax, cmap='crest', bins=50)
    ax.set_xlabel(defocus_v.replace('_rln', ''))
    ax.set_ylabel(ctf_figure_of_merit)
    ax.set_title(f"{defocus_v} / {ctf_figure_of_merit}")
    shortcodes.append('### Defocus / Figure of merit histogram')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='def_fom'))

    return shortcodes


def plot_motioncorr_stats(star, HUGO_FOLDER, job_name):

    sns.set_style('whitegrid')

    shortcodes = []
    star_data = star[1]

    '''Motion per index'''
    fig, ax = plt.subplots()
    meta_names = ['_rlnAccumMotionTotal',
                  '_rlnAccumMotionEarly', '_rlnAccumMotionLate']

    for meta in meta_names:
        data_list = np.clip(star_data[meta].astype(float), None, 2000)
        sns.lineplot(x=np.arange(0, len(data_list)), y=data_list,
                     label=meta.replace('_rln', ''), linewidth=1, ax=ax)

    ax.set_xlabel('Index')
    ax.set_ylabel('Motion')
    ax.set_title('MotionCorr statistics')
    ax.legend(loc='upper right')

    shortcodes.append(
        '### Motion statistics per micrograph. Limitted to 2000 max')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='index'))

    plt.close()

    # Motion histograms
    fig, ax = plt.subplots()
    sns.set_style('whitegrid')

    colors = sns.color_palette(n_colors=len(meta_names), palette='crest')

    data_list = []

    for idx, meta in enumerate(meta_names):
        data_list = np.clip(star_data[meta].astype(float), None, 100)
        sns.histplot(data_list, alpha=0.5, bins=np.arange(min(data_list), max(data_list), 0.2), label=meta.replace(
            '_rln', ''), kde=False, ax=ax, linewidth=1.5, element='step', color=colors[idx])

    ax.set_xlabel('Motion')
    ax.set_ylabel('Number')
    ax.set_title('MotionCorr statistics. Limitted to 100 max')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.02))

    shortcodes.append('### Motion histograms')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='hist'))

    plt.close()

    return shortcodes


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized_array = (array - min_value) / (max_value - min_value)
    return normalized_array


def plot_3dclasses(files, conc_axis=0):
    class_averages = []

    if len(files) == 1:
        conc_axis = 1

    for n, class_ in enumerate(files):

        with mrcfile.mmap(class_) as mrc_stack:
            mrcs_file = mrc_stack.data
            z, x, y = mrc_stack.data.shape

            average_top = np.mean(mrcs_file, axis=0)
            try:
                average_top = normalize_array(average_top)
            except:
                pass

            average_front = np.mean(mrcs_file, axis=1)

            try:
                average_front = normalize_array(average_front)

            except:
                pass

            average_side = np.mean(mrcs_file, axis=2)

            try:
                average_side = normalize_array(average_side)
            except:
                pass

            # join 3 views together if joined in 3D classification or Ab Initio, otherwise return projections (for mask)
            if conc_axis == 0:
                try:
                    average_class = np.concatenate(
                        (average_top, average_front, average_side), conc_axis)

                except ValueError:
                    max_size = max(
                        average_top.shape[1], average_front.shape[1], average_side.shape[1])

                    padded_top = np.pad(
                        average_top, ((0, 0), (0, max_size - average_top.shape[1])))
                    padded_front = np.pad(
                        average_front, ((0, 0), (0, max_size - average_front.shape[1])))
                    padded_side = np.pad(
                        average_side, ((0, 0), (0, max_size - average_side.shape[1])))

                    average_class = np.concatenate(
                        (padded_top, padded_front, padded_side), conc_axis)

                class_averages.append(average_class)
            else:
                class_averages = [average_top, average_front, average_side]

    try:
        if conc_axis == 0:
            final_average = np.concatenate(class_averages, axis=1)
        else:
            final_average = class_averages

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
    x1, x2 = int((volume_.shape[0] - new_size) /
                 2), volume_.shape[0] - int((volume_.shape[0] - new_size) / 2)

    fft_shift_new = fft_shift[x1:x2, x1:x2, x1:x2]

    # Apply spherical mask
    lx, ly, lz = fft_shift_new.shape
    X, Y, Z = np.ogrid[0:lx, 0:ly, 0:lz]
    dist_from_center = np.sqrt(
        (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 + (Z - lz / 2) ** 2)
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
    fig_volume.data[0].update(
        lighting=dict(
            ambient=0.5,  # Ambient light intensity (default value: 0.8)
            diffuse=0.9,  # Diffuse light intensity (default value: 0.8)
            fresnel=0.5,  # Fresnel reflectivity (default value: 0.2)
            specular=0.2,  # Specular light intensity (default value: 0.05)
            roughness=0.8,  # Surface roughness (default value: 0.5)
            facenormalsepsilon=0,  # Face normal epsilon (default value: 1e-6)
            # Vertex normal epsilon (default value: 1e-12)
            vertexnormalsepsilon=0
        )
    )
    fig_volume.data[0].update(
        lightposition=dict(
            x=-100,  # x-coordinate of the light source
            y=-200,  # y-coordinate of the light source
            z=-300   # z-coordinate of the light source
        )
    )
    fig_volume.update_scenes(
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=-1, z=0)
        )
    )
    fig_.add_trace(fig_volume['data'][0])

    return fig_


def write_plot_get_shortcode(fig, json_name, job_name, HUGO_FOLDER, fig_height=500):

    log = initialize_logger(DEBUG)

    fig_name = json_name + job_name.replace('/', '_')
    plotly_url = f"{fig_name.lower()}.json"
    log.debug(f'plotly_url: {plotly_url}')

    plotly_file = os.path.join(HUGO_FOLDER, plotly_url)
    fig.write_json(plotly_file)
    log.debug(f'Saved plotly json: {plotly_file}')

    shortcode = f'{{{{< plotly json="{plotly_url}" height="{fig_height}px" >}}}}'
    log.debug(f'{job_name} shortcode: {shortcode}')

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

    data_star = glob.glob(os.path.join(path_,'', '*data.star'))
    data_star.sort(key=os.path.getmtime)

    last_data_star = data_star[-1]

    rot_angles = parse_star_data(last_data_star, '_rlnAngleRot').astype(float)
    tilt_angles = parse_star_data(
        last_data_star, '_rlnAngleTilt').astype(float)
    psi_angles = parse_star_data(last_data_star, '_rlnAnglePsi').astype(float)

    return rot_angles, tilt_angles, psi_angles


def plot_combined_classes(class_paths, job_name, HUGO_FOLDER):
    num_classes = len(class_paths)
    cols = min(5, num_classes)
    rows = math.ceil(num_classes / 5)

    fig = sp.make_subplots(rows=rows, cols=cols, specs=[[{'type': 'scene'}] * cols] * rows)

    annotations = []

    for n, cls_path in enumerate(class_paths):
        mrc_cls_data = mrcfile.open(cls_path, permissive=True).data
        fig_ = plot_volume(go.Figure(), mrc_cls_data)

        if fig_ is not None:
            row, col = divmod(n, cols)
            row += 1
            col += 1

            trace = fig_.data[0]
            fig.add_trace(trace, row=row, col=col)

            fig.update_layout(
                **{f'scene{n + 1}': dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=dict(eye=dict(x=3, y=3, z=3))
                )},
                margin=dict(l=0, r=0, t=0, b=0)
            )

            # Set the position for each subplot title
            title_x = (col - 1) / cols + 0.5 / cols
            title_y = 1 - (row - 1) * 0.9 / rows
            annotations.append(dict(text=f"Class {n + 1}", x=title_x, y=title_y, showarrow=False, font=dict(size=16),
                                    xref="paper", yref="paper", xanchor="center", yanchor="top"))

    # Update the layout with the annotations
    fig.update_layout(annotations=annotations)

    shortcode = write_plot_get_shortcode(
        fig, 'cls3d_combined_', job_name, HUGO_FOLDER, fig_height=rows*300)

    return "#### Combined Classes:", shortcode


def plot_projections(path_data, HUGO_FOLDER, job_name, class_paths, class_dist_=None, string='Class', cmap='gray', extra_name=''):

    if class_paths is not None and len(class_paths) > 1:
        projections = plot_3dclasses(class_paths)

        plt.close()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.grid(False)

        # Display the cls3d_projections image
        ax.imshow(projections, cmap=cmap)

        # Configure X-axis ticks and labels. If mask or Refine skip adding distributions
        labels_positions_x = np.linspace(1 / len(class_paths) * projections.shape[1], projections.shape[1],
                                         len(class_paths)) - 0.5 * 1 / len(class_paths) * projections.shape[1]

        if class_dist_ is not None:
            class_dist_ = np.array(class_dist_)
            try:
                labels_x = [
                    f"{string} {x+1} ({round(float(class_dist_[:, -1][x]) * 100, 2)}%)" for x, cls in enumerate(class_paths)]
            except:
                labels_x = [
                    f"{string} {x+1} ({round(float(class_dist_[x]) * 100, 2)}%)" for x, cls in enumerate(class_paths)]
        else:
            labels_x = [f"{string} {x+1}" for x, cls in enumerate(class_paths)]

        ax.set_xticks(labels_positions_x)
        ax.set_xticklabels(labels_x)
        ax.xaxis.tick_top()

        # Configure Y-axis ticks and labels
        labels_positions_y = np.linspace(1 / 3 * projections.shape[0], projections.shape[0],
                                         3) - 0.5 * 1 / 3 * projections.shape[0]
        labels_y = ["Z", "X", "Y"]
        ax.set_yticks(labels_positions_y)
        ax.set_yticklabels(labels_y, fontweight='bold')

        plt.close(fig)

        shortcode = write_plot_get_shortcode_pyplot(
            job_name, HUGO_FOLDER, fig, extra_name=extra_name)

        return "#### Class Projections:", shortcode

    else:
        projections = plot_3dclasses(class_paths, conc_axis=1)
        shortcodes = []

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        projection_titles = ["Z Projection", "X Projection", "Y Projection"]

        for idx, ax in enumerate(axes):
            ax.imshow(projections[idx], cmap='gray')
            ax.set_title(projection_titles[idx], fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        shortcode = write_plot_get_shortcode_pyplot(
            job_name, HUGO_FOLDER, fig, extra_name=extra_name)
        plt.close()
        return "#### Class Projections:", shortcode

def plot_projections(path_data, HUGO_FOLDER, job_name, class_paths, class_dist_=None, string='Class', cmap='gray', extra_name=''):

    if class_paths is not None and len(class_paths) > 1:
        projections = plot_3dclasses(class_paths)

        plt.close()
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.grid(False)

        # Display the cls3d_projections image
        ax.imshow(projections, cmap=cmap)

        # Configure X-axis ticks and labels. If mask or Refine skip adding distributions
        labels_positions_x = np.linspace(1 / len(class_paths) * projections.shape[1], projections.shape[1],
                                         len(class_paths)) - 0.5 * 1 / len(class_paths) * projections.shape[1]

        if class_dist_ is not None:
            class_dist_ = np.array(class_dist_)
            try:
                labels_x = [
                    f"{string} {x+1} ({round(float(class_dist_[:, -1][x]) * 100, 2)}%)" for x, cls in enumerate(class_paths)]
            except:
                labels_x = [
                    f"{string} {x+1} ({round(float(class_dist_[x]) * 100, 2)}%)" for x, cls in enumerate(class_paths)]
        else:
            labels_x = [f"{string} {x+1}" for x, cls in enumerate(class_paths)]

        ax.set_xticks(labels_positions_x)
        ax.set_xticklabels(labels_x, rotation=90 if len(class_paths) > 8 else 0)
        ax.xaxis.tick_top()

        # Configure Y-axis ticks and labels
        labels_positions_y = np.linspace(1 / 3 * projections.shape[0], projections.shape[0],
                                         3) - 0.5 * 1 / 3 * projections.shape[0]
        labels_y = ["Z", "X", "Y"]
        ax.set_yticks(labels_positions_y)
        ax.set_yticklabels(labels_y, fontweight='bold')

        plt.close(fig)

        shortcode = write_plot_get_shortcode_pyplot(
            job_name, HUGO_FOLDER, fig, extra_name=extra_name)

        return "#### Class Projections:", shortcode

    else:
        projections = plot_3dclasses(class_paths, conc_axis=1)
        shortcodes = []

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        projection_titles = ["Z Projection", "X Projection", "Y Projection"]

        for idx, ax in enumerate(axes):
            ax.imshow(projections[idx], cmap='gray')
            ax.set_title(projection_titles[idx], fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        shortcode = write_plot_get_shortcode_pyplot(
            job_name, HUGO_FOLDER, fig, extra_name=extra_name)
        plt.close()
        return "#### Class Projections:", shortcode

def plot_class_distribution(class_dist_, job_name, HUGO_FOLDER, PLOT_HEIGHT):
    fig = go.Figure()
    for n, class_ in enumerate(class_dist_):
        class_ = class_.astype(float)*100
        x = np.arange(0, class_dist_.shape[1])
        fig.add_trace(go.Scatter(x=x, y=class_, name=f'Class {n + 1}',
                                 showlegend=True, hovertemplate=f"Class {n + 1}<br>inter: %{{x}}<br>Cls dist: %{{y:.2f}}{'%'}<extra></extra>", mode='lines', stackgroup='one'))

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Class distribution")

    fig.update_layout(title="Class distribution")
    fig.update_layout(hovermode="x unified")

    shortcode = write_plot_get_shortcode(
        fig, 'cls3d_dist_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)

    return shortcode


def plot_class_resolution(class_res_, job_name, HUGO_FOLDER, PLOT_HEIGHT):
    fig = go.Figure()
    for n, class_ in enumerate(class_res_):
        class_ = class_.astype(float)
        x = np.arange(0, class_res_.shape[1])

        fig.add_scatter(x=x, y=class_, name='Class {}'.format(n + 1), showlegend=True,
                        hovertemplate=f"Class {n + 1}<br>inter: %{{x}}<br>Cls res:%{{y:.2f}}A<extra></extra>",)

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Class Resolution [A]")

    fig.update_layout(title="Class Resolution [A]")
    fig.update_layout(hovermode="x unified")

    shortcode = write_plot_get_shortcode(
        fig, 'cls3d_res_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)

    return shortcode


def plot_angular_distribution(psi, rot, tilt, job_name, HUGO_FOLDER, PLOT_HEIGHT):

    angles_data = pd.DataFrame({"Psi": psi, "Rotation": rot, "Tilt": tilt})

    # Set Seaborn style for a modern look
    sns.set(style="whitegrid")

    # Create a custom PairGrid
    g = sns.PairGrid(angles_data, corner=True)

    g.fig.set_size_inches(10, 7)
    # Set up the 2D histograms using sns.histplot
    g.map_diag(sns.histplot, bins=20, kde=True)
    g.map_lower(sns.histplot, bins=20, cmap="magma", cbar=False)
    g.map_upper(sns.kdeplot)

    # Set the title for the 2D histogram matrix
    g.fig.suptitle(
        "2D Histogram Matrix of Rotation, Tilt, and Phi Angles", y=1.05)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1,
                        top=0.9, wspace=0.4, hspace=0.4)

    plt.tight_layout()

    shortcode = write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, g.fig, extra_name='angular_dist')

    return shortcode


def plot_cls3d_stats(rln_folder, HUGO_FOLDER, job_name):

    # Get the path for the data folder
    path_data = os.path.join(rln_folder,'', job_name)

    # Get model files
    model_files = glob.glob(os.path.join(path_data, "*model.star"))
    model_files.sort(key=os.path.getmtime)
    n_inter = len(model_files)
    shortcodes = []

    if n_inter != 0:
        (class_paths, n_classes_, iter_, class_dist_,
         class_res_) = get_classes(path_data, model_files)

        combined_classes = plot_combined_classes(
            class_paths, job_name, HUGO_FOLDER)
        shortcodes.extend(combined_classes)

        class_projections = plot_projections(
            path_data, HUGO_FOLDER, job_name, class_paths, class_dist_)
        shortcodes.append(class_projections[0])
        shortcodes.append(class_projections[1])

        class_distribution = plot_class_distribution(
            class_dist_, job_name, HUGO_FOLDER, PLOT_HEIGHT)
        shortcodes.append("#### Class distribution:")
        shortcodes.append(class_distribution)

        class_resolution = plot_class_resolution(
            class_res_, job_name, HUGO_FOLDER, PLOT_HEIGHT)
        shortcodes.append("#### Class Resolution [A]:")
        shortcodes.append(class_resolution)

        # Plot Angular distribution
        rot, tilt, psi = get_angles(path_data)
        angular_distribution = plot_angular_distribution(
            psi, rot, tilt, job_name, HUGO_FOLDER, PLOT_HEIGHT)
        shortcodes.append("#### Angular distribution matrix:")
        shortcodes.append(angular_distribution)

    else:
        shortcodes = ['No *_model.star found in the folder']

    return shortcodes


def get_note(path_):
    if os.path.exists(path_):
        file = open(path_)
        file_data = str(file.read())
        file_data = file_data.replace('++++', '\n')
        file_data = file_data.replace("`", '')

        file_data = file_data.replace('which', '\nwhich')

        file.close()

    else:
        file_data = 'Process queuing. Waiting for the run.'
    return file_data


def plot_mask(mask_path, HUGO_FOLDER, job_name):

    #path_data = os.path.join(rln_folder, job_name)
    mask_projections = plot_3dclasses([mask_path], conc_axis=1)
    shortcodes = []

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    projection_titles = ["Z Projection", "X Projection", "Y Projection"]

    for idx, ax in enumerate(axes):
        ax.imshow(mask_projections[idx], cmap='gray')
        ax.set_title(projection_titles[idx], fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    # Save the figure and generate the shortcode
    job = job_name.split('/')[1]
    fig_filename = f"maskcreate_{job}.png"

    fig.savefig(os.path.join(os.getcwd(), HUGO_FOLDER,
                fig_filename), bbox_inches='tight', pad_inches=0, dpi=150)

    shortcode = f'{{{{< img src="{fig_filename}" >}}}}'
    shortcodes.append(shortcode)

    # Close the figure
    plt.close()

    return shortcodes


def rotate_volume(volume, angle, rotaxes_):
    from scipy import ndimage

    volume = np.array(volume)
    volume = ndimage.interpolation.rotate(
        volume, angle, reshape=False, axes=rotaxes_)
    return volume


def plot_2d_projections(class_paths, job_name, HUGO_FOLDER):
    cls_path = class_paths[-1]
    volume_data = mrcfile.mmap(cls_path, permissive=True).data
    volume_100px = resize_3d(volume_data, new_size=100)

    job = job_name.split('/')[1]

    for n, angle in enumerate(np.arange(0, 360, 10)):
        projection = np.mean(rotate_volume(
            volume_100px, angle, [1, 2]), axis=2)
        plt.imsave(os.path.join(HUGO_FOLDER, job,
                   f'{n}.jpg'), projection, cmap='gray')
        last_n = n

    js_code = f'''
    <div class="center">
    <p>Volume projections preview:<p>
    <input id="valR" type="range" min="0" max="{last_n}" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="250">
    </div>

    <script>
        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + ".jpg";
            function showVal(newVal){{
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }}
    </script>
    <br>
    '''

    js_string = f"{{{{< rawhtml >}}}} js_code {{{{< /rawhtml >}}}}"

    return js_string


def plot_3d_model(class_paths, job_name, HUGO_FOLDER, shorcode_txt='', fig_height=600):
    cls_path = class_paths[-1]
    volume_data = mrcfile.mmap(cls_path, permissive=True).data
    volume_100px = resize_3d(volume_data, new_size=100)
    fig_ = go.Figure()
    fig_ = plot_volume(fig_, volume_100px)
    fig_['layout'].update(scene=dict(xaxis=dict(visible=False), yaxis=dict(
        visible=False), zaxis=dict(visible=False)))

    shortcode = write_plot_get_shortcode(
        fig_, shorcode_txt, job_name, HUGO_FOLDER, fig_height=fig_height)

    return shortcode


def plot_refine3d(rlnFOLDER, HUGO_FOLDER, job_name):

    path_data = os.path.join(rlnFOLDER,'', job_name)
    model_files = glob.glob(os.path.join(path_data, "*model.star"))
    model_files.sort(key=os.path.getmtime)

    n_iter = len(model_files)

    if n_iter != 0:
        ref3d_shortcodes = []

        # Get the classes from the path_data and model_files
        (class_paths, n_classes_, iter_, class_dist_,
         class_res_) = get_classes(path_data, model_files)

        # Plot Refine3D in 3D
        ref3d_3d_model = plot_3d_model(class_paths, job_name, HUGO_FOLDER)
        ref3d_shortcodes.append("#### Class Preview:")
        ref3d_shortcodes.append(ref3d_3d_model)

        # Plot Refine3D in 2D projections
        ref3d_projections = plot_projections(
            path_data, HUGO_FOLDER, job_name, class_paths, class_dist_)
        ref3d_shortcodes.append(ref3d_projections[0])
        ref3d_shortcodes.append(ref3d_projections[1])

        # Plot Class resolution
        class_resolution = plot_class_resolution(
            class_res_, job_name, HUGO_FOLDER, PLOT_HEIGHT)
        ref3d_shortcodes.append("#### Class Resolution [A]:")
        ref3d_shortcodes.append(class_resolution)

        # Plot Angular distribution
        rot, tilt, psi = get_angles(path_data)
        angular_distribution = plot_angular_distribution(
            psi, rot, tilt, job_name, HUGO_FOLDER, PLOT_HEIGHT)
        ref3d_shortcodes.append("#### Angular distribution matrix:")
        ref3d_shortcodes.append(angular_distribution)

    else:
        ref3d_shortcodes = ['No *model.star found in the folder']

    return ref3d_shortcodes


def plot_picks(rln_folder, HUGO_FOLDER, job_name, img_resize_fac=0.2):

    path_data = os.path.join(rln_folder,'', job_name)

    autopick_shorcode = []
    suffix = ''
    coord_paths = ''

    #coordinate_files = glob.glob(os.path.join(path_data, "*.star"))
    # coordinate_files.sort(key=os.path.getmtime)

    # Relion 4 has much easier coordinate handling
    if os.path.exists(os.path.join(path_data,'', 'autopick.star')) and os.path.getsize(os.path.join(path_data,'', 'autopick.star')) > 0:
        pick_old = False

        autopick_star = parse_star_whole(os.path.join(path_data,'', 'autopick.star'))['coordinate_files']
        mics_paths = autopick_star['_rlnMicrographName']
        coord_paths = autopick_star['_rlnMicrographCoordinates']
        if coord_paths.shape[0] != 1:
            coord_paths = np.squeeze(coord_paths.to_numpy())

    elif os.path.exists(os.path.join(path_data,'', 'manualpick.star')) and os.path.getsize(os.path.join(path_data,'', 'manualpick.star')) > 0:
        pick_old = False

        manpick_star = parse_star_whole(os.path.join(path_data,'', 'manualpick.star'))['coordinate_files']
        mics_paths = manpick_star['_rlnMicrographName']
        coord_paths = manpick_star['_rlnMicrographCoordinates']
        if coord_paths.shape[0] != 1:
            coord_paths = np.squeeze(coord_paths.to_numpy())

    elif glob.glob(os.path.join(path_data, 'coords_suffix_*')) != []:
        # Get the coordinates from subfolders
        pick_old = True

        # get suffix firsts

        suffix_file = glob.glob(os.path.join(path_data,'', 'coords_suffix_*'))[0]
        suffix = os.path.basename(suffix_file).replace(
            'coords_suffix_', '').replace('.star', '')

        # Get the folder with micrographs
        mics_data_path = open(
            glob.glob(os.path.join(path_data,'', 'coords_suffix_*'))[0]).readlines()[0].replace('\n', '')

        all_mics_paths = parse_star_data(os.path.join(rln_folder,'', mics_data_path), '_rlnMicrographName')

        mics_paths = []
        for name in all_mics_paths:
            mics_paths.append(os.path.join(rln_folder,'', name))

    # Autopick topaz training job
    elif glob.glob(os.path.join(path_data, 'model_training.txt')) != []:

        topaz_training_txt = glob.glob(os.path.join(path_data, 'model_training.txt'))[0]

        data = pd.read_csv(topaz_training_txt, delimiter='\t')
        data_test = data[data['split'] == 'test']

        x = data_test['epoch']
        data_test = data_test.drop(['iter', 'split', 'ge_penalty'], axis=1)

        fig_ = go.Figure()

        for n, column in enumerate(data_test.columns):
            if column != 'epoch':
                y = data_test[column]
                fig_.add_scatter(
                    x=x, y=y, name=f'{column}', hovertemplate=f"{column}<br>Epoch: %{{x}}<br>Y: %{{y:.2f}}<extra></extra>")

        fig_.update_xaxes(title_text="Epoch")
        fig_.update_yaxes(title_text="Statistics")

        fig_.update_layout(
            title="Topaz training stats. Best model: {}".format(data_test[data_test['auprc'].astype(float) ==
                                                                          np.max(data_test['auprc'].astype(float))][
                'epoch'].values)
        )

        fig_.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig_.update_layout(hovermode="x unified")

        shortcode = write_plot_get_shortcode(
            fig_, 'topaz_train_', job_name, HUGO_FOLDER, fig_height=PLOT_HEIGHT)

        return [shortcode]

    else:
        return ['Something went wrong. Files do not exist or job failed']

    # use the shuffle function to shuffle the files instead of hardcoding the random order
    # Mainly if number of files is smaller than MAX_MICS
    indices = np.arange(0, np.array(mics_paths).shape[0])
    np.random.shuffle(indices)

    processed = -1

    cpu_count = (int(os.cpu_count()) + 1) // 2
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = []
        for n, file in enumerate(np.array(mics_paths)[indices]):
            if processed == MAX_MICS:
                break

            future = executor.submit(process_micrograph, processed + 1, file, rln_folder,
                                     coord_paths, indices, pick_old, suffix, path_data, HUGO_FOLDER)
            futures.append(future)
            processed += 1

        for future in futures:
            future.result()

    js_code = f'''
    <div class="center">
    <p>Preview of random autopicked micrograph:<p>
    <input id="valR" type="range" min="0" max="{processed}" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="600">
    </div>

    <script>

        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + ".jpg";
            function showVal(newVal){{
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }}
    </script>
    <br>
    '''

    js_string = f"{{{{<rawhtml >}}}} {js_code} {{{{< /rawhtml >}}}}"
    return [js_string]


def process_micrograph(n, file, rln_folder, coord_paths, indices, pick_old, suffix, path_data, HUGO_FOLDER):
    
    log = initialize_logger(DEBUG)
    log.debug(f"Processing micrograph: {file}")
    
    try:
        plt.close()

        if pick_old:
            pick_star_name = os.path.basename(file).replace(
                '.mrc', '_{}.star'.format(suffix))
            pick_star_name_path = glob.glob(os.path.join(path_data, '**', pick_star_name))

            if pick_star_name_path != []:
                micrograph = mrcfile.mmap(os.path.join(rln_folder,'', file), permissive=True).data
                coords_file = parse_star(pick_star_name_path[0])[1]
                coords_x = coords_file['_rlnCoordinateX']
                coords_y = coords_file['_rlnCoordinateY']
                autopick_fom = coords_file['_rlnAutopickFigureOfMerit']

        else:
            micrograph = mrcfile.mmap(os.path.join(rln_folder,'', file), permissive=True).data
            coords_file = parse_star(os.path.join(rln_folder,'', coord_paths[indices[n]]))[1]
            coords_x = coords_file['_rlnCoordinateX']
            coords_y = coords_file['_rlnCoordinateY']

        if micrograph.shape[0] > 500:
            img_resize_fac = 500/micrograph.shape[0]
            mic_red = rescale(micrograph.astype(float), img_resize_fac)
        else:
            img_resize_fac = 1
            mic_red = micrograph.astype(float)

        p1, p2 = np.percentile(mic_red, (0.1, 99.8))
        mic_red = np.array(exposure.rescale_intensity(mic_red, in_range=(p1, p2)))

        plt.close()
        sns.set_style("white")
        fig, ax = plt.subplots()

        ax.imshow(mic_red, cmap='gray')
        ax.axis('off')

        marker_style = dict(linestyle='-', linewidth=1, marker='o',
                            s=120, facecolors='none', linewidths=0.5)
        ax.scatter(coords_x.astype(float) * img_resize_fac, coords_y.astype(float) * img_resize_fac,
                edgecolor="limegreen", **marker_style)

        fig.savefig(os.path.join(
            HUGO_FOLDER, f'{n}.jpg'), bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close(fig)
    except Exception as e:
        log.debug(f'File not found: {e}')


def plot_ctf_refine(node_files, FOLDER, HUGO_FOLDER, job_name):

    log = initialize_logger(DEBUG)
    job_name_path = job_name.replace('/', PATH_CHARACTER)

    shortcodes = []

    ctf_data = glob.glob(os.path.join(FOLDER,'', job_name_path, '*.mrc'))
    log.debug(f'ctf_data: {ctf_data}')

    n_images = len(ctf_data)
    log.debug(f'Number of images: {n_images}')

    if n_images == 0:
        note = get_note(os.path.join(FOLDER,'', job_name_path, 'note.txt'))
        log.debug(f'note: {note}')

        # get last refine job
        try:
            refine_path = re.search(r"--i\s([\w\d/]+\.star)", note).group(1)
            log.debug(f'refine_path 1: {refine_path}')

            # If CtfRefine is used as in tutorial, then 3 runs are done. The last one should have Refine3D path
            if 'CtfRefine' in refine_path:
                refine_path = refine_path.split('/')

                # search another job
                note = get_note(os.path.join(
                    FOLDER, refine_path[0], refine_path[1], 'note.txt'))

                refine_path = re.search(
                    r"--i\s([\w\d/]+\.star)", note).group(1)

                log.debug(f'refine_path 2: {refine_path}')

                if 'CtfRefine' in refine_path:
                    refine_path = refine_path.split('/')
                    note = get_note(os.path.join(
                        FOLDER, refine_path[0], refine_path[1], 'note.txt'))

                    refine_path = re.search(
                        r"--i\s([\w\d/]+\.star)", note).group(1)

                    log.debug(f'refine_path 2: {refine_path}')
                    if 'Refine3D' in refine_path:
                        refine_path = refine_path
                else:
                    refine_path = ''

        except Exception as e:
            log.info(f'CTF refine error: {e}')
            refine_path = ''

        if refine_path != '':

            refine_data = parse_star_whole(
                os.path.join(FOLDER, refine_path.replace('/', PATH_CHARACTER)))['particles']
            refine_data_U = refine_data['_rlnDefocusU'].values.astype(float)
            refine_data_V = refine_data['_rlnDefocusV'].values.astype(float)

            # node 1 is ctf refined star
            ctf_refine_data = parse_star_whole(
                os.path.join(FOLDER, node_files[1].replace('/', PATH_CHARACTER)))['particles']
            ctf_refine_data_U = ctf_refine_data['_rlnDefocusU'].values.astype(
                float)
            ctf_refine_data_V = ctf_refine_data['_rlnDefocusV'].values.astype(
                float)

            ctf_U = refine_data_U - ctf_refine_data_U
            ctf_V = refine_data_V - ctf_refine_data_V

            # Set Seaborn style
            sns.set_style("whitegrid")

            # Set the figure size and DPI
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100)

            # Plot _rlnDefocusU Change histogram
            sns.histplot(ctf_U, bins=100, ax=axes[0], color="tab:blue")
            axes[0].set_title('DefocusU Change')
            axes[0].set_ylabel('Count')
            axes[0].set_xlabel('Defocus change, Å')

            # Plot _rlnDefocusV Change histogram
            sns.histplot(ctf_V, bins=100, ax=axes[1], color="tab:orange")
            axes[1].set_title('DefocusV Change')
            axes[1].set_ylabel('Count')
            axes[1].set_xlabel('Defocus change, Å')

            # Plot _rlnDefocusU/V Change 2D histogram
            h = axes[2].hist2d(ctf_U, ctf_V, bins=100, cmap="viridis")
            axes[2].set_title('DefocusU/V Change')
            axes[2].set_xlabel('DefocusU change, Å')
            axes[2].set_ylabel('DefocusV change, Å')

            # Add a colorbar to the 2D histogram
            cb = plt.colorbar(h[3], ax=axes[2])
            cb.set_label('Count')

            # Adjust the layout
            fig.tight_layout()
            shortcodes.append('### CtfRefine values change')
            shortcodes.append(write_plot_get_shortcode_pyplot(
                job_name, HUGO_FOLDER, fig, extra_name='hist'))
            plt.close()

            return shortcodes

        else:
            log.info(
                'Per particle CTF estimation. No Refine3D *data.star provided. Nothing to plot')
            return ['']

    param_names = [os.path.basename(file).replace(
        '.mrc', '') for file in ctf_data]
    ctf_images = [mrcfile.open(img).data for img in ctf_data]

    fig = display_ctf_stats(ctf_images, param_names)

    shortcodes.append('### CtfRefine Plots')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='hist'))

    return shortcodes


def plot_polish(node_files, FOLDER, HUGO_FOLDER, job_name):
    shortcodes = []

    log = initialize_logger(DEBUG)

    if any('opt_params_all_groups.txt' in element for element in node_files):
        with open(os.path.join(FOLDER, node_files[0]), 'r') as f:
            parameters = f.readlines()[0].replace('\n', '').split(' ')
            shortcodes.append('### Optimal parameters')
            shortcodes.append(
                f'--s_vel {parameters[0]} --s_div {parameters[1]} --s_acc {parameters[2]}')

    elif any('shiny.star' in element for element in node_files):

        # plot bfactors per frame
        try:
            bafactors_star_path = glob.glob(os.path.join(
                FOLDER, job_name.replace('/', PATH_CHARACTER), 'bfactors.star'))[0]
            log.debug(f'bafactors_star_path: {bafactors_star_path}')
        except Exception as e:
            log.info(f'Could not open bfactors.star. Error: {e}')

        bfactors_data = parse_star_whole(bafactors_star_path)[
            'perframe_bfactors'].astype(float)

        # Seaborn plot setup
        sns.set(style="whitegrid")

        # Create the plot figure
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # First Y-axis plot (_rlnBfactorUsedForSharpening)
        sns.lineplot(x='_rlnMovieFrameNumber', y='_rlnBfactorUsedForSharpening',
                     data=bfactors_data, color='black', ax=ax1)

        # Set labels for the first Y-axis
        ax1.set_ylabel('_rlnBfactorUsedForSharpening'.replace(
            '_rln', ''), color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Create the second Y-axis that shares the same X-axis
        ax2 = ax1.twinx()

        # Second Y-axis plot (_rlnFittedInterceptGuinierPlot)
        sns.lineplot(x='_rlnMovieFrameNumber', y='_rlnFittedInterceptGuinierPlot',
                     data=bfactors_data, color='green', ax=ax2)

        # Set labels for the second Y-axis
        ax2.set_ylabel('_rlnFittedInterceptGuinierPlot'.replace(
            '_rln', ''), color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Set the X-axis label
        ax1.set_xlabel('_rlnMovieFrameNumber'.replace('_rln', ''))

        # # Set the figure size and DPI
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # for n, file in enumerate(eps_files):

        #     try:
        #         image = Image.open(file)
        #         image = np.asarray(image)

        #     except Exception as e:
        #         log = initialize_logger(DEBUG)
        #         log.info(f"Are you using Windows without Ghostscript? Could not open EPS file: {file}. Using other library.")
        #         log.info(e)

        #         return [f'Ghostscript installation not found in the path. Could not render EPS file: {file}.']

        #     axes[n].imshow(image)
        #     axes[n].set_title(os.path.basename(file).replace('.eps', ''))
        #     axes[n].axis('off')

        # Adjust the layout
        fig.tight_layout()
        shortcodes.append('### Polish job statistics')
        shortcodes.append(write_plot_get_shortcode_pyplot(
            job_name, HUGO_FOLDER, fig, extra_name=''))
        plt.close()

    return shortcodes


def display_ctf_stats(
        images, file_names,
        columns=2, width=8, height=5,
        label_wrap_length=50, label_font_size=8, label=True):

    if file_names:
        labels = []
        for n, name in enumerate(file_names):
            labels.append(name)

    # Set Seaborn style
    sns.set_style("whitegrid")

    max_images = len(images)
    height = max(height, int(len(images) / columns) * height)

   # Set the figure size
    num_rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=columns, figsize=(width, height))

    for i, image in enumerate(images):
        ax = axes[i // columns, i % columns]
        ax.imshow(adjust_contrast(image, 5, 95), cmap='magma')

        if label:
            title = labels[i]
            title = textwrap.wrap(title, label_wrap_length)
            title = "\n".join(title)
            ax.set_title(title, fontsize=label_font_size)
        ax.axis('off')

    # Hide any empty subplots
    for i in range(len(images), num_rows * columns):
        axes[i // columns, i % columns].axis('off')

    # Adjust the layout
    fig.tight_layout()

    return fig


def plot_import(rln_folder, star_data, HUGO_FOLDER, job_name):
    shortcode = []

    star_data = star_data[1]

    file_names = star_data.get(
        '_rlnMicrographMovieName', star_data.get('_rlnMicrographName', None))

    if file_names is None:
        return ['No Micrographs / Movies found']

    # Import by time
    file_mod_times = []

    for file in file_names:
        file_path = os.path.join(rln_folder, file)
        if os.path.exists(file_path):
            file_mod_times.append(datetime.fromtimestamp(
                os.path.getmtime(file_path)))
        else:
            pass
            #print(f"File not found: {file_path}")

    if not file_mod_times:
        return ['No valid files found']

    fig = go.Figure()

    fig.add_scatter(x=np.arange(len(file_mod_times)),
                    y=file_mod_times, name='Time stamp')

    fig.update_xaxes(title_text="Index")
    fig.update_yaxes(title_text="Time stamp")

    fig.update_layout(title="Imported micrographs timeline")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    name_json = 'import_'
    import_string = write_plot_get_shortcode(
        fig, name_json, job_name, HUGO_FOLDER, fig_height=500)
    shortcode.append(import_string)

    return shortcode


def plot_locres(node_files, FOLDER, HUGO_FOLDER, job_name):
    #    data_folder, jobname, mask_job='':

    # nodes 0 PDF, 1 localresfiltered, 2 localresmap

    shortcodes = []

    #localresfiltered = mrcfile.mmap(os.path.join(FOLDER, node_files[1])).data
    localresmap = mrcfile.mmap(os.path.join(FOLDER, node_files[2])).data
    data_shape = localresmap.shape

    note = get_note(os.path.join(FOLDER,'', job_name, 'note.txt'))

    # get mask
    try:
        mask_path = re.search(
            r"--mask\s([\w\d/]+\.mrc)", note).group(1)
    except:
        mask_path = ''

    if mask_path != '':
        mask = mrcfile.open(os.path.join(FOLDER,'', mask_path)).data.copy()
        mask[mask > 0] = 1
        localresmap = localresmap.copy() * mask

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), dpi=150)

    axes[0].imshow(localresmap[int(data_shape[0] / 2), :, :])
    cbar0 = fig.colorbar(mappable=axes[0].images[0], ax=axes[0])
    cbar0.set_label('Local Resolution, Å')
    axes[0].set_title('Slice Z')
    axes[0].axis('off')

    axes[1].imshow(localresmap[:, int(data_shape[0] / 2)])
    cbar1 = fig.colorbar(mappable=axes[1].images[0], ax=axes[1])
    cbar1.set_label('Local Resolution, Å')
    axes[1].set_title('Slice X')
    axes[1].axis('off')

    axes[2].imshow(localresmap[:, :, int(data_shape[0] / 2)])
    cbar2 = fig.colorbar(mappable=axes[2].images[0], ax=axes[2])
    cbar2.set_label('Local Resolution, Å')
    axes[2].set_title('Slice Y')
    axes[2].axis('off')

    plt.tight_layout()

    shortcodes.append('### Local Resolution Volume slice')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='slices'))
    plt.close()

    # Flatten the localresmap numpy array
    data = localresmap.flatten()
    data = data[data != 0]

    # Create a histogram using Seaborn
    sns.set_style('whitegrid')  # Apply a modern style to the plot
    sns.set(font_scale=1.3)  # Adjust the font scale for larger fonts

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data, kde=False, bins=100, ax=ax)

    # Set axis labels and make them bold
    ax.set_xlabel('Resolution [Å]', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')

    # Set the plot title and make it bold
    ax.set_title('Histogram of local resolution', fontweight='bold')

    shortcodes.append('### Local Resolution Histogram')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='hist'))

    return shortcodes


def plot_postprocess(rln_folder, nodes, HUGO_FOLDER, job_name, plot_relevant=False, dpi=150):

    res_ticks = (1.5,1.6, 1.7, 1.8, 2.0, 2.1, 2.3, 2.5, 3.0, 3.3, 4, 5, 7, 10, 20, 50)
    shortcodes = []

    '''Plot FSC curves from postprocess.star'''

    postprocess_star_path = os.path.join(rln_folder,'', nodes[3])
    if os.path.exists(postprocess_star_path):
        postprocess_star_data = parse_star_whole(postprocess_star_path)
    else:
        return ['No postprocess.star file found']

    res_ticks = np.array(res_ticks)

    fsc_data = postprocess_star_data['fsc']
    guinier_data = postprocess_star_data['guinier']

    fsc_x = fsc_data['_rlnAngstromResolution'].astype(float)
    fsc_x_min, fsc_x_max = np.min(fsc_x), np.max(fsc_x)
    
    if not plot_relevant:
        fsc_to_plot = ['_rlnFourierShellCorrelationCorrected', '_rlnFourierShellCorrelationUnmaskedMaps',
                       '_rlnFourierShellCorrelationMaskedMaps',
                       '_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps']
    else:
        fsc_to_plot = ['_rlnFourierShellCorrelationUnmaskedMaps',
                       '_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps']

    # Set Seaborn style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    for meta in fsc_to_plot:
        ax.plot(1 / fsc_x, fsc_data[meta].astype(float), label=meta.replace('_rln', ''))

    fsc143 = 0.143
    resolution_143_idx = np.argmin(np.abs(fsc_data['_rlnFourierShellCorrelationCorrected'].astype(float) - fsc143))
    
    fsc05 = 0.5
    resolution_05_idx = np.argmin(np.abs(fsc_data['_rlnFourierShellCorrelationCorrected'].astype(float) - fsc05))
    
    array_inv = 1 / res_ticks

    ax.set_xticks(array_inv)
    ax.set_xticklabels(np.around(1 / array_inv, 2))
    ax.axhline(0.5, linestyle='--', c='black')
    ax.axhline(0.143, linestyle='--', c='black')

    ax.set_yticks([0, 0.143, 0.2, 0.4, 0.5, 0.6, 0.8, 1])

    # 1/A
    x_min, x_max = ax.get_xlim()
    ax.annotate('0.143', (x_max, 0.16))
    ax.annotate('0.5', (x_max, 0.52))

    # Use Seaborn to style the legend
    ax.legend(loc='upper right', fontsize=8, frameon=True,
              facecolor='white', edgecolor='black', bbox_to_anchor=(1.0, 1.2))

    # Use a larger font size for axis labels
    ax.set_xlabel('Resolution, Å', fontsize=18)
    ax.set_ylabel('FSC', fontsize=18)

    # Make the ticks larger
    ax.tick_params(axis='both', labelsize=12)

    ax.set_xlim(1 / res_ticks[-1], 1 / np.array(fsc_x)[-1])
    ax.set_ylim(-0.05, 1.05)

    # Remove top and right spines for a cleaner look
    sns.despine()
    
    shortcodes.append(f'### Reported resolution @ FSC=0.143: __{round(fsc_x[resolution_143_idx],2)} Å__')
    shortcodes.append(f'### Reported resolution @ FSC=0.5: __{round(fsc_x[resolution_05_idx],2)} Å__')
    shortcodes.append('### FSC curve')
    
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='FSC'))

    plt.close()

    """Plot Gunier curves from postprocess.star"""

    guiner_x = guinier_data['_rlnResolutionSquared'].astype(float)

    guinier_to_plot = ['_rlnLogAmplitudesOriginal', '_rlnLogAmplitudesMTFCorrected',
                       '_rlnLogAmplitudesWeighted',
                       '_rlnLogAmplitudesSharpened', '_rlnLogAmplitudesIntercept']

    sns.set(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(10, 6))

    for meta in guinier_to_plot:
        try:
            y_data = guinier_data[meta].astype(float)
            y_data[y_data == -99] = float('nan')
            ax.plot(guiner_x, y_data, label=meta.replace('_rln', ''))
        except:
            pass

    ax.set_xlabel("Resolution Squared, [1/Å²]")
    ax.set_ylabel("Ln(Amplitutes)")
    ax.set_title("Guinier plot")
    ax.legend(loc='upper right', fontsize=8, frameon=True,
              facecolor='white', edgecolor='black', bbox_to_anchor=(1.0, 1.2))

    sns.despine()
    plt.tight_layout()

    shortcodes.append('### Gunier plot')
    shortcodes.append(write_plot_get_shortcode_pyplot(
        job_name, HUGO_FOLDER, fig, extra_name='Gunier'))

    """Plot masked postprocess max projection along XY with rotation"""

    volume_path = os.path.join(rln_folder,'', nodes[1])
    volume_data = mrcfile.mmap(volume_path).data

    note = get_note(os.path.join(rln_folder,'', job_name, 'note.txt'))

    # get mask
    try:
        mask_path = re.search(
            r"--mask\s([\w\d/]+\.mrc)", note).group(1)
    except:
        mask_path = ''

    if mask_path != '':
        mask = mrcfile.open(os.path.join(rln_folder,'', mask_path)).data.copy()
        mask[mask > 0] = 1
        volume_data = volume_data.copy() * mask
    
    angles = np.arange(0, 360, 10)
    with ProcessPoolExecutor() as executor:
        indices = list(range(len(angles)))
        results = list(executor.map(save_projection, indices, [volume_data]*len(angles), angles, [HUGO_FOLDER]*len(angles)))

    last_n = max(results)

    # Slices along Z axis are not very useful
    # for n, slice in enumerate(volume_data):
    #     plt.imsave(os.path.join(HUGO_FOLDER, f'{n}.jpg'), slice, cmap='gray')
    #     last_n = n

    js_code = f'''

            <div class="center">
            <p>Masked Volume slices preview:<p>
            <input id="valR" type="range" min="0" max="{last_n}" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
            <span id="range">0</span>
            <img id="img" width="350">
            </div>

            <script>

                var val = document.getElementById("valR").value;
                    document.getElementById("range").innerHTML=val;
                    document.getElementById("img").src = val + ".jpg";
                    function showVal(newVal){{
                      document.getElementById("range").innerHTML=newVal;
                      document.getElementById("img").src = newVal+ ".jpg";
                    }}
            </script>
            <br>
            '''

    js_string = f"{{{{< rawhtml >}}}} {js_code} {{{{< /rawhtml >}}}}"

    shortcodes.append(js_string)

    return shortcodes

def save_projection(index, volume_data, angle, output_folder):
    projection = np.mean(rotate_volume(volume_data, angle, [1, 2]), axis=2)
    plt.imsave(os.path.join(output_folder, f'{index}.jpg'), projection, cmap='gray')
    return index

def plot_selected_classes(node_files, FOLDER, HUGO_FOLDER, job_name, columns_=6, width_=8):
    shortcodes = []
    try:
        # For 2D class selection
        if any('class_averages.star' in element for element in node_files):

            # node files= 0 -> particles, 1 -> class averages (model), 2 -> autoselect?
            cls_averages_meta_path = node_files[1]
            selected_class_star_path = os.path.join(
                FOLDER, cls_averages_meta_path)

            note = get_note(os.path.join(FOLDER, job_name, 'note.txt'))
            try:
                # standard select 2D
                source_job = re.search(
                    r"--i\s([\w\d/]+\.star)", note).group(1).replace('optimiser', 'data')
            except:
                try:
                    # fancy relion autoselect that nobody uses
                    source_job = re.search(
                        r"--opt\s([\w\d/]+\.star)", note).group(1).replace('optimiser', 'data')
                except:
                    return ['']

            num_particles_source = parse_star_whole(os.path.join(FOLDER, source_job))[
                'particles'].shape[0]
            try:
                num_particles_selected = parse_star_whole(os.path.join(FOLDER, node_files[0]))[
                    'particles'].shape[0]
            except Exception as e:
                # no particles selected
                num_particles_selected = 0

            shortcodes.append(
                f'Selected __{num_particles_selected}__ (__{round(num_particles_selected/num_particles_source*100, 2)}%__) particles from __{num_particles_source}__ particles in {source_job}')

            # select star has no name for first loop
            try:
                cls_averages_data = parse_star_whole(
                    selected_class_star_path)['#']

            except Exception as e:
                # all classes taken from 2D classification
                return ['### All classes selected']

            # Get selected classes
            cls_images = []
            for n, cls_ in enumerate(cls_averages_data['_rlnReferenceImage']):
                cls_data = cls_.split('@')
                cls_images.append(mrcfile.mmap(os.path.join(FOLDER,'', cls_data[1])).data[int(cls_data[0]) - 1])
                last_n = n

            shortcodes.append('### Selected classes')
            shortcodes.append(display_2dclasses(
                cls_images, HUGO_FOLDER, job_name, model_star=selected_class_star_path))

            shortcodes.append('### Selected classes preview')
            shortcodes.append(generate_slider_html(last_n))

        # for 3D classification
        elif any('particles.star' in element for element in node_files) and len(node_files) == 1:

            note = get_note(os.path.join(FOLDER, job_name, 'note.txt'))
            try:
                # standard select 3D
                source_job = re.search(
                    r"--i\s([\w\d/]+\.star)", note).group(1).replace('optimiser', 'data')

            except:
                return ['']

            selected_star = parse_star_whole(
                os.path.join(FOLDER, node_files[0]))['particles']
            num_particles_source = parse_star_whole(os.path.join(FOLDER, source_job))[
                'particles'].shape[0]
            num_particles_selected = selected_star.shape[0]

            shortcodes.append(
                f'Selected __{num_particles_selected}__ (__{round(num_particles_selected/num_particles_source*100, 2)}%__) particles from __{num_particles_source}__ particles in {source_job}')

            selected_classes = np.unique(
                selected_star['_rlnClassNumber']).astype(int)

            source_job_folder = source_job.split(PATH_CHARACTER)
            model_files = glob.glob(os.path.join(
                FOLDER, source_job_folder[0], source_job_folder[1], "*model.star"))
            model_files.sort(key=os.path.getmtime)
            last_model_star_path = model_files[-1]

            parent_star = parse_star_whole(os.path.join(
                FOLDER, last_model_star_path))['model_classes']
            mrcs_paths = parent_star['_rlnReferenceImage']
            class_dist = parent_star['_rlnClassDistribution']

            selected_rows = mrcs_paths.str.contains(
                "|".join(f"class{class_num:03d}" for class_num in selected_classes))
            filtered_df = mrcs_paths[selected_rows]
            class_dist = class_dist[selected_rows].tolist()

            class_paths = []
            for n, class_name in enumerate(filtered_df.values):
                class_name = os.path.join(FOLDER, class_name)

                # Insert only new classes, in 2D only single file
                if class_name not in class_paths:
                    class_paths.append(class_name)

            shortcode = plot_projections(
                FOLDER, HUGO_FOLDER, job_name, class_paths, class_dist)
            shortcodes.append(shortcode[0])
            shortcodes.append(shortcode[1])

        else:
            shortcodes.append('### Select job generated:')
            for f in node_files:
                shortcodes.append(f'* {f}')

        return shortcodes
    except Exception as e:
        return ['Could not read star file']


def process_image(n, slice, tomo_idx, hugo_job_path):
    if slice.shape[1] > 500:
        slice = rescale(slice.astype(float), float(500 / slice.shape[1]))

    p1, p2 = np.percentile(slice, (1, 99))
    slice = np.array(exposure.rescale_intensity(slice, in_range=(p1, p2)))

    slice = skimage.filters.gaussian(slice, sigma=1)
    plt.imsave(os.path.join(hugo_job_path,
               f'{tomo_idx}_{n}.jpg'), slice, cmap='gray')


def plot_import_tomo(node_files, rlnFOLDER, hugo_job_path, job_name):
    shortcodes = []

    if any('tomograms.star' in element for element in node_files):
        tomograms_star_path = node_files[0]
        tomogram_star = parse_star_whole(os.path.join(
            rlnFOLDER, tomograms_star_path))['global']
        tomo_name = tomogram_star['_rlnTomoName']
        shortcodes.append(f'### Imported tomo files:')

        for file in tomo_name:
            shortcodes.append(f'* {file}')

        # project tomo
        tilt_series = tomogram_star['_rlnTomoTiltSeriesName']

        # this is a killer to open and write tilt series. However, ones computer can be powerful enough to handle multiple tilt series
        # change here manually number of tilt series to be posessed
        TILT_SERIES_TO_PROCESS = 3 if len(
            tilt_series) > 2 else len(tilt_series)
        selected_tilt_series = np.random.choice(
            tilt_series, TILT_SERIES_TO_PROCESS)

        for tomo_idx, single_tilt_series in enumerate(selected_tilt_series):
            try:
                volume_data = mrcfile.mmap(os.path.join(
                    rlnFOLDER, single_tilt_series)).data
            except Exception as e:
                log = initialize_logger(DEBUG)
                log.info(f'Could not read tomogram file {single_tilt_series}')
                return ['Could not read tomogram file. Are they in the right folder? (Only internal Relion folders are supported)']

            # Get half of the maximum available CPU count (rounded up if odd)
            half_cpu_count = (int(os.cpu_count()) + 1) // 2
            with ThreadPoolExecutor(max_workers=half_cpu_count) as executor:
                futures = []
                for n, slice in enumerate(volume_data):
                    future = executor.submit(
                        process_image, n, slice, tomo_idx, hugo_job_path)
                    futures.append(future)

                # Wait for all images to be processed
                for future in futures:
                    future.result()

            last_n = len(volume_data) - 1

            js_code = f'''

            <div class="center">
            <p>Tilt series preview:<p>
            <input id="valR_{tomo_idx}" type="range" min="0" max="{last_n}" value="0" step="1" oninput="showVal_{tomo_idx}(this.value)" onchange="showVal_{tomo_idx}(this.value)" />
            <span id="range_{tomo_idx}">0</span>
            <img id="img_{tomo_idx}" width="350">
            </div>

            <script>

                function showVal_{tomo_idx}(newVal){{
                    document.getElementById("range_{tomo_idx}").innerHTML=newVal;
                    document.getElementById("img_{tomo_idx}").src = "{tomo_idx}_"+ newVal + ".jpg";
                }}

                // Initialize the slider and image on page load
                var initial_val_{tomo_idx} = document.getElementById("valR_{tomo_idx}").value;
                showVal_{tomo_idx}(initial_val_{tomo_idx});

            </script>
            <br>
            '''

            js_string = f"{{{{< rawhtml >}}}} {js_code} {{{{< /rawhtml >}}}}"

            shortcodes.append(
                f'### Projected tomogram {single_tilt_series} preview')
            shortcodes.append(js_string)

    elif any('particles.star' in element for element in node_files):
        particles_data = parse_star_whole(
            os.path.join(rlnFOLDER, node_files[0]))['particles']

        tomo_names = np.unique(particles_data['_rlnTomoName'])

        if tomo_names.shape[0] > 3:
            selected_tomo = np.random.choice(tomo_names, 3)

        else:
            selected_tomo = np.random.choice(tomo_names)

        shortcodes.append(f'### Particles location preview:')

        for name in selected_tomo:

            sigle_tomo = particles_data[particles_data['_rlnTomoName'] == name].drop(
                '_rlnTomoName', axis=1)
            sigle_tomo = sigle_tomo.apply(
                lambda x: pd.to_numeric(x, errors='ignore'))

            fig = px.scatter_3d(sigle_tomo, x='_rlnCoordinateX',
                                y='_rlnCoordinateY', z='_rlnCoordinateZ')

            fig.update_scenes(aspectmode='data')
            fig.update_traces(marker_size=2)

            shortcodes.append(f'##### Preview of {name}')
            shortcodes.append(write_plot_get_shortcode(
                fig, f'{name}', job_name, hugo_job_path, fig_height=PLOT_HEIGHT))

    return shortcodes


def plot_reconstruct_tomo(node_files, rlnFOLDER, hugo_job_path, job_name):
    shortcodes = []

    if any('merged.mrc' in element for element in node_files):
        merged_mrc_path = node_files[0]

        #volume_data = mrcfile.mmap(os.path.join(rlnFOLDER, merged_mrc_path)).data
        shortcodes.append(f'### Reconstructed particles:')
        shortcodes.append(plot_3d_model([os.path.join(rlnFOLDER, merged_mrc_path)],
                          job_name, hugo_job_path, shorcode_txt='reconstructed_tomo', fig_height=600))

        projections = plot_projections('', hugo_job_path, job_name, [
                                       os.path.join(rlnFOLDER, merged_mrc_path)], class_dist_=None)
        shortcodes.append(f'### Reconstructed particles projection:')
        shortcodes.append(projections[1])

    return shortcodes


def plot_ctfrefine_tomo(node_files, rlnFOLDER, hugo_job_path, job_name):

    return ['Job preview not supported yet..']


def plot_frame_align(node_files, rlnFOLDER, hugo_job_path, job_name):

    return ['Job preview not supported yet..']


def plot_pseudosubtomo(node_files, rlnFOLDER, hugo_job_path, job_name):
    shortcodes = []

    if any('particles.star' in element for element in node_files):
        particles_path = os.path.join(rlnFOLDER, node_files[1])
        star = parse_star_whole(particles_path)['particles']
        shortcodes.append(
            f'### Generated Pseudosubtomo particles: {star.shape[0]}')

        selection = np.random.randint(0, star['_rlnImageName'].shape[0], 10)
        pseudotomo_data_names = star['_rlnImageName'][selection]
        pseudotomo_ctf_names = star['_rlnCtfImage'][selection]

        data_paths = [os.path.join(rlnFOLDER, name)
                      for name in pseudotomo_data_names]
        ctf_paths = [os.path.join(rlnFOLDER, name)
                     for name in pseudotomo_ctf_names]

        projections_data = plot_projections('', hugo_job_path, job_name, data_paths,
                                            class_dist_=None, string='Particle', cmap='gray', extra_name='particles')
        shortcodes.append(f'### Generated Pseudosubtomo projections:')
        shortcodes.append(projections_data[1])

        projections_ctf = plot_projections(
            '', hugo_job_path, job_name, ctf_paths, class_dist_=None, string='Particle', cmap='viridis', extra_name='ctf')
        shortcodes.append(f'### Generated Pseudosubtomo CTF projections:')
        shortcodes.append(projections_ctf[1])

        return shortcodes
