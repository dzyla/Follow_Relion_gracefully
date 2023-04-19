# Standard library imports
import os
import sys
import time
import argparse
from datetime import datetime

# Parallel and asynchronous processing
from joblib import Parallel, delayed

# Import custom library functions
from follow_relion_gracefully_lib import *


# Change the folder set if Hugo working on windows / linux. Not sure if it even matters.
PATH_CHARACTER = os.path.sep
HOSTNAME = '/'
FOLDER = ''
HUGO_FOLDER = 'content/jobs/'
FORCE_PROCESS = False
DEFAULT_N_CPUs = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description='Follow Relion Gracefully: web-based job GUI')
    parser.add_argument('--i', nargs='+', type=str,
                        help='One or more Relion folder paths. Required!', required=True)
    parser.add_argument('--h', type=str, default='/',
                        help='Hostname for HUGO website. For remotly hosted Hugo server (for example another workstation or Github). Use IP adress of the remote machine but make sure the ports are open. For local hosting leave it default (localhost)')
    parser.add_argument('--n', type=int, default=DEFAULT_N_CPUs,
                        help='Number of CPUs for processing. Use less (2-4) if you have less than 16 GB of RAM')
    parser.add_argument('--t', type=int, default=30,
                        help='Wait time between folder checks for changes used for continuous updates.')
    parser.add_argument('--single', action='store_true',
                        help='Single run, do not do continuous updates. Useful for the first-time usage')
    parser.add_argument('--new', action='store_true', default=False,
                        help='Start new Follow Relion Gracefully project and remove previous job previews. Removes whole /content/ folder! Use after downloading from Github or when acually want to start a new project')
    parser.add_argument('--download_hugo', action='store_true', default=False,
                        help='Download HUGO executable if not present. This is operating system specific')
    parser.add_argument('--server', action='store_true', default=False,
                        help='Automatically start HUGO server')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Change log level to debug. Helpful for checking if something goes wrong')
    args = parser.parse_args()

    return args


def process_job_if_updated(job_name, rln_path):

    # Initialize logger with debug argument to log messages in the specified level of detail
    log = initialize_logger(DEBUG)

    # Define the path for the Hugo server job
    hugo_job_path = os.path.join('content', os.path.basename(os.path.normpath(
        rln_path)), 'jobs', job_name.split('/')[1].replace('/', PATH_CHARACTER))
    log.debug(f'Hugo job path : {hugo_job_path}')  # Log the Hugo job path

    # Create a folder if not exists and set permissions
    os.makedirs(hugo_job_path, mode=0o777, exist_ok=True)
    # Log the creation of directory
    log.debug(f'Hugo job path folder created: {hugo_job_path}')

    # Get the last modified file from the folder and its modification time
    last_file, modTimesinceEpoc, _ = FolderMonitor(
        os.path.join(rln_path, job_name)).get_last_modified_file()
    time_file = time.strftime('%Y%m%d_%H%M%S', time.localtime(
        modTimesinceEpoc)) + '.time'  # Define the time file name
    log.debug(f'last_file: {last_file}')  # Log the last modified file
    log.debug(f'time_file name : {time_file}')  # Log the time file name

    # Convert the modification timestamp to date and time format
    modificationDate = time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
    modificationTime = time.strftime(
        '%H:%M:%S', time.localtime(modTimesinceEpoc))
    # Log the modification date and time
    log.debug(f'modificationDate: {modificationDate, modificationTime}')

    # Get the old time file from the folder and check if it exists
    time_old_path = os.path.join(os.getcwd(), hugo_job_path, '*.time')
    log.debug(f"time_old path: {time_old_path}")  # Log the old time file path

    time_old = glob.glob(time_old_path)
    log.debug(f'time_old glob: {time_old}')  # Log the old time file

    if not time_old:
        time_old = None  # Set old time file to None if not present
    else:
        time_old = time_old[0]  # Get the first element of the list
        time_old = os.path.basename(time_old)  # Get the filename

    # Log the old time file if exists
    log.debug(f'time_old if exists: {time_old}')

    # Check if the files are different, remove old one if present and write the new one. Set process_job True or false
    # if the same file found
    process_job = True

    if time_old != time_file:
        if time_old is not None:  # Check if the old time file is present
            try:
                # Remove the old time file
                os.remove(os.path.join(hugo_job_path, time_old))
            except PermissionError as e:  # Catch and log any permission errors while removing file
                log.info(f'Error in removing time file : {e}')
                pass

        process_job = True  # Set process_job to true

        # Write the time file
        open(os.path.join(hugo_job_path, time_file), 'w').close()
        log.debug(f'Written time file : {time_file}')

    elif time_old == time_file:  # Check if old time file and new time file are same
        process_job = False  # Set process_job to false
        # Log if the time files are same
        log.debug(f'Time file same: {time_old == time_file}')

    return process_job, modificationDate, modificationTime, time_file, hugo_job_path


def generate_per_job(job_name, rlnFOLDER, node_files, hugo_job_path):

    # Get job type from job name
    log = initialize_logger(DEBUG)

    job_type = job_name.split('/')[0]
    log.debug(f'job_type : {job_type}')

    # Log folder paths and node files
    log.debug(f'HUGO_FOLDER : {hugo_job_path}')
    log.debug(f'FOLDER : {rlnFOLDER}')
    log.debug(f'node_files : {" ".join(node_files)}')

    # Create dictionary for different job types and their functions
    job_functions = {
        'Class2D': lambda: plot_2d_pyplot(rlnFOLDER, hugo_job_path, job_name),
        'Extract': lambda: process_extract(node_files, rlnFOLDER, hugo_job_path, job_name),
        'CtfFind': lambda: plot_ctf_stats(parse_star(os.path.join(rlnFOLDER,'', node_files[0])), hugo_job_path, job_name),
        'MotionCorr': lambda: plot_motioncorr_stats(parse_star(os.path.join(rlnFOLDER,'', node_files[0])), hugo_job_path, job_name),
        'Import': lambda: plot_import(rlnFOLDER, parse_star(os.path.join(rlnFOLDER,'', node_files[0])), hugo_job_path, job_name),
        'Class3D': lambda: plot_cls3d_stats(rlnFOLDER, hugo_job_path, job_name),
        'InitialModel': lambda: plot_cls3d_stats(rlnFOLDER, hugo_job_path, job_name),
        'MaskCreate': lambda: plot_mask(os.path.join(rlnFOLDER,'', node_files[0]), hugo_job_path, job_name),
        'Refine3D': lambda: plot_refine3d(rlnFOLDER, hugo_job_path, job_name),
        'AutoPick': lambda: plot_picks(rlnFOLDER, hugo_job_path, job_name),
        'ManualPick': lambda: plot_picks(rlnFOLDER, hugo_job_path, job_name),
        'CtfRefine': lambda: plot_ctf_refine(node_files, rlnFOLDER, hugo_job_path, job_name),
        'Polish': lambda: plot_polish(node_files, rlnFOLDER, hugo_job_path, job_name),
        'LocalRes': lambda: plot_locres(node_files, rlnFOLDER, hugo_job_path, job_name),
        'PostProcess': lambda: plot_postprocess(rlnFOLDER, node_files, hugo_job_path, job_name),
        'Select': lambda: plot_selected_classes(node_files, rlnFOLDER, hugo_job_path, job_name),
        'ImportTomo': lambda: plot_import_tomo(node_files, rlnFOLDER, hugo_job_path, job_name),
        'ReconstructParticleTomo': lambda: plot_reconstruct_tomo(node_files, rlnFOLDER, hugo_job_path, job_name),
        'CtfRefineTomo': lambda: plot_ctfrefine_tomo(node_files, rlnFOLDER, hugo_job_path, job_name),
        'FrameAlignTomo': lambda: plot_frame_align(node_files, rlnFOLDER, hugo_job_path, job_name),
        'PseudoSubtomo': lambda: plot_pseudosubtomo(node_files, rlnFOLDER, hugo_job_path, job_name)
    }

    # If job type is present in job functions, return log of its function
    if job_type in job_functions:
        return '\n'.join(job_functions[job_type]())
    else:
        return ''


def run_job(quequing_elem_):
    # Initialize a logger to track the program progress
    time_start = time.time()
    log = initialize_logger(DEBUG)

    # Get job name, number of nodes and path for the project
    job_name, n, rln_project_path, process_nodes, process_status, process_alias = quequing_elem_
    log.debug(
        f'job_name, n, rln_project_path: {job_name, n, rln_project_path}')

    # Flag to identify process failure
    failed = False

    # Get list of files associated with the job
    node_files = []
    for node in process_nodes:
        if job_name in node:
            node_files.append(node)

    # Check if the job needs processing
    process_job, modificationDate, modificationTime, time_file, hugo_job_path = process_job_if_updated(
        job_name, rln_project_path)
    log.debug(f'process_job: {process_job}')

    # Generate shortcode string for job when processing is required
    if process_job or FORCE_PROCESS:
        shortcode = ''

        log.info(f'job_name: {job_name}')

        # Get path for job data
        path_data = os.path.join(rln_project_path,'', job_name)
        log.debug(f'path_data : {path_data}')

        # Get note associated with the job
        note = get_note(os.path.join(path_data,'', 'note.txt'))
        log.debug(f'note: {note}')

        # Run job processing
        try:
            shortcode = generate_per_job(
                job_name, rln_project_path, node_files, hugo_job_path)
            log.debug(f'plotly_string: {shortcode}')

        # Handle processing failures by writing error to file and removing time tag for rerun
        except Exception as e:
            log.exception(f'Error with job {job_name}: {e}')
            plotly_string = f'Something went wrong: \n\n  {e}'
            print(f'Something went wrong with {job_name}')
            try:
                os.remove(os.path.join(hugo_job_path, time_file))
            except Exception as e:
                log.debug(e)
            failed = True

        project_name = os.path.basename(os.path.normpath(rln_project_path))
        # Write MD file
        with open(os.path.join(hugo_job_path, 'index.md'), mode='w') as file:

            print(f"""\
---
title: {job_name.split('/')[1]}
jobname: {job_name}
status: {process_status}
date: {modificationDate}
time: {modificationTime}
jobtype: [{job_name.split('/')[0]}]
project: [{project_name}]
---

#### Job alias: {process_alias}

{shortcode}

#### Job command(s):

```bash
{note}
""", file=file)

        # Print job name on successful completion
        if process_job or FORCE_PROCESS:
            log.info(
                f'Job {job_name} took {round(time.time() - time_start, 3)} seconds.')
            if not failed:
                print(job_name)

        log.info(f'-------------------------------------------------------------------')
        return 1


if __name__ == "__main__":

    # The code below checks if the server is running and is a first run, and then parses the arguments
    SERVER_RUNNING = False
    FIRST_RUN = True

    time_beginning = time.time()  # Record the start time of the program
    args = parse_args()  # Parse command line arguments
    
    # Assign parameters from command line arguments to variables
    N_CPUs = int(args.n)
    FOLDER = args.i
    WAIT = int(args.t)
    DEBUG = args.debug

    set_debug(DEBUG)
    
    log = initialize_logger(DEBUG)  # Initialize the logger
    log.info(
        '''--------------NEW RUN--------------''')  # Log that a new run has started
    log.info(
        f"Command line arguments: {args}")  # Log command line arguments used


    # Check if the server is running, if the user specified to download Hugo, if a
    # new project needs to be created, and determine the folder watcher variables
    if args.download_hugo:
        download_hugo()
        log.info('HUGO executable downloaded')

    if args.new:
        new_project()
        FORCE_PROCESS = True
        log.info('Hugo content folder removed')

    # Print number of CPUs being used
    print('Using {} CPUs for processing!'.format(N_CPUs))
    log.info(f'FOLDER: {FOLDER}')  # Log folder information

    # Create folder watchers and log the latest modified file and its timestamp
    with ThreadPoolExecutor() as executor:
        folder_watchers = list(executor.map(monitor_folder, range(len(FOLDER)), FOLDER, [log]*len(FOLDER)))

    # Generate content/projects.txt file based on FOLDER list. Add to existing project
    with open('resources/project_paths.txt', mode='a') as file:
        for path in FOLDER:
            file.write(f'{path}\n')
    log.info(f'Wrote resources/project_paths.txt')

    # Change hostname in the config.toml file and log the total elapsed time
    write_config(hostname=args.h)
    log.info(
        f'Written config. Total time of initializing: {round(time.time() - time_beginning, 2)} seconds')

    try:
        # main loop
        while True:
            total_time1 = time.time()

            # Load pipeline star and check processes
            # Get each folder from FOLDER list one by one
            for idx, project_folder in enumerate(FOLDER):
                # Record the current timestamp to calculate loop execution time
                loop_start = time.time()
                print(f'Processing folder {idx} {project_folder}')
                
                log.debug(f'idx, project_folder: {idx, project_folder}')

                # Only process files when the folder changes
                # Check if folder content has changed or its first run
                if folder_watchers[idx].check_changes() or FIRST_RUN:
                    log.debug(
                        f'folder_watcher.file_modified: {folder_watchers[idx].latest_modified_file}')

                    # load pipeline star from current directory
                    # Create path to pipeline star file
                    pipeline = os.path.join(FOLDER[idx],'', 'default_pipeline.star')
                    log.info(f'pipeline: {pipeline}')

                    # try to open pipeline star
                    try:
                        pipeline_star = parse_star_whole(
                            pipeline)  # Parse pipeline star file
                        # Get process names from parsed file
                        process_name = pipeline_star['pipeline_processes']['_rlnPipeLineProcessName']
                        log.debug(f'process_name: {process_name.shape}')

                    except Exception as e:  # Handle exceptions
                        print(
                            f'default_pipeline.star does not exist in {project_folder}. Skipping...')
                        log.info(e)
                        continue

                    try:
                        # Relion 4.0
                        # Get process status label from parsed file
                        process_status = pipeline_star['pipeline_processes']['_rlnPipeLineProcessStatusLabel']
                        log.debug(
                            f'process_status Relion 4.+: , {process_status.shape}')

                    except KeyError:
                        # Relion 3.1
                        # Get process status from parsed file
                        process_status = pipeline_star['pipeline_processes']['_rlnPipeLineProcessStatus']
                        log.debug(
                            f'process_status Relion 3: , {process_status.shape}')

                    # Those are globally defined or something. Still works tho.
                    # Get process nodes from parsed file
                    process_nodes = pipeline_star['pipeline_nodes']['_rlnPipeLineNodeName']
                    log.debug(f'process_nodes:, {process_nodes.shape}')
                    # Get process alias from parsed file
                    process_alias = pipeline_star['pipeline_processes']['_rlnPipeLineProcessAlias']
                    log.debug(f'process_alias: , {process_alias.shape}')

                    data_queue = [(process_name[i], idx, FOLDER[idx], process_nodes,
                                   process_status[i], process_alias[i]) for i in range(0, len(process_name))]  # Create data queue

                    # Execute in parallel using N_CPUs
                    # Run jobs stored in data queue using N_CPUs in parallel
                    results = Parallel(n_jobs=N_CPUs)(delayed(run_job)(
                        queue_element) for queue_element in data_queue)
                    log.info(
                        f'Loop {project_folder} took {round(time.time() - loop_start, 3)} seconds')

            ### Server running check ###

            # check Hugo server running
            # When too many write requests are received Hugo likes to crash.
            if args.server:  # If Hugo server flag is set
                if FIRST_RUN:
                    # kill server
                    check_kill_process_by_name('hugo')

                # Check if server is running
                SERVER_RUNNING = check_kill_process_by_name(
                    'hugo', report=True)  # Check if Hugo server is running
                log.debug(f'Checking server status. Running: {SERVER_RUNNING}')

            # Generate projects.txt file with paths corresponding to project folders in content folder
            find_project_paths()

            # Run the Hugo server if requested
            if args.server and not SERVER_RUNNING:  # If Hugo server flag is set and it's not running
                if args.h == '/':  # If argument h equals /
                    host = 'localhost'  # Set host to localhost
                else:
                    host = args.h  # Else set host as argument h

                # Hugo server command
                hugo_folder = 'Hugo'
                hugo_executable = 'hugo'

                if sys.platform == "win32":
                    hugo_folder = 'Hugo'
                    hugo_executable = 'hugo.exe'

                # Create Hugo server command
                hugo_command = f'{os.path.join(hugo_folder, hugo_executable)} server -D -E -F --disableLiveReload --bind {host} --baseURL http://{host}/'
                log.info(f'hugo_command: {hugo_command}')

                # Download Hugo if not present
                if not os.path.exists(os.path.join(hugo_folder, hugo_executable)):
                    download_hugo()  # Download Hugo if it's not present
                    log.info(f'Downloading Hugo')

                # Run Hugo server
                hugo_server = start_detached_Hugo_server(
                    hugo_command)  # Start detached Hugo server
                SERVER_RUNNING = True  # Set SERVER_RUNNING flag to true
                log.info(f'SERVER_RUNNING : {SERVER_RUNNING}')


            if args.single:
                print(f'Single run complete. Exiting...')
                quit()  # Quit process

            time.sleep(WAIT)  # Wait for WAIT seconds for next iteration
            FIRST_RUN = False  # Set FIRST_RUN flag to false
        
            log.info(f'Total time for all folders: {time.time() - total_time1}')
        
    except KeyboardInterrupt:  # Handle keyboard interrupt exception
        print("Exiting...")
        log.info(f'Exiting by keyboard interrupt')

        try:
            check_kill_process_by_name('hugo')  # Kill Hugo server

        except Exception as e:  # Handle exception
            log.info(f'e : {e}')