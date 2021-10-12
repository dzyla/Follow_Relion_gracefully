from joblib import Parallel, delayed
from follow_relion_gracefully_lib import *
import multiprocessing
import time
import argparse

'''
Written by Dawid Zyla, LJI under Non-Profit Open Software License 3.0 (NPOSL-3.0).
Github: https://github.com/dzyla/Follow_Relion_gracefully/
v3
'''

parser = argparse.ArgumentParser(
        description='Follow Relion Gracefully: web based job GUI')
parser.add_argument('--i', type=str, help='Relion folder path')
parser.add_argument('--h', type=str, default='/',
                        help='Hostname for HUGO website. Only for hosted website')
parser.add_argument('--o', type=str, default='content/jobs/',
                        help='Output directory for HUGO website.')
parser.add_argument('--force', action='store_true',
                        help='Force redo all folders')
parser.add_argument('--n', type=int, default=0,
                        help='Number of CPUs for processing. Use less (1-2) if RAM is the issue')
parser.add_argument('--debug', action='store_true',
                        help='Debug')

args = parser.parse_args()


# Change the folder set if Hugo working on windows / linux.
PATH_CHARACTER = '/' if os.name != 'nt' else '\\'

HOSTNAME= args.h if not args.debug else 'https://dzyla.github.io/Follow_Relion_gracefully/'
FOLDER = args.i if not args.debug else '/mnt/staging/Dawid/test/relion40_tutorial_precalculated_results/'
HUGO_FOLDER = args.o if not args.debug else 'content/jobs/'
N_CPUs = args.n

# Force process of all folders, even if the same. Good for development.
FORCE_PROCESS = args.force

if __name__ == "__main__":
    # Check how many processors available
    if N_CPUs == 0:
        N_CPUs = multiprocessing.cpu_count()

    print('Using {} CPUs for processing!'.format(N_CPUs))

    # Load pipeline star and check processes
    pipeline = FOLDER + 'default_pipeline.star'

    # Change hostname in the config.toml file
    config_file = open('config.toml', 'r')
    config_new = []
    save_new_config = False
    config_ = config_file.readlines()
    config_file.close()
    for line in config_:
        if 'baseurl' in line:
            if line != 'baseurl = "{}"\n'.format(HOSTNAME):
                line = 'baseurl = "{}"\n'.format(HOSTNAME)
                save_new_config = True

        config_new.append(line)

    if save_new_config:
        config_new_file = open('config.toml', 'w')
        for line in config_new:
            print(line, file=config_new_file, end='')


    # Define the function for parallel execution
    def run_job(quequing_elem_):
        job_name, n = quequing_elem_

        # Define the path for each job for Hugo server
        path = HUGO_FOLDER + job_name.split('/')[1].replace('/', PATH_CHARACTER) + PATH_CHARACTER
        os.makedirs(path, mode=0o777, exist_ok=True)

        # Get the last file time:
        folder_files = glob.glob(FOLDER + job_name + '/*')

        folder_files.sort(key=os.path.getmtime)

        try:
            last_file = folder_files[-1]
        except IndexError:
            return

        modTimesinceEpoc = os.path.getmtime(last_file)
        modificationDate = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
        modificationTime = time.strftime('%H:%M:%S', time.localtime(modTimesinceEpoc))

        # Check time stamps. Continue if the new file is in the folder
        time_file = time.strftime('%Y%m%d_%H%M%S', time.localtime(modTimesinceEpoc)) + '.time'

        # Get the old file from the folder, if not present just set None
        try:
            time_old = glob.glob(path + '*.time')

            if time_old == []:
                time_old = None
            else:
                time_old = time_old[0]
                time_old = os.path.basename(time_old)


        except FileNotFoundError or IndexError:
            time_old = None

        # Check if the files are different, remove old one if present and write the new one. Set process_job True or false
        # if the same file found
        process_job = True

        if time_old != time_file:
            if time_old != None:
                try:
                    os.remove(path+time_old)
                except PermissionError as e:
                    print(e)
                    pass

            process_job = True

        elif time_old == time_file:
            process_job = False

        # Get job files:
        node_files = []
        for node in process_nodes:
            if job_name in node:
                node_files.append(node)

        # Plot accordingly
        if process_job or FORCE_PROCESS:
            plotly_string = ''
            print(job_name)

            path_data = FOLDER + job_name
            note = get_note(path_data + 'note.txt')

            try:
                if job_name.split('/')[0] == 'Class2D':

                    plotly_string = plot_2d_ply(path_data, HUGO_FOLDER, job_name)
                    plotly_string = '\n'.join(plotly_string)


                elif job_name.split('/')[0] == 'Extract':

                    # For Relion 3 only single output from Extract
                    if len(node_files) == 1:
                        if os.path.exists(FOLDER + node_files[0]):
                            star_path = FOLDER + node_files[0]
                            rnd_particles = show_random_particles(star_path, FOLDER, random_size=100, r=1, adj_contrast=False)

                            plotly_string = plot_extract_js(rnd_particles, HUGO_FOLDER, job_name)

                    # Relion 4 has more nodes?
                    elif len(node_files) > 1:

                        #File 1 is the particles star?
                        if os.path.exists(FOLDER + node_files[1]):
                            star_path = FOLDER + node_files[1]
                            rnd_particles = show_random_particles(star_path, FOLDER, random_size=100, r=1, adj_contrast=False)

                            plotly_string = plot_extract_js(rnd_particles, HUGO_FOLDER, job_name)


                elif job_name.split('/')[0] == 'CtfFind':

                    if os.path.exists(FOLDER + node_files[0]):
                        star_path = parse_star(FOLDER + node_files[0])

                        plotly_string = plot_ctf_stats(star_path, HUGO_FOLDER, job_name)
                        plotly_string = '\n'.join(plotly_string)


                elif job_name.split('/')[0] == 'MotionCorr':

                    if os.path.exists(FOLDER + node_files[0]):
                        star_path = parse_star(FOLDER + node_files[0])
                        plotly_string = plot_motioncorr_stats(star_path, HUGO_FOLDER, job_name)
                        plotly_string = '\n'.join(plotly_string)


                elif job_name.split('/')[0] == 'Import':
                    if os.path.exists(FOLDER + node_files[0]):
                        star_path = parse_star(FOLDER + node_files[0])
                        plotly_string = plot_import(FOLDER, star_path, HUGO_FOLDER, job_name)
                        plotly_string = '\n'.join(plotly_string)


                elif job_name.split('/')[0] == 'Class3D' or job_name.split('/')[0] == 'InitialModel':

                    plotly_string = plot_cls3d_stats(path_data, HUGO_FOLDER, job_name)
                    plotly_string = '\n'.join(plotly_string)

                elif job_name.split('/')[0] == 'MaskCreate':

                    if os.path.exists(FOLDER + node_files[0]):
                        mask_path = FOLDER + node_files[0]
                        plotly_string = plot_mask(mask_path, HUGO_FOLDER, job_name)
                        plotly_string = '\n'.join(plotly_string)

                elif job_name.split('/')[0] == 'Refine3D':

                    plotly_string = plot_refine3d(path_data, HUGO_FOLDER, job_name)
                    plotly_string = '\n'.join(plotly_string)

                elif job_name.split('/')[0] == 'AutoPick' or job_name.split('/')[0] == 'ManualPick':
                    plotly_string = plot_picks_plotly(FOLDER, path_data, HUGO_FOLDER, job_name)


                elif job_name.split('/')[0] == 'CtfRefine' or job_name.split('/')[0] == 'Polish':
                    plotly_string = plot_ctf_refine(path_data, HUGO_FOLDER, job_name)
                    plotly_string = '\n'.join(plotly_string)

            except:
                plotly_string = 'Something went wrong'
                print('Something went wrong with {}'.format(job_name))


            # write the MD file
            file = open(path + 'index.md', mode='w')


            print('''
---
title: {}
jobname: {}
status: {}
date: {}
time: {}
categories: [{}]
---

#### Job alias: {}

{}

#### Job command(s):

{}
'''.format(job_name.split('/')[1],
                   job_name,
                   process_status[n],
                   modificationDate,
                   modificationTime,
                   job_name.split('/')[0],
                   str(process_alias[n]),  # '\n\n'.join(node_files)
                   plotly_string,
                   note
                   ), file=file)

        # Write the time stamp if the job completed properly
        time_file = open(path + time_file, 'w')



    if os.path.exists(pipeline):
        pipeline_star = parse_star_whole(pipeline)
        process_name = pipeline_star['pipeline_processes']['_rlnPipeLineProcessName']
    else:
        print('\ndefault_pipeline.star does not exist or the folder path is incorrect. Aborting..\n')
        quit()

    try:
        # Relion 4.0
        process_status = pipeline_star['pipeline_processes']['_rlnPipeLineProcessStatusLabel']

    except KeyError:
        # Relion 3.1
        process_status = pipeline_star['pipeline_processes']['_rlnPipeLineProcessStatus']

    process_nodes = pipeline_star['pipeline_nodes']['_rlnPipeLineNodeName']
    process_alias = pipeline_star['pipeline_processes']['_rlnPipeLineProcessAlias']


    # Define the data queue for parallel execution
    data_queue = np.stack([process_name, np.arange(0, len(process_name))], axis=1)

    # Execute in parallel using N_CPUs
    results = Parallel(n_jobs=N_CPUs)(delayed(run_job)(queue_element) for queue_element in data_queue)