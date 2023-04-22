
# Follow Relion Gracefully :microscope::rocket::globe_with_meridians: 

**v4: Show them all! & partial :sparkles: `#teamtomo` :sparkles: and multi project support!**

  

* **Live demo: https://dzyla.github.io/Follow_Relion_gracefully/**

  

* **Data coming from [Relion4 tutorial](https://relion.readthedocs.io/en/release-4.0/) and [Relion4 STA](https://relion.readthedocs.io/en/release-4.0/STA_tutorial/index.html)**

  

* **Non-Profit Open Software License 3.0 (NPOSL-3.0)**

  
### :sparkles: Liked it and contributed to your research? You can cite my work! :sparkles:

  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7844516.svg)](https://doi.org/10.5281/zenodo.7844516)

#### Dawid Zyla. (2023). dzyla/Follow_Relion_gracefully: v4 (Version update4). Zenodo. https://doi.org/10.5281/zenodo.7844516

  
  

## Description:microscope:

This is the 4th iteration of my Relion job preview script. Previous versions have focused on running just one job and its associated statistics. Now, since v3,  users can preview all jobs in a web browser interface while still being able to monitor the progress of running jobs.

In this latest version, the Python script can parse Relion's default_pipeline.star files originating from multiple projects and generate job results in parallel, rendering them quickly and visually appealing. It then retrieves the data and plots job-related stats; many of the standard SPA jobs are already supported while most of the STA jobs are implemented, but some of them will be included soon - due to lack of access to a complete STA project :sweat_smile: . 

Data plotting is tailored to my own needs, but I'm open to modifying or adding any new features and plots. If you have a suggestion for something you'd like to see, don't hesitate to let me know! :sparkles:

What's more, v4 automatically downloads the relevant Hugo executable and executes it when prompted. Additionally, the server should now be more stable, with the IP address bound to the workstation facilitating smooth running and deployment.

## v4 features:dizzy:
* Python script to parse Relion's folder structure and generate job results preview that enables better understanding of the data and allows the user to access data quality without having to open a single file
* The Hugo server offers a straightforward web browser interface to view job results in an aesthetically pleasing way
* Multi platform support (Windows, Linux, Mac OS [not tasted, should work like Linux, hopefully])
* Convenience of Multi-Project Support:twisted_rightwards_arrows: - keeping all of your Relion data in one place! Generate comprehensive plots and processing statistics from multiple projects effortlesly. Works well as Relion electronic notebook:notebook:!
* Relion 4 Tomography support: do not limit yourself to SPA, STA is here for you (partcially). `#teamtomo`
* Rewritten code for speed and removed redundancys. Hopefully made code more readable and stable. #ChatGPT #GPT4
* Publication-ready figures (FSC, Class projections, and angular distrubution plots)
* Monitor the particles from Select and Extract processes: visualize selected particles and find out numbers of selected/extracted particles.
* Debugger support: if something fails `--debug` will print problems in the log file. Might be useful for identifying problems.
* Dark mode🌜! No need to strain your eyes while working night shifts!
* `#OpenSoftwareAcceleratesScience`

## Installation:rocket:

Minor changes were made from `v3`, though a few new libraries have been added. The code has been tested on `Windows 10/11`, `WSL2` and `Ubuntu 22.04`. 

For `CentOS 7`, the Python part should run as normal (assuming Python 3.10 and conda are being used); however,`Hugo v111+` requires newer versions of GCC for extended version, and might have problems running. However, the version automatically downloaded should be compatible with CentOS7. Alas, it's time to move on from CentOS 7. 

  

### Install dependencies:snake:

Before executing the program, all dependencies must be installed into a conda environment. As the root Python version needs to be 3.10, a virtual environment is no longer supported and tested (it mights still work tho).

  

#### Conda way

If conda is already installed, the process should be relatively straightforward and move directly to step 2. Otherwise, if not:

  

1. Install miniconda3 (*no root required*):

```bash
wget -q  -P  .  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash ./Miniconda3-latest-Linux-x86_64.sh
```

  

Agree to the license agreement  `yes` and add conda to the PATH. Activate conda for bash:



```bash

conda init  bash
```

  

and restart shell or type:

```bash
bash
```

  

You should see the (base) in front of your username now:
```bash
(base) dzyla@GPU0
```

  

1. Create conda environment and install dependencies. 

```bash
conda create  -y  --name  FollowRelion  python=3.10

conda activate  FollowRelion
```
Now you should see:
```bash
(FollowRelion) dzyla@GPU0
```
Installing dependencies:
```bash
pip install  joblib  matplotlib  mrcfile  pandas  plotly  gemmi  scikit-image  seaborn  requests  psutil
```

  

3. Download GitHub repository and enter the folder:

```bash
git clone  https://github.com/dzyla/Follow_Relion_gracefully.git

cd  Follow_Relion_gracefully
```

  

The downloading process of Hugo is now managed through a Python function, which should ensure that you receive the version corresponding to your operating system and platform.

5. Type:


```bash

python follow_relion_gracefully_run.py
```

If you see:

```bash
python follow_relion_gracefully_run.py
usage: follow_relion_gracefully_run.py [-h] --i I [I ...] [--h H] [--n N] [--t T] [--single] [--new] [--download_hugo] [--server] [--debug]
follow_relion_gracefully_run.py: error: the following arguments are required: --i
```

  

 :sparkles:**You are good to go!** :sparkles:

  

## Usage:computer:

```text
python follow_relion_gracefully_run.py --help
usage: follow_relion_gracefully_run.py [-h] --i I [I ...] [--h H] [--n N] [--t T] [--single] [--new] [--download_hugo] [--server] [--debug]

Follow Relion Gracefully: web-based job GUI

optional arguments:
  -h, --help       show this help message and exit
  --i I [I ...]    One or more Relion folder paths. Required!
  --h H            Hostname for HUGO website. For remotly hosted Hugo server (for example another workstation or Github). Use IP adress of
                   the remote machine but make sure the ports are open. For local hosting leave it default (localhost)
  --n N            Number of CPUs for processing. Use less (2-4) if you have less than 16 GB of RAM
  --t T            Wait time between folder checks for changes used for continuous updates.
  --single         Single run, do not do continuous updates. Useful for the first-time usage
  --new            Start new Follow Relion Gracefully project and remove previous job previews. Removes whole /content/ folder! Use after
                   downloading from Github or when acually want to start a new project
  --download_hugo  Download HUGO executable if not present. This is operating system specific
  --server         Automatically start HUGO server
  --debug          Change log level to debug. Helpful for checking if something goes wrong
  --p P            Port for Hugo server
```

  

##### An example of how to use live server updates and set up a new project:

Activate conda enviroment if not activated yet
```bash
conda activate  FollowRelion
```

Use multiple projects by --i /folder1/ /folder2/ /folder3/
```bash
python follow_relion_gracefully_run.py --i /mnt/f/linux/relion40_tutorial_precalculated_results/ /mnt/f/linux/relion40_sta_tutorial_data/ --server --new
```
Hugo now refreshes its content according to the changes made in the Relion directories.

## Accesing Hugo server:doughnut:
By default, Hugo will be bound to http://localhost:1313/. If you're running Follow Relion Gracefully on your local computer, simply open your preferred browser and enter http://localhost:1313/ in the address bar.

When executing the Python script remotely, things get a bit more complicated. If the remote workstation is not specified with the `--h` flag, it will still be bound to http://localhost:1313/. In this case, the Hugo server will only be accessible from a web browser on the remote machine. To access the server, open a new terminal on the remote machine and start your favorite web browser with http://localhost:1313/.

Alternatively, create a ssh tunnel that will connect the remote machine port 1313 to your local computer port 1313:
```bash

ssh -f  username@workstation -L  1313:localhost:1313  -N
```
This will allow you access the remote Hugo server on your local computer.

However, this solution can be slow, and it's often better to bind the Hugo server to the remote workstation by providing the hostname with the `--h` flag. To find the IP address of your workstation, type the following command in the terminal:
```bash
ifconfig
```

That will give the information about the workstation IP:
```bash
enp65s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.10.120  netmask 255.255.0.0  broadcast 10.0.255.255
        ...
```
In this example, the IP address is 10.0.10.120. Rerun the script using this IP address:
```bash
python follow_relion_gracefully_run.py --i /mnt/f/linux/relion40_tutorial_precalculated_results/ /mnt/f/linux/relion40_sta_tutorial_data/ --server --new --h 10.0.10.120
```
This command binds the server to the specified IP address, allowing you to access the results in your local computer's web browser by entering: http://10.0.10.120:1313/

However, there's a catch. If the remote workstation is part of a cluster or has a firewall, the port may not be accessible from outside. You can try opening port 1313, but this requires root access.

An alternative solution is to run the Python script on the remote workstation but within a network-shared directory (such as a Samba share) and mount the directory on your local computer. Then, run the Hugo server on your local computer (./Hugo/hugo server), and you should be able to access the results at http://localhost:1313/.

The `--p` argument can also be helpful, as it allows you to specify the port number on which the server will run. If port 1313 is already in use, you can try another available port number. For example, you could choose a random free port number within a specific range, such as 5000-6000. Make sure to update the port number in the Python script command and when accessing the server through your web browser

  
## Hosting own GitHub page:globe_with_meridians:

  

For this you will need to fork the repository. It should build on its own at the `https://yourgithub.github.io/Follow_Relion_gracefully/`

  

This will work, however, only if it is public repository. You don't want others to access your secret project, do you?:lock: Hosting a private repo website is available only in the paid option for Github.

*However, it would be intesting to see the Relion project workflow for published structures hosted on GitHub.*


Next, clone your forked repository. To start a new project use `--new`. To generate previews instead of continutes updates use `--single`

```bash

#run script to generate new website content with proper hostname

python follow_relion_gracefully_run.py  --i  /mnt/f/linux/relion40_tutorial_precalculated_results/ /mnt/f/linux/relion40_sta_tutorial_data/ --h  yourgithub.github.io/Follow_Relion_gracefully --single --new
```
  

* Commit changes and sync project content to GitHub
```
git add  . && git commit  -m  "my first commit" && git push
```

  

This should send the newly processed files to Github and build a website. On your forked repository check actions and see whether site-building is running.

  
## Troubleshooting:wrench:

  

* If a job fails with an error, it will be reported on the generated webpage. If this happens regularly, try running the script with the `--debug` option and check `FollowRelion.log`. This may be due to the job not yet being finished or files missing. If there is anything persistent that you notice, don't hesitate to let me know - I may be able to fix it or rewrite the code. 
* The first run can take a while. Let it run for some time (project file discovery part). Once files are indexed, processing should start.

* When it comes to volume rendering not displaying what you want, this is mainly because Plotly script I wrote forces Fourier cropping of volumes to 100px and Skimage's Marching Cubes function to calculate the isosurface at the optimal contour level. This limitation is tied to static website generators, so there is no way to change the contour level in browser. It would require a website-Python connection, which unfortunately is not possible for static websites with Hugo. The restriction on 100px mesh rendering is necessary to enable quick loading in browser. Bigger maps typically crash browsers. Auto contour level works in most cases but sometimes doesn't work as desired. If anyone has ideas on how to improve this, please do let me know!
 
* It's important to make sure that no other environment is activated when working with Python. Check if `FollowRelion` is enabled in your environment.

* Also note that jobs run manually (*via command line*) won't be processed by the script as it reads from the *default_pipeline.star* file. Some jobs are also hard-coded to follow the classical workflow of Relion. This limitation might cause the script to fail in some cases.

* If Hugo displays incorrect content, it could be due to issues in the markdown files (*.md) in each job folder. Specifically, the host `--h` should be checked.

* 2D class previews may not be ideal for more than 100 classes. Try opening the class image in a new window, and check the class previews below; they should be ordered based on the class distribution.

* If you're running Hugo remotely but don't have write permission to the folder, the website won't update. Try other options such as changing the folder permissions, running Hugo on your computer, or connecting via port tunneling.
* If you don't like how the plots look like (colors, FSC scales, etc) you can try modifying the `follow_relion_gracefully_lib.py` file. All jobs should relatively well described.

* Mac support was not tested. I assume that is similar to Linux enough that once conda envirment is set it should work as expected. If not, please let me know!
* Finally, not all Tomo jobs are currently supported, but I hope to have access to pre-calculated data soon so I can include them.
  
  

## To-do:memo:

* Preview of the remaining Tomo jobs (need access to fully calculated project)
* Preview of non-default jobs (Multi-Body, External, Particle subtractions)

* Better volume preview
* Better data visualization, more statistics, everything publication-ready
* Possibility to download volumes (*ala* cryoSPARC) (Is it really necessary?)

* Job flow chart overview. Who is father of whom and which jobs are realted.

* Can you run Relion *via* static website generator? (probably not)
* Optimization of speed and RAM usage
* Selection of jobs inside given project


  

## Questions/suggestions?:email:

Dawid Zyla, La Jolla Institute for Immunology

[Twitter](https://twitter.com/DawidZyla)

[dzyla@lji.org](mailto:dzyla@lji.org)
