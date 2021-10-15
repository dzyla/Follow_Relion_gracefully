# Follow Relion Gracefully
**v3: Show them all!**

**Live demo: https://dzyla.github.io/Follow_Relion_gracefully/**

**Data coming from [Relion4 tutorial](https://relion.readthedocs.io/en/release-4.0/)**

## Description
This is a 3rd iteration of my Relion job preview script. Previous versions focused on a single job and its statistics. Now 
all jobs can be previewed using friendly web browser interface.

The main changes include the addition of static website generator
[HUGO](https://gohugo.io/). Python script reads the Relion's default_pipeline.star
file, fetches the data and plots job-related stats. Almost all jobs are
supported, some of them will be supported soon.

HUGO takes Python written markdown files and generates a website
that can be seen either locally or via GitHub pages. 

## Installation

Due to the major changes in the code, some new libraries are required as well as a HUGO instance.
Code was tested on Windows 10/11 with and without WSL2, on Ubuntu 20.04 and CentOS7.

### Install dependencies
Before running the program one needs to install all dependencies in either conda environment or Python
virtual environment.

#### Python Environment way

#### Conda way
If conda is already installed it should be relatively easy and follow part 2. If not, then:

1. Install miniconda3:
```bash
wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
```

Agree for to the license agreement and add conda to the PATH. Activate conda for bash:

```bash
conda init bash
```

and restart shell or type:
```bash
bash
```

You should see the (base) in front of your username now.

2. Create conda environment and Install dependencies. Agree [y] on everything.
```bash
conda create --name FollowRelion python=3.9
conda activate FollowRelion

pip install joblib matplotlib mrcfile pandas plotly gemmi scikit-image
```

(!!Optional!!) Instead of conda, use Python virtual environment:
```bash
python3 -m venv FollowRelion
source FollowRelion/bin/activate
pip install joblib matplotlib mrcfile pandas plotly gemmi scikit-image
```

Python virtual environment will need to compile gemmi.
For this python3-dev package is required but has to be installed via sudo. If this fails, 
go conda.

3. Download GitHub repository and enter the folder:
```bash
git clone https://github.com/dzyla/Follow_Relion_gracefully.git
cd Follow_Relion_gracefully
```

4. Download Hugo inside the folder:
```bash
wget https://github.com/gohugoio/hugo/releases/download/v0.88.1/hugo_0.88.1_Linux-64bit.tar.gz
# Check https://github.com/gohugoio/hugo/releases for new releases

tar xvf hugo_0.88.1_Linux-64bit.tar.gz
# Start HUGO server
./hugo server --disableLiveReload
```
Hugo should start and show:
```bash
(FollowRelion) [dzyla@chris Follow_Relion_gracefully]$ ./hugo server --disableLiveReload
Start building sites … 
hugo v0.88.1-5BC54738 linux/amd64 BuildDate=2021-09-04T09:39:19Z VendorInfo=gohugoio

                   | EN   
-------------------+------
  Pages            |  76  
  Paginator pages  |   7  
  Non-page files   | 729  
  Static files     |   0  
  Processed images |   0  
  Aliases          |  20  
  Sitemaps         |   1  
  Cleaned          |   0  

Built in 153 ms
Watching for changes in /home/dzyla/Follow_Relion_gracefully/{archetypes,assets,content,layouts}
Watching for config changes in /home/dzyla/Follow_Relion_gracefully/config.toml
Environment: "development"
Serving pages from memory
Web Server is available at http://localhost:1313/Follow_Relion_gracefully/ (bind address 127.0.0.1)
Press Ctrl+C to stop
```

Check if the webserver is up and connect to server display in the terminal, e.g.: http://localhost:1313/Follow_Relion_gracefully/

**If everything works, you are ready to go!**

## Usage
```text
python follow_relion_gracefully_run.py --help
usage: follow_relion_gracefully_run.py [-h] --i I [--h H] [--o O] [--force]
                                       [--n N] [--debug] [--t T] [--single]

Follow Relion Gracefully: web based job GUI

optional arguments:
  -h, --help  show this help message and exit
  --i I       Relion folder path
  --h H       Hostname for HUGO website. Only for hosted website
  --o O       Output directory for HUGO website.
  --force     Force redo all folders
  --n N       Number of CPUs for processing. Use less (1-2) if RAM is the
              issue
  --debug     Debug
  --t T       Wait time between folder comparisons for the continues updates
  --single    Single run, do not update
```

* Example usage:

```bash
#if not done previously
conda activate FollowRelion
python follow_relion_gracefully_run.py --i /mnt/staging/Dawid/test/relion40_tutorial_precalculated_results/ --single
./hugo server --disableLiveReload
```

* For continues updates it is slightly more tricky:

```bash
# Use & at the end of the python command, so it can run in the background
python follow_relion_gracefully_run.py --i /mnt/staging/Dawid/test/relion40_tutorial_precalculated_results/ &
./hugo server --disableLiveReload
```
Now Hugo should update content based on changes in the Relion folder.

* Hosting own GitHub page

For this you will need to fork the repository. It should build on its own at the https://yourgithub.github.io/Follow_Relion_gracefully/

This will work, however, only if it is public repo. You don't want it for your secret project, do you?
Hosting private repo website is only paid option for Github.

Next, clone your forked repo, remove content/jobs:
```bash
rm -rf content/jobs/
#run script to generate new website content with proper hostname
python follow_relion_gracefully_run.py --i /mnt/staging/Dawid/test/relion40_tutorial_precalculated_results/ --h https://yourgithub.github.io/Follow_Relion_gracefully/ --single

#Add to your GitHub
git add . && git commit -m "my first commit" && git push
```

This should send the newly processed files to Github and host a website. On your forked repo check 
actions and see whether site building is running.

* Working from cluster etc.
You can theoretically do this from cluster and connect via your browser as soon as
port forwarding can be established. To do the forwarding, try:
```bash
ssh -f username@cluster -L 1313:localhost:1313 -N
```
and check your https://localhost:1313 for forwarding.

## Troubleshooting

* Whenever job finishes with error, it will say on the generated website that something
went wrong. It can be theoretically everything. The most common are jobs that were cancelled
and do not have required files for plotting.

* Volume rendering is not showing what I want --> this is the main drawback of plotly that to make it smooth I had to force fourier cropping of volumes to 100px. Then, marching cubes from skimage is used to calculate the isosurface at the best contour level. Everything what is automatic doesn't usually work well. If you had ideas how to fix that let me know!
* If something with Python doesn't work make sure that any other environment is not activated!
* Because this script reads *default_pipeline.star* file manually run jobs won't be processed by script. There should be a way to add the to the pipeline but not sure how.
* If HUGO shows wrong content check the markdown files (*.md) in the job folders. The host description might be the issue.


## To-do
* Better volume preview
* Possibility to download volumes (*ala* cryoSPARC)
* Follow job flow (what is input and output from current job)
* Preview of the Select job
* Can you run Relion *via* static website generator?

## Contact
Dawid Zyla, La Jolla Institute for Immunology

[dzyla@lji.org](mailto:dzyla@lji.org)

