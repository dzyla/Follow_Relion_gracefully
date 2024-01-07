# Follow Relion Gracefully :microscope::rocket::globe_with_meridians:
---
**v5: An almost complete dashboard for easy interaction with your cryo-EM data in Relion, now with partial :sparkles: `#teamtomo` :sparkles: support!**

* **Data sourced from [Relion5 tutorial](https://relion.readthedocs.io/en/latest/SPA_tutorial/index.html) and [Relion4 STA](https://relion.readthedocs.io/en/release-4.0/STA_tutorial/index.html)**
* **Licensed under Non-Profit Open Software License 3.0 (NPOSL-3.0)**
  
https://user-images.githubusercontent.com/20625527/233797797-c4a93f69-9abd-4636-aab2-958419bcec9f.mp4

  
#### :sparkles: Found this helpful in your research? Cite my work! :sparkles:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10465899.svg)](https://doi.org/10.5281/zenodo.10465899)


#### Dawid Zyla. (2024). dzyla/Follow_Relion_gracefully: v5 (Version v5). Zenodo. https://doi.org/10.5281/zenodo.10465899
<a href="https://www.buymeacoffee.com/dzyla" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>


---

## Description :microscope:

#### v5: :high_brightness:
Version 5 improves the job preview by adopting a dynamic approach. Using [Streamlit](https://streamlit.io/), it allows users to interact directly with their data. The underlying Python framework facilitates real-time computation of statistics and data from most jobs, enabling users to engage with metadata and select preferred statistics for download and further analysis.

#### v4:
Version 4 introduced support for multiple projects and job visualization through an online interface using the Hugo framework. While this static job generator enabled job display with example data, it lacked interactive capabilities due to its static nature.

## v5 features :dizzy:
* Python script recognizes `default_pipeline.star` (only Relion 4 and 5 supported) and generates job previews, enhancing data understanding and quality access without opening individual files.
* Streamlit simplifies the setup (no extra software downloads needed) and is operated through a single file `follow_relion_gracefully.py`.
* :sparkles: (New) Relion Live dashboard: Monitor live Relion sessions and select MotionCorr/CtfFind statistics. This functionality is inspired and partially based on [CNIO_Relion_Tools](https://github.com/cryoEM-CNIO/CNIO_Relion_Tools).
* :sparkles: (New) *Experimental* support for running Relion in-browser`*`! Access and operate all Relion programs directly via a web interface. Currently a Linux-only technical demo, but functioning well.
* :sparkles: (New) Job selection displays rejected particle locations (both SPA and STA), aiding in data quality assessment.
* :sparkles: (New) Downloadable results: Access all final MRC and PDF files directly, eliminating the need for network drive mounting.
* :sparkles: (New) Volume slider: Interactively adjust volume thresholds.
* :sparkles: (New) Enhanced picking job support: Interact with all micrographs and review picking statistics.
* :sparkles: (New) View your job flowchart in-browser. While not aesthetically pleasing, it is informative.
* :sparkles: (New) Interactive metadata plotting and selection tool: Choose data directly from plots and download selected STAR files. Upload your own STAR files to customize statistics.
* :sparkles: (New) ModelAngelo support: Visualize final protein structures with your maps in-browser.
* :sparkles: (New) Direct 2D class selection in-browser.
* :sparkles: (New) Increased security with password-protected dashboard access.
* Multi-platform support (Windows, Linux, Mac OS [untested, but expected to work similarly to Linux]).
* Multi-Project Support :twisted_rightwards_arrows: - Manage all your Relion data in one place. Generate detailed plots and processing statistics from multiple projects effortlessly. Functions well as a Relion electronic notebook:notebook:.
* Partial support for Relion 4 Tomography workflow `#teamtomo`
* Code rewritten for enhanced speed and stability. #ChatGPT #GPT4
* Publication-ready figures (FSC, class projections, and angular distribution plots).
* Monitoring capabilities for Select and Extract processes: visualize and quantify selected/extracted particles.
* Dark mode🌜: Reduce eye strain during night shifts, enabled via Streamlit settings.
* `#OpenSoftwareAcceleratesScience`

 `* not acually running in the browser but allowing to start the job that runs on Linux workstation` 

https://user-images.githubusercontent.com/20625527/233797847-8d1200e1-b592-415b-bedb-2be7776d5aa4.mp4


## Installation :rocket:

Minor changes from `v4`, with a few new libraries added. Tested on `Windows 10/11`, `WSL2`, `CentOS`, and `Ubuntu 22.04`.

### Install Dependencies :snake:

Install dependencies in a conda environment, as Python 3.11 is required and virtual environments are no longer supported (though they might still work).

#### Conda Instructions

1. Install miniconda3 (*no root access required*, only if not installed already):

```bash
wget -q  -P  .  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash ./Miniconda3-latest-Linux-x86_64.sh -b -f
```

Activate conda for bash:

```bash
conda init bash
```

Restart the shell or type `bash` to see the (base) prompt:

```bash
(base) dzyla@GPU0
```

2. Clone the GitHub repository and navigate to the folder:

```bash
git clone https://github.com/dzyla/Follow_Relion_gracefully.git

cd Follow_Relion_gracefully
```

3. Create a conda environment and install dependencies using the `environment.yml` file:

```bash
conda env create --file environment.yml

conda activate FollowRelion
```

You should now see:

```bash
(FollowRelion) dzyla@GPU0
```

:sparkles:**Ready to start!** :sparkles:

## Usage :computer:

```text
streamlit run follow_relion_gracefully.py
```

Additional command line parameters for extra features:

```
-h, --help            Show this help message and exit.
--i I, --folder I     Path to the default folder.
--path PATH, --relion-path PATH
                        Path to the RELION bin folder.
--p P, --password P   Password for securing your instance.
```

##### Example for live server updates and setting up a new project:

To use command line parameters with streamlit, add `--` before the parameters:

```
conda activate FollowRelion

streamlit run follow_relion_gracefully.py -- --p MyPassword$221#& --i /mnt/staging/240105_NewProcessing --path /usr/local/relion5/build/bin/
```

This sets a password, default processing folder, and specifies the path to Relion executables.

## Accessing the Dashboard via Browser :chart_with_upwards_trend:

The dashboard should open automatically in your browser. For remote workstations, access it using the provided network URL, ensuring the port is not firewall-blocked.

Remote access example:

```
(FollowRelion) dzyla@PC-HOME:~/Follow_Relion_gracefully$ streamlit run follow_relion_gracefully.py --server.port 8501 -- --p 1234 --i /mnt/f/linux/Tutorial5.0/

  Local URL: http://localhost:8501
  Network URL: http://172.21.222.176:8501
```

Open the network URL in your browser to access the dashboard.

For firewall issues, create an SSH tunnel:

```bash
ssh -f username@workstation -L 8501:localhost:8501 -N
```

This allows remote dashboard access on your local computer: http://localhost:8501.

## Hosting a GitHub Page :globe_with_meridians: (Deprecated)

The real-time data calculation in the new implementation makes pre-calculated server hosting obsolete. Advanced users can still use tools like [ngrok](https://ngrok.com/) for remote dashboard access.

## Troubleshooting :wrench:

* Issues with certain parameters causing errors? Start a new GitHub issue with the problematic STAR files.
* Large volumes (500px+) load slowly, especially for multiple class-3D classifications. Downloading them is suggested.
* Ensure the correct environment is activated (`FollowRelion`). Deactivate others with `conda deactivate`.
* Jobs run manually may not be processed, as the script reads from `default_pipeline.star`. The exception is the Relion Live dashboard.
* Rendering issues in the browser can often be resolved by refreshing (`F5`).
* Import job previews and ice thickness calculations in Relion Live may be slow.
* Mac support is untested, but it's assumed to work similarly to Linux. Please report any issues!
* Not all Tomo jobs are currently supported, but future updates are planned.
* Please note that this code was developed by a Python enthusiast, not a professional developer. It has been tested under standard scenarios to ensure reliability. However, as the author, I cannot be held responsible for any issues or damages that may arise from its use. Users are encouraged to review and test the code thoroughly before implementation in their projects.

## To-do :memo:

* Preview of the remaining Tomo jobs (need access to fully calculated project) :grey_question:
* Preview of non-default jobs (Multi-Body, External, Particle subtractions, DynaMight) :grey_question:
* Better volume preview :white_check_mark:
* Better data visualization, more statistics, everything publication-ready :white_check_mark:
* Possibility to download volumes (*ala* cryoSPARC) (Is it really necessary?) :white_check_mark:

* Job flow chart overview. Who is father of whom and which jobs are realted. :white_check_mark:

* Can you run Relion *via* static website generator? (~~probably not~~) :white_check_mark:
* Optimization of speed and RAM usage :grey_question:
* Selection of jobs inside given project :grey_question:
* Add job templates for running Relion via browser (basically copying Relion GUI)
* A way to modify and write jobs to `default_pipeline.star`
* Relion Live job creation (New Movie->Import->MotionCorr->CtfFind->update dashboard->Repeat)
* Add manual picking job that would save picks for further use.


  

## Questions/suggestions?:email:

Dawid Zyla, La Jolla Institute for Immunology

[Twitter](https://twitter.com/DawidZyla)

[dzyla@lji.org](mailto:dzyla@lji.org)
