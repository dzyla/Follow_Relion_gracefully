# Follow Relion Gracefully :microscope::rocket::globe_with_meridians:
---
**v6: A complete dashboard for easy interaction with your cryo-EM data in Relion, now with ~~partial~~ full :sparkles: `#teamtomo` :sparkles: support!**

* **Data sourced from [Relion5 tutorial](https://relion.readthedocs.io/en/latest/SPA_tutorial/index.html), [Relion4 STA](https://relion.readthedocs.io/en/release-4.0/STA_tutorial/index.html), and [Relion5 STA](https://zenodo.org/records/11068319)**
* **Licensed under Non-Profit Open Software License 3.0 (NPOSL-3.0)**
  



https://github.com/dzyla/Follow_Relion_gracefully/assets/20625527/5b14ca38-b318-4b69-ac72-749dcfa12559



  
#### :sparkles: Found this helpful in your research? Cite my work! :sparkles:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10465899.svg)](https://doi.org/10.5281/zenodo.10465899)


#### Dawid Zyla. (2024). dzyla/Follow_Relion_gracefully: v5 (Version v5). Zenodo. https://doi.org/10.5281/zenodo.10465899
<a href="https://www.buymeacoffee.com/dzyla" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-blue.png" alt="Buy Me A Coffee" height="41" width="174"></a>


---

## Description :microscope:

#### v6: :high_brightness:
Version 6 introduces a complete dashboard for easy interaction with your cryo-EM data in Relion, now with full `#teamtomo` support! It allows users to visualize and analyze their data in real-time, providing a comprehensive overview of their projects. The new version also includes improved job previews, enhanced data visualization, and the ability to download volumes directly from the dashboard.

#### v5: 
Version 5 improves the job preview by adopting a dynamic approach. Using [Streamlit](https://streamlit.io/), it allows users to interact directly with their data. The underlying Python framework facilitates real-time computation of statistics and data from most jobs, enabling users to engage with metadata and select preferred statistics for download and further analysis.

#### v4:
Version 4 introduced support for multiple projects and job visualization through an online interface using the Hugo framework. While this static job generator enabled job display with example data, it lacked interactive capabilities due to its static nature.

## v6 features :dizzy:
* All the features of v5 but with improvements
* Full support for `#teamtomo` jobs
* Better job previews, including volume previews, statistics, and download options
* Improved data visualization and publication-ready statistics
* Ability to download volumes directly from the dashboard
* Job flow chart overview, showing relationships between jobs
* Improved speed (but increased RAM usage)
* Plenty of QOL improvements
* `#OpenSoftwareAcceleratesScience`




## Installation :rocket:

Minor changes from `v5`, with a few new libraries added. Tested on `Windows 10/11`, `WSL2`, and `Ubuntu 22.04`.

### Install Dependencies :snake:

Install dependencies in a conda environment, as Python 3.12 is required and virtual environments are no longer supported (though they might still work).

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

#### UV Instructions
UV is a package manager for Python that allows you to install and manage Python packages easily. It is similar to pip but is ultra-fast. To install UV, follow these steps:

1. Install UV using pip:

```bash
pip install uv
```

or if you don't have python/pip/conda installed, use the following command:

```bash
# with curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# alternatively, with wget
wget -qO- https://astral.sh/uv/install.sh | sh

# or on Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Clone the GitHub repository and navigate to the folder:

```bash
git clone https://github.com/dzyla/Follow_Relion_gracefully.git

cd Follow_Relion_gracefully
```

3. Create a new virtual environment using UV:

```bash
uv venv FollowRelion --python 3.12
```
4. Activate the virtual environment:

```bash
source FollowRelion/bin/activate
```


5. Install the required packages using the `requirements.txt` file:

```bash
uv pip install -r requirements.txt
```
5. After the installation is complete, you should see a message indicating that the packages have been installed successfully.



:sparkles:**Ready to start!** :sparkles:

## Usage :computer:

```text
streamlit run follow_relion_gracefully.py
```

Additional command line parameters for extra features:

```
-h, --help            Show this help message and exit.
-i I, --folder I     Path to the default folder.
-p P, --password P   Password for securing your instance.
```

##### Example for live server updates and setting up a new project:

To use command line parameters with streamlit, add `--` before the parameters:

```
conda activate FollowRelion

streamlit run follow_relion_gracefully.py -- -p MyPassword$221#& -i /mnt/staging/240105_NewProcessing
```

This sets a password and default processing folder.

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

## Troubleshooting :wrench:

* As previous versions, the code is a hobby project and may not work perfectly. Please report any issues on GitHub.
* Large volumes (500px+) load slowly, especially for multiple class-3D classifications. Plotly does an excellent job with plotting but too large data can slow down the browser.
* Ensure the correct environment is activated (`FollowRelion`). Deactivate others with `conda deactivate`.
* Jobs run manually may not be processed, as the script reads from `default_pipeline.star`.
* There are some cases with warning about the session state. They can be ignored.
* Rendering issues in the browser can often be resolved by refreshing (`F5`).
* Mac support is untested, but it's assumed to work similarly to Linux. Please report any issues!
* Please note that this code was developed by a Python enthusiast, not a professional developer. It has been tested under standard scenarios to ensure reliability. However, as the author, I cannot be held responsible for any issues or damages that may arise from its use. Users are encouraged to review and test the code thoroughly before implementation in their projects.


## To-do :memo:

* Add support for `DynaMight`
* Speed up processing for some jobs


  

## Questions/suggestions?:email:

Dawid Zyla, La Jolla Institute for Immunology

[Twitter](https://twitter.com/DawidZyla)

[dzyla@lji.org](mailto:dzyla@lji.org)
