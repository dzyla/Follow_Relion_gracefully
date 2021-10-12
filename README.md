# Follow Relion Gracefully
v3: Show them all!

## Description
This is a 3rd iteration of Relion job preview script. Previous versions focused on a single job and its statistics. Now 
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

and restart shell.

2. Create conda environment and Install dependencies. Agree [y] on everything.
```bash
conda create --name FollowRelion python=3.9
conda activate FollowRelion

pip install joblib matplotlib mrcfile pandas plotly gemmi scikit-image
```

Download GitHub repository:
```bash
git clone https://github.com/dzyla/Follow_Relion_gracefully.git
```

## Usage

## Troubleshooting

## To-do

