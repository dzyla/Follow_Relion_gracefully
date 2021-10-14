# Follow Relion Gracefully
v3: Show them all!
Live demo: https://dzyla.github.io/Follow_Relion_gracefully/

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



## Troubleshooting

## To-do

