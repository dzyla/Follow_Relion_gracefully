# Relion-2D-and-3D-live-preview (aka Follow Relion Gracefully [FRG])
New script to follow Relion classification in a browser!

Simple python script which enables to follow the progress of 2D/3D classification and refinement in Relion.

---Requirements---
* Python 3+ with:
+ matplotlib
+ gemmi
+ mrcfile
+ numpy
+ pandas
+ skimage
+ plotly

---How to run---

Install own python enviroment:
```
python3 -m venv new-env
```
Activate enviroment:
```
source new-env/bin/activate
```
Install packages:
```
pip install numpy matplotlib pandas gemmi mrcfile plotly scikit-image
```
Run script from command line:
```
python3 relion-2D-and-3D-live-preview.py --i /path/to/the/classification/ --w 300
```
Help can be accesed by:
```
python relion-2D-and-3D-live-preview.py  --h
usage: relion-2D-and-3D-live-preview.py [-h] [--i I] [--w W]

Real-time preview Relion classification output from Class2D, Class3D and Refine3D jobs, including volume projections, class distributions and estimated resolution plots

optional arguments:
  -h, --help  show this help message and exit
  --i I       Classification folder path
  --w W       Wait time in seconds between refreshes. Adjust accordingly to the expected time of the iteration. If the time is up and there no new classification results plots will freeze
```

Run from command line with the destnation directory. Browser window should open with the classification statistics and will update as soon as the iteration is finished.

---Update---

200420: 
+ Now single script avaiable for all classification runs, including plotting of class distribution and class estimated resolution. All previous scripts are obsolete. 

210505: Major update and V2
+ Changed from Matplotlib to Plotly and web browser interface
+ Added extra dependencies (Plotly and scikit-image)
+ Shows 2D classification, 3D classification, Initial Model and 3D refinement
+ 2D/3D and Initial model are shown as projections from 3 axes
+ 3D refinement show 3D map (scaled to 100px)
+ Automatic updates in browser!
