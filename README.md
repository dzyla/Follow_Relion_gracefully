# Follow Relion Gracefully
New script to follow Relion classification in a browser!

Simple python script which enables following the progress of 2D/3D classification and refinement in Relion.

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

Install own python environment:
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
Run script from the command line:
```
python3 follow_relion_gracefully.py --i /path/to/the/classification/ --w 300
```
Help can be accessed by:
```
python follow_relion_gracefully.py  --h
usage: follow_relion_gracefully.py [-h] [--i I] [--w W]

Real-time preview Relion classification output from Class2D, Class3D, and Refine3D jobs, including volume projections, class distributions, and estimated resolution plots

optional arguments:
  -h, --help  show this help message and exit
  --i I       Classification folder path
  --w W       Wait time in seconds between refreshes. Adjust accordingly to the expected time of the iteration. If the time is up and there no new classification results plots will freeze
```

Run from the command line with the destination directory. The browser window should open with the classification statistics and will update as soon as the iteration is finished.

---Update---

200420: 
+ Now single script available for all classification runs, including plotting of class distribution and class estimated resolution. All previous scripts are obsolete. 

210505: Major update and V2
+ Changed from Matplotlib to Plotly and web browser interface
+ Added extra dependencies (Plotly and scikit-image)
+ Shows 2D classification, 3D classification, Initial Model, and 3D refinement
+ 2D classification is shown as classes ordered by class number
+ 3D classification and Initial model are shown as projections from 3 axes
+ 3D refinement show 3D map (scaled to 100px)
+ Automatic updates in the browser!


https://user-images.githubusercontent.com/20625527/117906929-b3667780-b28a-11eb-8334-e14789121b95.mp4



![Screenshot from 2021-05-11 18-53-12](https://user-images.githubusercontent.com/20625527/117906888-9e89e400-b28a-11eb-9067-43a533543836.png)
![Screenshot from 2021-05-11 18-53-56](https://user-images.githubusercontent.com/20625527/117906898-a21d6b00-b28a-11eb-82ff-e9f1400594e8.png)
![Screenshot from 2021-05-11 18-54-39](https://user-images.githubusercontent.com/20625527/117906913-ab0e3c80-b28a-11eb-97dc-a650cb979c0d.png)



