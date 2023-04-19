---
title: job006
jobname: AutoPick/job006/
status: Succeeded
date: 2021-05-25 09:11:48
time: 09:11:48
jobtype: [AutoPick]
project: [230416_Relion4_precalc]
---

#### Job alias: None

{{<rawhtml >}} 
    <div class="center">
    <p>Preview of random autopicked micrograph:<p>
    <input id="valR" type="range" min="0" max="9" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="600">
    </div>

    <script>

        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + ".jpg";
            function showVal(newVal){
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }
    </script>
    <br>
     {{< /rawhtml >}}

#### Job command(s):

```bash

 
 Executing new job on Tue May 25 17:11:42 2021
 
 with the following command(s): 

which relion_autopick_mpi --i Select/job005/micrographs_split1.star --odir AutoPick/job006/ --pickname autopick --LoG  --LoG_diam_min 150 --LoG_diam_max 180 --shrink 0 --lowpass 20 --LoG_adjust_threshold 0 --LoG_upper_threshold 5  --pipeline_control AutoPick/job006/
 
 


