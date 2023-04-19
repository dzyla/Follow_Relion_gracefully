---
title: job011
jobname: AutoPick/job011/
status: Succeeded
date: 2021-05-26 06:58:50
time: 06:58:50
jobtype: [AutoPick]
project: [230416_Relion4_precalc]
---

#### Job alias: None

{{<rawhtml >}} 
    <div class="center">
    <p>Preview of random autopicked micrograph:<p>
    <input id="valR" type="range" min="0" max="23" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
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

 
 Executing new job on Wed May 26 14:54:48 2021
 
 with the following command(s): 

which relion_autopick --i CtfFind/job003/micrographs_ctf.star --odir AutoPick/job011/ --pickname autopick --topaz_exe /public/EM/RELION/topaz --topaz_nr_particles 300 --particle_diameter 180 --topaz_extract --topaz_model AutoPick/job010/model_epoch10.sav --gpu "1"  --pipeline_control AutoPick/job011/
 
 


