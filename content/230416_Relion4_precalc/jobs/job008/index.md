---
title: job008
jobname: Class2D/job008/
status: Succeeded
date: 2021-05-25 09:27:23
time: 09:27:23
jobtype: [Class2D]
project: [230416_Relion4_precalc]
---

#### Job alias: None

{{< img src="Class2D_job008_all.png" style="width: 100%;">}}
{{<rawhtml >}} 
    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="50" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="250">
    </div>

    <script>

        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + 1 + ".jpg";
            function showVal(newVal){
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }
    </script>
    <br>
     {{< /rawhtml >}}
{{< img src="Class2D_job008_dist.png" >}}
{{< img src="Class2D_job008_res.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Tue May 25 17:24:51 2021
 
 with the following command(s): 

which relion_refine_mpi --o Class2D/job008/run --i Extract/job007/particles.star --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --ctf  --iter 25 --tau2_fudge 2 --particle_diameter 200 --K 50 --flatten_solvent  --zero_mask  --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 6 --gpu "0:1:2:3"  --pipeline_control Class2D/job008/
 
 


