---
title: job033
jobname: Class2D/job033/
status: Succeeded
date: 2023-04-05 22:05:25
time: 22:05:25
jobtype: [Class2D]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

{{< img src="Class2D_job033_all.png" style="width: 100%;">}}
{{<rawhtml >}} 
    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="40" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
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
{{< img src="Class2D_job033_dist.png" >}}
{{< img src="Class2D_job033_res.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Wed Apr  5 21:39:23 2023
 
 with the following command(s): 

which relion_refine --o Class2D/job033/run --grad --class_inactivity_threshold 0.1 --grad_write_iter 10 --iter 200 --i Refine3D/job029/run_data.star --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --ctf  --tau2_fudge 2 --particle_diameter 200 --K 40 --flatten_solvent  --zero_mask  --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --allow_coarser_sampling --norm --scale  --j 10 --gpu ""  --pipeline_control Class2D/job033/
 
 

 
 Executing new job on Wed Apr  5 21:53:57 2023
 
 with the following command(s): 

which relion_refine --continue Class2D/job033/run_it200_optimiser.star --o Class2D/job033/run --grad --class_inactivity_threshold 0.1 --grad_write_iter 10 --iter 300 --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --tau2_fudge 10 --particle_diameter 200 --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --allow_coarser_sampling --j 10 --gpu ""  --pipeline_control Class2D/job033/
 
 


