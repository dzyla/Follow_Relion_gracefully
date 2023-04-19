---
title: job032
jobname: Class2D/job032/
status: Succeeded
date: 2023-04-04 21:02:24
time: 21:02:24
jobtype: [Class2D]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

{{< img src="Class2D_job032_all.png" style="width: 100%;">}}
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
{{< img src="Class2D_job032_dist.png" >}}
{{< img src="Class2D_job032_res.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Tue Apr  4 20:45:34 2023
 
 with the following command(s): 

which relion_refine --o Class2D/job032/run --grad --class_inactivity_threshold 0.1 --grad_write_iter 10 --iter 200 --i Extract/job012/particles.star --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --ctf  --tau2_fudge 2 --particle_diameter 200 --K 40 --flatten_solvent  --zero_mask  --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --allow_coarser_sampling --norm --scale  --j 10 --gpu ""  --pipeline_control Class2D/job032/
 
 

 
 Executing new job on Tue Apr  4 20:46:58 2023
 
 with the following command(s): 

which relion_refine --continue Class2D/job032/run_it020_optimiser.star --o Class2D/job032/run --grad --class_inactivity_threshold 0.1 --grad_write_iter 10 --iter 200 --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --tau2_fudge 2 --particle_diameter 200 --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --allow_coarser_sampling --j 10 --gpu ""  --pipeline_control Class2D/job032/
 
 


