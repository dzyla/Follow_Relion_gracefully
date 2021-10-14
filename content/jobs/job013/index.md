
---
title: job013
jobname: Class2D/job013/
status: Succeeded
date: 2021-06-01 06:42:06
time: 06:42:06
categories: [Class2D]
---

#### Job alias: None

{{< plotly json="/jobs/job013/cls2d_Class2D_job013_.json" height="800px" >}}
{{<rawhtml >}} 

    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="100" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
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
{{< plotly json="/jobs/job013/cls2d_dist_Class2D_job013_.json" height="600px" >}}
{{< plotly json="/jobs/job013/cls2d_res_Class2D_job013_.json" height="600px" >}}

#### Job command(s):


 
 Executing new job on Tue Jun  1 14:39:37 2021
 
 with the following command(s): 

which relion_refine --o Class2D/job013/run --i Extract/job012/particles.star --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --ctf  --iter 100 --grad --class_inactivity_threshold 0.1  --init_blobs --tau2_fudge 2 --particle_diameter 200 --K 100 --flatten_solvent  --zero_mask  --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 12 --gpu "4,5,6,7"  --pipeline_control Class2D/job013/
 
 


