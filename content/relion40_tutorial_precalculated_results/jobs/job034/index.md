---
title: job034
jobname: Select/job034/
status: Succeeded
date: 2023-04-09 14:53:48
time: 14:53:48
jobtype: [Select]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

Selected __4688__ (__98.74%__) particles from __4748__ particles in Class2D/job033/run_it300_data.star
### Selected classes
{{< img src="Select_job034_all.png" style="width: 100%;">}}
### Selected classes preview
{{<rawhtml >}} 
    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="32" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
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

#### Job command(s):

```bash

 
 Executing new job on Sun Apr  9 14:53:23 2023
 
 with the following command(s): 

which relion_display --gui --i Class2D/job033/run_it300_optimiser.star --allow_save --fn_parts Select/job034/particles.star --fn_imgs Select/job034/class_averages.star  --pipeline_control Select/job034/
 
 


