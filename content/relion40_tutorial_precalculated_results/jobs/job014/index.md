---
title: job014
jobname: Select/job014/
status: Succeeded
date: 2021-06-01 06:44:09
time: 06:44:09
jobtype: [Select]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

Selected __5812__ (__63.24%__) particles from __9191__ particles in Class2D/job013/run_it100_data.star
### Selected classes
{{< img src="Select_job014_all.png" style="width: 100%;">}}
### Selected classes preview
{{<rawhtml >}} 
    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="35" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
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

 
 Executing new job on Tue Jun  1 14:42:46 2021
 
 with the following command(s): 

which relion_class_ranker --opt Class2D/job013/run_it100_optimiser.star --o Select/job014/ --fn_sel_parts particles.star --fn_sel_classavgs class_averages.star --python /public/EM/anaconda3/envs/topaz/bin/python --fn_root rank --do_granularity_features  --auto_select  --min_score 0.25  --pipeline_control Select/job014/
 
 


