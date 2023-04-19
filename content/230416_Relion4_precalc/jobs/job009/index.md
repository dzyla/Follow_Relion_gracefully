---
title: job009
jobname: Select/job009/
status: Succeeded
date: 2021-05-26 06:29:37
time: 06:29:37
jobtype: [Select]
project: [230416_Relion4_precalc]
---

#### Job alias: None

Selected __1122__ (__47.2%__) particles from __2377__ particles in Class2D/job008/run_it025_data.star
### Selected classes
{{< img src="Select_job009_all.png" style="width: 100%;">}}
### Selected classes preview
{{<rawhtml >}} 
    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="4" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
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

 
 Executing new job on Wed May 26 14:29:00 2021
 
 with the following command(s): 

which relion_class_ranker --opt Class2D/job008/run_it025_optimiser.star --o Select/job009/ --fn_sel_parts particles.star --fn_sel_classavgs class_averages.star --python /public/EM/anaconda3/envs/topaz/bin/python --fn_root rank --do_granularity_features  --auto_select  --min_score 0.5  --pipeline_control Select/job009/
 
 


