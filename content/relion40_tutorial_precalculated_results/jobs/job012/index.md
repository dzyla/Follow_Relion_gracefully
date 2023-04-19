---
title: job012
jobname: Extract/job012/
status: Succeeded
date: 2021-06-01 00:57:28
time: 00:57:28
jobtype: [Extract]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

### Extracted 9191 particles
{{< rawhtml >}} 
   
<div class="center">
<p>Preview of random extracted particles:<p>
<input id="valR" type="range" min="0" max="98" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
<span id="range">0</span>
<img id="img" width="200">
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

 
 Executing new job on Tue Jun  1 08:56:53 2021
 
 with the following command(s): 

which relion_preprocess --i CtfFind/job003/micrographs_ctf.star --coord_list AutoPick/job011/autopick.star --part_star Extract/job012/particles.star --part_dir Extract/job012/ --extract --extract_size 256 --float16  --scale 64 --norm --bg_radius 25 --white_dust -1 --black_dust -1 --invert_contrast   --pipeline_control Extract/job012/
 
 


