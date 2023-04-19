---
title: job018
jobname: Extract/job018/
status: Succeeded
date: 2021-06-01 07:22:28
time: 07:22:28
jobtype: [Extract]
project: [230416_Relion4_precalc]
---

#### Job alias: None

### Extracted 4748 particles
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

 
 Executing new job on Tue Jun  1 15:21:46 2021
 
 with the following command(s): 

which relion_preprocess --i CtfFind/job003/micrographs_ctf.star --reextract_data_star Select/job017/particles.star --recenter --recenter_x 0 --recenter_y 0 --recenter_z 0 --part_star Extract/job018/particles.star --pick_star Extract/job018/extractpick.star --part_dir Extract/job018/ --extract --extract_size 360 --float16  --scale 256 --norm --bg_radius 71 --white_dust -1 --black_dust -1 --invert_contrast   --pipeline_control Extract/job018/
 
 


