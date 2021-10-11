
---
title: job007
jobname: Extract/job007/
status: Succeeded
date: 2021-05-25 09:14:25
time: 09:14:25
categories: [Extract]
---

#### Job alias: None

{{<rawhtml >}} 
   
<div class="center">
<p>Preview of random extracted particles:<p>
<input id="valR" type="range" min="0" max="99" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
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


 
 Executing new job on Tue May 25 17:14:13 2021
 
 with the following command(s): 

which relion_preprocess --i CtfFind/job003/micrographs_ctf.star --coord_list AutoPick/job006/autopick.star --part_star Extract/job007/particles.star --part_dir Extract/job007/ --extract --extract_size 256 --float16  --scale 64 --norm --bg_radius 25 --white_dust -1 --black_dust -1 --invert_contrast   --pipeline_control Extract/job007/
 
 


