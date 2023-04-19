---
title: job021
jobname: PostProcess/job021/
status: Succeeded
date: 2021-06-01 08:27:30
time: 08:27:30
jobtype: [PostProcess]
project: [230416_Relion4_precalc]
---

#### Job alias: None

### Reported resolution @ FSC=0.143: __3.03 Å__
### Reported resolution @ FSC=0.5: __3.54 Å__
### FSC curve
{{< img src="PostProcess_job021_FSC.png" >}}
### Gunier plot
{{< img src="PostProcess_job021_Gunier.png" >}}
{{< rawhtml >}} 

            <div class="center">
            <p>Masked Volume slices preview:<p>
            <input id="valR" type="range" min="0" max="35" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
            <span id="range">0</span>
            <img id="img" width="350">
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

 
 Executing new job on Tue Jun  1 16:27:18 2021
 
 with the following command(s): 

which relion_postprocess --mask MaskCreate/job020/mask.mrc --i Refine3D/job019/run_half1_class001_unfil.mrc --o PostProcess/job021/postprocess  --angpix 1.244 --mtf mtf_k2_200kV.star --mtf_angpix 0.885 --auto_bfac  --autob_lowres 10  --pipeline_control PostProcess/job021/
 
 


