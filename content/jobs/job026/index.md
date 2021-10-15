
---
title: job026
jobname: PostProcess/job026/
status: Succeeded
date: 2021-06-01 08:59:24
time: 08:59:24
categories: [PostProcess]
---

#### Job alias: None

{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job026/postprocess_PostProcess_job026_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job026/postprocess1_PostProcess_job026_.json" height="500px" >}}
{{<rawhtml >}} 

            <div class="center">
            <p>Masked Volume slices preview:<p>
            <input id="valR" type="range" min="0" max="255" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
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


 
 Executing new job on Tue Jun  1 16:59:13 2021
 
 with the following command(s): 

which relion_postprocess --mask MaskCreate/job020/mask.mrc --i Refine3D/job025/run_half1_class001_unfil.mrc --o PostProcess/job026/postprocess  --angpix 1.244 --mtf mtf_k2_200kV.star --mtf_angpix 0.885 --auto_bfac  --autob_lowres 10  --pipeline_control PostProcess/job026/
 
 


