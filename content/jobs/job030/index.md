
---
title: job030
jobname: PostProcess/job030/
status: Succeeded
date: 2021-06-02 01:28:32
time: 01:28:32
categories: [PostProcess]
---

#### Job alias: None

{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job030/postprocess_PostProcess_job030_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job030/postprocess1_PostProcess_job030_.json" height="500px" >}}
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


 
 Executing new job on Wed Jun  2 09:28:20 2021
 
 with the following command(s): 

which relion_postprocess --mask MaskCreate/job020/mask.mrc --i Refine3D/job029/run_half1_class001_unfil.mrc --o PostProcess/job030/postprocess  --angpix 1.244 --mtf mtf_k2_200kV.star --mtf_angpix 0.885 --auto_bfac  --autob_lowres 10  --pipeline_control PostProcess/job030/
 
 


