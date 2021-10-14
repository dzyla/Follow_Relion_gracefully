
---
title: job019
jobname: Refine3D/job019/
status: Succeeded
date: 2021-06-01 08:25:55
time: 08:25:55
categories: [Refine3D]
---

#### Job alias: None

{{< plotly json="/jobs/job019/refine3d_model_Refine3D_job019_.json" height="800px" >}}
{{<rawhtml >}} 

        <div class="center">
        <p>Volume projections preview:<p>
        <input id="valR" type="range" min="0" max="35" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
        <span id="range">0</span>
        <img id="img" width="250">
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
#### Class Projections:
{{< plotly json="/jobs/job019/cls3d_projection_Refine3D_job019_.json" height="800px" >}}
{{< plotly json="/jobs/job019/cls3d_res_Refine3D_job019_.json" height="500px" >}}
{{< plotly json="/jobs/job019/cls3d_psi_rot_Refine3D_job019_.json" height="500px" >}}
{{< plotly json="/jobs/job019/cls3d_psi_tilt_Refine3D_job019_.json" height="500px" >}}

#### Job command(s):


 
 Executing new job on Tue Jun  1 16:19:22 2021
 
 with the following command(s): 

which relion_refine_mpi --o Refine3D/job019/run --auto_refine --split_random_halves --i Extract/job018/particles.star --ref Class3D/job016/run_it025_class004_box256.mrc --firstiter_cc --ini_high 50 --dont_combine_weights_via_disc --scratch_dir /ssd/scheres --pool 3 --pad 2  --skip_gridding  --auto_ignore_angles --auto_resol_angles --ctf --particle_diameter 200 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym D2 --low_resol_join_halves 40 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Refine3D/job019/
 
 


