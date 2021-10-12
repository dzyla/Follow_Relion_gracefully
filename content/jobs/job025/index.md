
---
title: job025
jobname: Refine3D/job025/
status: Succeeded
date: 2021-06-01 08:44:51
time: 08:44:51
categories: [Refine3D]
---

#### Job alias: None

{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job025/refine3d_model_Refine3D_job025_.json" height="800px" >}}
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
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job025/cls3d_projection_Refine3D_job025_.json" height="800px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job025/cls3d_res_Refine3D_job025_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job025/cls3d_psi_rot_Refine3D_job025_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job025/cls3d_psi_tilt_Refine3D_job025_.json" height="500px" >}}

#### Job command(s):


 
 Executing new job on Tue Jun  1 16:36:38 2021
 
 with the following command(s): 

which relion_refine_mpi --o Refine3D/job025/run --auto_refine --split_random_halves --i CtfRefine/job024/particles_ctf_refine.star --ref Refine3D/job019/run_class001.mrc --ini_high 50 --dont_combine_weights_via_disc --scratch_dir /ssd/scheres --pool 3 --pad 2  --skip_gridding  --auto_ignore_angles --auto_resol_angles --ctf --particle_diameter 200 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym D2 --low_resol_join_halves 40 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Refine3D/job025/
 
 

 
 Executing new job on Tue Jun  1 16:36:46 2021
 
 with the following command(s): 

which relion_refine_mpi --o Refine3D/job025/run --auto_refine --split_random_halves --i CtfRefine/job024/particles_ctf_refine.star --ref Refine3D/job019/run_class001.mrc --ini_high 50 --dont_combine_weights_via_disc --scratch_dir /ssd/scheres --pool 3 --pad 2  --skip_gridding  --auto_ignore_angles --auto_resol_angles --ctf --particle_diameter 200 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym D2 --low_resol_join_halves 40 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Refine3D/job025/
 
 


