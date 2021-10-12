
---
title: job029
jobname: Refine3D/job029/
status: Succeeded
date: 2021-06-02 01:24:41
time: 01:24:41
categories: [Refine3D]
---

#### Job alias: None

{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job029/refine3d_model_Refine3D_job029_.json" height="800px" >}}
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
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job029/cls3d_projection_Refine3D_job029_.json" height="800px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job029/cls3d_res_Refine3D_job029_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job029/cls3d_psi_rot_Refine3D_job029_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job029/cls3d_psi_tilt_Refine3D_job029_.json" height="500px" >}}

#### Job command(s):


 
 Executing new job on Wed Jun  2 09:04:21 2021
 
 with the following command(s): 

which relion_refine_mpi --o Refine3D/job029/run --auto_refine --split_random_halves --i Polish/job028/shiny.star --ref Refine3D/job025/run_half1_class001_unfil.mrc --ini_high 8 --dont_combine_weights_via_disc --scratch_dir /ssd/scheres --pool 3 --pad 2  --skip_gridding  --ctf --particle_diameter 200 --flatten_solvent --zero_mask --solvent_mask MaskCreate/job020/mask.mrc --solvent_correct_fsc  --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym D2 --low_resol_join_halves 40 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Refine3D/job029/
 
 


