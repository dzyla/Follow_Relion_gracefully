---
title: job029
jobname: Refine3D/job029/
status: Succeeded
date: 2021-06-02 01:24:41
time: 01:24:41
jobtype: [Refine3D]
project: [230416_Relion4_precalc]
---

#### Job alias: None

#### Class Preview:
{{< plotly json="refine3d_job029_.json" height="600px" >}}
#### Class Projections:
{{< img src="Refine3D_job029_.png" >}}
#### Class Resolution [A]:
{{< plotly json="cls3d_res_refine3d_job029_.json" height="500px" >}}
#### Angular distribution matrix:
{{< img src="Refine3D_job029_angular_dist.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Wed Jun  2 09:04:21 2021
 
 with the following command(s): 

which relion_refine_mpi --o Refine3D/job029/run --auto_refine --split_random_halves --i Polish/job028/shiny.star --ref Refine3D/job025/run_half1_class001_unfil.mrc --ini_high 8 --dont_combine_weights_via_disc --scratch_dir /ssd/scheres --pool 3 --pad 2  --skip_gridding  --ctf --particle_diameter 200 --flatten_solvent --zero_mask --solvent_mask MaskCreate/job020/mask.mrc --solvent_correct_fsc  --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym D2 --low_resol_join_halves 40 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Refine3D/job029/
 
 

