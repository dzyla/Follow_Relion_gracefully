---
title: job025
jobname: Refine3D/job025/
status: Succeeded
date: 2023-04-16 15:33:29
time: 15:33:29
jobtype: [Refine3D]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

#### Class Preview:
{{< plotly json="refine3d_job025_.json" height="600px" >}}
#### Class Projections:
{{< img src="Refine3D_job025_.png" >}}
#### Class Resolution [A]:
{{< plotly json="cls3d_res_refine3d_job025_.json" height="500px" >}}
#### Angular distribution matrix:
{{< img src="Refine3D_job025_angular_dist.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Tue Jun  1 16:36:38 2021
 
 with the following command(s): 

which relion_refine_mpi --o Refine3D/job025/run --auto_refine --split_random_halves --i CtfRefine/job024/particles_ctf_refine.star --ref Refine3D/job019/run_class001.mrc --ini_high 50 --dont_combine_weights_via_disc --scratch_dir /ssd/scheres --pool 3 --pad 2  --skip_gridding  --auto_ignore_angles --auto_resol_angles --ctf --particle_diameter 200 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym D2 --low_resol_join_halves 40 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Refine3D/job025/
 
 

 
 Executing new job on Tue Jun  1 16:36:46 2021
 
 with the following command(s): 

which relion_refine_mpi --o Refine3D/job025/run --auto_refine --split_random_halves --i CtfRefine/job024/particles_ctf_refine.star --ref Refine3D/job019/run_class001.mrc --ini_high 50 --dont_combine_weights_via_disc --scratch_dir /ssd/scheres --pool 3 --pad 2  --skip_gridding  --auto_ignore_angles --auto_resol_angles --ctf --particle_diameter 200 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym D2 --low_resol_join_halves 40 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Refine3D/job025/
 
 


