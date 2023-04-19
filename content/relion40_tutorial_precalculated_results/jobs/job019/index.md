---
title: job019
jobname: Refine3D/job019/
status: Succeeded
date: 2021-06-01 08:25:55
time: 08:25:55
jobtype: [Refine3D]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

#### Class Preview:
{{< plotly json="refine3d_job019_.json" height="600px" >}}
#### Class Projections:
{{< img src="Refine3D_job019_.png" >}}
#### Class Resolution [A]:
{{< plotly json="cls3d_res_refine3d_job019_.json" height="500px" >}}
#### Angular distribution matrix:
{{< img src="Refine3D_job019_angular_dist.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Tue Jun  1 16:19:22 2021
 
 with the following command(s): 

which relion_refine_mpi --o Refine3D/job019/run --auto_refine --split_random_halves --i Extract/job018/particles.star --ref Class3D/job016/run_it025_class004_box256.mrc --firstiter_cc --ini_high 50 --dont_combine_weights_via_disc --scratch_dir /ssd/scheres --pool 3 --pad 2  --skip_gridding  --auto_ignore_angles --auto_resol_angles --ctf --particle_diameter 200 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym D2 --low_resol_join_halves 40 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Refine3D/job019/
 
 


