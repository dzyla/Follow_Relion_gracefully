---
title: job035
jobname: Class3D/job035/
status: Succeeded
date: 2023-04-09 20:35:45
time: 20:35:45
jobtype: [Class3D]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

#### Combined Classes:
{{< plotly json="cls3d_combined_class3d_job035_.json" height="400px" >}}
#### Class Projections:
{{< img src="Class3D_job035_.png" >}}
#### Class distribution:
{{< plotly json="cls3d_dist_class3d_job035_.json" height="500px" >}}
#### Class Resolution [A]:
{{< plotly json="cls3d_res_class3d_job035_.json" height="500px" >}}
#### Angular distribution matrix:
{{< img src="Class3D_job035_angular_dist.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Sun Apr  9 20:08:10 2023
 
 with the following command(s): 

which relion_refine --o Class3D/job035/run --i Refine3D/job029/run_data.star --ref Refine3D/job029/run_class001.mrc --ini_high 10 --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --ctf --iter 50 --tau2_fudge 4 --particle_diameter 200 --K 4 --flatten_solvent --zero_mask --solvent_mask MaskCreate/job020/mask.mrc --oversampling 1 --healpix_order 4 --sigma_ang 1.66667 --offset_range 5 --offset_step 2 --sym D2 --norm --scale  --j 12 --gpu ""  --pipeline_control Class3D/job035/
 
 


