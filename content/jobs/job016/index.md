
---
title: job016
jobname: Class3D/job016/
status: Succeeded
date: 2021-06-01 07:23:39
time: 07:23:39
categories: [Class3D]
---

#### Job alias: None

#### Class 0:
{{< plotly json="/jobs/job016/cls3d_model0_Class3D_job016_.json" height="600px" >}}
#### Class 1:
{{< plotly json="/jobs/job016/cls3d_model1_Class3D_job016_.json" height="600px" >}}
#### Class 2:
{{< plotly json="/jobs/job016/cls3d_model2_Class3D_job016_.json" height="600px" >}}
#### Class 3:
{{< plotly json="/jobs/job016/cls3d_model3_Class3D_job016_.json" height="600px" >}}
#### Class Projections:
{{< plotly json="/jobs/job016/cls3d_projection_Class3D_job016_.json" height="800px" >}}
{{< plotly json="/jobs/job016/cls3d_dist_Class3D_job016_.json" height="500px" >}}
{{< plotly json="/jobs/job016/cls3d_res_Class3D_job016_.json" height="500px" >}}
{{< plotly json="/jobs/job016/cls3d_psi_rot_Class3D_job016_.json" height="500px" >}}
{{< plotly json="/jobs/job016/cls3d_psi_tilt_Class3D_job016_.json" height="500px" >}}

#### Job command(s):


 
 Executing new job on Tue Jun  1 14:47:32 2021
 
 with the following command(s): 

which relion_refine_mpi --o Class3D/job016/run --i Select/job014/particles.star --ref InitialModel/job015/initial_model.mrc --ini_high 50 --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --skip_gridding  --ctf --iter 25 --tau2_fudge 4 --particle_diameter 200 --K 4 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 --sym C1 --norm --scale  --j 6 --gpu "4:5:6:7"  --pipeline_control Class3D/job016/
 
 


