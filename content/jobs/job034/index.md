
---
title: job034
jobname: InitialModel/job034/
status: Running
date: 2021-10-13 19:37:10
time: 19:37:10
categories: [InitialModel]
---

#### Job alias: None

#### Class 0:
{{< plotly json="/jobs/job034/cls3d_model0_InitialModel_job034_.json" height="600px" >}}
#### Class Projections:
{{< plotly json="/jobs/job034/cls3d_projection_InitialModel_job034_.json" height="800px" >}}
{{< plotly json="/jobs/job034/cls3d_dist_InitialModel_job034_.json" height="500px" >}}
{{< plotly json="/jobs/job034/cls3d_res_InitialModel_job034_.json" height="500px" >}}
{{< plotly json="/jobs/job034/cls3d_psi_rot_InitialModel_job034_.json" height="500px" >}}
{{< plotly json="/jobs/job034/cls3d_psi_tilt_InitialModel_job034_.json" height="500px" >}}

#### Job command(s):


 
 Executing new job on Wed Oct 13 19:27:25 2021
 
 with the following command(s): 

which relion_refine --o InitialModel/job034/run --iter 100 --grad --denovo_3dref  --i Class2D/job033/run_it200_data.star --ctf --K 1 --sym C1  --flatten_solvent  --zero_mask  --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 1  --skip_gridding  --particle_diameter 200 --oversampling 1  --healpix_order 1  --offset_range 6  --offset_step 2 --auto_sampling  --tau2_fudge 4 --j 16 --gpu "01"  --pipeline_control InitialModel/job034/
rm -f InitialModel/job034/RELION_JOB_EXIT_SUCCESS

which relion_align_symmetry --i InitialModel/job034/run_it100_model.star --o InitialModel/job034/initial_model.mrc --sym D2 --apply_sym --select_largest_class  --pipeline_control InitialModel/job034/
touch InitialModel/job034/RELION_JOB_EXIT_SUCCESS
 
 


