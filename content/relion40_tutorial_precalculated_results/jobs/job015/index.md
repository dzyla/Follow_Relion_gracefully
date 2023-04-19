---
title: job015
jobname: InitialModel/job015/
status: Succeeded
date: 2021-06-01 06:47:06
time: 06:47:06
jobtype: [InitialModel]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

#### Combined Classes:
{{< plotly json="cls3d_combined_initialmodel_job015_.json" height="400px" >}}
#### Class Projections:
{{< img src="InitialModel_job015_.png" >}}
#### Class distribution:
{{< plotly json="cls3d_dist_initialmodel_job015_.json" height="500px" >}}
#### Class Resolution [A]:
{{< plotly json="cls3d_res_initialmodel_job015_.json" height="500px" >}}
#### Angular distribution matrix:
{{< img src="InitialModel_job015_angular_dist.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Tue Jun  1 14:45:43 2021
 
 with the following command(s): 

which relion_refine --o InitialModel/job015/run --iter 100 --grad_write_iter 10  --grad --init_blobs --denovo_3dref  --i Select/job014/particles.star --ctf --K 1 --sym C1  --flatten_solvent  --zero_mask  --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 1  --skip_gridding  --particle_diameter 200 --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 4 --j 12 --gpu "4,5,6,7"  --pipeline_control InitialModel/job015/
rm -f InitialModel/job015/RELION_JOB_EXIT_SUCCESS

which relion_align_symmetry --i InitialModel/job015/run_it100_class001.mrc --o InitialModel/job015/symmetry_aligned.mrc --sym D2 --pipeline_control InitialModel/job015/

which relion_image_handler --i InitialModel/job015/symmetry_aligned.mrc --o InitialModel/job015/initial_model.mrc --sym D2 --pipeline_control InitialModel/job015/
touch InitialModel/job015/RELION_JOB_EXIT_SUCCESS
 
 


