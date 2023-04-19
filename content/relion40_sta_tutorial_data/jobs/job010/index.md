---
title: job010
jobname: Class3D/job010/
status: Aborted
date: 2023-04-15 20:57:24
time: 20:57:24
jobtype: [Class3D]
project: [relion40_sta_tutorial_data]
---

#### Job alias: None

#### Combined Classes:
{{< plotly json="cls3d_combined_class3d_job010_.json" height="400px" >}}
#### Class Projections:
{{< img src="Class3D_job010_.png" >}}
#### Class distribution:
{{< plotly json="cls3d_dist_class3d_job010_.json" height="500px" >}}
#### Class Resolution [A]:
{{< plotly json="cls3d_res_class3d_job010_.json" height="500px" >}}
#### Angular distribution matrix:
{{< img src="Class3D_job010_angular_dist.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Sat Apr 15 20:54:58 2023
 
 with the following command(s): 

which relion_refine_mpi --o Class3D/job010/run --i PseudoSubtomo/job003/particles.star --ref ReconstructParticleTomo/job005/half1.mrc --firstiter_cc --ini_high 20 --dont_combine_weights_via_disc --pool 3 --pad 2  --ctf --iter 15 --tau2_fudge 4 --particle_diameter 230 --K 1 --flatten_solvent --zero_mask --solvent_mask MaskCreate/job006/mask.mrc --oversampling 1 --healpix_order 1 --offset_range 5 --offset_step 2 --allow_coarser_sampling --sym C6 --norm --scale  --j 4 --gpu ""  --pipeline_control Class3D/job010/
 
 


