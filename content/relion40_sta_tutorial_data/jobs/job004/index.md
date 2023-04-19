---
title: job004
jobname: InitialModel/job004/
status: Aborted
date: 2023-04-11 22:58:12
time: 22:58:12
jobtype: [InitialModel]
project: [relion40_sta_tutorial_data]
---

#### Job alias: None

#### Combined Classes:
{{< plotly json="cls3d_combined_initialmodel_job004_.json" height="400px" >}}
#### Class Projections:
{{< img src="InitialModel_job004_.png" >}}
#### Class distribution:
{{< plotly json="cls3d_dist_initialmodel_job004_.json" height="500px" >}}
#### Class Resolution [A]:
{{< plotly json="cls3d_res_initialmodel_job004_.json" height="500px" >}}
#### Angular distribution matrix:
{{< img src="InitialModel_job004_angular_dist.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Tue Apr 11 22:08:19 2023
 
 with the following command(s): 

which relion_refine --o InitialModel/job004/run --iter 100 --grad --denovo_3dref  --i PseudoSubtomo/job003/particles.star --ctf --K 1 --sym C1  --flatten_solvent  --zero_mask  --dont_combine_weights_via_disc --pool 3 --pad 1  --particle_diameter 230 --oversampling 1  --healpix_order 1  --offset_range 6  --offset_step 2 --auto_sampling  --tau2_fudge 4 --j 8 --gpu ""  --pipeline_control InitialModel/job004/
rm -f InitialModel/job004/RELION_JOB_EXIT_SUCCESS

which relion_align_symmetry --i InitialModel/job004/run_it100_model.star --o InitialModel/job004/initial_model.mrc --sym C6 --apply_sym --select_largest_class  --pipeline_control InitialModel/job004/
touch InitialModel/job004/RELION_JOB_EXIT_SUCCESS
 
 


