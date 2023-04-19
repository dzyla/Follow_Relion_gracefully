---
title: job005
jobname: ReconstructParticleTomo/job005/
status: Succeeded
date: 2023-04-15 00:13:15
time: 00:13:15
jobtype: [ReconstructParticleTomo]
project: [relion40_sta_tutorial_data]
---

#### Job alias: None

### Reconstructed particles:
{{< plotly json="reconstructed_tomoreconstructparticletomo_job005_.json" height="600px" >}}
### Reconstructed particles projection:
{{< img src="ReconstructParticleTomo_job005_.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Fri Apr 14 23:54:11 2023
 
 with the following command(s): 

which relion_tomo_reconstruct_particle_mpi --i PseudoSubtomo/job003/optimisation_set.star --theme classic --o ReconstructParticleTomo/job005/ --b 192 --crop 96 --bin 4 --j 4 --j_out 4 --j_in 1  --sym C1  --pipeline_control ReconstructParticleTomo/job005/
 
 


