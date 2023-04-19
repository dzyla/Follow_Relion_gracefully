---
title: job027
jobname: Polish/job027/
status: Succeeded
date: 2021-06-01 10:05:13
time: 10:05:13
jobtype: [Polish]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

### Optimal parameters
--s_vel 0.4035 --s_div 1155 --s_acc 2.715

#### Job command(s):

```bash

 
 Executing new job on Tue Jun  1 17:01:55 2021
 
 with the following command(s): 

which relion_motion_refine --i Refine3D/job025/run_data.star --f PostProcess/job026/postprocess.star --corr_mic MotionCorr/job002/corrected_micrographs.star --first_frame 1 --last_frame -1 --o Polish/job027/ --float16  --min_p 3500 --eval_frac 0.5 --align_frac 0.5 --params3  --j 16  --pipeline_control Polish/job027/
 
 


