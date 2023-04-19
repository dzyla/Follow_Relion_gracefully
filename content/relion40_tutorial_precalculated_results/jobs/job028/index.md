---
title: job028
jobname: Polish/job028/
status: Succeeded
date: 2021-06-02 01:01:03
time: 01:01:03
jobtype: [Polish]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

### Polish job statistics
{{< img src="Polish_job028_.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Wed Jun  2 08:42:09 2021
 
 with the following command(s): 

which relion_motion_refine --i Refine3D/job025/run_data.star --f PostProcess/job026/postprocess.star --corr_mic MotionCorr/job002/corrected_micrographs.star --first_frame 1 --last_frame -1 --o Polish/job028/ --float16  --params_file Polish/job027/opt_params_all_groups.txt --combine_frames --bfac_minfreq 20 --bfac_maxfreq -1 --j 16  --pipeline_control Polish/job028/
 
 


