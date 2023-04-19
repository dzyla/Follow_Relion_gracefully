---
title: job003
jobname: CtfFind/job003/
status: Succeeded
date: 2021-05-25 07:30:05
time: 07:30:05
jobtype: [CtfFind]
project: [relion40_tutorial_precalculated_results]
---

#### Job alias: None

### Defocus per micrograph
{{< img src="CtfFind_job003_index.png" >}}
### Max resolution per micrograph
{{< img src="CtfFind_job003_res.png" >}}
### Defocus histogram
{{< img src="CtfFind_job003_def_hist.png" >}}
### Astigmatism histogram
{{< img src="CtfFind_job003_ast_hist.png" >}}
### Max resolution histogram
{{< img src="CtfFind_job003_res_hist.png" >}}
### Defocus / Max resolution histogram
{{< img src="CtfFind_job003_def_res.png" >}}
### Defocus / Figure of merit histogram
{{< img src="CtfFind_job003_def_fom.png" >}}

#### Job command(s):

```bash

 
 Executing new job on Tue May 25 15:30:00 2021
 
 with the following command(s): 

which relion_run_ctffind_mpi --i MotionCorr/job002/corrected_micrographs.star --o CtfFind/job003/ --Box 512 --ResMin 30 --ResMax 5 --dFMin 5000 --dFMax 50000 --FStep 500 --dAst 100 --ctffind_exe /public/EM/ctffind/ctffind.exe --ctfWin -1 --is_ctffind4  --fast_search  --use_given_ps   --pipeline_control CtfFind/job003/
 
 


