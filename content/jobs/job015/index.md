
    ---
    title: job015
    jobname: InitialModel/job015/
    status: Succeeded
    date: 2021-06-01 06:47:06
    time: 06:47:06
    categories: [InitialModel]
    ---
    
    #### Job alias: None
    
    #### Class 0:
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job015/cls3d_model0_InitialModel_job015_.json" height="600px" >}}
#### Class Projections:
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job015/cls3d_projection_InitialModel_job015_.json" height="800px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job015/cls3d_dist_InitialModel_job015_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job015/cls3d_res_InitialModel_job015_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job015/cls3d_psi_rot_InitialModel_job015_.json" height="500px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job015/cls3d_psi_tilt_InitialModel_job015_.json" height="500px" >}}
    
    #### Job command(s):
    
    
 
 Executing new job on Tue Jun  1 14:45:43 2021
 
 with the following command(s): 

which relion_refine --o InitialModel/job015/run --iter 100 --grad_write_iter 10  --grad --init_blobs --denovo_3dref  --i Select/job014/particles.star --ctf --K 1 --sym C1  --flatten_solvent  --zero_mask  --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 1  --skip_gridding  --particle_diameter 200 --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 4 --j 12 --gpu "4,5,6,7"  --pipeline_control InitialModel/job015/
rm -f InitialModel/job015/RELION_JOB_EXIT_SUCCESS

which relion_align_symmetry --i InitialModel/job015/run_it100_class001.mrc --o InitialModel/job015/symmetry_aligned.mrc --sym D2 --pipeline_control InitialModel/job015/

which relion_image_handler --i InitialModel/job015/symmetry_aligned.mrc --o InitialModel/job015/initial_model.mrc --sym D2 --pipeline_control InitialModel/job015/
touch InitialModel/job015/RELION_JOB_EXIT_SUCCESS
 
 

    
