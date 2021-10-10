
    ---
    title: job008
    jobname: Class2D/job008/
    status: Succeeded
    date: 2021-05-25 09:27:23
    time: 09:27:23
    categories: [Class2D]
    ---
    
    #### Job alias: None
    
    {{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job008/cls2d1_Class2D_job008_Class2D_job008_.json" height="800px" >}}
{{<rawhtml >}} 

    <div class="center">
    <p>Class preview:<p>
    <input id="valR" type="range" min="1" max="50" value="1" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="250">
    </div>

    <script>

        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + 1 + ".jpg";
            function showVal(newVal){
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }
    </script>
    <br>
     {{< /rawhtml >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job008/cls2d_dist_Class2D_job008_Class2D_job008_.json" height="600px" >}}
{{< plotly json="https://dzyla.github.io/Follow_Relion_gracefully/jobs/job008/cls2d_res_Class2D_job008_Class2D_job008_.json" height="600px" >}}
    
    #### Job command(s):
    
    
 
 Executing new job on Tue May 25 17:24:51 2021
 
 with the following command(s): 

which relion_refine_mpi --o Class2D/job008/run --i Extract/job007/particles.star --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --ctf  --iter 25 --tau2_fudge 2 --particle_diameter 200 --K 50 --flatten_solvent  --zero_mask  --center_classes  --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale  --j 6 --gpu "0:1:2:3"  --pipeline_control Class2D/job008/
 
 

    
