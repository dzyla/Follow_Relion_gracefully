
    ---
    title: job004
    jobname: ManualPick/job004/
    status: Succeeded
    date: 2021-05-25 08:50:33
    time: 08:50:33
    categories: [ManualPick]
    ---
    
    #### Job alias: None
    
    {{<rawhtml >}} 


    <div class="center">
    <p>Preview of random extracted particles:<p>
    <input id="valR" type="range" min="0" max="0" value="0" step="1" oninput="showVal(this.value)" onchange="showVal(this.value)" />
    <span id="range">0</span>
    <img id="img" width="800">
    </div>

    <script>

        var val = document.getElementById("valR").value;
            document.getElementById("range").innerHTML=val;
            document.getElementById("img").src = val + ".jpg";
            function showVal(newVal){
              document.getElementById("range").innerHTML=newVal;
              document.getElementById("img").src = newVal+ ".jpg";
            }
    </script>
    <br>
     {{< /rawhtml >}}
    
    #### Job command(s):
    
    
 
 Executing new job on Tue May 25 16:48:27 2021
 
 with the following command(s): 

which relion_manualpick --i CtfFind/job003/micrographs_ctf.star --odir ManualPick/job004/ --pickname manualpick --allow_save   --fast_save --selection ManualPick/job004/micrographs_selected.star --scale 0.25 --sigma_contrast 3 --black 0 --white 0 --topaz_denoise --topaz_exe /public/EM/RELION/topaz --particle_diameter 200  --pipeline_control ManualPick/job004/
 
 

 
 Executing new job on Tue May 25 16:50:11 2021
 
 with the following command(s): 

which relion_manualpick --i CtfFind/job003/micrographs_ctf.star --odir ManualPick/job004/ --pickname manualpick --allow_save   --fast_save --selection ManualPick/job004/micrographs_selected.star --scale 0.25 --sigma_contrast 3 --black 0 --white 0 --topaz_denoise --topaz_exe /public/EM/RELION/topaz --particle_diameter 200  --pipeline_control ManualPick/job004/
 
 

    
