---
title: job001
jobname: ImportTomo/job001/
status: Succeeded
date: 2023-04-11 21:31:43
time: 21:31:43
jobtype: [ImportTomo]
project: [relion40_sta_tutorial_data]
---

#### Job alias: None

### Imported tomo files:
* TS_01
* TS_03
* TS_43
* TS_45
* TS_54
### Projected tomogram tomograms/TS_03/03.mrc preview
{{< rawhtml >}} 

            <div class="center">
            <p>Tilt series preview:<p>
            <input id="valR_0" type="range" min="0" max="39" value="0" step="1" oninput="showVal_0(this.value)" onchange="showVal_0(this.value)" />
            <span id="range_0">0</span>
            <img id="img_0" width="350">
            </div>

            <script>

                function showVal_0(newVal){
                    document.getElementById("range_0").innerHTML=newVal;
                    document.getElementById("img_0").src = "0_"+ newVal + ".jpg";
                }

                // Initialize the slider and image on page load
                var initial_val_0 = document.getElementById("valR_0").value;
                showVal_0(initial_val_0);

            </script>
            <br>
             {{< /rawhtml >}}
### Projected tomogram tomograms/TS_43/43.mrc preview
{{< rawhtml >}} 

            <div class="center">
            <p>Tilt series preview:<p>
            <input id="valR_1" type="range" min="0" max="40" value="0" step="1" oninput="showVal_1(this.value)" onchange="showVal_1(this.value)" />
            <span id="range_1">0</span>
            <img id="img_1" width="350">
            </div>

            <script>

                function showVal_1(newVal){
                    document.getElementById("range_1").innerHTML=newVal;
                    document.getElementById("img_1").src = "1_"+ newVal + ".jpg";
                }

                // Initialize the slider and image on page load
                var initial_val_1 = document.getElementById("valR_1").value;
                showVal_1(initial_val_1);

            </script>
            <br>
             {{< /rawhtml >}}
### Projected tomogram tomograms/TS_03/03.mrc preview
{{< rawhtml >}} 

            <div class="center">
            <p>Tilt series preview:<p>
            <input id="valR_2" type="range" min="0" max="39" value="0" step="1" oninput="showVal_2(this.value)" onchange="showVal_2(this.value)" />
            <span id="range_2">0</span>
            <img id="img_2" width="350">
            </div>

            <script>

                function showVal_2(newVal){
                    document.getElementById("range_2").innerHTML=newVal;
                    document.getElementById("img_2").src = "2_"+ newVal + ".jpg";
                }

                // Initialize the slider and image on page load
                var initial_val_2 = document.getElementById("valR_2").value;
                showVal_2(initial_val_2);

            </script>
            <br>
             {{< /rawhtml >}}

#### Job command(s):

```bash

 
 Executing new job on Tue Apr 11 21:31:43 2023
 
 with the following command(s): 
relion_tomo_import_tomograms  --i input/tomograms_descr.star --o ImportTomo/job001/tomograms.star --angpix 1.35 --voltage 300 --Cs 2.7 --Q0 0.07 --ol input/order_list.csv --flipYZ  --flipZ  --hand -1  --pipeline_control ImportTomo/job001/
 
 


