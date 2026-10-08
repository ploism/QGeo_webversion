# Geology Quantifier

## COPYRIGTH
 Copyright 2023 Dr Pia Lois-Morales and Dr Kimie Suzuki

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

## Description
This repository contains the implementation of a basic geological quantifier that uses computer vision techniques to detect minerals in a rock sample and calculate rock statistics.

## Installation

The installation process is quite simple, just go to the *releases* section and click on the version of the software desired. After that, click on the `gui.exe` file to download the app.

Note that the `.exe` **only runs at windows based systems**.

## Usage
The workflow of this app is quite sequential, this because the user will need to experience a series of stages before the quantification takes place. Indeed, the stages can be summarized as follows:

**Stage 1: Selecting an image**

After launching the app, the user will find himself in front of a single button with "*Seleccionar imagen*" as label. If clicked, this button will open the OS file manager and will request the user to provide an image of a rock sample.

**Stage 2: Extracting the rock sample**

Once an image is provided, the app will show a new screen where the user will be able to configure and perform the extraction of the rock sample. 

The app will provide an **extraction interface** that will consist of a modifiable area overlaying the image. The user will be able to both translate and reshape this area by moving the vertices that define it. It's important to note that the extraction interface counts with 3 different modes, where each of them applies specific constraints to the area.

In this screen, the app will also provide a series of buttons, each belonging to a specific group. The buttons, divided by group, are as follows:

* **Commands & Image Transformations**
  1. **"Recortar"** : when clicked, this button cuts the area of the extraction interface and the app jumps to the next stage.
    
  2. **"Girar"** : this button allows the user to rotate the image in 90° counterclockwise.
   
  3. **"Rotar imagen L"** : if clicked, this button will trigger a small counterclockwise rotation of the image, i.e., a rotation to the left.
   
  4. **"Rotar imagen R"** : when clicked, the image will be slightly rotated clockwise, i.e., the app will perform a rotation to the right.
   
  5. **"Restablecer"** : this button will reset the image and extraction interface to the default configuration.
   
  6. **"Seleccionar todo"** : when clicked, this button will automatically move the vertices of the extraction interface in order to select the whole image.
   
   
* **Extraction Mode**
  1. **"Modo rectangular"** : this button allows the user to shift to the **rectangular** extraction mode, where the overlaying exrtaction area maintains it's signature rectangular aspect ratio when modified.
   
  2. **"Modo unwrapping"** : when clicked, this button will change the extraction mode of the extraction interface to **unwrapping**. In this mode, the user now has 6 vertices available instead of 4 (2 vertices at the top and bottom edges). The movement of any corner vertex will trigger an expansion/contraction of the whole area that will maintain the aspect ratio. In the other hand, if an edge vertex is moved, it will do it freely without affecting the position of the others.
   
  3. **"Modo libre"** : this button sets the extraction mode to **free**. This means that the usual 4 vertices will be able to move freely without affecting the rest.
   
* **Rock Sample Parametrization**
  1. **"altura"** : this button takes the height at the field next to it and sets it as the height of the rock sample to extract.

**Stage 3: Analyzing the rock sample**
Once the rock sample has been extracted from the image, the user moves to this stage, where all the analysis happens. The expected workflow consists in initially running a color segmentation where the user must provide the desired number of color clusters to find. The user will be able to merge, delete or even **analyze** (calculating statistics) this clusters. To achieve this workflow, this stage offers the following 3 groups of buttons:

* **Color Segmentation**
  1. **"Separar"** : when clicked, this button takes the amount of desired clusters and runs K-Means as a color segmentation algorithm. The resulting clusters are showed at the GUI as images and can be selected for further processing.
  2. **"Combinar"** : when clicked, this button makes the app take all the selected clusters and merge them into one single cluster.
  3. **"Eliminar** : this button triggers a deletion of all the selected clusters.
* **Image Modification**
  1. **"Actualizar"** : when clicked, this button updates all the images in the GUI.
  2. **"Deshacer"** : this button resets the screen, taking the GUI to it's initial form (all processing done is deleted).
* **Analysis & Results Generation**
  1. **"3D"** : when clicked, this button launches a window with a 3D visualization of the rock sample.
  2. **"Analizar"** : if this button is pressed, the app will run an analysis over the selected cluster/image. Although the state of the "*Segmentar*" toggle button may make variate a bit the analysis, the statistics calculated in it will be basically the same.
  3. **"Segmentar"** : this toggle button allows the user to decide whether to detect shapes in the clusters/images that will be analyzed. If the button is off, there won't be any shape detection and all the statistics will be calculated over the cluster as a whole. In the other hand, if the button is on, a shape detection will be performed and the statistics of the analysis will be calculated for each of them.
