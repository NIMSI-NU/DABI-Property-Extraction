# DABI-Property-Extraction
Codes related to solving the inverse elastoplastic and creep problems for the Dimple Array Bulge Instrument (DABI)

## Data PostProcessing
Includes files used for pressure alignment (between DIC data and pressure test data) and files for DIC post-processing. 

The post-processing files were used to perform rigid body motion removal, fit the data to a Gaussian surface, visualize the aligned pressure displacement data, and select frames to be used for inverse extraction. The "DIC_Data_Postprocessing_D1.ipynb" script was ran using Google Collab and requires the DIC data to run.

The pressure alignment script code can be run using the file "Pressure_alignment_HT250818.py" with the additional python files containing functions used for alignment. These files were ran using a local system with python.

## Mesh Generation
This folder contains the two main scripts which were used to generate the meshes for this project.

The first script "Dimple_thickness_calculation.ipynb" is used to visualize the dimple thickness and back surface scans of the manufactured dimple plate provided in text files. This script was ran using Google Collab. Curve fitting is performed to approximate the dimple center of the point cloud using an ellipsoidal approximation and the profile thickness is determined and reported to generate a new geometry in Abaqus.

The second script "abaqus_new_mesh_generation.py" was run in the Abaqus GUI to generate the new geometries for each dimple with the measure profile thickness. The node sets, surface definitions, and meshes needed for the JAX-FEM inverse code are also generated using this script.

## Static Extraction
The following folder contains the script required to perform the inverse static extraction using the script located in "Static_Extraction/inverse_code/InverseExtraction_GaussianFit_PlasticityVoce_HT250818.py". The HT_250818 folder contains the mesh files and the Gaussian fit data which can be used to perform the inverse extraction.
It is recommended to run this code using a machine with an NVIDIA GPU and the latest version of JAX-FEM must be installed: https://github.com/deepmodeling/jax-fem
The Static_Extraction folder can be directly placed into the jax-fem/applications folder as is when running.
This code uses the pypardiso solver which also requires this library to be installed: https://pypi.org/project/pypardiso/

## Creep Extraction
The creep extraction folder contains the 3D models needed to perform the creep property extraction; an ABAQUS installation is required to run the codes.

## Thermal and Pressure Interface Scripts
Contains codes needed to plot DABI data, merge and synchronize DABI output files, and perform emissivity calibration via IR images.
