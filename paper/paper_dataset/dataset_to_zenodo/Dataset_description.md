# FishSizing: Stereo Vision Dataset for Non-Invasive Underwater Fish Length Estimation

## Overview
This dataset provides stereoscopic recordings and biological ground-truth data designed for the development and evaluation of underwater Computer Vision models. It is the official dataset associated with the paper: **"Autonomous underwater stereo vision system for non-invasive fish length estimation in marine environments"**.

The data encompasses varied underwater scenarios, featuring several marine species (European Seabass, Atlantic Mackerel, Bogue, and Blue Whiting) recorded under different conditions. Due to the large file size of the raw data, the dataset is structured into multiple volumes linked within this Zenodo Community.

**This specific repository (Core & Metadata) contains:**
* Global Ground-Truth physical measurements (CSV format).
* Detailed bagfile summary indexes containing frame rates, durations, and present identities.
* Stereo camera calibration parameters (`left.yaml` and `right.yaml`) for 3D reconstruction from stereo_pairs.
* Reference image gallery of the individual fishes.

## Dataset Structure
The dataset is logically divided into two main subsets based on the recording sessions and experimental complexity:
* **Dataset A:** Contains sequences featuring single and multiple fish recorded post-mortem simulating free-swimming behaviour, including *Dicentrarchus labrax*, *Scomber scombrus*, *Boops boops*, and *Micromesistius poutassou*.
* **Dataset B:** Contains sequences of multiple *D. labrax* individuals swimming simultaneously, visually identifiable via physical tags (Red Tag, Black Tag, Unmarked). 

## Ground Truth Methodology
Physical measurements (Total Length, $L_T$) of all participating fish were taken manually prior to their immersion in the experimental tanks. These absolute measurements are mapped to their visual identities in the provided `fish_gt_measurements.csv` file, allowing evaluation of any methodology that wishes to use the dataset.


## Tracking Annotations and Model Weights
To facilitate reproducibility and establish a benchmark for future research, this repository includes our extended ground truth for visual tracking:
* **`model_tracking_annotations.json`**: A dictionary mapping specific video frames to the physical identity of the fish (e.g., matching a visual track in bagfile X to "Fish 2" or "Red Tag"). These annotations correspond to the tracking outputs evaluated in our study.
* **Pre-trained Models:** We provide the model weights used in our pipeline within the `model/` directory, allowing researchers to deploy or fine-tune our approach directly.


## Camera Calibration
The calibration matrices (Intrinsic matrix $K$, Distortion coefficients $D$, Rectification $R$, and Projection $P$) provided in the `calibrations/` directory are calculated for the **full-resolution** images stored in the raw ROS bagfiles. 


## Raw Data Access (ROS Bagfiles)
The heavy raw data (`.bag` files containing synchronized left and right image topics) are hosted in linked Zenodo repositories within this community to bypass file size limitations. Please refer to the "Related works" section below to download the specific raw data volumes.

To decompress the data or test our algorithm refer to our github for guidance: https://github.com/Caterina1996/fish_sizing.git

## Citation
If you use this dataset, metadata, or associated code in your research, please cite our paper:
> *(Note: Pending to update)* > [Muntaner-Gonzalez, Caterina, Martin-Abadal Miguel, Gonzalez-Cid Yolanda]. "Autonomous underwater stereo vision system for non-invasive fish length estimation in marine environments". [TO BE UPDATED], 2026.

## License
This dataset is released under the **Creative Commons Attribution 4.0 International (CC-BY 4.0)** license. You are free to share and adapt the material for any purpose, even commercially, as long as appropriate credit is given to the original authors.

## Animal welfare statement

All animal care procedures were approved by the Animal Experimentation Ethics Committee of the University of the Balearic Islands (CEEA-UIB, ref. 254-03-25) and authorized by the Animal Health and Welfare Service of the General Directorate for Agriculture, Livestock and Rural Development of the Government of the Balearic Islands (exp. SSBA 11/2025 AEXP). All procedures were carried out by trained and competent personnel, in accordance with European Directive 2010/63/EU and Spanish Royal Decree RD53/2013 to ensure good practices for animal care, health and welfare.
