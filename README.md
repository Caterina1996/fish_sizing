# Fish Sizing Pipeline 🐟

This repository contains a modular and automated pipeline for the detection, tracking, and 3D measurement of fish using stereo cameras. The system is designed with a **modular architecture**, allowing each sub-module or pipeline stage to be modified, upgraded, or replaced independently to adapt to different research needs.

## 🏗️ Pipeline Architecture

The pipeline is organized into distinct functional blocks that handle the data flow from raw input to final biometric aggregation:

### 1. Image Extraction
* **ROS Integration**: Capable of streaming and pairing stereo images directly from `.bag` files using customizable topics.
* **Image Sets**: Supports processing from pre-existing sets of stereo image pairs stored in local directories.

### 2. Rectification & Decimation
* **Stereo Rectification**: Aligns left and right images based on camera calibration (K, D, R, P matrices) to ensure epipolar geometry.
* **Decimation**: Adjustable image scaling to optimize processing speed without sacrificing measurement accuracy.

### 3. Fish Detection, Segmentation & Tracking
* **YOLOv11 Inference**: Utilizes a high-performance model to perform real-time detection, classification, and instance segmentation of fish.
* **Tracking**: Implements persistent tracking (e.g., BoT-SORT) to maintain fish identities across frames.

### 4. Image Processing & Enhancement
* **Modular Pipelines**: Includes a dedicated `ImageProcessor` to apply specific enhancements or filters to the rectified images before stereo matching.

### 5. Stereo Calculation & Pointcloud Generation
* **Disparity Mapping**: Uses Semi-Global Block Matching (SGBM) optimized for aquatic environments.
* **WLS Filtering**: Applies Weighted Least Squares filtering to produce high-density, low-noise disparity maps.
* **3D Reprojection**: Transforms disparity data into a structured 3D point cloud.

### 6. Pointcloud Filtering
* **Adaptive Outlier Removal**: Combines Statistical Outlier Removal (SOR) and HDBSCAN clustering to isolate the fish body from water noise or floating particles.
* **Geometric Clipping**: Uses PCA-based thickness analysis to remove ghost points and artifacts.

### 7. Fish Measurement
* **Curved Spine length**: Calculates fish length through polynomial fitting and mathematical arc length integration, accounting for body curvature.
* **Pose Estimation**: Computes Azimuth and Elevation angles to assess the fish's orientation relative to the camera.

### 8. Measures Aggregation
* **Data Synthesis**: Consolidates individual frame measurements into a master database.
* **Smart Filtering**: Generates cleaned-up reports by filtering detections based on quality metrics like border proximity, overlap, and aspect ratio.



## ⚙️ Operation Modes

The pipeline offers flexibility through modular skips and diverse inputs:
* **Standard**: Full execution from `.bag` extraction to CSV export.
* **Batch Processing**: Recursive processing of multiple directories with intelligent state management to skip already-processed files.
* **Modular Skips**: Use `--skip_inference` or `--skip_pc` to reuse previous results and speed up specific analysis stages.

## 📊 Generated Outputs

Results are organized systematically for every execution:
* **`results/`**: Master CSV files (`all_fish_info_raw.csv`, `resume_filtered_smart.csv`) and interactive 3D HTML plots.
* **`frame_XXXX/`**: Local data including the `FrameScene` pickle, full scene `.ply` files, and individual filtered fish point clouds.
* **`run_config.yaml`**: A complete audit trail of parameters, environment mappings, and execution statistics.

## 🐋 Docker Support

The project is fully containerized to ensure cross-platform compatibility and seamless GPU acceleration setup.

> [!IMPORTANT]
> **Detailed Docker Instructions**: Please refer to [DOCKER.md](./DOCKER_SETUP.md) for build and runtime configuration.

---
*Developed by Caterina Muntaner-Gonzalez as a part of the PhD research.