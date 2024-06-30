# GaussFusion: Seamless 3D Object Integration using Gaussian Splatting
**Computer Vision Final Project on 3D Gaussian Splatting**

This project provides a streamlined pipeline to create a Gaussian Splatting model from two types of data sources: a 3D model of an object (downloaded from the internet) and a video scan of your environment. Our toolkit **facilitates easy preprocessing** of data into a format compatible with the Gaussian Splatting algorithm. Additionally, we offer **an intuitive and interactive visualizer** to merge and view the object in a new background, allowing for easy navigation and control over the object's relative position and scale.

## 🌟 Visualizer Features

1. 🔮 **Multi-Model Integration**: Supports trained Gaussian Splatting models in `ply` format, compatible with various 3DGS training repositories.

2. 🎨 **Multi-Model Support**: Visualize one or two Gaussian Splatting models simultaneously.

3. 🕹️ **Interactive Object Manipulation**: Easily adjust position, scale, and orientation of objects in real-time for perfect scene composition.

4. 🖥️ **Adaptive Rendering**: Toggle between different resolutions on-the-fly, balancing visual quality and performance to suit your hardware capabilities.


<!-- <div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; max-width: 100%;">
    <div style="flex: 1; max-width: 45%; text-align: center;">
        <img src="assets/bonsai_on_desktop.png" alt="bonsai on desktop" style="width: 100%; height: auto;"/>
        <div>Figure 1a: Bonsai on Desktop</div>
    </div>
    <div style="flex: 1; max-width: 45%; text-align: center;">
        <img src="assets/lamborghini_in_garage.png" alt="lamborghini in garage" style="width: 100%; height: auto;"/>
        <div>Figure 1b: Lamborghini in Garage</div>
    </div>
</div>
<p style="text-align: center;">Our plug-and-play merging visualizer supporting versatile object integration across different scenes.</p> -->
<p align="center">
  <img src="assets/bonsai_on_desktop.png" alt="bonsai on desktop" width="45%" />
  <img src="assets/lamborghini_in_garage.png" alt="lamborghini in garage" width="45%" />
</p>

<p align="center">
  <em>Figure 1a: Bonsai on Desktop &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 1b: Lamborghini in Garage</em>
</p>

<p align="center">
  Our plug-and-play merging visualizer supports versatile object integration across different scenes.
</p>



<!-- <div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; max-width: 100%;">
    <div style="flex: 1; max-width: 45%; text-align: center;">
        <img src="assets/multi_res_384_fps_60.png" alt="multi_res_384_fps_60" style="width: 100%; height: auto;"/>
        <div>Figure 2a: 384p @ 60 FPS</div>
    </div>
    <div style="flex: 1; max-width: 45%; text-align: center;">
        <img src="assets/multi_res_1024_fps_35.png" alt="multi_res_1024_fps_35" style="width: 100%; height: auto;"/>
        <div>Figure 2b: 1024p @ 35 FPS</div>
    </div>
    <div style="flex: 1; max-width: 45%; text-align: center;">
        <img src="assets/multi_res_3072_fps_8.png" alt="multi_res_3072_fps_8" style="width: 100%; height: auto;"/>
        <div>Figure 2c: 3072p @ 8 FPS</div>
    </div>
</div>
<p style="text-align: center;">Multi-resolution rendering enables trade-off between visual quality and performance.</p> -->

<p align="center">
  <img src="assets/multi_res_384_fps_60.png" alt="multi_res_384_fps_60" width="32%" />
  <img src="assets/multi_res_1024_fps_35.png" alt="multi_res_1024_fps_35" width="32%" />
    <img src="assets/multi_res_3072_fps_8.png" alt="multi_res_3072_fps_8" width="32%" />
</p>

<p align="center">
  <em>Figure 2a: 384p @ 60 FPS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 2b: 1024p @ 35 FPS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 2c: 3072p @ 8 FPS</em>
</p>

<p align="center">
  Multi-resolution rendering enables trade-off between visual quality and performance.
</p>

## Table of Contents
- [GaussFusion: Seamless 3D Object Integration using Gaussian Splatting](#gaussfusion-seamless-3d-object-integration-using-gaussian-splatting)
  - [🌟 Visualizer Features](#-visualizer-features)
  - [Table of Contents](#table-of-contents)
  - [Data Collection](#data-collection)
    - [3D Models](#3d-models)
    - [Video Scans](#video-scans)
  - [Data Preprocessing](#data-preprocessing)
    - [Image Generation](#image-generation)
    - [COLMAP Feature Matching and Reconstruction](#colmap-feature-matching-and-reconstruction)
  - [Gaussian Splatting Training](#gaussian-splatting-training)
  - [Visualization](#visualization)

## Data Collection

### 3D Models

We recommend using [Sketchfab](https://sketchfab.com/) to download 3D models of objects. The website offers a wide variety of models, and most of them are free to download. 

<!-- After downloading the model, extract the files to a folder and ensure the folder contains the following files:

```
object/
    model.obj
    texture.jpg
``` -->

### Video Scans

To create a video scan of your environment, you can use your smartphone or a camera to capture a video of the object from different angles. 

1. Ensure smooth and consistent motion while capturing the video, do not move too fast or too slow.
2. Capture the object from different angles.
3. Make sure in the first few seconds, the camera is vertical. This is crucial for consistent coordinate system between different data sources.

## Data Preprocessing

For each scene, the final output will be organized in the following directory structure:

```
scene/
    bonsai/
        database.db     --- COLMAP database file
        cameras.json    --- Camera parameters (only available for 3D model)
        images/         --- Images from the 3D model or video scan
            0000.jpg
            0001.jpg
            ...
        sparse/         --- Sparse reconstruction by COLMAP
            0/
                cameras.bin
                points3D.bin
                images.bin
    ...
```

Set the environment variable `$PROJECT_PATH` to the path of your scene folder, e.g. `export PROJECT_PATH=scene/bonsai`.

### Image Generation

**Generate Images from 3D Models**

To generate images from different views of a downloaded 3D model, use the following script:

```bash
python data_processing/model_view_renderer.py \
    -i /path/to/your/3d/model.gltf \
    -o $PROJECT_PATH
```

This script also generates a `cameras.json` file containing the camera parameters for each view.

**Generate Images from Video Scans**

To generate images from a video scan, run:

```bash
python data_processing/video_frame_extractor.py \
    -i /path/to/your/video.mp4 \
    -o $PROJECT_PATH
```

### COLMAP Feature Matching and Reconstruction

To create the sparse model with 3D keypoints and camera parameters, use the [COLMAP](https://github.com/colmap/colmap) library. The initial steps are the same for both 3D models and video scans:

```bash
colmap feature_extractor \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images \
    --ImageReader.camera_model=SIMPLE_PINHOLE \
    --ImageReader.single_camera=true

colmap sequential_matcher \
    --database_path $PROJECT_PATH/database.db
```

If your colmap installation does not support GPU, add `--SiftExtraction.use_gpu=false` to the `feature_extractor` command and `--SiftMatching.use_gpu=false` to the `sequential_matcher` command.

If your 3D model images are not captured at contiguous locations, use `exhaustive_matcher` instead of `sequential_matcher`.

**3D Model**

Ensure the `cameras.json` file is generated before proceeding:

```bash
python data_processing/model_to_sparse_colmap.py \
    -i $PROJECT_PATH \
    -o $PROJECT_PATH/sparse/predefined

mkdir -p $PROJECT_PATH/sparse/0
colmap point_triangulator \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images \
    --input_path $PROJECT_PATH/sparse/predefined \
    --output_path $PROJECT_PATH/sparse/0
```

The first command creates a COLMAP-compatible sparse model from the `cameras.json` file and reorders the images to align with the COLMAP database. The second command triangulates the 3D keypoints using the known camera parameters.

**Video Scan**

```bash
mkdir -p $PROJECT_PATH/sparse/raw

colmap mapper \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images \
    --output_path $PROJECT_PATH/sparse/raw
```

This command creates a sparse model from the video scan images. We need a further step to make the coordinate of sparse model align with the actual world coordinate, e.g. gravity points to the negative z-axis. We can use the following script to rotate the sparse model:

```bash
mkdir -p $PROJECT_PATH/sparse/0
python data_processing/video_rotate_sparse_model.py \
    -i $PROJECT_PATH/sparse/raw \
    -o $PROJECT_PATH/sparse/0
```

**Visualize Sparse Model**

Now the sparse model will be created under the `sparse_transformed/0,1,2,...` directory. You can visualize the sparse model using colmap gui (`File > import model` under Ubuntu) to check the validity of the sparse model.

<div style="text-align:center">
  <img src="assets/bonsai_colmap.png" alt="bonsai colmap" width="45%" />
  <img src="assets/desktop_colmap.png" alt="desktop colmap" width="45%" />
</div>
<p align="center">
  <em>Figure 3a: Bonsai Sparse Model (3D synthetic model) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 3b: Desktop Sparse Model (Video scan)</em>
</p>
<p align="center">
  Example visualizations of the sparse model in COLMAP GUI.
</p>

## Gaussian Splatting Training

For training the Gaussian Splatting model, please refer to [Grendel-GS](https://github.com/nyu-systems/Grendel-GS).

Since the distributed system saves one `.ply` file for each GPU, you can use the following script to merge them in to a single `.ply` file:

```bash
python visualizer/merge_ply.py \
    -i /path/to/the/folder/that/contains/your/ply/files \
    -o /path/to/your/output/ply/file.ply
```


## Visualization

To visualize the Gaussian Splatting model, we offer a simple visualizer. Ensure you have a CUDA-compatible GPU and sufficient memory:

```bash
python visualizer/visualize.py \
    --background /path/to/background/gaussian/splatting/model.ply \
    --object /path/to/object/gaussian/splatting/model.ply \
    --port 8080
```

Open your browser and navigate to `http://localhost:8080` to view the visualization.

<div style="text-align:center">
  <img src="assets/bonsai_30k_vs_50k.png" alt="visualizer demo" width="48%" />
  <img src="assets/lamborghini_30k_vs_50k.png" alt="visualizer demo" width="48%" />
  <img src="assets/bonsai_3dgs.png" alt="visualizer demo" width="48%" />
  <img src="assets/visualizer_demo.png" alt="visualizer demo" width="48%" />
</div>
