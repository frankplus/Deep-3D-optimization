# Deep 3D Optimization
*Elena Camuffo, Federica Battisti, Francesco Pham, Simone Milani - EUVIP 2022 [paper]*

![pipeline_my](https://user-images.githubusercontent.com/63043735/174303186-6cc17e57-3c83-4a2a-835b-bf433db31b89.png)

## Abstract 
The growing diffusion of immersive and interactive applications is posing new challenges in the multimedia processing chain. When dealing with AR and VR applications, the most relevant aspects to consider are the (1) **quality** of the visualized 3D objects and (2) the **fluidity** in the visualization in case the user is moving in the environment. In this framework, we propose a deep learning based approach that estimates the optimal model parameters to be used in relation to the viewer’s movement and the model characteristics and quality. The performed tests show the effectiveness of the proposed approach.

## General Requirements
As general requirements to install this project you need a .conda environment with Python 3.7 (with Pytorch 1.11) and Unity3D 2020.1.11.f1.
Move in  `PythonScripts`  folder and install all the packages using the command:

```
pip install -r PythonScripts/requirements.txt
```

## (1) Positions generation
Generate a set of uniformly distributed poisitions, around `(0,0,0)`. Use:
- `generate_clustered_positions` function to generate position pairs for an inter-view environment.
- `generate_positions` function to generate position pairs for an intra-view environment.

```
python random_walk_uniform.py
```

The generated positions are stored in `positions.txt`.

## (2) Scene Setup
The Unity environment is contained in the  `Assets`  folder. Import your models with different LODs here.
- To capture the renders from different viewpoints use the scene  `CameraPathScreenshots.unity`
- To compute intra-view SSIM use the scene  `SsimPredict.unity`
- To obtain OTC-projections use the scene  `SurfaceCountPorjections.unity`

## (3) Training the Network
To generate your own dataset use the script  `build_dataset.py`  or use ours `dataset.json`.
To train the network run the command:

```
python dynamic_ssim_multimodel/multimodel_predictor.py
```
