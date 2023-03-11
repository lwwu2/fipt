# FIPT: Factorized Inverse Path Tracing

This repo contains the training code for FIPT

## Setup

* python 3.8
* pytorch 1.13
* pytorch-lightning 1.2.10
* torch-scatter 2.1.0
* mitsuba 3.1.3
* CUDA 11.7

## Data preparation

Our data loader for real world scene supports reading folder with the following format:

```
scene.obj           # scene geometry
cam.txt             # camera look-at vector and up vector list
K_list.txt          # camera instrinsics list
Image/ID_0001.exr   # HDR image
segmentation/ID.exr # semantic segmentation
```

The camera intrinsic and extrinsic are in OpenCV coordinate system.

The semantic segmentation mask is obtained using [Mask2Former](https://github.com/facebookresearch/Mask2Former), then fused to scene geometry by running:

```
python utils/fuse_segmentation.py --scene SCENE_PATH --dataset DATASET_TYPE
```

## Train the model

Edit `configs/config.py` to configure hyper parameters.

Edit `train.sh` to specify dataset folder.

Run:

```
sh train.sh
```

The script contains 4 subroutines:

1. Initialize shadings:

   ```
   python bake_shading.py --scene SCENE_PATH --output OUTPUT_PATH \
                          --dataset DATASET_TYPE
   ```

2. Optimize BRDF and emission mask:

   ```
   python train.py --experiment_name EXPERIMENT_NAME --device DEVICE_ID \
                   --max_epochs MAX_EPOCHS
   ```

3. Extract emitters:

   ```
   python extract_emitter.py --scene SCENE_PATH --output OUTPUT_PATH\
                             --ckpt NETWORK_CHECKPOINT --dataset DATASET_TYPE
   ```

4. Shading refinement:

   ```
   python refine_shading.py --scene SCENE_PATH --output OUTPUT_PATH\
                            --ckpt NETWORK_CHECKPOINT --dataset DATASET_TYPE\
                            --ft REFINEMENT_STAGE
   ```

## Notebook demo

*  `demo/brdf-emission.ipynb` : visualize BRDF, emission, and rendering for selected views.
* `demo/relighting.ipymn`: customizable relighting and object insertion. The scene are written in mitsuba3.