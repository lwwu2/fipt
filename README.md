# FIPT: Factorized Inverse Path Tracing

This repo contains the demo code for [FIPT](). Full release is pending (see #TODO section)

### [Project Page]() | [Paper]() | [Data]()


## Setup

* python 3.8
* pytorch 1.13
* pytorch-lightning 1.2.10
* torch-scatter 2.1.0
* mitsuba 3.1.3
* CUDA 11.7

Set up the environment via:

``` bash
conda create --name fipt python=3.8 pip
conda activate fipt
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch # tested with tinycudann-1.7
```

Also, install `torch_scatter`. See the #FAQ section for detailed instructions.

## Data

For the current release, we provide demo of one synthetic scene and one real world scene. They may not exactly match the results reported in the paper.

You can download the processed data from [here](https://drive.google.com/drive/folders/1N8H1yR41MykUuSTyHvKGsZcuV2VjtWGr?usp=share_link) (download zip or folders). Or you can check [fipt-data (TODO: not ready; check back later)](https://github.com/Jerrypiglet/fipt-data) for generating data (synthetic/real) from scratch (i.e. synthetic scenes from [XML files](https://benedikt-bitterli.me/resources/), or real world data from RAW HDR captures.).

Organize the data as:

<!-- https://tree.nathanfriend.io -->

<!-- - data
  - indoor_synthetic
    - kitchen
      - scene.obj             # scene geometry
      - train | val
        - transforms.json     # camera parameters
        - Image/ID_0001.exr   # HDR image
        - segmentation/ID.exr # semantic segmentation
  - real
    - ClassRoom
      - scene.obj           # scene geometry
      - cam.txt             # each row of each camera is: origin, lookat location, and up; OpenCV convention (right-down-forward)
      - K_list.txt          # camera instrinsics list
      - Image/ID_0001.exr   # HDR image
      - segmentation/ID.exr # semantic segmentation

- pretrained
  - kitchen
    - last.ckpt # weights of BRDF and emission mask networks
    - emitter.pth # emitter parameters
    - vslf.npz # volumetric surface light field (radiance cache)
  - ClassRoom
    - ... -->



```
.
├── data/
│   ├── indoor_synthetic/
│   │   └── kitchen/
│   │       ├── scene.obj             # scene geometry
│   │       └── train | val
│   │           ├── transforms.json     # camera parameters
│   │           ├── Image/ID_0001.exr   # HDR image
│   │           └── segmentation/ID.exr # semantic segmentation
│   └── real/
│       └── ClassRoom/
│           ├── scene.obj             # scene geometry
│           ├── cam.txt             # each row of each camera is: origin, lookat location, and up; OpenCV convention (right-down-forward)
│           ├── K_list.txt          # camera instrinsics list
│           ├── Image/ID_0001.exr   # HDR image
│           └── segmentation/ID.exr # semantic segmentation
└── pretrained/
    ├── kitchen/
    │   ├── last.ckpt # weights of BRDF and emission mask networks
    │   ├── emitter.pth # emitter parameters
    │   └── vslf.npz # volumetric surface light field (radiance cache)
    └── ClassRoom/
        └── ...
```

<!-- The camera intrinsic and extrinsic are in OpenCV coordinate system (right-down-forward). -->

<!-- The semantic segmentation mask is obtained using [Mask2Former](https://github.com/facebookresearch/Mask2Former), then fused to scene geometry by running:

```
python utils/fuse_segmentation.py --scene SCENE_PATH --dataset DATASET_TYPE
``` -->

## Train the model

Edit `configs/config.py` to configure hyper parameters.

Edit `train.sh` to specify dataset folder.

Run:

``` bash
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

By default, the notebooks load pretrained model. To use your trained models, change all occurances of `pretrained/` to `outputs/`.

Make sure you select the right kernel which uses the correct Python environment as configured above.

* `demo/brdf-emission.ipynb` : visualize BRDF, emission, and rendering for selected views.
* `demo/relighting.ipynb`: customizable relighting and object insertion. The scene are written in mitsuba3.

## Citation

If you find our work is useful, please consider cite:

```
```

## TODO
- [ ] Full release of all scenes, and all benchmarks
  - [ ] Including relighting
- [ ] Implementation of adapted baseline methods & evaluation
  - [ ] Li'22
  - [ ] FVP

## FAQ

### Install torch_scatter

To install [torch_scatter](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), nagivate to your environment configuration, you will see something like:

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

Go to the link, find the `torch_scatter` wheel which matches your Python (`cp**`), torch (`pt***`) and cuda version (`cu***`). Get the link to the wheel, and run something like:

``` bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
```

Try this and make sure no error occurs:

```
python -c "import torch_scatter"
```
