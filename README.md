<!--Generate the TOC via: -->
<!-- (bash) ../gh-md-toc --insert README.md-->
<!--See https://github.com/ekalinin/github-markdown-toc#readme-->

<!--ts-->
- [FIPT: Factorized Inverse Path Tracing](#fipt-factorized-inverse-path-tracing)
    - [Project Page | Paper | Preprocessed Data Download | Scene Data Generation from Scratch](#project-page--paper--preprocessed-data-download--scene-data-generation-from-scratch)
  - [Setup](#setup)
  - [Data](#data)
  - [Train the model](#train-the-model)
  - [Notebook demo](#notebook-demo)
  - [Citation](#citation)
  - [TODO](#todo)
  - [FAQ](#faq)
    - [Install torch\_scatter](#install-torch_scatter)
    - [Install conda env `fipt` to Jupyter](#install-conda-env-fipt-to-jupyter)
    - [Reference training log](#reference-training-log)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: ruizhu, at: Thu Apr 13 04:43:32 PM PDT 2023 -->

<!--te-->

# FIPT: Factorized Inverse Path Tracing

This repo contains the demo code for [FIPT](https://jerrypiglet.github.io/fipt-ucsd/). Full release is pending (see [#TODO](#todo)).

### [Project Page](https://jerrypiglet.github.io/fipt-ucsd/) | [Paper](https://arxiv.org/abs/2304.05669) | [Preprocessed Data Download](https://mclab.ucsd.edu/FIPT_data_release/) | [Scene Data Generation from Scratch](https://github.com/Jerrypiglet/rui-indoorinv-data/tree/fipt)

## Setup

* python 3.8
* pytorch 1.13.1
* pytorch-lightning 1.9 # works with torch 1.13.1: https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix
* torch-scatter 2.1.0
* mitsuba 3.2.1
* CUDA 11.7

Set up the environment via:

``` bash
conda create --name fipt python=3.8 pip
conda activate fipt
# firstly, install torch==1.31.1 according to your platform; check: https://pytorch.org/get-started/previous-versions/
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch # tested with tinycudann-1.7
```

Also, install `torch_scatter`. See the [#FAQ](#faq) for detailed instructions.

## Data

For the current release, we provide demo of one synthetic scene and one real world scene. They may not exactly match the results reported in the paper.

You can download the processed data from [here](https://mclab.ucsd.edu/FIPT_data_release/) (download zip or folders). Or you can check [rui-indoorinv-data (branch: `fipt`)](https://github.com/Jerrypiglet/rui-indoorinv-data/tree/fipt) for generating data (synthetic/real) from scratch (i.e. synthetic scenes from [Mitsuba XML files by Benedikt Bitterli](https://benedikt-bitterli.me/resources/), or real world data from RAW HDR captures.).

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
│   │       ├── scene.obj               # scene geometry
│   │       └── train | val
│   │           ├── transforms.json     # camera parameters
│   │           ├── Image/ID_0001.exr   # HDR image
│   │           └── segmentation/ID.exr # semantic segmentation
│   └── real/
│       └── ClassRoom/
│           ├── scene.obj           # scene geometry
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

Make sure you select the right kernel which uses the correct Python environment as configured above. See [#FAQ](#install-conda-env-to-jupyter)

* `demo/brdf-emission.ipynb` : visualize BRDF, emission, and rendering for selected views.
* `demo/relighting.ipynb`: customizable relighting and object insertion. The scene are written in mitsuba3.

## Citation

If you find our work is useful, please consider cite:

```
@misc{fipt2023,
      title={Factorized Inverse Path Tracing for Efficient and Accurate Material-Lighting Estimation}, 
      author={Liwen Wu and Rui Zhu and Mustafa B. Yaldiz and Yinhao Zhu and Hong Cai and Janarbek Matai and Fatih Porikli and Tzu-Mao Li and Manmohan Chandraker and Ravi Ramamoorthi},
      year={2023},
      eprint={2304.05669},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
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

You can fist try to install it directly:

``` bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

Try this and hopefully no error occurs:

```
python -c "import torch_scatter"
```

If not, go to the link by the end of the first code block (e.g. https://data.pyg.org/whl/torch-1.13.0+cu117.html), find the `torch_scatter` wheel which matches your Python (`cp**`), torch (`pt***`) and cuda version (`cu***`). Get the link to the wheel, and run something like:

``` bash
pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
```

### Install conda env `fipt` to Jupyter

``` bash
conda activate fipt
python -m ipykernel install --user --name=fipt
# then launch Jupyter Notebook: (fipt) ~/Documents/Projects/fipt$ jupyter notebook
# select Kernel -> Change kernel -> fipt
```

### Reference training log

Training log on a 3090 GPU for the `kitchen` scene: see `train.log` (look for `...time (s): ...`).