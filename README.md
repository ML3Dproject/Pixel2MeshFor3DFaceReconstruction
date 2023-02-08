# Pixel2Mesh for 3D Face Reconstrcution

## Abstraction
Pixel2mesh is an end-to-end deep learning architecture that generates 3D triangular meshes from single-color images. The majority of previous works represent 3D shapes in volumes or point clouds, while the Pixel2Mesh network represents 3D shapes in meshes, which are essentially well graph suited for graph-based convolutional neural networks. Besides, this network has considerable potential to be applied in specific domains. Our group is committed to adapting pixel2mesh in the field of face generation and improving its performance through a series of optimizations. 

This is an implement of Pixel2Mesh for 3D Face Reconstruction. Our repository is based on the pytorch version of Pixel2Mesh in PyTorch of [this](https://github.com/noahcao/Pixel2Mesh).

## Our Work

- Focus on face reconstruction instead of  general object reconstruction.
- Improve the baseline model.
- Ablation study within our model and compare related criteria with original P2M.


## Get Started

### Environment

Current version only supports training and inference on GPU. It works well under dependencies as follows(must):

- Ubuntu 16.04 / 18.04
- Python 3.7
- PyTorch 1.1
- CUDA 9.0 (10.0 should also work)

Other conda environments can be found in the `requirements.txt`

After you have created and install the related dependencies, you should also done:

1. `git submodule update --init` to get [Neural Renderer](https://github.com/daniilidis-group/neural_renderer) ready.
2. `python setup.py install` in directory [external/chamfer](external/chamfer) and `external/neural_renderer` to compile the modules.

### Datasets
We use [AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) for model training and evaluation.You should organize your 'datasets' as following trees.
```
datasets/data
├── semi-sphere
│   ├── semi1.obj
│   ├── semi2.obj
│   ├── semi3.obj
|   ├── semi4.obj
│   └── semi.dat
├── pretrained
│   ... (.pth files)
└── AFLW2000-3D
    ├── AFLW2000
    │   ├── 02691156
    │   │   └── 3a123ae34379ea6871a70be9f12ce8b0_02.dat
    │   ├── 02828884
    │   └── ...
    ├── data_tf (standard data used in official implementation)
    │   ├── 02691156 (put the folders directly in data_tf)
    │   │   └── 10115655850468db78d106ce0a280f87
    │   ├── 02828884
    │   └── ...
    └── meta
        ...
```

### Our resources

- You can find our pre-processing AFLW2000-3D dataset [here](https://drive.google.com/file/d/1MKINKNRMQHitbQeM-yoqJppidUdFrVrB/view?usp=sharing).
- You can find our checkpoint [here](https://drive.google.com/file/d/1nEfYK0EfWyPJcfeuPvBKyDJWKI_dvbzF/view?usp=sharing).
- You can find pytorch-author's checkpoint [here](https://drive.google.com/file/d/1pZm_IIWDUDje6gRZHW-GDhx5FCDM2Qg_/view?usp=sharing)

### Usage

#### Configuration

You can modify configuration in a `yml` file for training/evaluation. It overrides dsefault settings in `options.py`. We provide some examples in the `experiments` directory. 

#### Training

```
python entrypoint_train.py --name xxx --options path/to/yaml
```
#### Training example
```
python entrypoint_train.py --name tmp --options experiments/default/tensorflow.yml
```

*P.S. To train on slurm clusters, we also provide settings reference. Refer to [slurm](slurm) folder for details.*

#### Evaluation

```shell
python entrypoint_eval.py --name xxx --options path/to/yml --checkpoint path/to/checkpoint
```

#### Inference

We recommand to use our jupyternotebook `yu_predict.ipynb` in the folder [Jupyter_simplify_file](Jupyter_simplify_file). And the file shold be removed outside.


You can do inference on your own images by a simple command:

``` 
python entrypoint_predict.py --options /path/to/yml --checkpoint /path/to/checkpoint --folder /path/to/images
```

*P.S. we only support do training/evaluation/inference with GPU by default.*


## Details of Improvement

Compared with data in ShapeNet dataset, in this human face dataset:
- 1.2D images are more noisy; 
- 2.3D point clouds contain more points and show more details. 
Due to these points as well as different topologies of 3D point clouds, we introduce a preprocessing network for denoising, add one more deformation process,  enabling our network be able to predict more details, and use semi-sphere as initial reference to match the new topology. Furthermore, we change the fixed graph unpooling to adaptive unpooling, which can accelerate training process.

## Acknowledgements

Our work is based on the official version of [Pixel2Mesh](https://github.com/noahcao/Pixel2Mesh); And thanks for authors of this [paper](https://www.sciencedirect.com/science/article/pii/S131915782200413X) to provide useful but not open-source information. Sincere thanks all of people helped us!
