import os

# dataset root
DATASET_ROOT = "datasets/data"
SHAPENET_ROOT = os.path.join(DATASET_ROOT, "shapenet")
IMAGENET_ROOT = os.path.join(DATASET_ROOT, "imagenet")
AFLW2000_ROOT = os.path.join(DATASET_ROOT, "AFLW2000-3D")

# ellipsoid path
ELLIPSOID_PATH = os.path.join(DATASET_ROOT, "semi-sphere/semi-sphere.dat")
# ELLIPSOID_PATH = os.path.join(DATASET_ROOT, "ellipsoid/semi-sphere.dat")pixel2mesh_aux_4stages


# pretrained weights path
PRETRAINED_WEIGHTS_PATH = {
    "vgg16": os.path.join(DATASET_ROOT, "pretrained/vgg16-397923af.pth"),
    "resnet50": os.path.join(DATASET_ROOT, "pretrained/resnet50-19c8e357.pth"),
    "vgg16p2m": os.path.join(DATASET_ROOT, "pretrained/vgg16-p2m.pth"),
    "vggface": os.path.join(DATASET_ROOT, "pretrained/resnet50_ft_weight.pkl"),
}

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 256
