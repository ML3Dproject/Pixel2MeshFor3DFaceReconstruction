import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate

import config
from datasets.base_dataset import BaseDataset


class ShapeNet(BaseDataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
        #file root is --datasets/data/shapenet
        # dataset root
        #   # DATASET_ROOT = "datasets/data"
        # SHAPENET_ROOT = os.path.join(DATASET_ROOT, "shapenet")#examples
        # IMAGENET_ROOT = os.path.join(DATASET_ROOT, "imagenet")

        super().__init__()
        self.file_root = file_root  #use in base.py file --config.SHAPENET_ROOT
        with open(os.path.join(self.file_root, "meta", "shapenet.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))#after load, the output is a dictionary,sort operate on the list only, sorted can work on dictionary
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}#self.labels_map,i = 0 1 2 3 ...?
        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1] #divide by the space!!
        self.tensorflow = "_tf" in file_list_name # tensorflow version of data
        self.normalization = normalization#options.dataset.normalization = True
        self.mesh_pos = mesh_pos#options.dataset.mesh_pos = [0., 0., -0.8]
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border#use edict to access the key, False

    def __getitem__(self, index):
        if self.tensorflow:#if is the official
            filename = self.file_names[index][17:]#04256520/cc644fad0b76a441d84c7dc40ac6d743/rendering/02.dat
            label = filename.split("/", maxsplit=1)[0]#maxsplit=1 ,divide into two parts 04256520
            pkl_path = os.path.join(self.file_root, "data_tf", filename)
            img_path = pkl_path[:-4] + ".png"#get rid of the .dat
            with open(pkl_path) as f:
                data = pickle.load(open(pkl_path, 'rb'), encoding="latin1")#transfer to python files, 'latin1' is the encoder method to the binary file and python object
            pts, normals = data[:, :3], data[:, 3:]#dat文件中，前三项目存储该点坐标数值，后三个为该点的normal vector数值
            img = io.imread(img_path)#137*137*4，α channel 不透明度
            img[np.where(img[:, :, 3] == 0)] = 255# why ==0 to 255?? if have values, it ranges in 0-255
            if self.resize_with_constant_border:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                       mode='constant', anti_aliasing=False)  # to match behavior of old version,224*224
            else:
                img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            img = img[:, :, :3].astype(np.float32)#4个channels
        else:
            label, filename = self.file_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")#transfer to python files, 'latin1' is the encoder method to the binary file and python object
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]#存储方式不一样，另一种不分开，直接存储，取的时候分两个channels

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,#image matrix, without 0 (is 255)
            "points": pts,#every points
            "normals": normals,#the normal vector of every points
            "labels": self.labels_map[label],# 0-12, class = 13
            "filename": filename,#"04256520/cc644fad0b76a441d84c7dc40ac6d743/rendering/02.dat"
            "length": length#8678
        }

    def __len__(self):
        return len(self.file_names)
    
    ### yu_git_test

# #aaa
# class AFLW2000(BaseDataset):
#     """
#     Dataset wrapping images and target meshes for AFLW2000.
#     """

#     def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
#         super().__init__()
#         self.file_root = file_root

#         #INPUT ARGUMENTS
#         #file_root: 
#         #  original:..datasets/data/shapenet
#         #  now:..datasets/data/AFLW2000-3D

#         #file_list_name:
#         #  original:e.g. train_all.txt
#         #  now: train_aflw.txt/ test_aflw.txt

#         #mesh_pos:
#         #  original:[0., 0., -0.8]
#         #  now:???do we need mesh_pos? how can we determine the mesh_pos?????????????

#         #normalization:...

#         #shapenet_options:...


#         #SELF VARIABLES
#         #self.labels_map: {'02691156': 0, '02828884': 1, '02933112': 2, '02958343': 3, '03001627': 4, '03211117': 5, '03636649': 6, '03691459': 7, '04090263': 8, '04256520': 9, '04379243': 10, '04401088': 11, '04530566': 12}
#         self.labels_map = {'face':0}

#         #self.file_names: ['02691156_fff513f407e00e85a9ced22d91ad7027_19.dat', '02691156_fff513f407e00e85a9ced22d91ad7027_20.dat', '02691156_fff513f407e00e85a9ced22d91ad7027_23.dat']
#         #self.file_names: ['image02795.jpg']
#         with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
#             self.file_names = fp.read().split("\n")

#         self.normalization = normalization #boolean 
#         self.mesh_pos = mesh_pos #mesh position


#     def __getitem__(self, index):

#         label = "face"
#         filename = self.file_names[index] #file name of the img. e.g.image00002.jpg

#         img_path =  self.file_root + "/AFLW2000/"+ filename
#         data_path = self.file_root + "/AFLW2000/"+ filename[:-4] + ".txt"

#         #np.loadtxt need file without "," !!!!!!!!
#         data = np.loadtxt(data_path)
#         #first 3 columns is point positions?
#         pts, normals = data[:, :3], data[:, 3:]

#         img = io.imread(img_path)

#         if self.resize_with_constant_border:
#             img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
#                                     mode='constant', anti_aliasing=False)  # to match behavior of old versions
#         else:
#             img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

#         img = img.astype(np.float32)

#         pts -= np.array(self.mesh_pos)
#         assert pts.shape[0] == normals.shape[0]
#         length = pts.shape[0]

#         img = torch.from_numpy(np.transpose(img, (2, 0, 1))) #turn data into [channels, height, width]
#         img_normalized = self.normalize_img(img) if self.normalization else img  #normalize_img哪来的????????????????


#         #OUTPUT QUESTIONS:
#         #labels: 0????????????
#         #filename:????????????

#         return {
#             "images": img_normalized,
#             "images_orig": img,
#             "points": pts,
#             "normals": normals,
#             "labels": 0, #means face
#             "filename": filename,#dont know if its used later?????
#             "length": length
#         }

#     def __len__(self):
#         return len(self.file_names)

class ShapeNetImageFolder(BaseDataset):

    def __init__(self, folder, normalization, shapenet_options):#folder:"/tmp"
        super().__init__()
        self.normalization = normalization
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.file_list = []
        for fl in os.listdir(folder):
            file_path = os.path.join(folder, fl)
            # check image before hand
            try:
                if file_path.endswith(".gif"):
                    raise ValueError("gif's are results. Not acceptable")
                Image.open(file_path)
                self.file_list.append(file_path)
            except (IOError, ValueError):
                print("=> Ignoring %s because it's not a valid image" % file_path)

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                   mode='constant', anti_aliasing=False)
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "filepath": self.file_list[item]
        }

    def __len__(self):
        return len(self.file_list)
#the methods are same as the before, just to demo the process of image processing

def get_shapenet_collate(num_points):#num_points = 3000
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    def shapenet_collate(batch):
        if len(batch) > 1:
            all_equal = True
            for t in batch:
                if t["length"] != batch[0]["length"]:#the number of the points in an example
                    all_equal = False
                    break
            points_orig, normals_orig = [], []
            if not all_equal:
                for t in batch:
                    pts, normal = t["points"], t["normals"]
                    length = pts.shape[0]
                    choices = np.resize(np.random.permutation(length), num_points)
                    t["points"], t["normals"] = pts[choices], normal[choices]
                    points_orig.append(torch.from_numpy(pts))
                    normals_orig.append(torch.from_numpy(normal))
                ret = default_collate(batch)
                ret["points_orig"] = points_orig
                ret["normals_orig"] = normals_orig
                return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret
  # the purpose is to make sure all batchs have same number of points, if not, it will change it to same 3000, else:default
    return shapenet_collate