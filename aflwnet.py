import json
import os

import config

import numpy as np
import torch

import torchvision.transforms as transforms

from skimage import io, transform
from torch.utils.data.dataloader import default_collate

from torch.utils.data.dataset import Dataset


class AFLW2000(Dataset):
    """
    Dataset wrapping images and target meshes for AFLW2000.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, aflw_options):  #shapenet_options):
        """
        file_root: 'datasets/data/AFLW2000-3D'
        file_list_name: 'train_tf_overf'
        """
        super().__init__()
        self.file_root = file_root
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")

        self.normalization = normalization #boolean 
        self.mesh_pos = mesh_pos #mesh position
        
        self.resize_with_constant_border = aflw_options.resize_with_constant_border
        # shapenet_options.resize_with_constant_border#use edict to access the key, False



    def __getitem__(self, index):

        filename = self.file_names[index] 
        img_path =  self.file_root + "/AFLW2000/"+ filename
        data_path = self.file_root + "/AFLW2000/"+ filename[:-4] + ".txt"

        data = np.loadtxt(data_path, delimiter=",") #现在还是numpy_array
        #torch.from_numpy()
        pts, normals = data[:, :3]/1000., data[:, 3:]
        pts, normals = pts.astype(np.float32),normals.astype(np.float32)
        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]
        # pts, normals是np.array

        img = io.imread(img_path)#137*137*4，α channel 不透明度
        # img[np.where(img[:, :, 3] == 0)] = 255# why ==0 to 255?? if have values, it ranges in 0-255
        # img现在是np.array

        #resize
        if self.resize_with_constant_border:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
                                       mode='constant', anti_aliasing=False)  # to match behavior of old version,224*224
        else:
            img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

        img = img.astype(np.float32) #4个channels
        
        #to_tensor
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        #normalize
        img_normalized = img
        if self.normalization:
            img_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return {
            "images": img_normalized, #normalized_tensor
            "images_orig": img, #tensor
            "points": pts, #np.array
            "normals": normals, #np.array
            "labels": 0,
            "filename": filename,
            "length": length
        }

    def __len__(self):
        return len(self.file_names)


def get_aflw_collate(num_points):#num_points = 20000
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    def shapenet_collate(batch):
        # if len(batch) > 1:
        #     all_equal = True
        #     for t in batch:
        #         if t["length"] != batch[0]["length"]:#the number of the points in an example
        #             all_equal = False
        #             break
        points_orig, normals_orig = [], []
        all_equal = False
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