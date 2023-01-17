import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection


#yu_git_test

class P2MModel(nn.Module):
    #camera_f[248,248], camera_c[111.5,111,5], meshpos[0,0,-0.8]
    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
        super(P2MModel, self).__init__()

        self.hidden_dim = options.hidden_dim #192
        self.coord_dim = options.coord_dim #3, namely x, y, z positional information
        self.last_hidden_dim = options.last_hidden_dim #192

        #shape n_pts*3
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options) 
        # TODO: 查看VGG16这两个class
        #encoder VGG16P2M, decoder VGG16RECONS

        self.features_dim = self.nn_encoder.features_dim + self.coord_dim # final features, 960+3

        #deformation block, 3 times
        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim, #6, 963, 192, 3
                        ellipsoid.adj_mat[0], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,#6, 963+192, 192, 3
                        ellipsoid.adj_mat[1], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,#6, 963+192, 192,192
                        ellipsoid.adj_mat[2], activation=self.gconv_activation)
        ])

        #graph unpooling, add more vertex, 2 times
        self.unpooling = nn.ModuleList([
            GUnpooling(ellipsoid.unpool_idx[0]), #462*2
            GUnpooling(ellipsoid.unpool_idx[1]) #1848*2
        ])

        # if options.align_with_tensorflow:
        #     self.projection = GProjection
        # else:
        #     self.projection = GProjection
        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold, #z_treshold=0
                                      tensorflow_compatible=options.align_with_tensorflow)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim, #in:192, out:3
                           adj_mat=ellipsoid.adj_mat[2])

    def forward(self, img):
        #img shape: batch_size 224 224 3
        batch_size = img.size(0)
        #包含[img2, img3, img4, img5]
        img_feats = self.nn_encoder(img)

        #return img size
        img_shape = self.projection.image_feature_shape(img)

        #把initial pts复制batch size个放在一个数组里, batchsize * pts * 3
        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        
        x = self.projection(img_shape, img_feats, init_pts)
        #x in shape [batch_size x num_points x feat_dim=963]
        #X1是新的坐标=3，x_hidden是新feature=192
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        #新坐标取中点，feature只有坐标， 对于coord的第一次加点
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        
        x = self.projection(img_shape, img_feats, x1)
        #x in shape [batch_size x num_points x feat_dim=963]

        #把x替换成增加中点后的mesh，feature维度是963+192， 对于feature的第一次加点
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        #963+192-> x2=3,x_hidden=192
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        #x2_up的feature只有坐标， 对于coord的第二次加点
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, img_feats, x2)
        #x in shape [batch_size x num_points x feat_dim=963]

        #把x替换成增加中点后的mesh，feature维度是963+192， 对于feature的第二次加点
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        
        #x3这次不是3维坐标，而是963+192
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        #经过relu后再经过一个gcn变成3维
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(img_feats)
        else:
            reconst = None

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst
        }
