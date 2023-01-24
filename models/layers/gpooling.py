# import torch
# import torch.nn as nn
# import numpy as np


# class GUnpooling(nn.Module):
#     """Graph Pooling layer, aims to add additional vertices to the graph.
#     The middle point of each edges are added, and its feature is simply
#     the average of the two edge vertices.
#     Three middle points are connected in each triangle.
#     """

#     def __init__(self, unpool_idx):
#         super(GUnpooling, self).__init__()
#         self.unpool_idx = unpool_idx #462*2， 1848*2
        
#         # #trainable new points
#         # self.num_edge=np.size(self.unpool_idx,0)
#         # new_pts_pos=np.full(self.num_edge,0.5)
#         # self.new_pts_pos=nn.Parameter(torch.from_numpy(new_pts_pos))
#         # save dim info
#         self.in_num = torch.max(unpool_idx).item()
#         self.out_num = self.in_num + len(unpool_idx)

#     def forward(self, inputs):
#         #unpooling_index可能存储edge的两个顶点
#         #new_features = inputs[:, self.unpool_idx].clone()
#         # pt1 = inputs[:, self.unpool_idx[:,0]].clone()
#         # pt2 = inputs[:, self.unpool_idx[:,1]].clone()
#         # for i in range(self.num_edge):
#         #     ll=self.new_pts_pos[i]
#         #     pt3 = ll*pt1+(1-ll)*pt2
#         # #new_vertices = 0.5 * new_features.sum(2)
#         # output = torch.cat([inputs, pt3], 1)
#         new_features = inputs[:, self.unpool_idx].clone()
#         new_vertices = 0.5 * new_features.sum(2)
#         output = torch.cat([inputs, new_vertices], 1)

#         return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_num) + ' -> ' \
#                + str(self.out_num) + ')'
import torch
import torch.nn as nn
import numpy as np


class GUnpooling(nn.Module):
    """Graph Pooling layer, aims to add additional vertices to the graph.
    The middle point of each edges are added, and its feature is simply
    the average of the two edge vertices.
    Three middle points are connected in each triangle.
    """

    def __init__(self, unpool_idx):
        super(GUnpooling, self).__init__()
        self.unpool_idx = unpool_idx #462*2， 1848*2
        
        #trainable new points
        self.num_edge=np.size(self.unpool_idx,0)
        new_pts_pos=np.full(self.num_edge,0.5,dtype=np.float32)
        self.new_pts_pos=nn.Parameter(torch.from_numpy(new_pts_pos))
        # save dim info
        self.in_num = torch.max(unpool_idx).item()
        self.out_num = self.in_num + len(unpool_idx)

    def forward(self, inputs):
        #unpooling_index可能存储edge的两个顶点
        new_features = inputs[:, self.unpool_idx].clone()
        # new_vertices = 0.5 * new_features.sum(2)#1*edges*3
        # print(self.new_pts_pos.reshape(-1, 1))
        # print("222",1 - self.new_pts_pos.reshape(-1, 1))
        
        torch.clamp(self.new_pts_pos, 0.0, 1.0, out=None)
        unpooling_wights = torch.cat([self.new_pts_pos.reshape(-1, 1), 1 - self.new_pts_pos.reshape(-1, 1)], dim=1)
        unpooling_wights_expand = unpooling_wights.view(new_features.shape[1], 2, 1)#462*2 -> 462*2*1
        unpooling_wights = unpooling_wights_expand.expand_as(new_features)#沿着第3维复制,新增第一维 462*2*1 ->batchsize*462*2*3
        features = torch.mul(new_features,unpooling_wights)#对应相乘
        new_vertices = torch.sum(features,dim=2)
        
        
        
        # torch.clamp(self.new_pts_pos, 0.0, 1.0, out=None)
        # for i in range(self.num_edge):
        #     ll=self.new_pts_pos[i]
        #     new_vertices[:,i] = ll * new_features[:,i,0] + (1-ll)*new_features[:,i,1]
        #     #new_vertices[:,i] = ll*new_features[:,i]+(1-ll)*new_features[:,i]
        
        output = torch.cat([inputs, new_vertices], 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_num) + ' -> ' \
               + str(self.out_num) + ')'
