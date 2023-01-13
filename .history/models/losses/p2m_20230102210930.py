import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.chamfer_wrapper import ChamferDist
import numpy as np


class P2MLoss(nn.Module):
    def __init__(self, options, ellipsoid):
        super().__init__()
        self.options = options#options.loss
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()
        self.laplace_idx = nn.ParameterList([
            nn.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx])##ellipsoid.laplace_idx include 3 parts，the first part is the information of first deformation(0-155 vertices,0-627,0-2455,same as the paper)
        self.edges = nn.ParameterList([
            nn.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges])

    def edge_regularization(self, pred, edges):#pred = pred_coord[i]
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        return self.l2_loss(pred[:, edges[:, 0]], pred[:, edges[:, 1]]) * pred.size(-1)#core:pred[:, edges[:, 0]], pred[:, edges[:, 1]]are two points,edges[:, 0] is the edge index of the point
        #combine (the first value) of edges[:, 0] and edges[:, 1], it will be an edge

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]!!!!

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """

        indices = lap_idx[:, :-2]#for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]!!!! here we get the max 8 neighbors of a point
        invalid_mask = indices < 0#return the indices,1 for <0, 0 for > 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices, I think negative means we don't have this neighbor point

        vertices = inputs[:, all_valid_indices] #choose the valid neighbor points,or vertices = inputs[all_valid_indices,:] 
        vertices[:, invalid_mask] = 0
        neighbor_sum = torch.sum(vertices, 2)#sum 3 coord. xyz
        neighbor_count = lap_idx[:, -1].float()#the total numbers of a neighbor
        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]#see the jupyter example

        return laplace

    def laplace_regularization(self, input1, input2, block_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """
        print('input1',input1.shape)
        print('#####################')

        lap1 = self.laplace_coord(input1, self.laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, self.laplace_idx[block_idx])
        laplace_loss = self.l2_loss(lap1, lap2) * lap1.size(-1)#a real number
        move_loss = self.l2_loss(input1, input2) * input1.size(-1) if block_idx > 0 else 0#input1.size(-1) = lap1.size(-1) =3
        return laplace_loss, move_loss

    def normal_loss(self, gt_normal, indices, pred_points, adj_list):#adj_list are the neighbors,indices is idx2, it is the indices corresponding to the pred in GT
        edges = F.normalize(pred_points[:, adj_list[:, 0]] - pred_points[:, adj_list[:, 1]], dim=2)#not know adj_list has what thing? p-q
        nearest_normals = torch.stack([t[i] for t, i in zip(gt_normal, indices.long())])
        print('nearest_normals',nearest_normals.shape)
        print('#####################')
        normals = F.normalize(nearest_normals[:, adj_list[:, 0]], dim=2)#现在这个点的对应GT的normal vector, the order is same as the adj_list[:, 0](the first points in edges)
        cosine = torch.abs(torch.sum(edges * normals, 2))
        return torch.mean(cosine)

    def image_loss(self, gt_img, pred_img):
        rect_loss = F.binary_cross_entropy(pred_img, gt_img)
        return rect_loss

    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """

        chamfer_loss, edge_loss, normal_loss, lap_loss, move_loss = 0., 0., 0., 0., 0.
        lap_const = [0.2, 1., 1.]

        gt_coord, gt_normal, gt_images = targets["points"], targets["normals"], targets["images"]#the dataset has normals of each face
        pred_coord, pred_coord_before_deform = outputs["pred_coord"], outputs["pred_coord_before_deform"]#
        image_loss = 0.
        if outputs["reconst"] is not None and self.options.weights.reconst != 0:
            image_loss = self.image_loss(gt_images, outputs["reconst"])

        for i in range(3):#3 is for 3 times deformation
            dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord, pred_coord[i])
            print('idx2.shape',idx2.shape)
            print('#####################')
            chamfer_loss += self.options.weights.chamfer[i] * (torch.mean(dist1) +
                                                               self.options.weights.chamfer_opposite * torch.mean(dist2))#here have another weight and get the mean  of the dist
            normal_loss += self.normal_loss(gt_normal, idx2, pred_coord[i], self.edges[i])
            edge_loss += self.edge_regularization(pred_coord[i], self.edges[i])
            lap, move = self.laplace_regularization(pred_coord_before_deform[i],
                                                                   pred_coord[i], i)
            lap_loss += lap_const[i] * lap
            move_loss += lap_const[i] * move

        loss = chamfer_loss + image_loss * self.options.weights.reconst + \
               self.options.weights.laplace * lap_loss + \
               self.options.weights.move * move_loss + \
               self.options.weights.edge * edge_loss + \
               self.options.weights.normal * normal_loss  #weights are not same as the paper...edge loss算的和move loss意义一样？

        loss = loss * self.options.weights.constant

        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            "loss_move": move_loss,
            "loss_normal": normal_loss,
        }
###need to know the form of 'idx2' and 'pred_coord'