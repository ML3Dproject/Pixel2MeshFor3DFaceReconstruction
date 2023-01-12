import torch.nn as nn
import torch.nn.functional as F

from models.layers.gconv import GConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adj_mat, activation=None):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=in_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x = self.conv2(x)
        if self.activation:
            x = self.activation(x)

        return (inputs + x) * 0.5


class GBottleneck(nn.Module):
    
    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None):
        super(GBottleneck, self).__init__()

        resblock_layers = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat, activation=activation)
                           for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        #inputs batchsize*pts*963
        x = self.conv1(inputs)
        #此时x batchsize*pts*192
        #True
        if self.activation:
            x = self.activation(x)
        x_hidden = self.blocks(x)
        x_out = self.conv2(x_hidden)

        #x_out是新的vertex坐标dimension=3, x_hidden是新的feature
        return x_out, x_hidden
