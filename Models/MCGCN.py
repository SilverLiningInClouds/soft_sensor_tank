# MC-GCN: Multi-Channel Graph Convolutional Networks
import torch
import torch.nn as nn
from Base.utils import *


# Network
class MultiChannelGraphConvolutionalNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj, in_fc=(1024,), gc=(256,), out_fc=(256, 256)):
        super(MultiChannelGraphConvolutionalNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.adj = adj
        self.net_in_fc = [dim_X, ] + list(in_fc)
        self.net_gc = [in_fc[-1], ] + list(gc)
        self.net_out_fc = [gc[-1], ] + list(out_fc)

        # Input FC & Output FC
        self.in_fc = nn.ModuleList()
        self.out_fc = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(dim_y):
            self.in_fc.append(nn.ModuleList())
            
            for j in range(len(in_fc)):
                self.in_fc[-1].append(nn.Sequential(nn.Linear(self.net_in_fc[j], self.net_in_fc[j + 1]), self.act))
            self.out_fc.append(nn.ModuleList())
            
            for j in range(len(out_fc)):
                self.out_fc[-1].append(nn.Sequential(nn.Linear(self.net_out_fc[j], self.net_out_fc[j + 1]), self.act))
            self.out_fc[-1].append(nn.Linear(self.net_out_fc[-1], 1))

        # GC
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GraphConvolution(self.net_gc[i], self.net_gc[i + 1], self.adj))

    # Forward propagation
    def forward(self, X):
        feat_list = []
        res_list = []

        # Input FC
        for i in range(self.dim_y):
            feat = X
            import pdb;pdb.set_trace()
            for j in self.in_fc[i]:
                feat = j(feat)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=1)

        # GC
        for gc in self.gc:
            feat = gc(feat)
            feat = self.act(feat)

        # Output FC
        for i in range(self.dim_y):
            res = feat[:, i, :]
            
            for j in self.out_fc[i]:
                res = j(res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=1)

        return res


# Model
class McgcnModel(NeuralNetwork):

    # Initialization
    def __init__(self, in_fc=(1024,), gc=(256,), out_fc=(256, 256), **args):
        super(McgcnModel, self).__init__()

        # Parameter assignment
        self.in_fc = in_fc
        self.gc = gc
        self.out_fc = out_fc
        self.prob = 'regression'
        self.args['adj_mode'] = 'sc'
        self.args['graph_reg'] = 0.05
        self.args['self_con'] = 0.2
        self.args['scale'] = 0.4
        self.args['epsilon'] = 0.1
        self.args['need_pretraining'] = False
        self.args.update(args)

        # Set seed
        torch.manual_seed(self.args['seed'])

    # Train
    def fit(self, X, y):
        # Data creation
        self.data_create(X, y, adj_X=True)

        # Model creation
        self.model = MultiChannelGraphConvolutionalNetworks(self.dim_X, self.dim_y, self.adj, self.in_fc, self.gc,
                                                            self.out_fc)
        self.model_create()

        # Model training
        self.training()

        return self
