# AGCN: Adaptive Graph Convolutional Networks
from Base.utils import *
from Base.packages import *


# Temporal graph convolution
class TemporalGraphConvolution(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj=None, bias=False):
        super(TemporalGraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.adj = adj
        self.weight = Parameter(torch.FloatTensor(dim_X, dim_y))

        if bias:
            self.bias = Parameter(torch.FloatTensor(dim_y))

        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # Forward propagation
    def forward(self, X, adj=None):
        if self.adj is not None:
            res = torch.matmul(self.adj, torch.matmul(X, self.weight))

        elif adj is not None:
            res = torch.matmul(adj, torch.matmul(X, self.weight))

        else:
            raise Exception('No adjacency matrix available.')
        
        if self.bias is not None:
            return res + self.bias
        else:
            return res

    # Weight reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


# Spatial graph convolution
class SpatialGraphConvolution(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj=None, bias=False):
        super(SpatialGraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.adj = adj
        self.weight = Parameter(torch.FloatTensor(dim_X, dim_y))

        if bias:
            self.bias = Parameter(torch.FloatTensor(dim_y))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # Forward propagation
    def forward(self, X, adj=None):
        if self.adj is not None:
            res = torch.matmul(torch.matmul(X, self.adj), self.weight)

        elif adj is not None:
            res = torch.matmul(torch.matmul(X, adj), self.weight)

        else:
            raise Exception('No adjacency matrix available.')
        
        if self.bias is not None:
            return res + self.bias
        
        else:
            return res
    
    # Weight reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class SpatialAdaptiveGraphConvolution(nn.Module):
    def __init__(self, dim_X, dim_y, node_embedding_size, bias=False):
        super(SpatialAdaptiveGraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.node_embedding_size = node_embedding_size
        self.node_embeddings = Parameter(torch.FloatTensor(dim_X, node_embedding_size))

        if bias:
            self.bias = Parameter(torch.FloatTensor(dim_y))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.spatial_gc = SpatialGraphConvolution(dim_X, dim_y)
    

    def forward(self, X):
        adaptive_adj = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        output = self.spatial_gc(X, adaptive_adj)
        return output
    

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.node_embeddings.size(0))
        self.node_embeddings.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class TemporalAdaptiveGraphConvolution(nn.Module):
    def __init__(self, dim_X, dim_y, dim_W, node_embedding_size, bias=False):
        super(TemporalAdaptiveGraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.dim_W = dim_W
        self.node_embedding_size = node_embedding_size
        self.node_embeddings = Parameter(torch.FloatTensor(dim_X, node_embedding_size))

        if bias:
            self.bias = Parameter(torch.FloatTensor(dim_y))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.temporal_gc = TemporalGraphConvolution(dim_W, dim_y)
    

    def forward(self, X):
        adaptive_adj = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        output = self.temporal_gc(X, adaptive_adj)
        return output
    

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.node_embeddings.size(0))
        self.node_embeddings.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


# class TemporalAdaptiveGraphConvolution(nn.Module):
#     def __init__(self, dim_X, dim_y, dim_W, node_embedding_size, bias=False):
#         super(TemporalAdaptiveGraphConvolution, self).__init__()
#         self.dim_X = dim_X
#         self.dim_y = dim_y
#         self.dim_W = dim_W
#         self.node_embedding_size = node_embedding_size
#         self.node_embeddings = Parameter(torch.FloatTensor(dim_X, node_embedding_size))

#         if bias:
#             self.bias = Parameter(torch.FloatTensor(dim_y))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()
        
    

#     def forward(self, X):
#         adaptive_adj = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        
#         return output
    

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.node_embeddings.size(0))
#         self.node_embeddings.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)


# Network
class BidirectionalAdaptiveGraphConvolutionalNetworks(nn.Module):

    # Initialization
    def __init__(self, sp_dim_X, tp_dim_X, dim_y, node_embedding_size=128, gc=(256,), out_fc=(256, 256),
                 adj_mode='pearson', graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1,
                 prob='regression', mode='mvm'):
        super(BidirectionalAdaptiveGraphConvolutionalNetworks, self).__init__()

        # Parameter assignment
        self.tp_dim_X = tp_dim_X
        self.sp_dim_X = sp_dim_X
        self.dim_y = dim_y

        self.node_embedding_size = node_embedding_size
        self.net_tp_gc = [tp_dim_X, ] + list(gc)
        self.net_sp_gc = [sp_dim_X, ] + list(gc)
        self.net_out_fc = [gc[-1], ] + list(out_fc)

        self.adj_mode = adj_mode
        self.graph_reg = graph_reg
        self.self_con = self_con
        self.scale = scale
        self.epsilon = epsilon
        self.prob = prob
        self.mode = mode

        # Model creation
        self.tp_gc = nn.ModuleList()
        self.sp_gc = nn.ModuleList()
        self.out_fc = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.1)

        # Layer normalization
        self.tp_gc_norm = nn.LayerNorm(self.net_tp_gc[-1])
        self.sp_gc_norm = nn.LayerNorm(self.net_sp_gc[-1])

        if prob == 'regression':
            for i in range(dim_y):
                self.tp_gc.append(nn.ModuleList())
                self.sp_gc.append(nn.ModuleList())
                self.out_fc.append(nn.ModuleList())

                for j in range(len(gc)):
                    self.tp_gc[-1].append(TemporalAdaptiveGraphConvolution(self.net_tp_gc[j], self.net_tp_gc[j + 1], sp_dim_X, node_embedding_size))

                for j in range(len(gc)):
                    self.sp_gc[-1].append(SpatialAdaptiveGraphConvolution(self.net_sp_gc[j], self.net_sp_gc[j + 1], node_embedding_size))
                
                for j in range(len(out_fc)):
                    self.out_fc[-1].append(nn.Sequential(nn.Linear(self.net_out_fc[j], self.net_out_fc[j + 1]), self.act))
                    # self.out_fc[-1].append(nn.Linear(self.net_out_fc[j], self.net_out_fc[j + 1]))
                self.out_fc[-1].append(nn.Sequential(nn.Linear(self.net_out_fc[-1], 1), self.act))
        
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        
        # adj = adjacency_matrix(X.cpu().numpy(), self.adj_mode, self.graph_reg, self.self_con, self.scale, self.epsilon)
        # import pdb;pdb.set_trace()
        if self.prob == 'regression':
            res_list = []

            for i in range(self.dim_y):
                spatial_feat, temporal_feat = X, X

                for layer in self.sp_gc[i]:
                    spatial_feat = self.act(layer(spatial_feat))
                # spatial_feat = self.sp_gc_norm(spatial_feat)

                for layer in self.tp_gc[i]:
                    temporal_feat = self.act(layer(temporal_feat))
                # temporal_feat = self.sp_gc_norm(temporal_feat)
                
                feat = spatial_feat + temporal_feat
                # feat = torch.concat((spatial_feat, temporal_feat), axis=1)
                
                for layer in self.out_fc[i]:
                    feat = layer(feat)
                
                res_list.append(feat.squeeze())

            res = torch.stack(res_list, dim=-1)
        
        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class BiagcnModel(NeuralNetwork):

    # Initialization
    def __init__(self, node_embedding_size=128, gc=(1024,), out_fc=(256, 256), prob='regression', mode='mvm', **args):
        super(BiagcnModel, self).__init__()

        # Parameter assignment
        self.node_embedding_size = node_embedding_size
        self.gc = gc
        self.out_fc = out_fc
        self.prob = prob
        self.mode = mode
        self.args['seq_len'] = 64
        self.args['adj_mode'] = 'rbf'
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
        self.data_create(X, y)

        # Model creation
        self.model = BidirectionalAdaptiveGraphConvolutionalNetworks(
            self.dim_X, self.args['seq_len'], self.dim_y, self.node_embedding_size, self.gc, self.out_fc,
            self.args['adj_mode'], self.args['graph_reg'], self.args['self_con'], 
            self.args['scale'], self.args['epsilon'], self.prob, self.mode)
            
        self.model_create('MSE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training()

        return self
