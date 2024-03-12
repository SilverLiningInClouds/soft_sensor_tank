# BiGCN: Bi-directional Graph Convolutional Networks
from Base.utils import *
from Base.packages import *


# Graph convolution
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


class SpatialGraphConvolution(nn.Module):
    def __init__(self, dim_X, dim_y, adj=None, bias=False) -> None:
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


# Network
class BidirectionalGraphConvolutionalNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, gc=(1024,), adj_mode='rbf', graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1,
                 prob='regression'):
        super(BidirectionalGraphConvolutionalNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net = [dim_X, ] + list(gc)
        self.adj_mode = adj_mode
        self.graph_reg = graph_reg
        self.self_con = self_con
        self.scale = scale
        self.epsilon = epsilon
        self.prob = prob

        # Model creation
        self.temporal_gc = nn.ModuleList()
        self.spatial_gc = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.1)

        # Layer normalization
        self.spatial_gc_norm = nn.LayerNorm(self.net[-1])
        self.temporal_gc_norm = nn.LayerNorm(self.net[-1])

        if prob == 'regression':
            for i in range(dim_y):
                self.spatial_gc.append(nn.ModuleList())
                self.temporal_gc.append(nn.ModuleList())
                
                for j in range(len(gc)):
                    self.spatial_gc[-1].append(SpatialGraphConvolution(self.net[j], self.net[j + 1]))
                    self.temporal_gc[-1].append(TemporalGraphConvolution(self.net[j], self.net[j + 1]))
                
                # 全连接只有一层，有多层时需要重写
                self.fc.append(nn.Linear(gc[-1] * 2, 1))
        
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        
        spatial_adj = adjacency_matrix(X.T.cpu().numpy(), self.adj_mode, self.graph_reg, self.self_con, self.scale, self.epsilon)
        temporal_adj = adjacency_matrix(X.cpu().numpy(), self.adj_mode, self.graph_reg, self.self_con, self.scale, self.epsilon)
        
        if self.prob == 'regression':
            res_list = []

            for i in range(self.dim_y):
                spatial_feat, temporal_feat = X, X

                for layer in self.spatial_gc[i]:
                    # TODO: add norm layer
                    spatial_feat = self.act(layer(spatial_feat, spatial_adj))
                spatial_feat = self.spatial_gc_norm(spatial_feat)

                for layer in self.temporal_gc[i]:
                    temporal_feat = self.act(layer(temporal_feat, temporal_adj))
                temporal_feat = self.temporal_gc_norm(temporal_feat)
                
                feat = torch.concat((spatial_feat, temporal_feat), axis=1)
                feat = self.fc[i](feat)
                res_list.append(feat.squeeze())
                
            res = torch.stack(res_list, dim=-1)
        
        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class BigcnModel(NeuralNetwork):

    # Initialization
    def __init__(self, gc=(1024,), prob='regression', **args):
        super(BigcnModel, self).__init__()

        # Parameter assignment
        self.gc = gc
        self.prob = prob
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
        self.model = BidirectionalGraphConvolutionalNetworks(self.dim_X, self.dim_y, self.gc, self.args['adj_mode'], 
                                                    self.args['graph_reg'], self.args['self_con'], self.args['scale'], 
                                                    self.args['epsilon'], self.prob)
        self.model_create('MSE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training()

        return self
