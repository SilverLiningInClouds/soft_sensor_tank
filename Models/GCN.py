# GCN: Graph Convolutional Networks
from Base.utils import *
from Base.packages import *


# Graph convolution
class GraphConvolution(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj=None, bias=False):
        super(GraphConvolution, self).__init__()
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


# Network
class GraphConvolutionalNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, gc=(256,), adj_mode='pearson', graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1,
                 prob='regression'):
        super(GraphConvolutionalNetworks, self).__init__()

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
        self.gc = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.1)
        
        if prob == 'regression':
            for i in range(dim_y):
                self.gc.append(nn.ModuleList())
                for j in range(len(gc)):
                    self.gc[-1].append(GraphConvolution(self.net[j], self.net[j + 1]))
                self.fc.append(nn.Linear(gc[-1], 1))
        
        elif prob == 'classification':
            for i in range(len(gc)):
                self.gc.append(GraphConvolution(self.net[i], self.net[i + 1]))
            self.fc = nn.Linear(gc[-1], dim_y)
        
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        
        adj = adjacency_matrix(X.cpu().numpy(), self.adj_mode, self.graph_reg, self.self_con, self.scale, self.epsilon)
        
        if self.prob == 'regression':
            res_list = []

            for i in range(self.dim_y):
                feat = X
                for layer in self.gc[i]:
                    # TODO: add BN layer
                    feat = self.act(layer(feat, adj))
                feat = self.fc[i](feat)
                res_list.append(feat.squeeze())
            
            res = torch.stack(res_list, dim=-1)
        
        elif self.prob == 'classification':
            res = X
            for i in self.gc:
                res = i(res, adj)
            res = self.fc(res)
        
        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class GcnModel(NeuralNetwork):

    # Initialization
    def __init__(self, gc=(1024,), prob='regression', **args):
        super(GcnModel, self).__init__()

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
        self.model = GraphConvolutionalNetworks(self.dim_X, self.dim_y, self.gc, self.args['adj_mode'],
                                                self.args['graph_reg'], self.args['self_con'], self.args['scale'],
                                                self.args['epsilon'], self.prob)
        self.model_create('MSE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training()

        return self
