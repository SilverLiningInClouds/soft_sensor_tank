# RMAGCN: Recurrent Multi Adaptive Graph Convolutional Networks
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


class MultiAdaptiveGraphConvolution(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, dim_hidden, order, node_embedding_size, bias=False):
        super(MultiAdaptiveGraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.order = order
        self.node_embedding_size = node_embedding_size
        self.node_embeddings_list = []
        
        for _ in range(order):
            self.node_embeddings = Parameter(torch.FloatTensor(dim_X, node_embedding_size))            
            self.reset_parameters()
            self.node_embeddings_list.append(self.node_embeddings)

        self.spatial_gc = SpatialGraphConvolution(dim_X, dim_hidden)
        self.gc_layer_norm = nn.LayerNorm(dim_hidden)
        self.out_fc = nn.Sequential(nn.Linear(dim_hidden * order, dim_y), nn.ReLU())
    
    # Forward propagation
    def forward(self, X):
        output_list = []
        
        for node_embedding in self.node_embeddings_list:
            adaptive_adj = F.softmax(F.relu(torch.mm(node_embedding, node_embedding.transpose(0, 1))), dim=1)
            output = self.spatial_gc(X, adaptive_adj)
            # output = self.gc_layer_norm(output)
            output_list.append(output)
        
        output_feat = torch.concat(output_list, dim=1)
        output_feat = self.out_fc(output_feat)
        return output_feat
    
    # Weight reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim_X)
        self.node_embeddings.data.uniform_(-stdv, stdv)


# Network
class RecurrentMultiAdaptiveGraphConvolutionalNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, dim_hidden=256, order=3, node_embedding_size=128, 
                 rnn=(1024,), gc=(1024,), out_fc=(256, 256),
                 adj_mode='pearson', graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1,
                 prob='regression'):
        super(RecurrentMultiAdaptiveGraphConvolutionalNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.dim_hidden = dim_hidden
        self.order = order
        self.node_embedding_size = node_embedding_size

        self.net_rnn = [dim_X, ] + list(rnn)
        self.net_gc = [dim_X, ] + list(gc)
        self.net_out_fc = [gc[-1] + rnn[-1], ] + list(out_fc)

        self.adj_mode = adj_mode
        self.graph_reg = graph_reg
        self.self_con = self_con
        self.scale = scale
        self.epsilon = epsilon
        self.prob = prob

        # Model creation
        self.rnn = nn.ModuleList()
        self.gc = nn.ModuleList()
        self.out_fc = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.gc_layer_norm = nn.LayerNorm(self.net_gc[-1])

        if prob == 'regression':
            for _ in range(dim_y):
                self.rnn.append(nn.ModuleList())
                for j in range(len(rnn)):
                    self.rnn[-1].append(nn.GRU(self.net_rnn[j], self.net_rnn[j + 1]))

                self.gc.append(nn.ModuleList())
                for j in range(len(gc)):
                    self.gc[-1].append(MultiAdaptiveGraphConvolution(self.net_gc[j], self.net_gc[j + 1], self.dim_hidden, self.order, self.node_embedding_size))
                
                self.out_fc.append(nn.ModuleList())
                for j in range(len(out_fc)):
                    self.out_fc[-1].append(nn.Sequential(nn.Linear(self.net_out_fc[j], self.net_out_fc[j + 1]), self.act))
                self.out_fc[-1].append(nn.Linear(self.net_out_fc[-1], 1))
        
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        # adj = adjacency_matrix(X.cpu().numpy(), self.adj_mode, self.graph_reg, self.self_con, self.scale, self.epsilon)
        
        if self.prob == 'regression':
            feat_list = []
            res_list = []
            
            for i in range(self.dim_y):
                rnn_feat = X
                for layer in self.rnn[i]:
                    rnn_feat, _ = layer(rnn_feat)

                gc_feat = X
                for layer in self.gc[i]:
                    gc_feat = self.act(layer(gc_feat))
                gc_feat = self.gc_layer_norm(gc_feat)
                
                feat = torch.concat((rnn_feat, gc_feat), axis=1)
                feat_list.append(feat)
            
            feat = torch.stack(feat_list, dim=1)
            
            for i in range(self.dim_y):
                res = feat[:, i, :]
                for fc_layer in self.out_fc[i]:
                    res = fc_layer(res)
                res_list.append(res.squeeze())
            res = torch.stack(res_list, dim=-1)
        
        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class RmagcnModel(NeuralNetwork):

    # Initialization
    def __init__(self, dim_hidden=256, order=3, node_embedding_size=128, rnn=(1024,), gc=(1024,), out_fc=(256, 256), prob='regression', **args):
        super(RmagcnModel, self).__init__()

        # Parameter assignment
        self.dim_hidden = dim_hidden
        self.order = order
        self.node_embedding_size = node_embedding_size
        self.rnn = rnn
        self.gc = gc
        self.out_fc = out_fc
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
        self.model = RecurrentMultiAdaptiveGraphConvolutionalNetworks(
            self.dim_X, self.dim_y, self.dim_hidden, self.order, self.node_embedding_size, 
            self.rnn, self.gc, self.out_fc,
            self.args['adj_mode'], self.args['graph_reg'], self.args['self_con'], 
            self.args['scale'], self.args['epsilon'], self.prob)
            
        self.model_create('MSE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training()

        return self
