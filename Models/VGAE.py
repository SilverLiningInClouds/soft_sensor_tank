# VGAE: Variational Graph Auto-Encoder
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


# Network
class VariationalGraphAutoEncoderNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, dim_hidden, dim_latent, 
                 base_adj_mode='mic', latent_adj_mode='sc', graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1):
        super(VariationalGraphAutoEncoderNetworks, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent

        self.base_adj_mode = base_adj_mode
        self.latent_adj_mode = latent_adj_mode
        self.graph_reg = graph_reg
        self.self_con = self_con
        self.scale = scale
        self.epsilon = epsilon

        self.base_gcn = TemporalGraphConvolution(dim_X, dim_hidden)
        self.mean_gcn = TemporalGraphConvolution(dim_hidden, dim_latent)
        self.std_gcn = TemporalGraphConvolution(dim_hidden, dim_latent)
        self.proj = TemporalGraphConvolution(dim_latent, dim_y)
    
    # Encoding: sampling
    def encoder(self, X):
        self.base_adj = adjacency_matrix(X.cpu().numpy(), self.base_adj_mode, self.graph_reg, self.self_con, self.scale, self.epsilon)
        hidden = self.base_gcn(X, self.base_adj)
        # latent_adj = adjacency_matrix(hidden.detach().numpy(), self.latent_adj_mode, self.graph_reg, self.self_con, self.scale, self.epsilon)
        self.mean = self.mean_gcn(hidden, self.base_adj)
        self.std = self.std_gcn(hidden, self.base_adj)
        gaussian_noise = torch.randn(X.size(0), self.dim_latent)
        sampled_z = gaussian_noise * torch.exp(self.std) + self.mean

        return sampled_z
    
    # Decoding: reparameterizing
    def decoder(self, z):
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))

        return adj_pred

    # Forward propagation
    def forward(self, X):
        z = self.encoder(X)
        adj_pred = self.decoder(z)
        output = self.proj(z, adj_pred)

        return output, self.mean, self.std
    

# Model
class VgaeModel(NeuralNetwork):

    # Initialization
    def __init__(self, dim_hidden=256, dim_latent=3, prob='regression', **args):
        super(VgaeModel, self).__init__()

        # Parameter assignment
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        
        self.args['alpha'] = 0.1
        self.args['base_adj_mode'] = 'rbf'
        self.args['latent_adj_mode'] = 'mic'
        self.args['graph_reg'] = 0.05
        self.args['self_con'] = 0.2
        self.args['scale'] = 0.4
        self.args['epsilon'] = 0.1
        self.args['need_pretraining'] = False

        self.prob = prob
        self.args.update(args)

        # Set seed
        torch.manual_seed(self.args['seed'])

    # Train
    def fit(self, X, y):
        # Data creation
        self.data_create(X, y)

        # Model creation
        self.model = VariationalGraphAutoEncoderNetworks(
            self.dim_X, self.dim_y, self.dim_hidden, self.dim_latent, self.args['base_adj_mode'], self.args['latent_adj_mode'], 
            self.args['graph_reg'], self.args['self_con'], self.args['scale'], self.args['epsilon'])
            
        self.model_create('VAE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training()

        return self