# VAE: Variational Auto-Encoders
import torch
import torch.nn as nn
from Base.utils import *


# Network
class VariationalAutoEncoders(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_z=128, hidden_layers=(256,)):
        super(VariationalAutoEncoders, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_z = dim_z
        self.hidden_layers = hidden_layers
        self.net_encoder = [dim_X, ] + list(hidden_layers)
        self.net_decoder = [dim_z, ] + list(reversed(hidden_layers))

        # Model creation
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(len(hidden_layers)):
            self.encoder.append(nn.Sequential(nn.Linear(self.net_encoder[i], self.net_encoder[i + 1]), self.act))
            self.decoder.append(nn.Sequential(nn.Linear(self.net_decoder[i], self.net_decoder[i + 1]), self.act))
        self.decoder.append(nn.Linear(hidden_layers[0], dim_X))
        self.mu = nn.Linear(hidden_layers[-1], dim_z)
        self.logvar = nn.Linear(hidden_layers[-1], dim_z)
        self.proj = nn.Linear(dim_X, 1)


    # Forward propagation
    def forward(self, X, transform=False):

        # Encoder
        feat = X
        for i in self.encoder:
            feat = i(feat)
        
        mu = self.mu(feat)
        logvar = self.logvar(feat)

        # Reparameter
        z = self.reparameterize(mu, logvar)
        if transform:
            return z

        # Decoder
        res = z
        for i in self.decoder:
            res = i(res)
        res = self.proj(res)

        return res, mu, logvar

    # Reparameter trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


# Model
class VaeModel(NeuralNetwork):

    # Initialization
    def __init__(self, if_pretrain=False, **args):
        super(VaeModel, self).__init__()

        # Parameter assignment
        self.args['dim_z'] = 2
        self.args['hidden_layers'] = (256, 128)
        self.args['alpha'] = 0.1

        self.args['need_pretraining'] = if_pretrain
        # self.prob = 'dimensionality-reduction'
        self.prob = 'regression'
        self.args.update()

        # Set seed
        torch.manual_seed(self.args['seed'])

    # Fit
    def fit(self, X, y=None):
        # Data creation
        self.data_create(X, X)

        # Model creation
        self.model = VariationalAutoEncoders(self.dim_X, self.args['dim_z'], self.args['hidden_layers'])
        self.model_create('VAE')

        # Model training
        self.training()

        return self
