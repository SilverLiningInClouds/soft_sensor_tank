# AE: Auto-Encoders
from Base.utils import *
from Base.packages import *

# Network
class AutoEncoders(nn.Module):

    # Initialization
    def __init__(self, dim_X, hidden_layers=(256,)):
        super(AutoEncoders, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.hidden_layers = hidden_layers
        self.net = [dim_X, ] + list(hidden_layers) + list(reversed(hidden_layers[:-1]))

        # Model creation
        self.fc = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(2 * len(hidden_layers) - 1):
            self.fc.append(
                nn.Sequential(nn.Linear(self.net[i], self.net[i + 1]), self.act))
        self.fc.append(nn.Linear(hidden_layers[0], dim_X))

    # Forward propagation
    def forward(self, X, transform=False):
        res = X
        if transform:
            for i in range(len(self.hidden_layers)):
                res = self.fc[i](res)
        else:
            for i in self.fc:
                res = i(res)
        return res


# Model
class AeModel(NeuralNetwork):

    # Initialization
    def __init__(self, **args):
        super(AeModel, self).__init__()

        # Parameter assignment
        self.args['hidden_layers'] = (256, 128, 2)
        self.prob = 'dimensionality-reduction'
        self.args.update(args)

        # Set seed
        torch.manual_seed(self.args['seed'])

    # Fit
    def fit(self, X, y=None):
        # Data creation
        self.data_create(X, X)

        # Model creation
        self.model = AutoEncoders(self.dim_X, self.args['hidden_layers'])
        self.model_create()

        # Model training
        self.training()

        return self
