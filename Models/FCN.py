# FCN: Fully Connected Networks
from Base.utils import *
from Base.packages import *
from sklearn.neural_network import MLPRegressor, MLPClassifier


# Network
class FullyConnectedNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, hidden_layers=(256,), prob='regression'):
        super(FullyConnectedNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net = [dim_X, ] + list(hidden_layers)
        self.prob = prob

        # Model creation
        self.fc = nn.ModuleList()
        self.act = nn.ReLU()
        if prob == 'regression':
            for i in range(dim_y):
                self.fc.append(nn.ModuleList())
                for j in range(len(hidden_layers)):
                    self.fc[-1].append(
                        nn.Sequential(nn.Linear(self.net[j], self.net[j + 1]), self.act))
                self.fc[-1].append(nn.Linear(self.net[-1], 1))
        elif prob == 'classification':
            for i in range(len(hidden_layers)):
                self.fc.append(
                    nn.Sequential(nn.Linear(self.net[i], self.net[i + 1]), self.act))
            self.fc.append(nn.Linear(self.net[-1], dim_y))
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        if self.prob == 'regression':
            res_list = []

            for i in range(self.dim_y):
                feat = X
                for j in self.fc[i]:
                    feat = j(feat)
                res_list.append(feat.squeeze())

            res = torch.stack(res_list, dim=1)
        elif self.prob == 'classification':
            res = X
            for i in self.fc:
                res = i(res)
        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class FcnModel(NeuralNetwork):

    # Initialization
    def __init__(self, hidden_layer_sizes=(256,), prob='regression', **args):
        super(FcnModel, self).__init__()

        # Parameter assignment
        self.hidden_layer_sizes = hidden_layer_sizes
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
        self.model = FullyConnectedNetworks(self.dim_X, self.dim_y, self.hidden_layer_sizes, self.prob)
        # self.model = FullyConnectedNetworks(self.dim_X, self.dim_y, self.hidden_layer_sizes, self.prob).cuda(self.args['gpu'])
        self.model_create('MSE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training()

        return self
