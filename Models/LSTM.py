# LSTM: Long Short-Term Memory
import torch
import torch.nn as nn
from Base.utils import *


# Network
class LongShortTermMemory(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, lstm=(1024,), mode='mvo', prob='regression'):
        # mvm: multi input & multi output
        # mvo: multi input & one output
        super(LongShortTermMemory, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.net = [dim_X, ] + list(lstm)
        self.mode = mode
        self.prob = prob

        # Model creation
        self.lstm = nn.ModuleList()
        self.fc = nn.ModuleList()
        if prob == 'regression':
            for i in range(dim_y):
                self.lstm.append(nn.ModuleList())
                for j in range(len(lstm)):
                    self.lstm[-1].append(nn.GRU(self.net[j], self.net[j + 1], batch_first=True))
                self.fc.append(nn.Linear(lstm[-1], 1))
        elif prob == 'classification':
            for i in range(len(lstm)):
                self.lstm.append(nn.GRU(self.net[i], self.net[i + 1], batch_first=True))
            self.fc = nn.Linear(lstm[-1], dim_y)
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        if self.prob == 'regression':
            res_list = []

            for i in range(self.dim_y):
                feat = X
                for j in self.lstm[i]:
                    feat, _ = j(feat)
                if self.mode == 'mvm':
                    feat = self.fc[i](feat)
                elif self.mode == 'mvo':
                    feat = self.fc[i](_[0])
                else:
                    raise Exception('Wrong mode selection.')
                res_list.append(feat.squeeze())
            res = torch.stack(res_list, dim=-1)
        elif self.prob == 'classification':
            res = X
            for i in self.lstm:
                res, _ = i(res)
            if self.mode == 'mvm':
                res = self.fc(res).view(-1, self.dim_y)
            elif self.mode == 'mvo':
                res = self.fc(_[0]).squeeze()
            else:
                raise Exception('Wrong mode selection.')
        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class LstmModel(NeuralNetwork):

    # Initialization
    def __init__(self, lstm=(1024,), mode='mvm', prob='regression', **args):
        super(LstmModel, self).__init__()

        # Parameter assignment
        self.lstm = lstm
        self.mode = mode
        self.prob = prob
        self.args['seq_len'] = 30
        self.args['need_pretraining'] = False
        self.args.update(args)

        # Set seed
        torch.manual_seed(self.args['seed'])

    # Train
    def fit(self, X, y):
        # Data creation
        self.data_create(X, y)

        # Model creation
        # self.model = LongShortTermMemory(self.dim_X, self.dim_y, self.lstm, self.mode, self.prob).cuda(self.args['gpu'])
        self.model = LongShortTermMemory(self.dim_X, self.dim_y, self.lstm, self.mode, self.prob)
        self.model_create('MSE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training()

        return self
