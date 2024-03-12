# GC-LSTM: Graph Convolution Long Short-Term Memory
from Base.utils import *
from Base.packages import *


# Network
class GraphConvolutionLongShortTermMemory(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj, lstm=(1024,), gc=(256,), fc=(256, 256), mode='mvo'):
        super(GraphConvolutionLongShortTermMemory, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.adj = adj
        self.net_lstm = [dim_X, ] + list(lstm)
        self.net_gc = [lstm[-1], ] + list(gc)
        self.net_fc = [gc[-1], ] + list(fc)
        self.mode = mode

        # LSTM & FC
        self.lstm = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(dim_y):
            self.lstm.append(nn.ModuleList())
            for j in range(len(lstm)):
                self.lstm[-1].append(nn.LSTM(self.net_lstm[j], self.net_lstm[j + 1], batch_first=True))
            self.fc.append(nn.ModuleList())
            for j in range(len(fc)):
                self.fc[-1].append(nn.Sequential(nn.Linear(self.net_fc[j], self.net_fc[j + 1]), self.act))
            self.fc[-1].append(nn.Linear(self.net_fc[-1], 1))

        # GC
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GraphConvolution(self.net_gc[i], self.net_gc[i + 1], self.adj))

    # Forward propagation
    def forward(self, X):
        feat_list = []
        res_list = []
        
        # LSTM
        for i in range(self.dim_y):
            feat = X

            for j in self.lstm[i]:
                feat, _ = j(feat)

            if self.mode == 'mvm':
                feat_list.append(feat)

            elif self.mode == 'mvo':
                feat_list.append(_[0])

            else:
                raise Exception('Wrong mode selection.')
            
        feat = torch.stack(feat_list, dim=-2)
        
        # GC
        for gc in self.gc:
            feat = gc(feat)
            feat = self.act(feat)
        
        # FC
        for i in range(self.dim_y):
            res = feat[:, :, i, :]

            for j in self.fc[i]:
                res = j(res)

            res_list.append(res.squeeze())
            
        res = torch.stack(res_list, dim=-1)
        
        return res


# Model
class GclstmModel(NeuralNetwork):

    # Initialization
    def __init__(self, lstm=(256,), gc=(128,), fc=(128, 128), mode='mvm', **args):
        super(GclstmModel, self).__init__()

        # Parameter assignment
        self.lstm = lstm
        self.gc = gc
        self.fc = fc
        self.mode = mode
        self.prob = 'regression'
        self.args['seq_len'] = 30
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
        self.data_create(X, y, adj_X=False, adj_Y=True)

        # Model creation
        self.model = GraphConvolutionLongShortTermMemory(self.dim_X, self.dim_y, self.adj, self.lstm, self.gc, self.fc, self.mode)

        self.model_create()

        # Model training
        self.training()

        return self
