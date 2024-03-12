# SS-SAE: Semi-Supervised Stacked Auto-Encoder
from Base.utils import *
from Base.packages import *


# Auto-Encoder Networks
class AutoEncoder(nn.Module):
    def __init__(self, dim_X, dim_H):
        super(AutoEncoder, self).__init__()
        self.dim_X = dim_X
        self.dim_H = dim_H
        self.act = nn.Sigmoid()

        self.encoder = nn.Linear(dim_X, dim_H, bias=True)
        self.decoder = nn.Linear(dim_H, dim_X, bias=True)

    def forward(self, X, rep=False):
        H = self.act(self.encoder(X))
        if rep is False:
            return self.act(self.decoder(H))
        else:
            return H


# Stacked Auto-Encoder Networks
class StackedAutoEncoder(nn.Module):
    def __init__(self, layers_list):
        super(StackedAutoEncoder, self).__init__()
        self.num_layers = len(layers_list)
        self.layers = []

        for i in range(1, self.num_layers):
            self.layers.append(AutoEncoder(layers_list[i-1], layers_list[i]))

        self.proj = nn.Linear(layers_list[self.num_layers - 1], 1)

    def forward(self, X, layer_id, is_pretraining=False):
        """
        :param X: 进口参数
        :param NoL: 第几层
        :param PreTrain: 是不是无监督预训练
        :return:
        """
        out = X
        if is_pretraining:
            # SAE的预训练
            if layer_id == 0:
                return out, self.layers[layer_id](out)

            else:
                for i in range(layer_id):
                    # 第N层之前的参数给冻住
                    for param in self.layers[i].parameters():
                        param.requires_grad = False

                    out = self.layers[i](out, rep=True)
                # 训练第N层
                inputs = out
                out = self.layers[layer_id](out)
                return inputs, out
        else:
            for i in range(self.num_layers - 1):
                # 做微调
                for param in self.layers[i].parameters():
                    param.requires_grad = True

                out = self.layers[i](out, rep=True)
            out = torch.sigmoid(self.proj(out))
            return out


# Model
class SaeModel(NeuralNetwork):

    # Initialization
    def __init__(self, layers_list=[11, 7], unsupervised_epoch=500, supervised_epoch=500, 
                 unsupervised_bs=256, supervised_bs=256, unsupervised_lr=0.01, supervised_lr=0.01, 
                 if_pretrain=True, prob='regression', **args):
        super(SaeModel, self).__init__()

        # Parameter assignment
        self.layers_list = layers_list
        self.args['num_layers'] = len(layers_list) - 1

        self.args['unspv_epoch'] = unsupervised_epoch
        self.args['spv_epoch'] = supervised_epoch

        self.args['unspv_bs'] = unsupervised_bs
        self.args['spv_bs'] = supervised_bs

        self.args['unspv_lr'] = unsupervised_lr
        self.args['spv_lr'] = supervised_lr

        self.args['need_pretraining'] = if_pretrain
        self.prob = prob
        self.args.update(args)

        # Set seed
        torch.manual_seed(self.args['seed'])
    
    # Train
    def fit(self, X, y):
        # Data creation
        self.data_create(X, y)

        # Model creation
        self.model = StackedAutoEncoder(self.layers_list)

        self.model_create(need_pretrain=True, loss='MSE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training(need_pretrain=True)

        return self
