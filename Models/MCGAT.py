# MCGAT: Multi-Channel Graph Attention Networks
from Base.utils import *
from Base.packages import *

'''
    self.adj计算的是整个数据集的邻接矩阵
    adj计算的是一个batch内数据的邻接矩阵

'''


class GraphAttentionLayer(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 node_feature_size, 
                 node_embedding_size, 
                 dropout_prob, 
                 alpha) -> None:
        super(GraphAttentionLayer, self).__init__()
        self.num_nodes = num_nodes
        self.node_feature_size = node_feature_size
        self.node_embedding_size = node_embedding_size
        self.act_func = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout_prob)
        self.input_weights = nn.Parameter(torch.zeros(size=(node_feature_size, node_embedding_size)))
        nn.init.xavier_uniform_(self.input_weights.data, gain=1.414)  # xavier初始化
        self.attn_weights = nn.Parameter(torch.zeros(size=(2 * node_embedding_size, 1)))
        nn.init.xavier_uniform_(self.attn_weights.data, gain=1.414)  # xavier初始化


    def forward(self, x):
        # if add_input_weight:
        # TODO: consider to add adj matrix
        
        hidden_states = torch.matmul(x, self.input_weights)
        nodes_concat_feature = torch.cat([hidden_states.repeat(1, 1, self.num_nodes).view(x.shape[0], self.num_nodes ** 2, -1), \
                                          hidden_states.repeat(1, self.num_nodes, 1)], dim=-1). \
                                            view(x.shape[0], self.num_nodes, -1, 2 * self.node_embedding_size)
        e = self.act_func(torch.matmul(nodes_concat_feature, self.attn_weights))
        # TODO: consider to construct a mask matrix between adj matrix and attention map
        #       to make the attention map sparse
        # TODO: consider to add skip connections
        attention = F.softmax(e, dim=-2)
        attention = self.dropout(attention).squeeze(dim=-1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        output_feature = torch.matmul(attention, hidden_states)
        
        # output_feature = nn.Sigmoid(torch.sum(output_feature, dim=1))
        return output_feature
    

    def reset_input_parameters(self):
        std = 1. / math.sqrt(self.input_weights.size(0))
        self.input_weights.data.uniform_(-std, std)


    def reset_attn_parameters(self):
        std = 1. / math.sqrt(self.attn_weights.size(0))
        self.attn_weights.data.uniform_(-std, std)


# TODO: feature dimension GAT and temporal dimension GAT
class GAT(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 node_feature_size, 
                 node_embedding_size, 
                 output_size,
                 num_heads=1, 
                 dropout=0.2, 
                 alpha=0.2, 
                 ) -> None:
        super(GAT, self).__init__()
        self.gat_layers = [GraphAttentionLayer(num_nodes, node_feature_size, node_embedding_size, dropout, alpha) \
                           for _ in range(num_heads)]
        
        for idx, attn_map in enumerate(self.gat_layers):
            self.add_module('Attention Map {}'.format(idx), attn_map)

        self.out_attn_layer = GraphAttentionLayer(num_nodes, node_embedding_size * num_heads, output_size, dropout, alpha)
        

    def forward(self, x):
        
        gat_output = torch.cat([layer(x) for layer in self.gat_layers], dim=-1)
        gat_output = self.out_attn_layer(gat_output)
        return gat_output


# Network
class MultiChannelGraphAttentionNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj, in_fc=(1024,), gc=(256,), out_fc=(256, 256)):
        super(MultiChannelGraphAttentionNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.adj = adj
        self.net_in_fc = [dim_X, ] + list(in_fc)
        self.net_gc = [in_fc[-1], ] + list(gc)
        self.net_out_fc = [gc[-1], ] + list(out_fc)

        # Input FC & Output FC
        self.in_fc = nn.ModuleList()
        self.out_fc = nn.ModuleList()
        self.act = nn.ReLU()
        
        for i in range(dim_y):
            self.in_fc.append(nn.ModuleList())
            for j in range(len(in_fc)):
                self.in_fc[-1].append(nn.Sequential(nn.Linear(self.net_in_fc[j], self.net_in_fc[j + 1]), self.act))
            self.out_fc.append(nn.ModuleList())
            for j in range(len(out_fc)):
                self.out_fc[-1].append(nn.Sequential(nn.Linear(self.net_out_fc[j], self.net_out_fc[j + 1]), self.act))
            self.out_fc[-1].append(nn.Linear(self.net_out_fc[-1], 1))

        # GC
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GAT(self.dim_y, self.net_in_fc[-1], self.net_gc[i], self.net_gc[i + 1]))

    # Forward propagation
    def forward(self, X):
        feat_list = []
        res_list = []
        
        # Input FC
        for i in range(self.dim_y):
            feat = X
            for j in self.in_fc[i]:
                feat = j(feat)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=1)
        
        # GC
        for gc in self.gc:
            feat = gc(feat)
            feat = self.act(feat)

        # Output FC
        for i in range(self.dim_y):
            res = feat[:, i, :]
            for j in self.out_fc[i]:
                res = j(res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=1)

        return res


# Model
class McgatModel(NeuralNetwork):

    # Initialization
    def __init__(self, in_fc=(1024,), gc=(256,), out_fc=(256, 256), prob='regression', **args):
        super(McgatModel, self).__init__()

        # Parameter assignment
        self.in_fc = in_fc
        self.gc = gc
        self.out_fc = out_fc
        self.prob = prob
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
        self.data_create(X, y, adj_X=True, adj_Y=False)
        
        # Model creation
        self.model = MultiChannelGraphAttentionNetworks(self.dim_X, self.dim_y, self.adj, self.in_fc, self.gc, self.out_fc)
        self.model_create()

        # Model training
        self.training()

        return self