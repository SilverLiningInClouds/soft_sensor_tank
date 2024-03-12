# GCMCGAT
from Base.utils import *
from Base.packages import *

'''
    self.adj----整个数据集的邻接矩阵
    adj----one batch内数据的邻接矩阵

    性能最好的模型只能保留到本文件

    DONE:
    # TODO: 考虑是 <全局稀疏性> 还是 <局部稀疏性>
    # TODO: variable attention
    # TODO: Layer Normalization

'''

# TODO: 稀疏图矩阵只用作对于输入数据的 Mask 操作
# TODO: 输入数据经过 Sptial GC 和 Temporal GC 叠加作用，送至后续网络
# TODO: GC 尽可能改成 Spatial 模式，使用合适的邻接矩阵
# TODO: 调研 GAT 的改造模型，尝试是否能提升性能
# TODO: 三组图卷积也设计为并行结构
# TODO: lambda 修改为可学习的 tensor
# TODO: Decoder 侧的全连接层优化


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
    def __init__(self, dim_X, dim_y, adj=None, bias=False) -> None:
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
        self.input_weights = Parameter(torch.zeros(size=(node_feature_size, node_embedding_size)))
        nn.init.xavier_uniform_(self.input_weights.data, gain=1.414)  # xavier初始化
        self.attn_weights = Parameter(torch.zeros(size=(2 * node_embedding_size, 1)))
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


# Variable Attention Layer
class VariableAttentionLayer(nn.Module):
    def __init__(self, dim_X, epsilon):
        super(VariableAttentionLayer, self).__init__()
        self.hidden_dim = dim_X
        self.epsilon = epsilon
        self.W = Parameter(torch.Tensor(dim_X, dim_X), requires_grad=True)
        self.U = Parameter(torch.Tensor(dim_X, dim_X), requires_grad=True)
        self.bias = Parameter(torch.Tensor(dim_X), requires_grad=True)
        self.reset_parameters()

    def forward(self, X):
        e = torch.tanh(X @ self.U + self.bias) @ self.W
        alpha = F.softmax(e, dim=1)
        return self.U, self.W, alpha * X
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


# Network
class GraphConvolutionMultiChannelGraphAttentionNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj_mode='rbf', gc=(1024, ), in_fc=(512, ), gat=(256, ), out_fc=(256, 256), 
                 graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1):
        super(GraphConvolutionMultiChannelGraphAttentionNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y

        self.adj_mode = adj_mode
        self.graph_reg = graph_reg
        self.self_con = self_con
        self.scale = scale
        self.epsilon = epsilon

        # self.lambda_spca = Parameter(torch.Tensor(1), requires_grad=True)
        # self.lambda_U = Parameter(torch.Tensor(1), requires_grad=True)
        # self.lambda_W = Parameter(torch.Tensor(1), requires_grad=True)
        # self.reset_parameter()

        self.gc = [dim_X, ] + list(gc)
        self.net_in_fc = [gc[-1], ] + list(in_fc)
        self.net_gat = [in_fc[-1], ] + list(gat)
        self.net_out_fc = [gat[-1], ] + list(out_fc)
        self.var_attn_layer = VariableAttentionLayer(dim_X, epsilon)

        # layer output normalization
        self.gc_layer_norm = nn.LayerNorm(self.gc[-1])
        self.gat_layer_norm = nn.LayerNorm(self.net_gat[-1])
        
        # Input FC & Output FC
        self.temporal_gc = nn.ModuleList()
        self.spatial_gc = nn.ModuleList()
        self.in_fc = nn.ModuleList()
        self.out_fc = nn.ModuleList()
        self.act = nn.ReLU()
        

        for i in range(len(self.gc) - 1):
            self.temporal_gc.append(TemporalGraphConvolution(self.gc[i], self.gc[i + 1]))
        
        # for i in range(len(self.gc) - 1):
        #     self.spatial_gc.append(SpatialGraphConvolution(self.gc[i], self.gc[i + 1]))

        for i in range(dim_y):
            self.in_fc.append(nn.ModuleList())
            for j in range(len(in_fc)):
                self.in_fc[-1].append(nn.Sequential(nn.Linear(self.net_in_fc[j], self.net_in_fc[j + 1]), self.act))
            self.out_fc.append(nn.ModuleList())
            for j in range(len(out_fc)):
                self.out_fc[-1].append(nn.Sequential(nn.Linear(self.net_out_fc[j], self.net_out_fc[j + 1]), self.act))
            self.out_fc[-1].append(nn.Linear(self.net_out_fc[-1], 1))

        # GAT
        self.gat = nn.ModuleList()
        for i in range(len(gat)):
            self.gat.append(GAT(self.dim_y, self.net_in_fc[-1], self.net_gat[i], self.net_gat[i + 1]))

    # Forward propagation
    def forward(self, X):
        # TODO: 稀疏图矩阵只用作对于输入数据的 Mask 操作
        # TODO: 输入数据经过 Sptial GC 和 Temporal GC 叠加作用，送至后续网络
        # TODO: GC 尽可能改成 Spatial 模式，使用合适的邻接矩阵
        # TODO: 调研 GAT 的改造模型，尝试是否能提升性能
        # TODO: 三组图卷积也设计为并行结构
        # TODO: lambda 修改为可学习的 tensor
        # TODO: Decoder 侧的全连接层优化
        # 稀疏成分提取/融合，Sparse Component Extractor/ Sparse Component Fusion
        feat_list = []
        res_list = []
        
        sparse_mask = adjacency_matrix(X.cpu().numpy(), 'spca', self.graph_reg, self.self_con, self.scale, self.epsilon)
        temporal_adj = adjacency_matrix(X.cpu().numpy(), self.adj_mode, self.graph_reg, self.self_con, self.scale, self.epsilon)
        U, W, attn_X = self.var_attn_layer(X)
        
        zero_tensor = torch.zeros_like(U)
        U = torch.where(torch.abs(U) > self.epsilon, U, zero_tensor)
        W = torch.where(torch.abs(W) > self.epsilon, W, zero_tensor)
        
        X = X + 0.1 * torch.matmul(X, sparse_mask) + 0.05 * torch.matmul(X, U) + 0.05 * torch.matmul(X, W)
        # X = X + self.lambda_spca * torch.matmul(X, sparse_mask) + self.lambda_U * torch.matmul(X, U) + self.lambda_W * torch.matmul(X, W)
        # X = X + 0.1 * torch.matmul(X, sparse_mask) + 0.1 * torch.matmul(X, W) 
        
        # Input GC
        for gc in self.temporal_gc:
            graph_X = gc(X, temporal_adj)
        
        graph_X = self.gc_layer_norm(graph_X)

        # Multi FC
        for i in range(self.dim_y):
            feat = graph_X
            for j in self.in_fc[i]:
                feat = j(feat)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=1)
        
        # GAT
        for gat_layer in self.gat:
            feat = gat_layer(feat)
            feat = self.act(feat)
        
        feat = self.gat_layer_norm(feat)

        # Output FC
        for i in range(self.dim_y):
            res = feat[:, i, :]
            for j in self.out_fc[i]:
                res = j(res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=1)

        return res
    
    # def reset_parameter(self):
    #     self.lambda_spca = torch.tensor(0.1)
    #     self.lambda_U = torch.tensor(0.1)
    #     self.lambda_W = torch.tensor(0.1)


# Model
class GcmcgatModel(NeuralNetwork):

    # Initialization
    def __init__(self, gc=(1024, ), in_fc=(512, ), gat=(256, ), out_fc=(256, 256), prob='regression', **args):
        super(GcmcgatModel, self).__init__()

        # Parameter assignment
        self.gc = gc
        self.in_fc = in_fc
        self.gat = gat
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
        self.model = GraphConvolutionMultiChannelGraphAttentionNetworks(
            self.dim_X, self.dim_y, self.args['adj_mode'], self.gc, self.in_fc, self.gat, self.out_fc, 
            self.args['graph_reg'], self.args['self_con'], self.args['scale'], self.args['epsilon'])
        
        self.model_create()

        # Model training
        self.training()

        return self