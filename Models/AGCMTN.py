# AGCMTN: Adaptive graph convolution multi task networks (PLE: Progressive layers extraction)
# Down-to-date Baseline
from Base.utils import *
from Base.packages import *


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
    def __init__(self, dim_X, dim_hidden, dim_y, node_embedding_size, order, bias=False):
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
            output_list.append(output)
        
        output_feat = torch.concat(output_list, dim=1)
        
        output_feat = self.out_fc(output_feat)
        return output_feat
    
    # Weight reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim_X)
        self.node_embeddings.data.uniform_(-stdv, stdv)


class ExpertFullyConnectedNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_hidden, dim_y):
        super(ExpertFullyConnectedNetworks, self).__init__()
        self.fc1 = nn.Linear(dim_X, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_y)
        self.act = nn.ReLU()

    # Forward propagation
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)

        return out


class ExpertGraphNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_hidden, dim_y, node_embedding_size, order):
        super(ExpertGraphNetworks, self).__init__()
        self.layer = MultiAdaptiveGraphConvolution(dim_X, dim_hidden, dim_y, node_embedding_size, order)
    
    # Forward propagation
    def forward(self, X):

        return self.layer(X)


class GatesInteractionControlNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_hidden, dim_y, node_embedding_size, order, num_specific_experts, num_shared_experts, is_last=True):
        super(GatesInteractionControlNetworks, self).__init__()
        self.dim_X = dim_X
        self.dim_hidden = dim_hidden
        self.dim_y = dim_y
        self.node_embedding_size = node_embedding_size
        self.order = order

        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.is_last = is_last

        self.shared_experts = nn.ModuleList([ExpertGraphNetworks(self.dim_X, self.dim_hidden, self.dim_y, self.node_embedding_size, self.order) for _ in range(self.num_shared_experts)])
        self.task1_experts = nn.ModuleList([ExpertFullyConnectedNetworks(self.dim_X, self.dim_hidden, self.dim_y) for _ in range(self.num_specific_experts)])
        self.task2_experts = nn.ModuleList([ExpertFullyConnectedNetworks(self.dim_X, self.dim_hidden, self.dim_y) for _ in range(self.num_specific_experts)])

        self.gate_shared = nn.Sequential(nn.Linear(self.dim_X, self.num_specific_experts*2 + self.num_shared_experts), nn.Softmax(dim=1))
        self.gate_task1 = nn.Sequential(nn.Linear(self.dim_X, self.num_specific_experts + self.num_shared_experts), nn.Softmax(dim=1))
        self.gate_task2 = nn.Sequential(nn.Linear(self.dim_X, self.num_specific_experts + self.num_shared_experts), nn.Softmax(dim=1))
    
    # Forward propagation
    def forward(self, X):
        inputs_shared, inputs_task1, inputs_task2 = X, X, X
        
        shared_experts_output = [expert(inputs_shared) for expert in self.shared_experts]
        shared_experts_output = torch.stack(shared_experts_output)
        
        task1_experts_output = [expert(inputs_task1) for expert in self.task1_experts]
        task1_experts_output = torch.stack(task1_experts_output)
        
        task2_experts_output = [expert(inputs_task2) for expert in self.task2_experts]
        task2_experts_output = torch.stack(task2_experts_output)

        # Gate 1
        selected_task1 = self.gate_task1(inputs_task1)
        gate_expert_output1 = torch.cat((task1_experts_output, shared_experts_output), dim=0)
        gate_task1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected_task1)
        
        # Gate 2
        selected_task2 = self.gate_task2(inputs_task2)
        gate_expert_output2 = torch.cat((task2_experts_output, shared_experts_output), dim=0)
        gate_task2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected_task2)

        # Shared Gate
        selected_shared = self.gate_shared(inputs_shared)
        gate_expert_outputshared = torch.cat((task1_experts_output, task2_experts_output, shared_experts_output), dim=0)
        gate_shared_out = torch.einsum('abc, ba -> bc', gate_expert_outputshared, selected_shared)

        if self.is_last:
            return [gate_task1_out, gate_task2_out]
        else:
            return [gate_shared_out, gate_task1_out, gate_task2_out]


class TowerNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_hidden, dim_y):
        super(TowerNetworks, self).__init__()
        self.fc1 = nn.Linear(dim_X, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_y)
        self.sigmoid = nn.Sigmoid()
    
    # Forward propagation
    def forward(self, X):
        out = self.fc1(X)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out
        

# Network
class AdaptiveGraphConvolutionalMultiTaskNetworks(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, num_specific_experts, num_shared_experts,
                 experts_hidden=1024, experts_out=512, tower_hidden=256, order=3, node_embedding_size=128,
                 prob='regression'):
        super(AdaptiveGraphConvolutionalMultiTaskNetworks, self).__init__()

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.experts_hidden = experts_hidden
        self.experts_out = experts_out
        self.tower_hidden = tower_hidden
        self.order = order
        self.node_embedding_size = node_embedding_size
        self.prob = prob

        # Model creation
        if prob == 'regression':
            self.interact_gate = GatesInteractionControlNetworks(dim_X, experts_hidden, experts_out, node_embedding_size, order, num_specific_experts, num_shared_experts)
            self.tower1 = TowerNetworks(experts_out, tower_hidden, 1)
            self.tower2 = TowerNetworks(experts_out, tower_hidden, 1)
        
        else:
            raise Exception('Wrong problem type.')

    # Forward propagation
    def forward(self, X):
        
        if self.prob == 'regression':
            res_list = []
            gate_outputs = self.interact_gate(X)
            
            output1 = self.tower1(gate_outputs[0])
            res_list.append(output1.squeeze())

            output2 = self.tower2(gate_outputs[1])
            res_list.append(output2.squeeze())

            res = torch.stack(res_list, dim=1)

        else:
            raise Exception('Wrong problem type.')

        return res


# Model
class AgcmtnModel(NeuralNetwork):

    # Initialization
    def __init__(self, num_specific_experts=2, num_shared_experts=3, experts_hidden=1024, experts_out=512, 
                 tower_hidden=256, order=3, node_embedding_size=128, prob='regression', **args):
        super(AgcmtnModel, self).__init__()

        # Parameter assignment
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_hidden = experts_hidden
        self.experts_out = experts_out
        self.tower_hidden = tower_hidden

        self.order = order
        self.node_embedding_size = node_embedding_size
        
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
        self.model = AdaptiveGraphConvolutionalMultiTaskNetworks(
            self.dim_X, self.dim_y, self.num_specific_experts, self.num_shared_experts,
            self.experts_hidden, self.experts_out, self.tower_hidden, self.order, self.node_embedding_size, self.prob)
            
        self.model_create('UMSE' if self.prob == 'regression' else 'CE')

        # Model training
        self.training()

        return self
