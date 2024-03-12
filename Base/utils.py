# Some functions and classes for utilization
from .packages import *


CCF_DATA_PATH = r'.\Data\CCF.xlsx'
DHG_DATA_PATH = r'.\Data\DHG.xlsx'
PCG_DATA_PATH = r'.\Data\PCG_filtered.xlsx'
RMG_DATA_PATH = r'.\Data\RMG.xlsx'
DBT_DATA_PATH = r'.\Data\DBT.xlsx'
SRU_DATA_PATH = r'.\Data\SRU.xlsx'
VFA_DATA_PATH = r'.\Data\VFA.xlsx'


# Load calcium carbide furnace dataset
def load_ccf_data(data_path=CCF_DATA_PATH, test_size=0.3, seed=123, normalization=None):

    # Dataset type
    data = pd.read_excel(data_path, index_col=0)    # TODO: min-max and data norm
    X = data[[
                'I1', 'I2', 'I3', 'U1', 'U2', 'U3', 
                'P1', 'P2', 'P3', 'U12', 'U23', 'U31', 
                'I1/U1', 'I2/U2', 'I3/U3', 'U1/I1', 'U2/I2', 'U3/I3',
                'U1*I1', 'U2*I2', 'U3*I3', 'Uhat1', 'Uhat2','Uhat3',
                                                                    ]]
    y = data[['Depth1', 'Depth2', 'Depth3']]

    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Normalization type
    if normalization == 'SS':
        scaler = StandardScaler()
    elif normalization == 'MMS':
        scaler = MinMaxScaler()
    elif normalization is None:
        return X_train, X_test, y_train, y_test
    else:
        raise Exception('Wrong normalization type.')
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# Load diesel hydrogenation dataset
def load_dhg_data(data_path=DHG_DATA_PATH, test_size=0.3, seed=123, normalization=None):
    
    # Dataset type
    data = pd.read_excel(data_path, index_col=0)    # TODO: min-max and data norm
    X = data[[
                'A', 'B', 'C', 'D', 'E', 'F', 
                'G', 'H', 'I', 'J', 'K', 'L', 
                'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W',
                                            ]]
    y = data[['FP', 'FBP']]

    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Normalization type
    if normalization == 'SS':
        scaler = StandardScaler()
    elif normalization == 'MMS':
        scaler = MinMaxScaler()
    elif normalization is None:
        return X_train, X_test, y_train, y_test
    else:
        raise Exception('Wrong normalization type.')
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# Load pulverized coal gasification dataset
def load_pcg_data(data_path=PCG_DATA_PATH, test_size=0.3, seed=123, normalization=None):
    
    # Dataset type
    data = pd.read_excel(data_path, index_col=0)    # TODO: min-max and data norm
    X = data[[
                '煤线一煤粉质量流量', '煤线二煤粉质量流量', '煤线三煤粉质量流量',
                '氧气流量', '氧气主管压力', '给料罐与烧嘴压差', '盘管出口水密度',
                '煤磨', '气化炉出口温度', '有效气流量', '总给煤', '氧煤比', 
                                                                    ]]
    y = data[['CO成分', 'CO2报警', 'H2成分',  'CH4报警',]]

    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Normalization type
    if normalization == 'SS':
        scaler = StandardScaler()
    elif normalization == 'MMS':
        scaler = MinMaxScaler()
    elif normalization is None:
        return X_train, X_test, y_train, y_test
    else:
        raise Exception('Wrong normalization type.')
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# Load raw meal grinding dataset
def load_rmg_data(data_path=RMG_DATA_PATH, test_size=0.3, seed=123, normalization=None):
    
    # Dataset type
    data = pd.read_excel(data_path, index_col=0)    # TODO: min-max and data norm
    X = data[[
                '生料喂料量', '出口温度', '出口压力', 
                '返料斗提电流', '原料磨右侧入口压力', '原料磨左侧入口压力', 
                '生料磨电流', '入库斗提电流', '原料磨进出口压差', 
                '循环风机频率', '选粉机转速', '循环风机入口气体压力', 
                                            ]]
    y = data[['生料细度', '游离氧化钙']]
													
    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Normalization type
    if normalization == 'SS':
        scaler = StandardScaler()
    elif normalization == 'MMS':
        scaler = MinMaxScaler()
    elif normalization is None:
        return X_train, X_test, y_train, y_test
    else:
        raise Exception('Wrong normalization type.')
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# Load debutanizer column dataset
def load_dbt_data(data_path=DBT_DATA_PATH, test_size=0.3, seed=123, normalization=None):
    
    # Dataset type
    data = pd.read_excel(data_path, index_col=0)    # TODO: min-max and data norm
    X = data[['A', 'B', 'C', 'D', 'E', 'F', 'G']]
    y = data[['H']]	

    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Normalization type
    if normalization == 'SS':
        scaler = StandardScaler()
    elif normalization == 'MMS':
        scaler = MinMaxScaler()
    elif normalization is None:
        return X_train, X_test, y_train, y_test
    else:
        raise Exception('Wrong normalization type.')
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# def load_dbt_data(data_path=DBT_DATA_PATH, test_size=0.3, seed=123, normalization=None):
    
#     # Dataset type
#     data = pd.read_excel(data_path, index_col=0)    # TODO: min-max and data norm
#     data = data.values
#     x_temp = data[:, :7]
#     y_temp = data[:, 7]


#     x_new = np.zeros([2390, 13])
#     x_6 = x_temp[:, 4]
#     x_9 = (x_temp[:, 5] + x_temp[:, 6])/2
#     x_new[:, :5] = x_temp[4: 2394, :5]

#     x_new[:, 5] = x_6[3: 2393]
#     x_new[:, 6] = x_6[2: 2392]
#     x_new[:, 7] = x_6[1: 2391]
#     x_new[:, 8] = x_9[4: 2394]


#     x_new[:, 9] = y_temp[3: 2393]
#     x_new[:, 10] = y_temp[2: 2392]
#     x_new[:, 11] = y_temp[1:2391]
#     x_new[:, 12] = y_temp[:2390]
#     y_new = y_temp[4: 2394]
#     y_new = y_new.reshape([-1, 1])

#     #划分数据集
#     # x_new = torch.from_numpy(x_new).float()
#     # y_new = torch.from_numpy(y_new).float()
#     X_train = x_new[:1600, :]
#     y_train = y_new[:1600]

#     X_test = x_new[1600:2390, :]
#     y_test = y_new[1600:2390]

#     return X_train, X_test, y_train, y_test


# Load SRU dataset
def load_sru_data(data_path=SRU_DATA_PATH, test_size=0.3, seed=123, normalization=None):
    
    # Dataset type
    data = pd.read_excel(data_path, index_col=0)    # TODO: min-max and data norm
    X = data[['A', 'B', 'C', 'D', 'E', 'F']]
    y = data[['G']]
													
    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Normalization type
    if normalization == 'SS':
        scaler = StandardScaler()
    elif normalization == 'MMS':
        scaler = MinMaxScaler()
    elif normalization is None:
        return X_train, X_test, y_train, y_test
    else:
        raise Exception('Wrong normalization type.')
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# Load volatile fatty acid concentration dataset
def load_vfa_data(data_path=VFA_DATA_PATH, test_size=0.3, seed=123, normalization=None):
    
    # Dataset type
    data = pd.read_excel(data_path, index_col=0)    # TODO: min-max and data norm
    X = data[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']]
    y = data[['L']]
													
    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Normalization type
    if normalization == 'SS':
        scaler = StandardScaler()
    elif normalization == 'MMS':
        scaler = MinMaxScaler()
    elif normalization is None:
        return X_train, X_test, y_train, y_test
    else:
        raise Exception('Wrong normalization type.')
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# Mean filtering and data extraction
def mean_filtering(file_name, save_path, var_list, period, need_extraction=False):
    data = pd.read_excel(file_name)
    data_filtered = []

    # Mean filtering
    for var in tqdm.tqdm(var_list):
        data_filtered_var = []
        data_col = data[var]
        time = len(data_col)
        if var in ['T']:
            for i in range(time):
                data_filtered_var.append(str(data_col[i]))
            data_filtered.append(data_filtered_var)
        else:
            for i in range(time):
                if i < period:
                    data_filtered_var.append(data_col[i])
                else:
                    data_filtered_var.append(sum(data_col[i - period : i]) / period)
            data_filtered.append(data_filtered_var)
    
    data_filtered = np.array(data_filtered)

    # Data extraction
    if need_extraction:
        data_chosen = []
        i = 0
        for var in range(len(var_list)):
            data_var = []
            while i < len(data_filtered[var]):
                data_var.append(data_filtered[var][i])
                i += period
            data_chosen.append(data_var)
            i = 0
        data_chosen = np.array(data_chosen)
        data_save = data_chosen.transpose()

    else:
        data_save = data_filtered.transpose()
    
    save_name = os.path.join(save_path, os.path.basename(file_name).replace('.xlsx', '_filtered.xlsx'))
    if os.path.exists(save_name):
        os.remove(save_name)
    save_df = pd.DataFrame(data_save, columns=var_list)
    save_df.to_excel(save_name)


# Adjacency matrix
def adjacency_matrix(X, mode='sc', graph_reg=0.05, l2=0.5, self_con=0.2, scale=0.4, epsilon=0.1, mine=MINE(alpha=0.6, c=15), gpu=0):

    # RBF kernel function
    if mode == 'rbf':
        kernel = RBF(length_scale=scale)
        A = kernel(X, X)
        A[np.abs(A) < epsilon] = 0
        D = np.diag(np.sum(A, axis=1) ** (-0.5))
        A = np.matmul(np.matmul(D, A), D)

    # Pearson correlation coefficient
    elif mode == 'pearson':
        A = np.corrcoef(X.T)
        A[np.abs(A) < epsilon] = 0
        D = np.diag(np.sum(A, axis=1) ** (-0.5))
        A = np.matmul(np.matmul(D, A), D)

    # Sparse coding
    elif mode == 'sc':
        A = cp.Variable((X.shape[1], X.shape[1]))
        term1 = cp.norm(X * A - X, p='fro')
        term2 = cp.norm1(A)
        constraints = []
        for i in range(X.shape[1]):
            constraints.append(A[i, i] == 0)
            for j in range(X.shape[1]):
                constraints.append(A[i, j] >= 0)
        constraints.append(A == A.T)
        objective = cp.Minimize(term1 + graph_reg * term2)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        A = A.value
        A = A + self_con * np.eye(X.shape[1])
        A[np.abs(A) < epsilon] = 0
        D = np.diag(np.sum(A, axis=1) ** (-0.5))
        A = np.matmul(np.matmul(D, A), D)

    # Maximal information coefficient
    elif mode == 'mic':
        A = np.zeros([X.shape[1], X.shape[1]])
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                mine.compute_score(X[:, i], X[:, j])
                A[i, j] = mine.mic()
                A[j, i] = mine.mic()
        
        # Omit small values
        A[np.abs(A) < epsilon] = 0
        degree_count = copy.copy(A)
        degree_count[np.abs(degree_count) > 0] = 1
        
        # Normalization
        # D = np.diag(np.sum(degree_count, axis=1) ** (-0.5))
        D = np.diag(np.sum(A, axis=1) ** (-0.5))
        A = np.matmul(np.matmul(D, A), D)

    # Sparse PCA    global matrix
    elif mode == 'spca':
        k = X.shape[1]
        iter = 10
        U, sigma, VT = np.linalg.svd(np.matmul(X.T, X))
        alpha = VT[:k, :].T
        
        for i in range(iter):
            beta = np.zeros(alpha.shape)
            for j in range(k):
                net = ElasticNet(graph_reg + 2 * l2, graph_reg / (graph_reg + 2 * l2))
                net.fit(X, np.matmul(X, alpha[:, j]))
                beta_j = net.coef_
                beta[:, j] = beta_j.T
            
            U, sigma, VT = np.linalg.svd(np.matmul(np.matmul(X.T, X), beta))
            alpha = np.matmul(U, VT)

        A = np.matmul(beta, alpha.T)
        A[np.abs(A) < epsilon] = 0
        # D = np.diag(np.sum(A, axis=1) ** (-1)) + 1e-5 * np.eye(A.shape[0])
        # A = np.matmul(D, A)

    # TODO: LASSO
    elif mode == 'lasso':
        pass
    
    # TODO: Ridge
    elif mode == 'ridge':
        pass

    else:
        raise Exception('Wrong mode selection.')

    # sns.heatmap(A, annot=True)
    # plt.show()
    A = torch.tensor(A, dtype=torch.float32)

    return A


# Transform to 3d data
def transform3d(X, y, seq_len=30, mode='mvo'):
    X_3d = []
    y_3d = []

    for i in range(X.shape[0] - seq_len + 1):
        X_3d.append(X[i:i + seq_len])
        y_3d.append(y[i:i + seq_len])
    X_3d = np.stack(X_3d)

    if mode == 'mvo':
        y_3d = y[seq_len - 1:]

    elif mode == 'mvm':
        y_3d = np.stack(y_3d)
        
    else:
        raise Exception('Wrong mode selection.')

    return X_3d, y_3d


# MyDataset
class MyDataset(Dataset):

    # Initialization
    def __init__(self, data, label, prob='regression', gpu=0):
        super(MyDataset, self).__init__()
        self.gpu = gpu
        self.data = self.__transform__(data)
        self.label = self.__transform__(label, prob)

    # Transform
    def __transform__(self, data, prob='regression'):
        if prob in ['regression', 'dimensionality-reduction']:
            return torch.tensor(data, dtype=torch.float32)
        
        else:
            return torch.tensor(data, dtype=torch.long)

    # Get item
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    # Get length
    def __len__(self):
        return self.data.shape[0]


# VAE loss
class VaeLoss(nn.Module):

    # Initialization
    def __init__(self, alpha=1.0):
        super(VaeLoss, self).__init__()
        self.alpha = alpha
        self.mseloss = nn.MSELoss(reduction='sum')

    # Forward propagation
    def forward(self, output, x):
        x_hat, mu, logvar = output
        loss_rec = self.mseloss(x_hat, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_rec + self.alpha * loss_kld


# TODO: 计算二范数进行规约？
class WeighedLoss(nn.Module):
    def __init__(self, scale_coeff=0.5):
        super(WeighedLoss, self).__init__()
        self.mseloss = nn.MSELoss(reduction='sum')
        self.scale_coeff = scale_coeff
    
    def forward(self, output, y):
        loss = []
        total_loss = 0
        
        for i in range(len(output[0])):
            coeff = self.mseloss(output[:, i], y[:, i])
            loss.append(coeff)
        
        for j in range(len(loss)):
            loss_coeff = self.scale_coeff * loss[j] / sum(loss)
            total_loss = total_loss + loss[j] * loss_coeff
        
        return total_loss
    

class UncertaintyLoss(nn.Module):
    def __init__(self, dim_y, scale_coeff=0.85):
        super(UncertaintyLoss, self).__init__()
        self.dim_y = dim_y
        self.mseloss = nn.MSELoss(reduction='sum')
        self.scale_coeff = scale_coeff
        self.loss_uncertainties = Parameter(torch.FloatTensor(dim_y))
        self.reset_parameters()
    
    def forward(self, output, y):
        total_loss = 0
        for i in range(self.dim_y):
            total_loss += self.loss_uncertainties[i] * self.mseloss(output[:, i], y[:, i])
        
        return total_loss
    
    def reset_parameters(self):     # uniform_(1, self.dim_y)
        self.loss_uncertainties.data.uniform_(1, self.dim_y * self.scale_coeff) 
        # self.loss_uncertainties.data.uniform_(self.dim_y / self.scale_coeff, self.dim_y * self.scale_coeff) 
        # stdv = math.sqrt(self.dim_y)
        # self.loss_uncertainties.data.normal_(self.dim_y, stdv) 
        

# Graph convolution
class GraphConvolution(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, adj=None):
        super(GraphConvolution, self).__init__()
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.adj = adj
        self.weight = Parameter(torch.FloatTensor(dim_X, dim_y))
        self.reset_parameters()

    # Forward propagation
    def forward(self, X, adj=None):
        if self.adj is not None:
            res = torch.matmul(self.adj, torch.matmul(X, self.weight))

        elif adj is not None:
            res = torch.matmul(adj, torch.matmul(X, self.weight))

        else:
            raise Exception('No adjacency matrix available.')
        
        return res

    # Weight reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)


# Neural network
class NeuralNetwork(BaseEstimator):

    # Initialization
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.scaler = MinMaxScaler()
        self.args = {
            'n_epoch': 300,   # 200
            'batch_size': 64,
            'lr': 0.001,
            'weight_decay': 0.01,
            'step_size': 50,
            'gamma': 0.5,
            'gpu': 0,
            'seed': 123
        }


    # Data creation
    def data_create(self, X, y, adj_X=False, adj_Y=False):
        self.dim_X = X.shape[-1]
        # import pdb;pdb.set_trace()
        if self.prob == 'regression':
            y = self.scaler.fit_transform(y)
            self.dim_y = y.shape[-1]

        elif self.prob == 'classification':
            self.dim_y = np.unique(y).shape[0]

        if adj_X:
            self.adj = adjacency_matrix(X, self.args['adj_mode'], self.args['graph_reg'], self.args['self_con'],
                                        self.args['scale'], self.args['epsilon'], gpu=self.args['gpu'])
            
        if adj_Y:
            self.adj = adjacency_matrix(y, self.args['adj_mode'], self.args['graph_reg'], self.args['self_con'],
                                        self.args['scale'], self.args['epsilon'], gpu=self.args['gpu'])
            
        if 'mode' in self.__dict__:
            self.X, self.y = transform3d(X, y, self.args['seq_len'], self.mode)
            
        else:
            self.X = X
            self.y = y
        
        self.dataset = MyDataset(self.X, self.y, self.prob, self.args['gpu'])
        self.dataloader = DataLoader(self.dataset, batch_size=self.args['batch_size'], shuffle=False)


    # Model creation
    def model_create(self, loss='MSE', need_pretrain=False):
        if need_pretrain:
            self.loss_hist = np.zeros(self.args['spv_epoch'])
            optimizer_list = []
            optimizer_list.append({'params': self.model.parameters(), 'lr': self.args['unspv_lr']})
            for i in range(self.args['num_layers']):
                optimizer_list.append({'params': self.model.layers[i].parameters(), 'lr': self.args['spv_lr']})
            self.optimizer = optim.Adam(optimizer_list)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.args['step_size'], self.args['gamma'])

        else:
            self.loss_hist = np.zeros(self.args['n_epoch'])
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.args['step_size'], self.args['gamma'])
        
        # TODO: multi-task learning loss
        if loss == 'MSE':
            self.criterion = nn.MSELoss(reduction='sum')

        elif loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()

        elif loss == 'VAE':
            self.criterion = VaeLoss(alpha=self.args['alpha'])
        
        elif loss == 'WMSE':
            self.criterion = WeighedLoss()
        
        elif loss == 'UMSE':
            self.criterion = UncertaintyLoss(dim_y=self.dim_y)

        else:
            raise Exception('Wrong loss function.')


    # Unsupervised Pretraining
    def pretraining(self, model, dataloader, epochs, layer_id, lr):
        self.optimizer = torch.optim.Adam(model.layers[layer_id].parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        for i in range(epochs):
            sum_loss = 0
            for batch_X, batch_y in dataloader:
                hidden, hidden_reconst = model(batch_X, layer_id, is_pretraining=True)
                pretrain_loss = self.criterion(hidden, hidden_reconst)
                self.optimizer.zero_grad()
                pretrain_loss.backward()
                self.optimizer.step()
                sum_loss += pretrain_loss.detach().item()
            print('Unsupervised Pretraining: Layer {}, Epoch {}, Loss {}'.format(layer_id + 1, i + 1, pretrain_loss.data.numpy()))

        return model


    # Model training
    def training(self, need_pretrain=False):
        self.model.train()

        # Supervised Finetuning
        if need_pretrain:
            for layer_id in range(self.args['num_layers']):
                self.model = self.pretraining(self.model, self.dataloader, self.args['unspv_epoch'], layer_id, self.args['unspv_lr'])
            
            for i in range(self.args['spv_epoch']):
                start = time.time()
                
                for batch_X, batch_y in self.dataloader:
                    self.optimizer.zero_grad()
                    output = self.model(batch_X, 0)
                    loss = self.criterion(output, batch_y)
                    self.loss_hist[i] += loss.item()
                    loss.backward()
                    self.optimizer.step()
                
                self.scheduler.step()
                end = time.time()
                print('Supervised Finetuning: Epoch: {}, Loss: {:.4f}, Time: {:.2f}s'.format(i + 1, self.loss_hist[i], end - start))

        # Supervised Training
        else:
            for i in range(self.args['n_epoch']):
                start = time.time()

                for batch_X, batch_y in self.dataloader:
                    if self.prob == 'classification':
                        batch_y = batch_y.view(-1)
                    
                    self.optimizer.zero_grad()
                    output = self.model(batch_X)
                    loss = self.criterion(output, batch_y)
                    self.loss_hist[i] += loss.item()
                    loss.backward()
                    self.optimizer.step()

                self.scheduler.step()
                end = time.time()
                print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}s'.format(i + 1, self.loss_hist[i], end - start))


    # Test
    def predict(self, X):
        if 'mode' in self.__dict__:

            if self.mode == 'mvm':
                X, _ = transform3d(X, X, X.shape[0])

            elif self.mode == 'mvo':
                X, _ = transform3d(X, X, self.args['seq_len'])
        
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        
        with torch.no_grad():
            if self.prob == 'regression':
                
                if self.args['need_pretraining']:   # AE-series
                    result = self.model(X, 0)
                    y = self.scaler.inverse_transform(result.cpu().numpy())
                
                else:
                    result = self.model(X)
                    if type(result) == tuple:       # VAE-series
                        y = self.scaler.inverse_transform(result[0].cpu().numpy())
                    else:
                        y = self.scaler.inverse_transform(self.model(X).cpu().numpy())
                
                print('Predictions: ', y)

            else:
                y = np.argmax(self.model(X).cpu().numpy(), 1)
        
        return y

    # Score
    def score(self, X, y, index='r2'):
        y_pred = self.predict(X)
        
        if self.prob == 'regression':
            if index == 'r2':
                r2 = r2_score(y, y_pred)
                return r2
            
            elif index == 'rmse':
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                return rmse
            
            elif index == 'mae':
                mae = mean_absolute_error(y, y_pred)
                return mae
            
            else:
                raise Exception('Wrong index selection.')
        
        elif self.prob == 'classification':
            acc = accuracy_score(y, y_pred)
            return acc
        
        else:
            raise Exception('Wrong problem type.')

    # Transform
    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        
        with torch.no_grad():
            y = self.model(X, transform=True).cpu().numpy()
        
        return y

    # Fit & Transform
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
