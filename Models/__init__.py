# Initialization
__all__ = ['OLS', 'RR', 'LASSO', 'PLSR', 'GPR', 'ELM', 'MCGCN', 'GCLSTM', 'LR', 'FCN', 'LSTM', 'GCN', 'PCA', 'tSNE',
           'AE', 'VAE']

from .OLS import OlsModel
from .RR import RrModel
from .LASSO import LassoModel
from .PLSR import PlsrModel
from .GPR import GprModel
from .ELM import ElmModel
from .MCGCN import McgcnModel
from .GCLSTM import GclstmModel
from .LR import LrModel
from .FCN import FcnModel
from .LSTM import LstmModel
from .GCN import GcnModel
from .PCA import PcaModel
from .tSNE import TSNE
from .AE import AeModel
from .VAE import VaeModel
