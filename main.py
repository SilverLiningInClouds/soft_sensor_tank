"""
This is a collection of many useful machine learning models
Thus, it is named as 'Arsenal'!

The content of this collection is shown as below:

=====Regression=====
OLS: Ordinary Least Square
RR: Ridge Regression
LASSO: Least Absolute Shrinkage and Selection Operator
PLSR: Partial Least Square Regression
GPR: Gaussian Process Regression
ELM: Extreme Learning Machine
MC-GCN: Multi-Channel Graph Convolutional Networks
GC-LSTM: Graph Convolution Long Short-Term Memory

=====Classification=====
LR: Logistic Regression

=====Regression & Classification=====
FCN: Fully Connected Networks
LSTM: Long Short-Term Memory
GCN: Graph Convolutional Networks

=====Dimensionality Reduction=====
PCA: Principal Component Analysis
t-SNE: t-distributed Stochastic Neighbor Embedding
AE: Auto-Encoders
VAE: Variational Auto-Encoders

To be continued ...
"""

# Packages
from Base.plot import *
from Base.utils import *
from Base.hyper_params import *

# Ignore warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser('Arsenal for machine learning')
parser.add_argument('-prob', type=str, default='regression', help='regression / classification / dimensionality-reduction')
parser.add_argument('-data', type=str, default='ccf', help='ccf / dhg / pcg')
parser.add_argument('-normalize', type=str, default='SS', help='SS (StandardScaler) / MMS (MinMaxScaler)')
parser.add_argument('-model', type=str, default='OLS', help='OLS / RR / LASSO / PLSR / GPR / ELM / MCGCN / GCLSTM / LR / FCN / LSTM / GCN / PCA / tSNE / AE / VAE / RNNGAT')
parser.add_argument('-myself', type=bool, default=False, help='model implemented by myself or package')
parser.add_argument('-multi_y', type=bool, default=False, help='single or multiple y, only available in regression')
parser.add_argument('-mode', type=str, default='mvm', help='mvm or mvo')
parser.add_argument('-hpo', type=bool, default=False, help='whether to optimize hyper-parameters')
parser.add_argument('-hpo_method', type=str, default='GS', help='GS (Grid Search) / RS (Random Search)')
parser.add_argument('-seed', type=int, default=123)
args = parser.parse_args()


# Main function
def mainfunc():

    # Load data
    print('=====Loading ' + args.data + ' data=====')
    if args.data == 'ccf':
        X_train, X_test, y_train, y_test = load_ccf_data(seed=args.seed, normalization=args.normalize)
    
    elif args.data == 'dhg':
        X_train, X_test, y_train, y_test = load_dhg_data(seed=args.seed, normalization=args.normalize)

    elif args.data == 'pcg':
        X_train, X_test, y_train, y_test = load_pcg_data(seed=args.seed, normalization=args.normalize)
    
    elif args.data == 'rmg':
        X_train, X_test, y_train, y_test = load_rmg_data(seed=args.seed, normalization=args.normalize)
    
    elif args.data == 'dbt':
        X_train, X_test, y_train, y_test = load_dbt_data(seed=args.seed, normalization=args.normalize)
    
    elif args.data == 'sru':
        X_train, X_test, y_train, y_test = load_sru_data(seed=args.seed, normalization=args.normalize)
    
    elif args.data == 'vfa':
        X_train, X_test, y_train, y_test = load_vfa_data(seed=args.seed, normalization=args.normalize)
    
    else:
        raise Exception('Wrong data selection.')

    y_train = np.array(y_train)
    
    y_test = np.array(y_test)
    print('Dataset for {} problem has been loaded.\n'.format(args.prob))

    # Model construction
    print('=====Constructing model=====')
    if args.myself and args.model in model_myself[args.prob].keys():
        model = model_myself[args.prob][args.model]
        print('{} model by myself.'.format(args.model))
    
    elif not args.myself and args.model in model_package[args.prob].keys():
        model = model_package[args.prob][args.model]
        print('{} model by package.'.format(args.model))
    
    else:
        raise Exception('Wrong model selection.')
    
    if args.model in ['LSTM', 'GCLSTM', 'BiAGCN']:
        model.set_params(mode=args.mode)

    # HPO setting
    if args.hpo and args.model in hyper_params.keys():
        if args.hpo_method == 'GS':
            model = GridSearchCV(model, hyper_params[args.model], cv=hpo['GS']['cv'])
            print('Grid search for hpo.')
        elif args.hpo_method == 'RS':
            model = RandomizedSearchCV(model, hyper_params[args.model], cv=hpo['RS']['cv'], n_iter=hpo['RS']['cv'],
                                       random_state=args.seed)
            print('Random search for hpo.')
        else:
            raise Exception('Wrong method for hpo.')
        model.fit(X_train, y_train)
        print('Best hyper-params: {}'.format(model.best_params_))
    
    else:
        print('Default hyper-params for modelling.')
        model.fit(X_train, y_train)

    # Model prediction
    if args.prob in ['regression', 'classification']:
        # y_fit = model.predict(X_train)
        y_pred = model.predict(X_test)
    
    elif args.prob == 'dimensionality-reduction':
        if args.model == 'tSNE':
            X_train_trans = model.fit_transform(X_train)
            X_test_trans = model.fit_transform(X_test)
        else:
            X_train_trans = model.transform(X_train)
            X_test_trans = model.transform(X_test)
    
    else:
        raise Exception('Wrong problem type.')
    
    print('Modelling is finished.\n')

    # Model evaluation
    print('=====Evaluating model=====')
    if args.model in ['LSTM', 'GCLSTM', 'BiGCN'] and args.mode == 'mvo':
        y_train = y_train[model.args['seq_len'] - 1:]
        y_test = y_test[model.args['seq_len'] - 1:]
    
    if args.prob == 'regression':
        r2_test, rmse_test, mae_test = curve_scatter(y_test, y_pred, '{}-Test'.format(args.model))
        print('Predicting performance: R2: {}, RMSE: {}, MAE: {}'.format(r2_test, rmse_test, mae_test))
        # r2_train, rmse_train = curve_scatter(y_train, y_fit, '{}-Train'.format(args.model))
        # print('Fitting performance: R2: {}, RMSE: {}'.format(r2_train, rmse_train))
    
    elif args.prob == 'classification':
        acc_test = confusion(y_test, y_pred, '{}-Test'.format(args.model))
        print('Predicting performance: Acc: {}'.format(acc_test))
        # acc_train = confusion(y_train, y_fit, '{}-Train'.format(args.model))
        # print('Fitting performance: Acc: {}'.format(acc_train))
    
    elif args.prob == 'dimensionality-reduction':
        scatter(X_train_trans, y_train, '{}-Train'.format(args.model))
        scatter(X_test_trans, y_test, '{}-Test'.format(args.model))
    
    else:
        raise Exception('Wrong problem type.')
    
    print('Evaluating is finished.')


if __name__ == '__main__':
    mainfunc()
