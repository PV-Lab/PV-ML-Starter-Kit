import pandas as pd
import numpy as np
import GPy
import GPyOpt # Developed using GPyOpt version 1.2.6, must use scipy version 1.4.1 and python version 3.8
from GPyOpt.methods import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from utils import normalize, unnormalize

def BO_optimizer(df, number_of_predictions, type_of_optimization):
    '''
    Implement Bayesian optimization.
    Inputs:
    data                     := unnormalized dataset with parameters in columns 1 to N-1 and the target variable to optimized in column N.
    number_of_predictions    := desired output batch size for the suggested next locations

    Ouputs:
    df                       := A dataframe of predicted, normalized parameter values (B by N), where N are the control parameters and B is the batch size
    '''
    df_norm = normalize(df)
    X = np.array(df_norm.iloc[:, :-1])  # X data are the parameters
    Y = np.array(df_norm.iloc[:, -1])  # Y data is the last column
    if type_of_optimization == 'max':
        Y = -1 * np.array(df.iloc[:, -1])  # Y data is the last column
    else:
        Y = np.array(df.iloc[:, -1])  # Y data is the last column
    N = X.shape[1]  # number of parameters, N
    col_names = df.columns
    bds = [{'name': f'x{n + 1}', 'type': 'continuous', 'domain': (0, 1)} for n in range(N)]  # N-dimensions
    kernel = GPy.kern.Matern52(input_dim=len(bds),
                               ARD=True)  # Use the matern 5/2 kernel with automatic relevence detection enabled
    regr_RF = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=50)
    regr_RF.fit(X, Y)
    obj = regr_RF.predict
    optimizer = BayesianOptimization(f=obj,
                                     domain=bds,
                                     constraints=None,
                                     model_type='GP',  # gaussian process model
                                     acquisition_type='LCB',  # expected improvement acquisition
                                     acquisition_jitter=0.,  # tune to adjust exploration
                                     X=X,  # normalized parameter value data
                                     Y=Y.reshape(Y.shape[0], 1),
                                     evaluator_type='local_penalization',
                                     batch_size=number_of_predictions,  # batch size of predicted optima
                                     normalize_Y=False,
                                     kernel=kernel,  # select the kernel
                                     )
    # optimizer.run_optimization(max_iter=15)
    # optimizer.plot_convergence()
    suggested = optimizer.suggest_next_locations()  # get next parameter values to synthesize experimentally
    predicted = optimizer.model.predict(suggested)[0]
    predictions = pd.DataFrame(suggested, columns=[f'Suggested {col_names[n]}' for n in range(len(col_names) - 1)])
    if type_of_optimization == 'max':
        predictions[f'Predicted {col_names[-1]}'] = -1 * predicted
    else:
        predictions[f'Predicted {col_names[-1]}'] = predicted
    predictions = unnormalize(df, predictions)
    return predictions