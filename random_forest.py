import datetime as dt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import argparse
import pandas as pd
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
from sqlalchemy import create_engine, TIMESTAMP, TEXT, BIGINT, NUMERIC
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, precision_score, \
    recall_score, f1_score

from preprocess.load_data import load_data
import global_vals
import sys

rf_space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
    # 'n_estimators': hp.choice('n_estimators', [15, 50, 100]),
    'max_depth': hp.choice('max_depth', [8, 32, 64]),
    'min_samples_split': hp.choice('min_samples_split', [5, 10, 50, 100]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [5, 10, 50]),
    'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0, 1e-2, 1e-1]),
    'max_features': hp.choice('max_features',[0.5, 0.7, 0.9]),
    'min_impurity_decrease': 0,
    # 'max_samples': hp.choice('max_samples',[0.7, 0.9]),
    'ccp_alpha': hp.choice('ccp_alpha', [0, 1e-3]),
    'n_jobs': -1,
    # 'random_state': 666
}

hpot = {}

def rf_train(rf_space, rerun):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    main = sys.modules["__main__"]
    sample_set = main.sample_set

    params = rf_space.copy()
    for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
        params[k] = int(params[k])
    main.sql_result.update(params)
    params['bootstrap'] = False

    if main.sql_result['tree_type'] == 'extra':
        regr = ExtraTreesRegressor(criterion=main.sql_result['objective'], **params)
    elif main.sql_result['tree_type'] == 'rf':
        regr = RandomForestRegressor(criterion=main.sql_result['objective'], **params)

    if rerun:
        regr.fit(sample_set['train_x'], sample_set['train_y_final'])
    else:
        regr.fit(sample_set['train_xx'], sample_set['train_yy_final'])

    # prediction on all sets
    if rerun:
        Y_train_pred = regr.predict(sample_set['train_x'])
        Y_valid_pred = regr.predict(sample_set['valid_x'])
        Y_test_pred = regr.predict(sample_set['test_x'])
    else:
        Y_train_pred = regr.predict(sample_set['train_xx'])
        Y_valid_pred = regr.predict(sample_set['valid_x'])
        Y_test_pred = regr.predict(sample_set['test_x'])

    main.sql_result['feature_importance'], feature_importance_df = to_list_importance(regr)

    return Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df

def eval_test_return(actual, pred, Y_train_pred=[]):
    ''' test return based on test / train set quantile bins '''

    main = sys.modules["__main__"]
    p = np.linspace(0, 1, 4)
    if len(Y_train_pred)>0:
        bins = np.quantile(Y_train_pred, p)
    else:
        bins = np.quantile(pred, p)

    ret = []
    factor_name = []
    for i in range(3):
        ret.append(np.mean(actual[(pred >= bins[i]) & (pred < bins[i+1])]))

        factor_name.append([x[2:] for x in main.data.y_col][(pred >= bins[i]) & (pred < bins[i+1])])

    return ret, factor_name[2]

def eval_regressor(rf_space, rerun=False):
    ''' train & evaluate LightGBM on given rf_space by hyperopt trials with Regressiong model
    -------------------------------------------------
    This part haven't been modified for multi-label questions purpose
    '''

    main = sys.modules["__main__"]
    sample_set = main.sample_set
    main.sql_result['finish_timing'] = dt.datetime.now()

    Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df = rf_train(rf_space, rerun)

    if rerun: # save prediction bins for training as well
        p = np.linspace(0, 1, 10)
        main.sql_result['train_bins'] = list(np.quantile(Y_train_pred, p))

    if len(sample_set['test_y'])==0:    # for the actual prediction iteration
        sample_set['test_y'] = np.zeros(Y_test_pred)

    ret, best_factor = eval_test_return(sample_set['test_y'], Y_test_pred, Y_train_pred)
    if rerun:
        result = {'mae_train': mean_absolute_error(sample_set['train_y'], Y_train_pred),
                  'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
                  'mse_train': mean_squared_error(sample_set['train_y'], Y_train_pred),
                  'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
                  'r2_train': r2_score(sample_set['train_y'], Y_train_pred),
                  'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred),
                  'mae_test': mean_absolute_error(sample_set['test_y'], Y_test_pred),
                  'mse_test': mean_squared_error(sample_set['test_y'], Y_test_pred),
                  'net_ret': ret[2]
                  }
    else:
        result = {'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
                  'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
                  'mse_train': mean_squared_error(sample_set['train_yy'], Y_train_pred),
                  'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
                  'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
                  'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred),
                  'mae_test': mean_absolute_error(sample_set['test_y'], Y_test_pred),
                  'mse_test': mean_squared_error(sample_set['test_y'], Y_test_pred),
                  'net_ret': ret[2]
                  }

    main.sql_result.update(result)  # update result of model
    hpot['all_results'].append(main.sql_result.copy())

    if (result['mae_valid'] < hpot['best_score']) or (rerun): # update best_mae to the lowest value for Hyperopt
        hpot['best_score'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

    if rerun:
        print(f"RERUN --> {str(result['mse_train']*100)[:6]}, {str(result['mse_test']*100)[:6]}, {str(result['net_ret'])[:6]}, {best_factor}")
        return result['mse_train']
    else:
        print(f"HPOT --> {str(result['mse_valid']*100)[:6]}, {str(result['mse_test']*100)[:6]}, {str(result['net_ret'])[:6]}, {best_factor}")
        return result['mse_valid']

def to_sql_prediction(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    main = sys.modules["__main__"]
    main.sql_result['y_type'] = [x[2:] for x in main.data.y_col]
    df = pd.DataFrame(Y_test_pred, index=main.data.test['group'].to_list(), columns=main.sql_result['y_type'])
    df = df.unstack().reset_index(drop=False)
    df.columns = ['y_type', 'group', 'pred']
    df['actual'] = main.sample_set['test_y_final'].flatten(order='F')       # also write actual qcut to BD
    df['finish_timing'] = [main.sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return df

def to_list_importance(rf):
    ''' based on rf model -> records feature importance in DataFrame to be uploaded to DB '''

    df = pd.DataFrame()
    main = sys.modules["__main__"]
    df['name'] = main.data.x_col     # column names
    df['split'] = rf.feature_importances_
    df['finish_timing'] = [main.sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return ','.join(df.sort_values(by=['split'], ascending=False)['name'].to_list()), df

def rf_HPOT(rf_space, max_evals):
    ''' use hyperopt on each set '''

    global hpot
    main = sys.modules["__main__"]
    hpot.update(main.sql_result)
    hpot['all_results'] = []
    hpot['best_score'] = 10000  # record best training (min mae_valid) in each hyperopt

    trials = Trials()
    best = fmin(fn=eval_regressor, space=rf_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        lasso_bm = pd.read_sql(f"SELECT * FROM {global_vals.result_score_table}_lasso_prod "
                               f"WHERE \"group_code\"='{hpot['group_code']}' "
                               f"AND testing_period<'{dt.datetime.strftime(hpot['testing_period'], '%Y-%m-%d')}'", conn)
        # lasso_bm = lasso_bm.groupby(['group_code','testing_period'])['mse_test'].min()
        lasso_bm = lasso_bm['mse_train'].mean()
    global_vals.engine_ali.dispose()

    print('==============> BM mse_train', str(lasso_bm*100)[:6])

    i = 1
    while (main.sql_result['mse_train'] > lasso_bm) and (i<10):     # run re-evaluation round until results better than LASSO
        best_space = space_eval(rf_space, best)
        eval_regressor(best_space, rerun=True)
        i += 1

    # write score/prediction/feature to DB
    tbl_suffix = '_rf_reg'
    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
        hpot['best_stock_df'].to_sql(f"{global_vals.result_pred_table}{tbl_suffix}", **extra)
        pd.DataFrame(hpot['all_results']).to_sql(f"{global_vals.result_score_table}{tbl_suffix}", **extra)
        hpot['best_stock_feature'].to_sql(f"{global_vals.feature_importance_table}{tbl_suffix}", **extra)
    global_vals.engine_ali.dispose()


