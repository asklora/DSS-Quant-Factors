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
    sql_result = main.sql_result
    sample_set = main.sample_set

    params = rf_space.copy()
    for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
        params[k] = int(params[k])
    print('===== hyperrf_space =====', params)
    sql_result.update(params)

    params['bootstrap'] = False

    if sql_result['tree_type'] == 'extra':
        regr = ExtraTreesRegressor(criterion=sql_result['objective'], **params)
    elif sql_result['tree_type'] == 'rf':
        regr = RandomForestRegressor(criterion=sql_result['objective'], **params)

    regr.fit(sample_set['train_xx'], sample_set['train_yy_final'])

    # prediction on all sets
    Y_train_pred = regr.predict(sample_set['train_xx'])
    Y_valid_pred = regr.predict(sample_set['valid_x'])
    Y_test_pred = regr.predict(sample_set['test_x'])

    sql_result['feature_importance'], feature_importance_df = to_list_importance(regr)

    return Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df

def eval_regressor(rf_space):
    ''' train & evaluate LightGBM on given rf_space by hyperopt trials with Regressiong model
    -------------------------------------------------
    This part haven't been modified for multi-label questions purpose
    '''

    main = sys.modules["__main__"]
    sample_set = main.sample_set

    main.sql_result['finish_timing'] = dt.datetime.now()
    Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df = rf_train(rf_space)

    result = {'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
              'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
              'mse_train': mean_squared_error(sample_set['train_yy'], Y_train_pred),
              'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
              'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
              'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred)}

    try:    # for backtesting -> calculate MAE/MSE/R2 for testing set
        result_test = {'mae_test': mean_absolute_error(sample_set['test_y'], Y_test_pred),
                       'mse_test': mean_squared_error(sample_set['test_y'], Y_test_pred)}
        result.update(result_test)
    except Exception as e:  # for real_prediction -> no calculation
        print(e)
        pass

    main.sql_result.update(result)  # update result of model
    hpot['all_results'].append(main.sql_result.copy())

    if result['mae_valid'] < hpot['best_score']: # update best_mae to the lowest value for Hyperopt
        hpot['best_score'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

    if main.sql_result['objective'] == 'mae':
        return result['mae_valid']
    elif main.sql_result['objective'] == 'mse':
        return result['mse_valid']
    else:
        NameError('Objective not evaluated!')

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
    best_space = space_eval(rf_space, best)
    eval_regressor(best_space, rerun=True)

    # write score/prediction/feature to DB
    tbl_suffix = '_prod'
    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
        hpot['best_stock_df'].to_sql(f"{global_vals.result_pred_table}{tbl_suffix}", **extra)
        pd.DataFrame(hpot['all_results']).to_sql(f"{global_vals.result_score_table}{tbl_suffix}", **extra)
        hpot['best_stock_feature'].to_sql(f"{global_vals.feature_importance_table}{tbl_suffix}", **extra)
    global_vals.engine_ali.dispose()