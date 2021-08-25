import datetime as dt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import argparse
import pandas as pd
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
from sqlalchemy import create_engine, TIMESTAMP, TEXT, BIGINT, NUMERIC
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn import linear_model
from preprocess.load_data import load_data
import global_vals

sql_result = {}
sample_set = {}
hpot = {}  # storing data for best trials in each Hyperopt
data = load_data(use_biweekly_stock=False, stock_last_week_avg=True, mode='default')  # load_data (class) STEP 1

def lasso_train():
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    sql_result['finish_timing'] = dt.datetime.now()

    clf = linear_model.ElasticNet(alpha=sql_result['alpha'], l1_ratio=sql_result['l1_ratio']).fit(sample_set['train_x'], sample_set['train_y_final'])
    Y_train_pred = clf.predict(sample_set['train_x'])   # prediction on all sets
    Y_test_pred = clf.predict(sample_set['test_x'])

    sql_result['feature_importance'] = to_list_importance(clf)

    return Y_train_pred, Y_test_pred

def eval_regressor():
    ''' train & evaluate LightGBM on given space by hyperopt trials with Regressiong model '''

    Y_train_pred, Y_test_pred = lasso_train()

    result = {'mae_train': mean_absolute_error(sample_set['train_y'], Y_train_pred),
              'mse_train': mean_squared_error(sample_set['train_y'], Y_train_pred),
              'r2_train': r2_score(sample_set['train_y'], Y_train_pred),
              'train_len': len(sample_set['train_y']),
              'mae_test': mean_absolute_error(sample_set['test_y'], Y_test_pred),
              'mse_test': mean_squared_error(sample_set['test_y'], Y_test_pred),
    }

    sql_result.update(result)  # update result of model

    hpot['all_results'].append(sql_result.copy())
    hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)

    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
        hpot['best_stock_df'].to_sql(global_vals.result_pred_table + "_lasso_prod", **extra)
        pd.DataFrame(hpot['all_results']).to_sql(global_vals.result_score_table + "_lasso_prod", **extra)
    global_vals.engine_ali.dispose()

def to_sql_prediction(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame(Y_test_pred, index=data.test['group'].to_list(), columns=sql_result['y_type'])
    df = df.unstack().reset_index(drop=False)
    df.columns = ['y_type', 'group', 'pred']
    df['actual'] = sample_set['test_y_final'].flatten(order='F')       # also write actual qcut to BD
    df['finish_timing'] = [sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return df

def to_list_importance(model):
    ''' based on rf model -> records feature importance in DataFrame to be uploaded to DB '''

    df = pd.DataFrame({'name':[], 'split': []})
    df['name'] = data.x_col     # column names
    df['split'] = list(np.sum(model.coef_, axis=0))
    return ','.join(df.sort_values(by=['split'], ascending=False)['name'].to_list())

def start_lasso(testing_period_list, group_code_list, y_type):
    ''' running grid search on lasso and save best results to DB as benchmark'''

    sql_result['y_type'] = y_type
    alpha_list = [0, 0.0001, 0.001, 0.005]
    use_pca_list = [0, 0.2, 0.4, 0.6]
    l1_ratio_list = [0, 0.5, 1]

    for testing_period, group_code in zip(testing_period_list, group_code_list):
        sql_result['testing_period'] = testing_period
        sql_result['group_code'] = group_code
        data.split_group(group_code)  # load_data (class) STEP 2

        hpot['best_score'] = 10000
        hpot['all_results'] = []
        for alpha, use_pca, l1_ratio in zip(alpha_list, use_pca_list, l1_ratio_list):
            sql_result['alpha'] = alpha
            sql_result['use_pca'] = use_pca
            sql_result['l1_ratio'] = l1_ratio
            load_data_params = {'qcut_q': 0, 'y_type': y_type, 'valid_method': 'chron', 'use_median': False,
                                'n_splits': 1, 'test_change': False, 'use_pca': sql_result['use_pca']}
            sample_set_re, cv = data.split_all(testing_period, **load_data_params)  # load_data (class) STEP 3
            sample_set.update(sample_set_re)

            for k in ['train_x', 'test_x']:
                sample_set[k] = np.nan_to_num(sample_set[k], nan=0)
            sample_set['weight'] = np.array(range(len(sample_set['train_y'])))/len(sample_set['train_y'])
            sample_set['weight'] = np.tanh(sample_set['weight']-0.5)+0.5
            eval_regressor()


