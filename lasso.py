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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn import linear_model
from preprocess.load_data import load_data
from random_forest import eval_test_return
import global_vars
import sys
import itertools

sql_result_lasso = {}
hpot_lasso = {}  # storing data for best trials in each Hyperopt

class lasso_bm:

    def __init__(self, data):
        ''' '''
        self.sample_set = {}
        self.data = data

    def load_data_lasso(self, space, rerun=False):
        ''' Prepare train / te '''

        params = space.copy()
        sql_result_lasso.update(params)
        sql_result_lasso['finish_timing'] = dt.datetime.now()

        load_data_params = {'qcut_q': 0, 'y_type': sql_result_lasso['y_type'], 'valid_method': 'chron', 'use_median': False,
                            'n_splits': 1, 'test_change': False, 'use_pca': sql_result_lasso['use_pca']}

        sample_set, cv = self.data.split_all(sql_result_lasso['testing_period'], **load_data_params)  # load_data (class) STEP 3
        sql_result_lasso['cv_number'] = 1

        train_index, valid_index = cv[0][0], cv[0][1]
        sample_set['valid_x'] = sample_set['train_x'][valid_index]
        sample_set['train_xx'] = sample_set['train_x'][train_index]
        sample_set['valid_y'] = sample_set['train_y'][valid_index]
        sample_set['train_yy'] = sample_set['train_y'][train_index]
        sample_set['valid_y_final'] = sample_set['train_y_final'][valid_index]
        sample_set['train_yy_final'] = sample_set['train_y_final'][train_index]

        sql_result_lasso['train_len'] = len(sample_set['train_xx'])  # record length of training/validation sets
        sql_result_lasso['valid_len'] = len(sample_set['valid_x'])

        for k in ['valid_x', 'train_xx', 'test_x', 'train_x']:  # fill NaN to 0
            sample_set[k] = np.nan_to_num(sample_set[k], nan=0)

        sample_set['weight'] = np.array(range(len(sample_set['train_y']))) / len(sample_set['train_y'])
        sample_set['weight'] = np.tanh(sample_set['weight'] - 0.5) + 0.5
        sql_result_lasso['neg_factor'] = ','.join(self.data.neg_factor)
        self.sample_set.update(sample_set)
        self.eval_regressor(rerun)

        return sql_result_lasso['mse_valid']


    def lasso_train(self, rerun):
        ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

        if sql_result_lasso['alpha'] == 0:
            clf = linear_model.LinearRegression().fit(self.sample_set['train_x'],self.sample_set['train_y_final'])
        else:
            clf = linear_model.ElasticNet(alpha=sql_result_lasso['alpha'], l1_ratio=sql_result_lasso['l1_ratio'])\
                .fit(self.sample_set['train_x'], self.sample_set['train_y_final'])

        if rerun:
            Y_train_pred = clf.predict(self.sample_set['train_x'])   # prediction on all sets
            Y_valid_pred = clf.predict(self.sample_set['valid_x'])
            Y_test_pred = clf.predict(self.sample_set['test_x'])
        else:
            Y_train_pred = clf.predict(self.sample_set['train_xx'])   # prediction on all sets
            Y_valid_pred = clf.predict(self.sample_set['valid_x'])
            Y_test_pred = clf.predict(self.sample_set['test_x'])

        sql_result_lasso['feature_importance'] = self.to_list_importance(clf)

        return Y_train_pred, Y_valid_pred, Y_test_pred

    def eval_regressor(self, rerun):
        ''' train & evaluate LightGBM on given space by hyperopt trials with Regressiong model '''

        Y_train_pred, Y_valid_pred, Y_test_pred = self.lasso_train(rerun)
        sample_set = self.sample_set
        p = np.linspace(0, 1, 10)
        train_bins = np.quantile(Y_train_pred, p)

        ret, ret_col = eval_test_return(sample_set['test_y'], Y_test_pred, Y_train_pred)

        if rerun:
            result = {'mae_train': mean_absolute_error(sample_set['train_y'], Y_train_pred),
                      'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
                      'mse_train': mean_squared_error(sample_set['train_y'], Y_train_pred),
                      'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
                      'r2_train': r2_score(sample_set['train_y'], Y_train_pred),
                      'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred),
                      'mae_test': mean_absolute_error(sample_set['test_y'], Y_test_pred),
                      'mse_test': mean_squared_error(sample_set['test_y'], Y_test_pred),
                      'net_ret': ret[2] - ret[0]
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
                      'net_ret': ret[2] - ret[0]
                      }

        sql_result_lasso.update(result)  # update result of model

        if rerun:
            sql_result_lasso['train_bins'] = list(train_bins)
            hpot_lasso['all_results'].append(sql_result_lasso.copy())
            hpot_lasso['best_stock_df'] = self.to_sql_prediction(Y_test_pred)

            with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
                extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
                hpot_lasso['best_stock_df'].to_sql(global_vals.result_pred_table + "_lasso_prod", **extra)
                pd.DataFrame(hpot_lasso['all_results']).to_sql(global_vals.result_score_table + "_lasso_prod", **extra)
            global_vals.engine_ali.dispose()

        return result['mse_valid']

    def to_sql_prediction(self, Y_test_pred):
        ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

        sql_result_lasso['y_type'] = [x[2:] for x in self.data.y_col]
        df = pd.DataFrame(Y_test_pred, index=self.data.test['group'].to_list(), columns=sql_result_lasso['y_type'])
        df = df.unstack().reset_index(drop=False)
        df.columns = ['y_type', 'group', 'pred']
        df['actual'] = self.sample_set['test_y_final'].flatten(order='F')       # also write actual qcut to BD
        df['finish_timing'] = [sql_result_lasso['finish_timing']] * len(df)      # use finish time to distinguish dup pred
        return df

    def to_list_importance(self, model):
        ''' based on rf model -> records feature importance in DataFrame to be uploaded to DB '''

        df = pd.DataFrame({'name':[], 'split': []})
        df['name'] = self.data.x_col     # column names
        df['split'] = list(np.sum(model.coef_, axis=0))
        return ','.join(df.sort_values(by=['split'], ascending=False)['name'].to_list())

def lasso_HPOT(data):
    ''' Using hpot_lasso to go through alpha, l1, pca_ratio space in LASSO/RIDGE/Linear Regression '''

    global hpot_lasso, sql_result_lasso
    hpot_lasso.update(sql_result_lasso)
    hpot_lasso['all_results'] = []
    hpot_lasso['best_score'] = 10000  # record best training (min mae_valid) in each hyperopt

    lasso_space = {'alpha': hp.choice('alpha', [0.001, 0.005, 0.0001]),
                   'use_pca': hp.choice('use_pca', [0.2, 0.4, 0.6, 0.8]),
                   'l1_ratio': hp.choice('l1_ratio', [0.5, 1])}

    trials = Trials()
    best = fmin(fn=lasso_bm(data).load_data_lasso, space=lasso_space, algo=tpe.suggest, max_evals=10, trials=trials)

    best_space = space_eval(lasso_space, best)
    print(best_space)
    lasso_bm(data).load_data_lasso(best_space, rerun=True)

def start_lasso(data, testing_period_list, group_code_list, y_type):
    ''' running grid search on lasso and save best results to DB as benchmark'''

    sql_result_lasso['name_sql'] = f"prod_{dt.datetime.strftime(dt.datetime.today(),'%Y%m%d')}"
    sql_result_lasso['y_type'] = y_type
    sql_result_lasso['cv_number'] = 1

    for testing_period, group_code in itertools.product(testing_period_list, group_code_list):
        sql_result_lasso['testing_period'] = testing_period
        sql_result_lasso['group_code'] = group_code
        print(testing_period, group_code)
        data.split_group(group_code)  # load_data (class) STEP 2
        lasso_HPOT(data)

if __name__ == '__main__':
    testing_period_list = pd.date_range(dt.datetime(2017,8,31), dt.datetime(2021,6,30),freq='m')
    group_code_list = pd.read_sql('SELECT DISTINCT currency_code from universe WHERE currency_code IS NOT NULL', global_vals.engine.connect())['currency_code'].to_list()
    # group_code_list = ['HKD','EUR']
    y_type = pd.read_sql('SELECT name from factor_formula_ratios WHERE factors', global_vals.engine_ali.connect())['name'].to_list()
    start_lasso(data, testing_period_list, group_code_list, y_type)


