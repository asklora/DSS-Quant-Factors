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

method = 'lasso'

def rf_train():
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    sql_result['finish_timing'] = dt.datetime.now()

    # if method == 'lasso':
    #     clf = linear_model.Lasso(alpha=sql_result['alpha']).fit(sample_set['train_x'], sample_set['train_y_final'], sample_weight=sql_result['weight'])
    #     Y_train_pred = clf.predict(sample_set['train_x'])   # prediction on all sets
    #     Y_test_pred = clf.predict(sample_set['test_x'])
    # elif method == 'en':
    clf = linear_model.ElasticNet(alpha=sql_result['alpha'], l1_ratio=sql_result['l1_ratio']).fit(sample_set['train_x'], sample_set['train_y_final'])
    Y_train_pred = clf.predict(sample_set['train_x'])   # prediction on all sets
    Y_test_pred = clf.predict(sample_set['test_x'])

    sql_result['feature_importance'], feature_importance_df = to_list_importance(clf)
    print(feature_importance_df)

    return Y_train_pred, Y_test_pred, feature_importance_df

def eval_regressor():
    ''' train & evaluate LightGBM on given space by hyperopt trials with Regressiong model
    -------------------------------------------------
    This part haven't been modified for multi-label questions purpose
    '''

    Y_train_pred, Y_test_pred, feature_importance_df = rf_train()

    result = {'mae_train': mean_absolute_error(sample_set['train_y'], Y_train_pred),
              'mse_train': mean_squared_error(sample_set['train_y'], Y_train_pred),
              'r2_train': r2_score(sample_set['train_y'], Y_train_pred),
              'train_len': len(sample_set['train_y'])
              }

    try:    # for backtesting -> calculate MAE/MSE/R2 for testing set
        result_test = {
            'mae_test': mean_absolute_error(sample_set['test_y'], Y_test_pred),
            'mse_test': mean_squared_error(sample_set['test_y'], Y_test_pred),
            # 'r2_test': r2_score(test_df['actual'], test_df['pred'])
        }
        # result['test_len'] = len(test_df)
        result.update(result_test)
    except Exception as e:  # for real_prediction -> no calculation
        print(e)
        pass

    sql_result.update(result)  # update result of model

    hpot['all_results'].append(sql_result.copy())
    hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
    hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

    print(result['mse_train'])

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

    df = pd.DataFrame()
    df['name'] = data.x_col     # column names
    # x = model.coef_
    # print(np.sum(model.coef_, axis=0))
    try:
        df['split'] = list(np.sum(model.coef_, axis=0))
    except:
        df['split'] = list(model.coef_)

    df['finish_timing'] = [sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return ','.join(df.sort_values(by=['split'], ascending=False)['name'].to_list()), df

def HPOT():
    ''' use hyperopt on each set '''

    hpot['all_results'] = []
    eval_regressor()

    hpot['all_results'][0]['weight'] = 'tanh'

    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
        hpot['best_stock_df'].to_sql(global_vals.result_pred_table+"_lasso", **extra)
        pd.DataFrame(hpot['all_results']).to_sql(global_vals.result_score_table+"_lasso", **extra)
        hpot['best_stock_feature'].to_sql(global_vals.feature_importance_table+"_lasso", **extra)
    global_vals.engine_ali.dispose()

if __name__ == "__main__":

    sql_result = {}

    # --------------------------------- Different Config ------------------------------------------

    sql_result['name_sql'] = 'lasso_multipca1'
    use_biweekly_stock = False
    stock_last_week_avg = True
    valid_method = 'chron'
    group_code_list = ['USD']
    qcut_q = 0
    use_median = False
    n_splits = 1
    test_change = False
    sql_result['alpha'] = 0.001
    sql_result['l1_ratio'] = 1
    use_pca = 0.4

    # --------------------------------- Define Variables ------------------------------------------

    hpot = {}                   # storing data for best trials in each Hyperopt

    # create date list of all testing period
    if use_biweekly_stock:
        last_test_date = dt.datetime(2021,7,4)
        backtest_period = 100
        testing_period_list=[last_test_date+relativedelta(days=1) - i*relativedelta(weeks=2)
                             - relativedelta(days=1) for i in range(0, backtest_period+1)]
    else:
        last_test_date = dt.date.today() + MonthEnd(-2)     # Default last_test_date is month end of 2 month ago from today
        backtest_period = 46
        testing_period_list=[last_test_date+relativedelta(days=1) - i*relativedelta(months=1)
                             - relativedelta(days=1) for i in range(0, backtest_period+1)]

    # --------------------------------- Model Training ------------------------------------------

    data = load_data(use_biweekly_stock=use_biweekly_stock, stock_last_week_avg=stock_last_week_avg, mode='default')  # load_data (class) STEP 1
    factors_to_test = data.factor_list       # random forest model predict all factor at the same time
    # factors_to_test = ['vol_0_30','book_to_price','earnings_yield','market_cap_usd']
    print(f"===== test on y_type", len(factors_to_test), factors_to_test, "=====")

    # for f in factors_to_test:
    sql_result['y_type'] = factors_to_test
    # print(sql_result['y_type'])
    for a in [0.2, 0.4]:
        sql_result['use_pca'] = a
    # for y in factors_to_test:
    #     sql_result['y_type'] = [y]
        for group_code in group_code_list:
            sql_result['group_code'] = group_code
            data.split_group(group_code)  # load_data (class) STEP 2
            for testing_period in testing_period_list:
                sql_result['testing_period'] = testing_period
                load_data_params = {'qcut_q': qcut_q, 'y_type': sql_result['y_type'], 'valid_method': valid_method,
                                    'use_median': use_median, 'n_splits': n_splits, 'test_change': test_change, 'use_pca': sql_result['use_pca']}
                try:
                    sample_set, cv = data.split_all(testing_period, **load_data_params)  # load_data (class) STEP 3
                    print(list(data.x_col))

                    cv_number = 1  # represent which cross-validation sets
                    for train_index, valid_index in cv:  # roll over 5 cross validation set
                        sql_result['cv_number'] = cv_number

                        # sample_set['valid_x'] = sample_set['train_x'][valid_index]
                        # sample_set['train_xx'] = sample_set['train_x'][train_index]
                        # sample_set['valid_y'] = sample_set['train_y'][valid_index]
                        # sample_set['train_yy'] = sample_set['train_y'][train_index]
                        # sample_set['valid_y_final'] = sample_set['train_y_final'][valid_index]
                        # sample_set['train_yy_final'] = sample_set['train_y_final'][train_index]

                        # sample_set['train_y'] = sample_set['train_y'].flatten()
                        # sample_set['test_y'] = sample_set['test_y'].flatten()
                        # sample_set['train_y_final'] = sample_set['train_y_final'].flatten()
                        # sample_set['test_y_final'] = sample_set['test_y_final'].flatten()

                        sql_result['valid_group'] = ','.join(list(data.train['group'][valid_index].unique()))

                        for k in ['train_x', 'test_x']:
                            # sample_set[k][(np.abs(sample_set[k])==np.inf)] = np.nan
                            sample_set[k] = np.nan_to_num(sample_set[k], nan=0)

                        sql_result['weight'] = np.array(range(len(sample_set['train_y'])))/len(sample_set['train_y'])
                        sql_result['weight'] = np.tanh(sql_result['weight']-0.5)+0.5
                        # sql_result['weight'] = 1/(1+1/np.exp(sql_result['weight']-0.5))
                        # sql_result['weight'] = pd.cut(sql_result['weight'], bins=12, labels=False)
                        # print(sql_result['weight'])

                        print(group_code, testing_period, len(sample_set['train_y_final']))
                        HPOT()  # start hyperopt
                        cv_number += 1
                except Exception as e:
                    print(testing_period, e)
                    continue