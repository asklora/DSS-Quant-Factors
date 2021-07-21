import datetime as dt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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

from preprocess.load_data_lgbm import load_data
from hyperspace_rf import find_hyperspace
import global_vals

def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()
    for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
        params[k] = int(params[k])
    print(params)

    # params['n_jobs'] = 6

    if args.tree_type == 'extra':
        regr = ExtraTreesRegressor(criterion='mae', verbose=1, **params)
    if args.tree_type == 'rf':
        regr = RandomForestRegressor(criterion='mae', verbose=1, **params)

    regr.fit(X_train, Y_train)

    # prediction on all sets
    Y_train_pred = regr.predict(X_train)
    Y_valid_pred = regr.predict(X_valid)
    Y_test_pred = regr.predict(X_test)

    return Y_train_pred, Y_valid_pred, Y_test_pred

def eval(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials '''

    Y_train_pred, Y_valid_pred, Y_test_pred = lgbm_train(space)

    result = {'mae_train': mean_absolute_error(Y_train, Y_train_pred),
              'mae_valid': mean_absolute_error(Y_valid, Y_valid_pred),
              'mae_test': mean_absolute_error(Y_test, Y_test_pred),
              'mse_train': mean_squared_error(Y_train, Y_train_pred),
              'mse_valid': mean_squared_error(Y_valid, Y_valid_pred),
              'mse_test': mean_squared_error(Y_test, Y_test_pred),
              'r2_train': r2_score(Y_train, Y_train_pred),
              'r2_valid': r2_score(Y_valid, Y_valid_pred),
              'r2_test': r2_score(Y_test, Y_test_pred),
              'status': STATUS_OK}
    print(result)

    sql_result.update(space)  # update hyper-parameter used in model
    sql_result.update(result)  # update result of model
    sql_result['finish_timing'] = dt.datetime.now()

    hpot['all_results'].append(sql_result.copy())
    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] < hpot['best_mae']:  # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_trial'] = sql_result['trial_lgbm']

    sql_result['trial_lgbm'] += 1

    return result['mae_valid']

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    hpot['best_mae'] = 10000  # record best training (min mae_valid) in each hyperopt
    hpot['all_results'] = []

    trials = Trials()

    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(best)

    # write stock_pred for the best hyperopt records to sql
    with global_vals.engine_ali.connect() as conn:
        hpot['best_stock_df'].to_sql('results_randomforest_stock', con=conn, index=False, if_exists='append', method='multi')
        pd.DataFrame(hpot['all_results']).to_sql('results_randomforest', con=conn, index=False, if_exists='append', method='multi')
    global_vals.engine_ali.dispose()

    sql_result['trial_hpot'] += 1


def to_sql_prediction(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['identifier'] = test_id
    df['pred'] = Y_test_pred
    df['trial_lgbm'] = [sql_result['trial_lgbm']] * len(test_id)
    df['name'] = [sql_result['name']] * len(test_id)
    # print('stock-wise prediction: ', df)

    return df

def read_db_last(sql_result, results_table='results_randomforest'):
    ''' read last records on DB TABLE lightgbm_results for resume / trial_no counting '''

    try:
        with engine_ali.connect() as conn:
            db_last = pd.read_sql("SELECT * FROM {} Order by finish_timing desc LIMIT 1".format(results_table), conn)
        engine_ali.dispose()

        db_last_param = db_last[['icb_code', 'testing_period']].to_dict('index')[0]
        db_last_trial_hpot = int(db_last['trial_hpot'])
        db_last_trial_lgbm = int(db_last['trial_lgbm'])

        sql_result['trial_hpot'] = db_last_trial_hpot + 1  # trial_hpot = # of Hyperopt performed (n trials each)
        sql_result['trial_lgbm'] = db_last_trial_lgbm + 1  # trial_lgbm = # of Lightgbm performed
        print('if resume from: ', db_last_param, '; sql last trial_lgbm: ', sql_result['trial_lgbm'])
    except:
        db_last_param = None
        sql_result['trial_hpot'] = sql_result['trial_lgbm'] = 0

    return db_last_param, sql_result

if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_type', default='rf')
    parser.add_argument('--qcut_q', default=3, type=int)  # Default: Low, Mid, High
    args = parser.parse_args()

    # --------------------------------- Define Variables ------------------------------------------

    # create dict storing values/df used in training
    sql_result = vars(args)     # data write to DB TABLE lightgbm_results
    sql_result['name_sql'] = f'{args.y_type}_{dt.datetime.now()}_' + 'testing'
    hpot = {}                   # storing data for best trials in each Hyperopt

    # update additional base_space for Hyperopt
    base_space = {'verbose':0,
                  'objective': args.objective,
                  'eval_metric': 'mae',
                  'grow_policy': 'depthwise',
                  'num_threads': args.nthread}

    if sql_result['objective'] == 'multiclass':
        base_space['num_class'] = sql_result['qcut_q']
        base_space['metric'] = 'multi_error'

    last_test_date = dt.date.today() + MonthEnd(-2)     # Default last_test_date is month end of 2 month ago from today
    backtest_period = 22

    # create date list of all testing period
    testing_period_list=[last_test_date+relativedelta(days=1) - i*relativedelta(months=1)
                         - relativedelta(days=1) for i in range(0, backtest_period+1)]
    print(f"===== test on sample sets {testing_period_list[-1].strftime('%Y-%m-%d')} to "
          f"{testing_period_list[0].strftime('%Y-%m-%d')} ({len(testing_period_list)}) =====")

    # --------------------------------- Model Training ------------------------------------------

    data = load_data()  # load_data (class) STEP 1
    for group_code in ['industry', 'currency']:
        sql_result['group_code'] = group_code
        data.split_group(group_code)  # load_data (class) STEP 2
        # for f in data.factor_list:
        y_type = ['earnings_yield']
        print(y_type)
        for testing_period in reversed(testing_period_list):
            sql_result['testing_period'] = testing_period
            backtest = testing_period not in testing_period_list[0:4]
            load_data_params = {'qcut_q': args.qcut_q, 'y_type': y_type}

            try:
                sample_set, cv = data.split_all(testing_period, **load_data_params)  # load_data (class) STEP 3

                cv_number = 1  # represent which cross-validation sets
                for train_index, valid_index in cv:  # roll over 5 cross validation set
                    sql_result['cv_number'] = cv_number

                    sample_set['valid_x'] = sample_set['train_x'][valid_index]
                    sample_set['train_xx'] = sample_set['train_x'][train_index]
                    sample_set['valid_y'] = sample_set['train_y'][valid_index]
                    sample_set['train_yy'] = sample_set['train_y'][train_index]
                    sample_set['valid_y_final'] = sample_set['train_y_final'][valid_index]
                    sample_set['train_yy_final'] = sample_set['train_y_final'][train_index]

                    sample_set['valid_y_final'] = sample_set['valid_y_final'].flatten()
                    sample_set['train_yy_final'] = sample_set['train_yy_final'].flatten()
                    sample_set['test_y_final'] = sample_set['test_y_final'].flatten()

                    sql_result['train_len'] = len(sample_set['train_xx'])  # record length of training/validation sets
                    sql_result['valid_len'] = len(sample_set['valid_x'])

                    space = find_hyperspace(sql_result)
                    space.update(base_space)
                    print(group_code, testing_period, len(sample_set['train_yy_final']))
                    HPOT(space, max_evals=args.max_eval)  # start hyperopt
                    cv_number += 1
            except Exception as e:
                print(testing_period, e)
                continue
