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

from load_data_lgbm import load_data
from hyperspace_rf import find_hyperspace

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

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

    hpot['best_mae'] = 1  # record best training (min mae_valid) in each hyperopt
    hpot['all_results'] = []

    trials = Trials()

    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(best)

    # write stock_pred for the best hyperopt records to sql
    with engine.connect() as conn:
        hpot['best_stock_df'].to_sql('results_randomforest_stock', con=conn, index=False, if_exists='append', method='multi')
        pd.DataFrame(hpot['all_results']).to_sql('results_randomforest', con=conn, index=False, if_exists='append', method='multi')
    engine.dispose()

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
        with engine.connect() as conn:
            db_last = pd.read_sql("SELECT * FROM {} Order by finish_timing desc LIMIT 1".format(results_table), conn)
        engine.dispose()

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_sql', required=True)
    parser.add_argument('--sp_only', default=False, action='store_true')
    parser.add_argument('--exclude_stock', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--exclude_fwd', default=False, action='store_true')
    parser.add_argument('--tree_type', default='rf')
    parser.add_argument('--sample_type', default='industry')
    parser.add_argument('--sample_no', type=int, default=21)
    args = parser.parse_args()

    # training / testing sets split par
    if args.sample_type == 'industry':
        partitions = [20, 30, 35, 40, 45, 51, 60, 65]
    elif args.sample_type == 'sector':
        partitions = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010,
                      201020, 502030, 401010, 999999]  # icb_code with > 1300 samples + rests in single big model (999999)
    elif args.sample_type == 'entire':
        partitions = [0, 1, 2]
    else:
        NameError('wrong sample_type')

    period_1 = dt.datetime(2013, 3, 31)  # starting point for first testing set
    base_space = {'verbosity': 0,
                  'nthread': 12,
                  'eval_metric': 'mae',
                  'grow_policy':'depthwise'}

    # create dict storing values/df used in training
    sql_result = {}  # data write to DB TABLE lightgbm_results
    hpot = {}  # storing data cxe  best trials in each Hyperopt
    resume = args.resume  # change to True if want to resume from the last running as on DB TABLE lightgbm_results
    sample_no = args.sample_no  # number of training/testing period go over ( 25 = until 2019-3-31)

    load_data_params = {'exclude_fwd': args.exclude_fwd,
                        'use_median': True,
                        'chron_valid': False,
                        'y_type': 'ibes',
                        'qcut_q': 10,
                        'ibes_qcut_as_x': not (args.exclude_fwd),
                        'exclude_stock': args.exclude_stock}

    data = load_data(macro_monthly=True)  # load all data: create load_data.main = df for all samples - within data(CLASS)

    x_type_map = {True: 'fwdepsqcut', False: 'ni'}  # True/False based on exclude_fwd
    sql_result['x_type'] = x_type_map[args.exclude_fwd]
    sql_result['name'] = 'rf {} -sample_type {} -x_type {}'.format(args.name_sql, args.sample_type, sql_result['x_type'])  # label experiment

    # update load_data data
    sql_result['qcut_q'] = load_data_params['qcut_q']  # number of Y classes
    sql_result['y_type'] = load_data_params['y_type']

    ''' start roll over testing period(25) / icb_code(16) / cross-validation sets(5) for hyperopt '''

    db_last_param, sql_result = read_db_last(sql_result)  # update sql_result['trial_hpot'/'trial_lgbm'] & got params for resume (if True)

    for icb_code in partitions:  # roll over industries (first 2 icb code)

        data.split_industry(icb_code, combine_ind=True)
        sql_result['icb_code'] = icb_code

        for i in tqdm(range(sample_no)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sql_result['testing_period'] = testing_period

            # when setting resume = TRUE -> continue training from last records in DB results_lightgbm
            if resume == True:

                if {'icb_code': icb_code,
                    'testing_period': pd.Timestamp(testing_period)} == db_last_param:  # if current loop = last records
                    resume = False
                    print('---------> Resume Training', icb_code, testing_period)
                else:
                    print('Not yet resume: params done', icb_code, testing_period)
                    continue

            sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, **load_data_params)
            sql_result['exclude_fwd'] = load_data_params['exclude_fwd']
            space = find_hyperspace(sql_result)

            X_test = np.nan_to_num(sample_set['test_x'], nan=0)
            Y_test = sample_set['test_y']
            sql_result['number_features'] = X_test.shape[1]

            cv_number = 1
            for train_index, test_index in cv:
                sql_result['cv_number'] = cv_number

                X_train = np.nan_to_num(sample_set['train_x'][train_index], nan=0)
                Y_train = sample_set['train_y'][train_index]
                X_valid = np.nan_to_num(sample_set['train_x'][test_index], nan=0)
                Y_valid = sample_set['train_y'][test_index]

                HPOT(space, max_evals=10)  # start hyperopt
                cv_number += 1