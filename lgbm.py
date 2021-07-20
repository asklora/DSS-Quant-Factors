import datetime as dt
import lightgbm as lgb
import argparse
import pandas as pd
import numpy as np
from math import floor

from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, Trials
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from hyperspace_lgbm import find_hyperspace
from preprocess.load_data_lgbm import load_data
from pandas.tseries.offsets import MonthEnd

# from results_analysis.lgbm_merge import combine_pred, calc_mae_write, read_eval_best

import global_vals

def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()

    for k in ['max_bin', 'num_leaves', 'min_data_in_leaf','bagging_freq']:
        params[k] = int(params[k])
    sql_result.update(params)        # update hyper-parameter used in model
    print('===== hyperspace =====', params)

    lgb_train = lgb.Dataset(sample_set['train_xx'], label=sample_set['train_yy_final'], free_raw_data=False)
    lgb_eval = lgb.Dataset(sample_set['valid_x'], label=sample_set['valid_y_final'], free_raw_data=False, reference=lgb_train)

    evals_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_eval, lgb_train],
                    valid_names=['valid', 'train'],
                    num_boost_round=1000,
                    early_stopping_rounds=150,
                    feature_name=data.train.columns.to_list()[2:-1],
                    evals_result=evals_result)

    Y_train_pred = gbm.predict(sample_set['train_xx'], num_iteration=gbm.best_iteration)    # prediction on all sets
    Y_valid_pred = gbm.predict(sample_set['valid_x'], num_iteration=gbm.best_iteration)
    Y_test_pred = gbm.predict(sample_set['test_x'], num_iteration=gbm.best_iteration)

    sql_result['feature_importance'] = to_list_importance(gbm)

    return Y_train_pred, Y_valid_pred, Y_test_pred, evals_result

def eval(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials '''

    sql_result['finish_timing'] = dt.datetime.now()
    Y_train_pred, Y_valid_pred, Y_test_pred, evals_result = lgbm_train(space)

    result = {'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
              'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
              'mse_train': mean_squared_error(sample_set['train_yy'], Y_train_pred),
              'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
              'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
              'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred)}

    try:
        test_df = pd.DataFrame({'actual':sample_set['test_y'], 'pred': Y_test_pred})
        test_df = test_df.dropna(how='any')
        result_test = {
            'mae_test': mean_absolute_error(test_df['actual'], test_df['pred']),
            'mse_test': mean_squared_error(test_df['actual'], test_df['pred']),
            'r2_test': r2_score(test_df['actual'], test_df['pred'])}
        result['test_len'] = len(test_df)
        result.update(result_test)
    except:
        result_test = {
            'mae_test': np.nan,
            'mse_test': np.nan,
            'r2_test': np.nan}
        result['test_len'] = np.nan
        result.update(result_test)

    sql_result.update(result)  # update result of model

    hpot['all_results'].append(sql_result.copy())
    print(f'===== records written to DB {global_vals.test_score_results_table} ===== ', sql_result)

    if result['mae_valid'] < hpot['best_mae']: # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)

    if sql_result['objective'] == 'regression_l2':
        return result['mse_valid']
    elif sql_result['objective'] == 'regression_l1':
        return result['mae_valid']
    else:
        NameError('Objective not evaluated!')

def to_sql_prediction(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['group'] = data.test['group'].to_list()
    df['pred'] = Y_test_pred
    df['finish_timing'] = [sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return df

def to_list_importance(gbm):
    ''' based on gbm model -> records feature importance in DataFrame to be uploaded to DB '''

    df = pd.DataFrame()
    df['name'] = data.train.columns.to_list()[2:-1]     # column names
    df['split'] = gbm.feature_importance(importance_type='split')
    df['split'] = df['split'].rank(ascending=False)
    return ','.join(df.sort_values(by=['split'], ascending=True)['name'].to_list())

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    hpot['best_mae'] = 10000  # record best training (min mae_valid) in each hyperopt
    hpot['all_results'] = []

    trials = Trials()
    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print('===== best eval ===== ', best)

    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
        hpot['best_stock_df'].to_sql(global_vals.test_pred_results_table, **extra)
        pd.DataFrame(hpot['all_results']).to_sql(global_vals.test_score_results_table, **extra)
    global_vals.engine.dispose()

if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='regression_l2')     # OPTIONS: regression_l1 / regression_l2
    parser.add_argument('--y_type', default='all')                  # OPTIONS: ibes_yoy / ibes_qoq / rev_yoy
    parser.add_argument('--qcut_q', default=3, type=int)            # Default: Low, Mid, High
    # parser.add_argument('--backtest_period', default=12, type=int)
    # parser.add_argument('--last_quarter', default='')             # OPTIONS: 'YYYYMMDD' date format
    parser.add_argument('--max_eval', type=int, default=10)         # for hyperopt
    parser.add_argument('--nthread', default=12, type=int)          # for the best speed, set this to the number of real CPU cores
    args = parser.parse_args()
    print(args)

    # --------------------------------- Define Variables ------------------------------------------

    base_space = {'verbose': -1,
                  'objective': args.objective,
                  'num_threads': args.nthread}

    # create dict storing values/df used in training
    sql_result = vars(args)     # data write to DB TABLE lightgbm_results
    sql_result['name_sql'] = f'{args.y_type}_{dt.datetime.now()}_' + 'testing'
    hpot = {}                   # storing data for best trials in each Hyperopt

    last_test_date = dt.date.today() + MonthEnd(-2)     # Default last_test_date is month end of 2 month ago from today
    backtest_period = 22
    # self-defined last testing date from Parser
    # if args.last_quarter != '':
    #     last_test_date = dt.datetime.strptime(args.last_quarter, "%Y%m%d")
    # del sql_result['last_quarter']

    # create date list of all testing period
    testing_period_list=[last_test_date+relativedelta(days=1) - i*relativedelta(months=1)
                         - relativedelta(days=1) for i in range(0, backtest_period+1)]
    print(f"===== test on sample sets {testing_period_list[-1].strftime('%Y-%m-%d')} to "
          f"{testing_period_list[0].strftime('%Y-%m-%d')} ({len(testing_period_list)}) =====")

    # --------------------------------- Model Training ------------------------------------------

    data = load_data()                                                              # load_data (class) STEP 1
    for group_code in ['industry', 'currency']:
        sql_result['group_code'] = group_code
        data.split_group(group_code)                                                # load_data (class) STEP 2
        # for f in data.factor_list:
        for y_type in ['earnings_yield']:
            print(y_type)
            for testing_period in reversed(testing_period_list):
                sql_result['testing_period'] = testing_period
                backtest = testing_period not in testing_period_list[0:4]
                load_data_params = {'qcut_q': args.qcut_q,
                                    'y_type': y_type}

                if 1==1:
                # try:
                    sample_set, cv = data.split_all(testing_period, **load_data_params)  # load_data (class) STEP 3

                    cv_number = 1   # represent which cross-validation sets
                    for train_index, valid_index in cv:     # roll over 5 cross validation set
                        sql_result['cv_number'] = cv_number

                        sample_set['valid_x'] = sample_set['train_x'][valid_index]
                        sample_set['train_xx'] = sample_set['train_x'][train_index]
                        sample_set['valid_y'] = sample_set['train_y'][valid_index]
                        sample_set['train_yy'] = sample_set['train_y'][train_index]
                        sample_set['valid_y_final'] = sample_set['train_y_final'][valid_index]
                        sample_set['train_yy_final'] = sample_set['train_y_final'][train_index]

                        sql_result['train_len'] = len(sample_set['train_xx']) # record length of training/validation sets
                        sql_result['valid_len'] = len(sample_set['valid_x'])

                        space = find_hyperspace(sql_result)
                        space.update(base_space)
                        print(group_code, testing_period, len(sample_set['train_yy_final']))
                        HPOT(space, max_evals=args.max_eval)   # start hyperopt
                        cv_number += 1
                # except:
                #     print('ERROR ON ', testing_period)
                #     continue

    # --------------------------------- Results Analysis ------------------------------------------

    # data = load_data(y_type=args.y_type, first_test=testing_period_list[-1], restart_eval=True)     # restart evaluation
    # sql_result['name_sql'] = 'rev_yoy_2021-07-09 09:23:27.029713'                                      # restart evaluation

    # results = combine_pred(sql_result['name_sql'], restart_eval=False).combine_industry_market()     # write consolidated results to DB
    #
    # calc = calc_mae_write(sql_result['name_sql'], results, data.all_test, restart_eval=False)    # calculate MAE/MSE/R2 for backtesting
    #
    # best_pred = read_eval_best(pred=calc.all_results, eval=calc.all_metrices)   # best prediciton based on backtest MAE

    # print(best_pred)