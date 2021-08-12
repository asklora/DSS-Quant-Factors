import datetime as dt
import lightgbm as lgb
import argparse
import pandas as pd
import numpy as np
import numpy.ma as ma
import os
from math import floor
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, Trials
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from pandas.tseries.offsets import MonthEnd
# from results_analysis.lgbm_merge import combine_pred, calc_mae_write, read_eval_best

from preprocess.load_data import load_data
from hyperspace_lgbm import find_hyperspace
import global_vals

from utils import remove_tables_with_suffix

to_sql_suffix = ""

# ------------------------------------------- Train LightGBM ----------------------------------------------------

def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()

    for k in ['max_bin', 'num_leaves', 'min_data_in_leaf','bagging_freq']:
        params[k] = int(params[k])
    sql_result.update(params)        # update hyper-parameter used in model
    print('===== hyperspace =====', params)
    # params['is_unbalance'] = True
    params['min_hessian'] = 0
    params['first_metric_only'] = True
    params['verbose'] = -1

    lgb_train = lgb.Dataset(sample_set['train_xx'], label=sample_set['train_yy_final'], free_raw_data=False)
    lgb_eval = lgb.Dataset(sample_set['valid_x'], label=sample_set['valid_y_final'], free_raw_data=False,
                           reference=lgb_train)

    evals_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_eval, lgb_train],
                    valid_names=['valid', 'train'],
                    num_boost_round=1000,
                    early_stopping_rounds=150,
                    feature_name=data.x_col,
                    evals_result=evals_result)

    # prediction on all sets if using regression
    Y_train_pred_softmax = gbm.predict(sample_set['train_xx'], num_iteration=gbm.best_iteration)
    Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    Y_valid_pred_softmax = gbm.predict(sample_set['valid_x'], num_iteration=gbm.best_iteration)
    Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
    Y_test_pred_softmax = gbm.predict(sample_set['test_x'], num_iteration=gbm.best_iteration)
    Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]
    sql_result['feature_importance'], feature_importance_df = to_list_importance(gbm)
    return Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, feature_importance_df, Y_test_pred_softmax

# -------------------------------- Evaluate Results (Regression / Classification) -------------------------------------

def eval_classifier(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials with classification model '''

    sql_result['finish_timing'] = dt.datetime.now()
    Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, feature_importance_df, Y_test_pred_proba = lgbm_train(space)

    lb = LabelBinarizer().fit(list(range(len(sql_result['y_type']))))

    result = {'accuracy_train': accuracy_score(sample_set['train_yy_final'], Y_train_pred),
              'accuracy_valid': accuracy_score(sample_set['valid_y_final'], Y_valid_pred),
              'return_train': np.mean(sample_set['train_yy']*lb.transform(Y_train_pred)*4),
              'return_valid': np.mean(sample_set['valid_y']*lb.transform(Y_valid_pred)*4),
        }
    try:        # for backtesting -> calculate accuracy for testing set
        test_df = pd.DataFrame({'actual':sample_set['test_y_final'], 'pred': Y_test_pred})
        test_df = test_df.dropna(how='any')
        result_test = {
            'accuracy_test': accuracy_score(test_df['actual'], test_df['pred']),
            'return_test': np.mean(test_df['actual'].values * lb.transform(test_df['pred']) * 4),
        }

        result['test_len'] = len(test_df)
        result.update(result_test)
    except Exception as e:     # for real_prediction -> no calculation
        print(e)
        pass

    sql_result.update(result)  # update result of model

    hpot['all_results'].append(sql_result.copy())

    if result['accuracy_valid'] > hpot['best_score']:   # update best_mae to the lowest value for Hyperopt
        hpot['best_score'] = result['accuracy_valid']
        hpot['best_stock_df'] = to_sql_prediction(result_test['return'], Y_test_pred_proba)
        hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

    return 1 - result['accuracy_valid']

# -------------------------------------- Organize / Visualize Results -------------------------------------------

def to_sql_prediction(Y_test_pred, Y_test_pred_proba=None):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['group'] = data.test['group'].to_list()
    df['pred'] = Y_test_pred
    df['proba'] = [','.join([str(i) for i in x]) for x in Y_test_pred_proba]
    df['actual'] = np.max(sample_set['test_y_final'], axis=1)
    # df['y_type'] = sql_result['y_type']
    df['finish_timing'] = [sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return df

def to_list_importance(gbm):
    ''' based on gbm model -> records feature importance in DataFrame to be uploaded to DB '''

    df = pd.DataFrame()
    df['name'] = data.x_col     # column names
    df['split'] = gbm.feature_importance(importance_type='split')
    df['finish_timing'] = [sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return ','.join(df.sort_values(by=['split'], ascending=False)['name'].to_list()), df

# ----------------------------------- Hyperopt & Write Best Iteration to DB ----------------------------------------

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    hpot['all_results'] = []
    trials = Trials()

    hpot['best_score'] = 0  # record best training (max accuracy_valid) in each hyperopt
    best = fmin(fn=eval_classifier, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 1000}
        hpot['best_stock_df'].to_sql(global_vals.result_pred_table+"_lgbm_class"+to_sql_suffix, **extra)
        pd.DataFrame(hpot['all_results']).to_sql(global_vals.result_score_table+"_lgbm_class"+to_sql_suffix, **extra)
        hpot['best_stock_feature'].to_sql(global_vals.feature_importance_table+"_lgbm_class"+to_sql_suffix, **extra)
    global_vals.engine_ali.dispose()

    print('===== best eval ===== ', best)

if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='regression_l2')     # OPTIONS: regression_l1 / regression_l2 / multiclass
    parser.add_argument('--max_eval', type=int, default=10)         # for hyperopt
    parser.add_argument('--nthread', default=12, type=int)          # for the best speed, set this to the number of real CPU cores
    args = parser.parse_args()
    print(args)
    sql_result = vars(args)     # data write to DB TABLE lightgbm_results

    # --------------------------------- Different Config ------------------------------------------

    sql_result['name_sql'] = 'lastweekavg_rank'
    n_splits = 1
    use_biweekly_stock = False
    stock_last_week_avg = True
    # factors_to_test = ['stock_return_r6_2']
    valid_method = 'chron'     # cv/chron
    defined_cut_bins = []
    group_code_list = ['currency']
    use_median = False
    continue_test = False
    test_change = False

    # --------------------------------- Define Variables ------------------------------------------

    # create dict storing values/df used in training
    hpot = {}                   # storing data for best trials in each Hyperopt
    base_space = {}
    write_cutbins = True        # write cut bins to DB

    # update additional base_space for Hyperopt
    base_space['metric'] = 'multi_logloss' # multi_logloss
    base_space['objective'] = 'multiclass'

    # create date list of all testing period
    if use_biweekly_stock:
        last_test_date = dt.datetime(2021,7,4)
        backtest_period = 100
        testing_period_list=[last_test_date+relativedelta(days=1) - 2*i*relativedelta(weeks=2)
                             - relativedelta(days=1) for i in range(0, backtest_period+1)]
    else:
        last_test_date = dt.date.today() + MonthEnd(-2)     # Default last_test_date is month end of 2 month ago from today
        backtest_period = 46
        testing_period_list=[last_test_date+relativedelta(days=1) - i*relativedelta(months=1)
                             - relativedelta(days=1) for i in range(0, backtest_period+1)]

    print(f"===== test on sample sets {testing_period_list[-1].strftime('%Y-%m-%d')} to "
          f"{testing_period_list[0].strftime('%Y-%m-%d')} ({len(testing_period_list)}) =====")

    # read last record configuration for continue testing (continue_test = True)
    if continue_test:
        with global_vals.engine_ali.connect() as conn:
            last_record = pd.read_sql(f"SELECT y_type, group_code, testing_period FROM {global_vals.result_score_table}_lgbm_class "
                         f"WHERE name_sql='{sql_result['name_sql']}' ORDER BY finish_timing desc LIMIT 1", conn)       # download training history
        global_vals.engine_ali.dispose()
        last_record = last_record.iloc[0,:].to_list()

    if os.environ.get('FACTORS_LGBM_REMOVE_CACHE', 'false').lower() == 'true':
        print("FACTORS_LGBM_REMOVE_CACHE")
        remove_tables_with_suffix(global_vals.engine_ali, to_sql_suffix)

    # --------------------------------- Model Training ------------------------------------------

    data = load_data(use_biweekly_stock=use_biweekly_stock, stock_last_week_avg=stock_last_week_avg)  # load_data (class) STEP 1
    # factors_to_test = data.factor_list[1:]
    factors_to_test = ['market_cap_usd','vol_0_30','book_to_price','earnings_yield']

    print(f"===== test on y_type", len(factors_to_test), factors_to_test, "=====")
    sql_result['y_type'] = factors_to_test
    for group_code in ['currency']:
        sql_result['group_code'] = group_code
        data.split_group(group_code)  # load_data (class) STEP 2
        for testing_period in testing_period_list:
            sql_result['testing_period'] = testing_period
            load_data_params = {'y_type': sql_result['y_type'],
                                'valid_method': valid_method, 'defined_cut_bins': defined_cut_bins}
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

                    sample_set['test_y'] = np.argmax(sample_set['test_y'], axis=1)
                    sample_set['valid_y_final'] = np.argmax(sample_set['valid_y_final'], axis=1)
                    sample_set['train_yy_final'] = np.argmax(sample_set['train_yy_final'], axis=1)

                    sql_result['train_len'] = len(sample_set['train_xx'])  # record length of training/validation sets
                    sql_result['valid_len'] = len(sample_set['valid_x'])
                    sql_result['valid_group'] = ','.join(list(data.train['group'][valid_index].unique()))

                    print(group_code, testing_period, len(sample_set['train_yy_final']))
                    sql_result['objective'] = 'multiclass'
                    space = find_hyperspace(sql_result)
                    base_space['num_class'] = len(sql_result['y_type'])
                    space.update(base_space)
                    HPOT(space, max_evals=10)  # start hyperopt
                    cv_number += 1
            except Exception as e:
                print(testing_period, e)
                continue
