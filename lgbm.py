import datetime as dt
import lightgbm as lgb
import argparse
import pandas as pd
import numpy as np
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
    params['verbose'] = 2
    params['metric'] = 'l2'  # multi_logloss

    # if args.objective == 'multiclass':
    #     dict_weight = {0:2,1:1,2:1}
    #     lgb_train = lgb.Dataset(sample_set['train_xx'], label=sample_set['train_yy_final'],
    #                             weight=list(map(dict_weight.get, sample_set['train_yy_final'])), free_raw_data=False)
    #     lgb_eval = lgb.Dataset(sample_set['valid_x'], label=sample_set['valid_y_final'],
    #                            weight=list(map(dict_weight.get, sample_set['valid_y_final'])), free_raw_data=False, reference=lgb_train)
    # else:
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
    if sql_result['objective'] in ['regression_l1', 'regression_l2']:

        Y_train_pred = gbm.predict(sample_set['train_xx'], num_iteration=gbm.best_iteration)
        Y_valid_pred = gbm.predict(sample_set['valid_x'], num_iteration=gbm.best_iteration)
        Y_test_pred = gbm.predict(sample_set['test_x'], num_iteration=gbm.best_iteration)
        sql_result['feature_importance'], feature_importance_df = to_list_importance(gbm)
        return Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, feature_importance_df

    # prediction on all sets if using classification
    elif sql_result['objective'] in ['multiclass']:
        Y_train_pred_softmax = gbm.predict(sample_set['train_xx'], num_iteration=gbm.best_iteration)
        Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
        Y_valid_pred_softmax = gbm.predict(sample_set['valid_x'], num_iteration=gbm.best_iteration)
        Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
        Y_test_pred_softmax = gbm.predict(sample_set['test_x'], num_iteration=gbm.best_iteration)
        Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]
        sql_result['feature_importance'], feature_importance_df = to_list_importance(gbm)
        return Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, feature_importance_df, Y_test_pred_softmax

# -------------------------------- Evaluate Results (Regression / Classification) -------------------------------------

def eval_regressor(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials with Regressiong model '''

    sql_result['finish_timing'] = dt.datetime.now()
    Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, feature_importance_df = lgbm_train(space)

    def port_ret(ret, pred, set_name='test_x'):
        if test_change:
            pred = (pred+1)*sample_set[set_name][:,data.x_col.index(sql_result['y_type'])]
        return np.nanmean(ret[np.array(pred) > np.nanquantile(pred, 2/3)]) - np.nanmean(ret[np.array(pred) < np.nanquantile(pred, 1/3)])

    result = {'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
              'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
              'mse_train': mean_squared_error(sample_set['train_yy'], Y_train_pred),
              'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
              'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
              'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred),
              'return_train': port_ret(sample_set['train_yy'], Y_train_pred, 'train_xx'),
              'return_valid': port_ret(sample_set['valid_y'], Y_valid_pred, 'valid_x'),
              }

    try:    # for backtesting -> calculate MAE/MSE/R2 for testing set
        test_df = pd.DataFrame({'actual':sample_set['test_y'], 'pred': Y_test_pred})
        test_df = test_df.dropna(how='any')
        result_test = {
            'mae_test': mean_absolute_error(test_df['actual'], test_df['pred']),
            'mse_test': mean_squared_error(test_df['actual'], test_df['pred']),
            'r2_test': r2_score(test_df['actual'], test_df['pred']),
            'return_test':  port_ret(test_df['actual'], test_df['pred'], 'test_x'),
        }
        result['test_len'] = len(test_df)
        result.update(result_test)
    except Exception as e:  # for real_prediction -> no calculation
        print(e)
        pass

    sql_result.update(result)  # update result of model

    hpot['all_results'].append(sql_result.copy())

    if result['mae_valid'] < hpot['best_score']: # update best_mae to the lowest value for Hyperopt
        hpot['best_score'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_stock_feature'] = feature_importance_df

    if sql_result['objective'] == 'regression_l2':
        return 1-result['r2_test']
    elif sql_result['objective'] == 'regression_l1':
        return result['mae_valid']
    else:
        NameError('Objective not evaluated!')

def eval_classifier(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials with classification model '''

    sql_result['finish_timing'] = dt.datetime.now()
    Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, feature_importance_df, Y_test_pred_proba = lgbm_train(space)

    def class_ret(ret, pred, ret_class):
        x = np.nanmean(ret[np.array(pred) == ret_class])
        if np.isnan(x):
            x = 0
        return x

    dict_weight = {0:2,1:1,2:2}

    result = {'accuracy_train': accuracy_score(sample_set['train_yy_final'], Y_train_pred),
              'accuracy_valid': accuracy_score(sample_set['valid_y_final'], Y_valid_pred),
              'precision_train': precision_score(sample_set['train_yy_final'], Y_train_pred, average='macro',
                                                  sample_weight=list(map(dict_weight.get, sample_set['train_yy_final']))),
              'precision_valid': precision_score(sample_set['valid_y_final'], Y_valid_pred, average='macro',
                                                  sample_weight=list(map(dict_weight.get, sample_set['valid_y_final']))),
              'return_train': class_ret(sample_set['train_yy'], Y_train_pred, 2) - class_ret(sample_set['train_yy'],
                                                                                             Y_train_pred, 0),
              'return_valid': class_ret(sample_set['valid_y'], Y_valid_pred, 2) - class_ret(sample_set['valid_y'],
                                                                                            Y_valid_pred, 0)}
    try:        # for backtesting -> calculate accuracy for testing set
        test_df = pd.DataFrame({'actual':sample_set['test_y_final'], 'pred': Y_test_pred})
        test_df = test_df.dropna(how='any')
        result_test = {
            'accuracy_test': accuracy_score(test_df['actual'], test_df['pred']),
            'precision_test': precision_score(test_df['actual'], test_df['pred'], average='macro',
                                                sample_weight=list(map(dict_weight.get, test_df['actual'].values))),
            # 'mae_test': mean_absolute_error(test_df['actual'], test_df['pred']),
            # 'mse_test': mean_squared_error(test_df['actual'], test_df['pred']),
            # 'r2_test': r2_score(test_df['actual'], test_df['pred']),
        }
        # test_true_arr = LabelBinarizer().fit(list(range(sql_result['qcut_q']))).transform(test_df['actual'])
        # Y_test_pred_proba = Y_test_pred_proba[list(test_df.index),:]
        # result['auc_test'] = roc_auc_score(test_true_arr, Y_test_pred_proba, multi_class='ovr')
        result['test_len'] = len(test_df)
        result['return_test'] = class_ret(sample_set['test_y'], Y_test_pred, 2) - class_ret(sample_set['test_y'], Y_test_pred, 0)
        result.update(result_test)
    except Exception as e:     # for real_prediction -> no calculation
        print(e)
        pass

    sql_result.update(result)  # update result of model

    hpot['all_results'].append(sql_result.copy())

    if result['precision_valid'] > hpot['best_score']:   # update best_mae to the lowest value for Hyperopt
        hpot['best_score'] = result['accuracy_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred, Y_test_pred_proba)
        hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

    return 1 - result['precision_valid']

# -------------------------------------- Organize / Visualize Results -------------------------------------------

def to_sql_prediction(Y_test_pred, Y_test_pred_proba=None):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['group'] = data.test['group'].to_list()
    df['pred'] = Y_test_pred
    if sql_result['objective'] in ['regression_l1', 'regression_l2']:       # for regression use original (before qcut/convert to median)
        df['actual'] = sample_set['test_y']
    elif sql_result['objective'] in ['multiclass']:         # for classification use after qcut
        df['proba'] = [','.join([str(i) for i in x]) for x in Y_test_pred_proba]
        df['actual'] = sample_set['test_y_final']
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

    if sql_result['objective'] in ['regression_l1', 'regression_l2']:
        hpot['best_score'] = 10000  # record best training (min mae_valid) in each hyperopt
        best = fmin(fn=eval_regressor, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 1000}
            hpot['best_stock_df'].to_sql(global_vals.result_pred_table+"_lgbm_reg"+to_sql_suffix, **extra)
            pd.DataFrame(hpot['all_results']).to_sql(global_vals.result_score_table+"_lgbm_reg"+to_sql_suffix, **extra)
            hpot['best_stock_feature'].to_sql(global_vals.feature_importance_table+"_lgbm_reg"+to_sql_suffix, **extra)
        global_vals.engine_ali.dispose()

    elif sql_result['objective'] in ['multiclass']:
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
    parser.add_argument('--qcut_q', default=0, type=int)            # Default: Low, Mid, High
    # parser.add_argument('--backtest_period', default=12, type=int)
    # parser.add_argument('--last_quarter', default='')             # OPTIONS: 'YYYYMMDD' date format
    parser.add_argument('--max_eval', type=int, default=20)         # for hyperopt
    parser.add_argument('--nthread', default=12, type=int)          # for the best speed, set this to the number of real CPU cores
    args = parser.parse_args()
    print(args)
    sql_result = vars(args)     # data write to DB TABLE lightgbm_results

    # --------------------------------- Different Config ------------------------------------------

    sql_result['name_sql'] = 'newlastweekavg_dart'
    n_splits = 1
    use_biweekly_stock = False
    stock_last_week_avg = True
    # factors_to_test = ['stock_return_r6_2']
    valid_method = 'chron'     # cv/chron
    defined_cut_bins = []
    group_code_list = ['currency']
    use_median = True
    continue_test = False
    test_change = False

    # from preprocess.ratios_calculations import calc_factor_variables
    # from preprocess.premium_calculation import calc_premium_all

    # recalculate ratio & premium before rerun regression
    # calc_factor_variables(price_sample='last_week_avg', fill_method='fill_all', sample_interval='monthly',
    #                       use_cached=True, save=False, update=False)
    # calc_premium_all(stock_last_week_avg=True, use_biweekly_stock=False, update=False)

    # --------------------------------- Define Variables ------------------------------------------

    # create dict storing values/df used in training
    hpot = {}                   # storing data for best trials in each Hyperopt
    base_space = {}
    write_cutbins = True        # write cut bins to DB

    # update additional base_space for Hyperopt
    if args.objective == 'multiclass':
        base_space['num_class'] = sql_result['qcut_q']
        base_space['metric'] = 'multi_logloss' # multi_logloss

    # create date list of all testing period
    if use_biweekly_stock:
        last_test_date = dt.datetime(2021,6,27)
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
    factors_to_test = ['vol_0_30']
    print(f"===== test on y_type", len(factors_to_test), factors_to_test, "=====")
    for f in factors_to_test:
        sql_result['y_type'] = f
        print(sql_result['y_type'])
        for group_code in group_code_list:
            sql_result['group_code'] = group_code
            data.split_group(group_code)                                                # load_data (class) STEP 2
            for testing_period in testing_period_list:
                if continue_test:
                    if [f, group_code,testing_period] != last_record:
                        continue
                    else:
                        continue_test = False
                        print(' ----------------------------> Continue testing from', last_record)

                sql_result['testing_period'] = testing_period
                load_data_params = {'qcut_q': args.qcut_q, 'y_type': [sql_result['y_type']],
                                    'valid_method':valid_method, 'defined_cut_bins': defined_cut_bins,
                                    'use_median':use_median, 'n_splits':n_splits, 'test_change':test_change}
                try:
                    sample_set, cv = data.split_all(testing_period, **load_data_params)  # load_data (class) STEP 3

                    # write stock_pred for the best hyperopt records to sql
                    if (write_cutbins)&(args.objective=='multiclass'):
                        cut_bins_df = data.cut_bins_df
                        cut_bins_df['testing_period'] = testing_period
                        cut_bins_df['group_code'] = group_code
                        cut_bins_df['name_sql'] = sql_result['name_sql']

                        with global_vals.engine_ali.connect() as conn:
                            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
                            cut_bins_df.drop(['index'], axis=1).to_sql(global_vals.processed_cutbins_table, **extra)
                        global_vals.engine_ali.dispose()

                    cv_number = 1   # represent which cross-validation sets
                    for train_index, valid_index in cv:     # roll over 5 cross validation set
                        sql_result['cv_number'] = cv_number

                        sample_set['valid_x'] = sample_set['train_x'][valid_index]
                        sample_set['train_xx'] = sample_set['train_x'][train_index]
                        sample_set['valid_y'] = sample_set['train_y'][valid_index]
                        sample_set['train_yy'] = sample_set['train_y'][train_index]
                        sample_set['valid_y_final'] = sample_set['train_y_final'][valid_index]
                        sample_set['train_yy_final'] = sample_set['train_y_final'][train_index]

                        sample_set['train_yy'] = sample_set['train_yy'].flatten()
                        sample_set['valid_y'] = sample_set['valid_y'].flatten()
                        sample_set['test_y'] = sample_set['test_y'].flatten()
                        sample_set['valid_y_final'] = sample_set['valid_y_final'].flatten()
                        sample_set['train_yy_final'] = sample_set['train_yy_final'].flatten()
                        sample_set['test_y_final'] = sample_set['test_y_final'].flatten()

                        sql_result['train_len'] = len(sample_set['train_xx']) # record length of training/validation sets
                        sql_result['valid_len'] = len(sample_set['valid_x'])
                        sql_result['valid_group'] = ','.join(list(data.train['group'][valid_index].unique()))

                        space = find_hyperspace(sql_result)
                        space.update(base_space)
                        print(group_code, testing_period, len(sample_set['train_yy_final']))
                        HPOT(space, max_evals=args.max_eval)   # start hyperopt
                        cv_number += 1
                except Exception as e:
                    print(testing_period, e)
                    raise e
                    exit(2)
                    continue

        write_cutbins = False

    # --------------------------------- Results Analysis ------------------------------------------

    # data = load_data(y_type=args.y_type, first_test=testing_period_list[-1], restart_eval=True)     # restart evaluation
    # sql_result['name_sql'] = 'rev_yoy_2021-07-09 09:23:27.029713'                                      # restart evaluation

    # results = combine_pred(sql_result['name_sql'], restart_eval=False).combine_industry_market()     # write consolidated results to DB
    #
    # calc = calc_mae_write(sql_result['name_sql'], results, data.all_test, restart_eval=False)    # calculate MAE/MSE/R2 for backtesting
    #
    # best_pred = read_eval_best(pred=calc.all_results, eval=calc.all_metrices)   # best prediciton based on backtest MAE

    # print(best_pred)