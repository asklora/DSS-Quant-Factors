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

from preprocess.load_data import load_data
import global_vals
from results_analysis.lgbm_pred_merge_rotate import download_stock_pred

space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
    # 'n_estimators': hp.choice('n_estimators', [15, 50, 100]),
    'max_depth': hp.choice('max_depth', [8, 32, 64]),
    'min_samples_split': hp.choice('min_samples_split', [5, 10, 50, 100]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [5, 10, 50]),
    'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0, 1e-2, 5e-2, 1e-1]),
    'max_features': hp.choice('max_features',[0.5, 0.7, 0.9]),
    'min_impurity_decrease': 0,
    'max_samples': hp.choice('max_samples',[0.5, 0.7, 0.9]),
    'ccp_alpha': hp.choice('ccp_alpha',[0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
    'n_jobs': -1,
    # 'random_state': 666
}

def rf_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()
    for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
        params[k] = int(params[k])
    print('===== hyperspace =====', params)
    sql_result.update(params)

    params['bootstrap'] = False

    if args.objective in ['gini','entropy']:        # Classification problem
        if args.tree_type == 'extra':
            regr = ExtraTreesClassifier(criterion=args.objective, **params)
        elif args.tree_type == 'rf':
            regr = RandomForestClassifier(criterion=args.objective, **params)
    elif args.objective in ['mse','mae']:           # Regression problem
        if args.tree_type == 'extra':
            regr = ExtraTreesRegressor(criterion=args.objective, **params)
        elif args.tree_type == 'rf':
            regr = RandomForestRegressor(criterion=args.objective, **params)

    regr.fit(sample_set['train_xx'], sample_set['train_yy_final'], sample_weight=sample_set['weight'])

    # prediction on all sets
    Y_train_pred = regr.predict(sample_set['train_xx'])
    Y_valid_pred = regr.predict(sample_set['valid_x'])
    Y_test_pred = regr.predict(sample_set['test_x'])

    sql_result['feature_importance'], feature_importance_df = to_list_importance(regr)

    return Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df

def eval_regressor(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials with Regressiong model
    -------------------------------------------------
    This part haven't been modified for multi-label questions purpose
    '''

    sql_result['finish_timing'] = dt.datetime.now()
    Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df = rf_train(space)

    result = {'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
              'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
              'mse_train': mean_squared_error(sample_set['train_yy'], Y_train_pred),
              'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
              'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
              'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred)}

    try:    # for backtesting -> calculate MAE/MSE/R2 for testing set
        test_df = pd.DataFrame({'actual':sample_set['test_y'].flatten(), 'pred': Y_test_pred.flatten()})
        test_df = test_df.dropna(how='any')
        # test_df = pd.DataFrame({'actual':sample_set['test_y'], 'pred': Y_test_pred})
        # test_df = test_df.dropna(how='any')
        result_test = {
            'mae_test': mean_absolute_error(sample_set['test_y'], Y_test_pred),
            'mse_test': mean_squared_error(sample_set['test_y'], Y_test_pred),
            # 'r2_test': r2_score(sample_set['test_y'], Y_test_pred)
        }
        # result['test_len'] = len(test_df)
        result.update(result_test)
    except Exception as e:  # for real_prediction -> no calculation
        print(e)
        pass

    sql_result.update(result)  # update result of model

    hpot['all_results'].append(sql_result.copy())

    if result['mae_valid'] < hpot['best_score']: # update best_mae to the lowest value for Hyperopt
        hpot['best_score'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

    if sql_result['objective'] == 'mae':
        return result['mse_valid']
    elif sql_result['objective'] == 'mse':
        return result['mae_valid']
    else:
        NameError('Objective not evaluated!')

def eval_classifier(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials with classification model
    -----------------------------------------------------------------------------------------
    Modified for multi-label questions purpose with maximize average accuracy score
    '''

    sql_result['finish_timing'] = dt.datetime.now()
    Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df = rf_train(space)

    result = {}
    for i in range(Y_test_pred.shape[1]):
        result[sql_result['y_type'][i]] = {'accuracy_train': accuracy_score(sample_set['train_yy_final'][i], Y_train_pred[i]),
                  'accuracy_valid': accuracy_score(sample_set['valid_y_final'][i], Y_valid_pred[i])}

        try:        # for backtesting -> calculate accuracy for testing set
            test_df = pd.DataFrame({'actual':sample_set['test_y_final'][i], 'pred': Y_test_pred[i]})
            test_df = test_df.dropna(how='any')
            result_test = {
                'accuracy_test': accuracy_score(test_df['actual'], test_df['pred']),
                # 'precision_test': precision_score(test_df['actual'], test_df['pred'], average='micro'),
                # 'recall_test': recall_score(test_df['actual'], test_df['pred'], average='micro'),
                # 'f1_test': f1_score(test_df['actual'], test_df['pred'], average='micro')
            }
            result[sql_result['y_type'][i]]['test_len'] = len(test_df)
            result[sql_result['y_type'][i]].update(result_test)
        except Exception as e:     # for real_prediction -> no calculation
            print('ERROR on: ', e)
            pass

    result_comb = {k: i.tolist() for k, i in pd.DataFrame(result).transpose().to_dict(orient='series').items()}

    sql_result.update(result_comb)  # update result of model
    sql_result['accuracy_valid_mean'] = np.mean(result_comb['accuracy_valid'])

    hpot['all_results'].append(sql_result.copy())

    if sql_result['accuracy_valid_mean'] > hpot['best_score']:   # update best_mae to the lowest value for Hyperopt
        hpot['best_score'] = sql_result['accuracy_valid_mean']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

    return 1 - np.mean(result_comb['accuracy_valid'])

def to_sql_prediction(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''
    # sql_result['y_type'] = data.y_col
    df = pd.DataFrame(Y_test_pred, index=data.test['group'].to_list(), columns=sql_result['y_type'])
    df = df.unstack().reset_index(drop=False)
    df.columns = ['y_type', 'group', 'pred']
    df['actual'] = sample_set['test_y_final'].flatten(order='F')       # also write actual qcut to BD
    df['finish_timing'] = [sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return df

def to_list_importance(rf):
    ''' based on rf model -> records feature importance in DataFrame to be uploaded to DB '''

    df = pd.DataFrame()
    df['name'] = data.x_col     # column names
    df['split'] = rf.feature_importances_
    df['finish_timing'] = [sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return ','.join(df.sort_values(by=['split'], ascending=False)['name'].to_list()), df

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    hpot['all_results'] = []
    trials = Trials()

    if sql_result['objective'] in ['mae', 'mse', 'friedman_mse']:
        hpot['best_score'] = 10000  # record best training (min mae_valid) in each hyperopt
        best = fmin(fn=eval_regressor, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        tbl_suffix = '_rf_reg'

        with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
            hpot['best_stock_df'].to_sql(f"{global_vals.result_pred_table}{tbl_suffix}", **extra)
            pd.DataFrame(hpot['all_results']).to_sql(f"{global_vals.result_score_table}{tbl_suffix}", **extra)
            hpot['best_stock_feature'].to_sql(f"{global_vals.feature_importance_table}{tbl_suffix}", **extra)
        global_vals.engine_ali.dispose()

    elif sql_result['objective'] in ['gini','entropy']:
        hpot['best_score'] = 0  # record best training (max accuracy_valid) in each hyperopt
        best = fmin(fn=eval_classifier, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        tbl_suffix = '_rf_class'

        with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
            hpot['best_stock_df'].to_sql(f"{global_vals.result_pred_table}{tbl_suffix}", **extra)
            pd.DataFrame(hpot['all_results']).to_sql(f"{global_vals.result_score_table}{tbl_suffix}", **extra)
            hpot['best_stock_feature'].to_sql(f"{global_vals.feature_importance_table}{tbl_suffix}", **extra)
        global_vals.engine_ali.dispose()

    print('===== best eval ===== ', best)

if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_type', default='extra')
    parser.add_argument('--objective', default='mse')
    parser.add_argument('--qcut_q', default=0, type=int)  # Default: Low, Mid, High
    parser.add_argument('--mode', default='v2', type=str)
    args = parser.parse_args()
    sql_result = vars(args)     # data write to DB TABLE lightgbm_results

    # --------------------------------- Different Config ------------------------------------------

    sql_result['name_sql'] = 'pca_tryaddx_'
    use_biweekly_stock = False
    stock_last_week_avg = True
    valid_method = 'chron'
    n_splits = 1
    defined_cut_bins = []
    group_code_list = ['EUR','USD','HKD']
    # group_code_list = ['HKD']
    use_pca = 0.6
    use_median = False

    # --------------------------------- Define Variables ------------------------------------------

    hpot = {}                   # storing data for best trials in each Hyperopt

    # create date list of all testing period
    if use_biweekly_stock:
        last_test_date = dt.datetime(2021,7,11)
        backtest_period = 50
        testing_period_list=[last_test_date+relativedelta(days=1) - 2*i*relativedelta(weeks=2)
                             - relativedelta(days=1) for i in range(0, backtest_period+1)]
    else:
        last_test_date = dt.date.today() + MonthEnd(-2)     # Default last_test_date is month end of 2 month ago from today
        backtest_period = 46
        testing_period_list=[last_test_date+relativedelta(days=1) - i*relativedelta(months=1)
                             - relativedelta(days=1) for i in range(0, backtest_period+1)]

    # --------------------------------- Model Training ------------------------------------------

    data = load_data(use_biweekly_stock=use_biweekly_stock, stock_last_week_avg=stock_last_week_avg, mode=args.mode)  # load_data (class) STEP 1
    sql_result['y_type'] = y_type = data.factor_list[:12]       # random forest model predict all factor at the same time
    # sql_result['y_type'] = []       # random forest model predict all factor at the same time
    # other_y = [x for x in data.x_col_dict['factor'] if x not in sql_result['y_type']]
    other_y = data.factor_list[:13]

    # sql_result['y_type'] = y_type = ['vol_0_30','book_to_price','earnings_yield','market_cap_usd']
    print(f"===== test on y_type", len(y_type), y_type, "=====")

    r_mean = 0

    i=1
    for y in other_y:
        sql_result['name_sql'] = f'pca_trylessx_{i}'
        # sql_result['y_type'].append(y)
        if i>1:
            sql_result['y_type'].remove(y)
        print(sql_result['y_type'])
        i += 1
    # if 1==1:
        for tree_type in ['extra']:
            sql_result['tree_type'] = tree_type
            for group_code in group_code_list:
                sql_result['group_code'] = group_code
                data.split_group(group_code)  # load_data (class) STEP 2
                for testing_period in reversed(testing_period_list):
                    sql_result['testing_period'] = testing_period
                    load_data_params = {'qcut_q': args.qcut_q, 'y_type': sql_result['y_type'],
                                        'valid_method': valid_method, 'defined_cut_bins': defined_cut_bins,
                                        'use_median': use_median, 'use_pca':use_pca, 'n_splits':n_splits}
                    try:
                        sample_set, cv = data.split_all(testing_period, **load_data_params)  # load_data (class) STEP 3
                        # # write stock_pred for the best hyperopt records to sql
                        # if (write_cutbins) & (args.objective == 'multiclass'):
                        #     cut_bins_df = data.cut_bins_df
                        #     cut_bins_df['testing_period'] = testing_period
                        #     cut_bins_df['group_code'] = group_code
                        #     cut_bins_df['name_sql'] = sql_result['name_sql']
                        #
                        #     with global_vals.engine_ali.connect() as conn:
                        #         extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
                        #         cut_bins_df.drop(['index'], axis=1).to_sql(global_vals.processed_cutbins_table, **extra)
                        #     global_vals.engine_ali.dispose()
                        cv_number = 1  # represent which cross-validation sets
                        for train_index, valid_index in cv:  # roll over 5 cross validation set
                            sql_result['cv_number'] = cv_number

                            sample_set['valid_x'] = sample_set['train_x'][valid_index]
                            sample_set['train_xx'] = sample_set['train_x'][train_index]
                            sample_set['valid_y'] = sample_set['train_y'][valid_index]
                            sample_set['train_yy'] = sample_set['train_y'][train_index]
                            sample_set['valid_y_final'] = sample_set['train_y_final'][valid_index]
                            sample_set['train_yy_final'] = sample_set['train_y_final'][train_index]

                            sql_result['train_len'] = len(sample_set['train_xx'])  # record length of training/validation sets
                            sql_result['valid_len'] = len(sample_set['valid_x'])

                            for k in ['valid_x','train_xx','test_x']:
                                sample_set[k] = np.nan_to_num(sample_set[k], nan=0)

                            # sample_set['weight'] = np.array(range(len(sample_set['train_xx'])))/len(sample_set['train_xx'])
                            # sample_set['weight'] = np.tanh(sample_set['weight']-0.5)+0.5
                            sample_set['weight'] = np.ones((len(sample_set['train_xx']),))

                            # sql_result['weight'] = pd.cut(sql_result['weight'], bins=12, labels=False)
                            # print(sql_result['weight'])
                            print(data.x_col)
                            sql_result['neg_factor'] = ','.join(data.neg_factor)
                            print(group_code, testing_period, len(sample_set['train_yy_final']))
                            HPOT(space, max_evals=20)  # start hyperopt
                            cv_number += 1
                    except Exception as e:
                        print(testing_period, e)
                        continue
        r = download_stock_pred(4, sql_result['name_sql'], False, False)
        if r_mean < (r['max_ret'][0] - r['min_ret'][0]):
            r_mean = r['max_ret'][0] - r['min_ret'][0]
        else:
            print(y, sql_result['y_type'])
            # if i > 1:
            # sql_result['y_type'].remove(y)
            sql_result['y_type'].append(y)
            print(sql_result['y_type'])
