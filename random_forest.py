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

space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'max_depth': hp.choice('max_depth', [4, 8, 12]),
    'min_samples_split': hp.choice('min_samples_split', [5, 25, 100]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 5, 50]),
    'min_weight_fraction_leaf': 0.1,
    'max_features': hp.choice('max_features',[0.3, 0.5, 0.8]),
    'min_impurity_decrease': 0,
    'max_samples': hp.choice('max_samples',[0.3, 0.6, 0.9]),
    'n_jobs': -1,
    'random_state': 666}

def rf_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()
    for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
        params[k] = int(params[k])
    print('===== hyperspace =====', params)
    sql_result.update(params)

    # params['n_jobs'] = 6

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

    regr.fit(sample_set['train_xx'], sample_set['train_yy_final'])

    # prediction on all sets
    Y_train_pred = regr.predict(sample_set['train_xx'])
    Y_valid_pred = regr.predict(sample_set['valid_x'])
    Y_test_pred = regr.predict(sample_set['test_x'])

    sql_result['feature_importance'] = to_list_importance(regr)

    return Y_train_pred, Y_valid_pred, Y_test_pred

def eval_regressor(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials with Regressiong model
    -------------------------------------------------
    This part haven't been modified for multi-label questions purpose
    '''

    sql_result['finish_timing'] = dt.datetime.now()
    Y_train_pred, Y_valid_pred, Y_test_pred = rf_train(space)

    result = {'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
              'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
              'mse_train': mean_squared_error(sample_set['train_yy'], Y_train_pred),
              'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
              'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
              'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred)}

    try:    # for backtesting -> calculate MAE/MSE/R2 for testing set
        test_df = pd.DataFrame({'actual':sample_set['test_y'], 'pred': Y_test_pred})
        test_df = test_df.dropna(how='any')
        result_test = {
            'mae_test': mean_absolute_error(test_df['actual'], test_df['pred']),
            'mse_test': mean_squared_error(test_df['actual'], test_df['pred']),
            'r2_test': r2_score(test_df['actual'], test_df['pred'])}
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
    Y_train_pred, Y_valid_pred, Y_test_pred = rf_train(space)

    result = {}
    for i in range(Y_test_pred.shape[1]):

        result[i] = {'accuracy_train': accuracy_score(sample_set['train_yy_final'][i], Y_train_pred[i]),
                  'accuracy_valid': accuracy_score(sample_set['valid_y_final'][i], Y_valid_pred[i])}

        try:        # for backtesting -> calculate accuracy for testing set
            test_df = pd.DataFrame({'actual':sample_set['test_y_final'][i], 'pred': Y_test_pred[i]})
            test_df = test_df.dropna(how='any')
            result_test = {
                'accuracy_test': accuracy_score(test_df['actual'], test_df['pred']),
                'precision_test': precision_score(test_df['actual'], test_df['pred'], average='micro'),
                'recall_test': recall_score(test_df['actual'], test_df['pred'], average='micro'),
                'f1_test': f1_score(test_df['actual'], test_df['pred'], average='micro')}
            result[i]['test_len'] = len(test_df)
            result[i].update(result_test)
        except Exception as e:     # for real_prediction -> no calculation
            print(e)
            pass

    result_comb = {k: i.tolist() for k, i in pd.DataFrame(result).transpose().to_dict(orient='series').items()}

    sql_result.update(result_comb)  # update result of model

    hpot['all_results'].append(sql_result.copy())
    print(np.mean(result_comb['accuracy_valid']))

    if np.mean(result_comb['accuracy_valid']) > hpot['best_score']:   # update best_mae to the lowest value for Hyperopt
        hpot['best_score'] = np.mean(result_comb['accuracy_valid'])
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)

    return 1 - np.mean(result_comb['accuracy_valid'])

def to_sql_prediction(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame(Y_test_pred, index=data.test['group'].to_list(), columns=sql_result['y_type'])
    df = df.unstack().reset_index(drop=False)
    df.columns = ['y_type', 'group', 'pred']
    df['finish_timing'] = [sql_result['finish_timing']] * len(df)      # use finish time to distinguish dup pred
    return df

def to_list_importance(rf):
    ''' based on rf model -> records feature importance in DataFrame to be uploaded to DB '''

    df = pd.DataFrame()
    df['name'] = data.train.columns.to_list()[2:-len(sql_result['y_type'])]     # column names
    df['split'] = rf.feature_importances_
    df['split'] = df['split'].rank(ascending=False)
    return ','.join(df.sort_values(by=['split'], ascending=True)['name'].to_list())

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    hpot['all_results'] = []
    trials = Trials()

    if sql_result['objective'] in ['mae', 'mse', 'friedman_mse']:
        hpot['best_score'] = 10000  # record best training (min mae_valid) in each hyperopt
        best = fmin(fn=eval_regressor, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
            hpot['best_stock_df'].to_sql(global_vals.result_pred_table+"_rf_reg", **extra)
            pd.DataFrame(hpot['all_results']).to_sql(global_vals.result_score_table+"_rf_reg", **extra)
        global_vals.engine_ali.dispose()

    elif sql_result['objective'] in ['gini','entropy']:
        hpot['best_score'] = 0  # record best training (max accuracy_valid) in each hyperopt
        best = fmin(fn=eval_classifier, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
            hpot['best_stock_df'].to_sql(global_vals.result_pred_table+"_rf_class", **extra)
            pd.DataFrame(hpot['all_results']).to_sql(global_vals.result_score_table+"_rf_class", **extra)
        global_vals.engine_ali.dispose()

    print('===== best eval ===== ', best)

if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_type', default='rf')
    parser.add_argument('--objective', default='gini')
    parser.add_argument('--qcut_q', default=3, type=int)  # Default: Low, Mid, High
    args = parser.parse_args()

    # --------------------------------- Define Variables ------------------------------------------

    # create dict storing values/df used in training
    sql_result = vars(args)     # data write to DB TABLE lightgbm_results
    sql_result['name_sql'] = f'{dt.datetime.now()}_' + 'testing'
    hpot = {}                   # storing data for best trials in each Hyperopt

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
        sql_result['y_type'] = ['earnings_yield', 'market_cap_usd']
        print(sql_result['y_type'])
        for testing_period in reversed(testing_period_list):
            sql_result['testing_period'] = testing_period
            backtest = testing_period not in testing_period_list[0:4]
            load_data_params = {'qcut_q': args.qcut_q, 'y_type': sql_result['y_type']}

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

                    sql_result['train_len'] = len(sample_set['train_xx'])  # record length of training/validation sets
                    sql_result['valid_len'] = len(sample_set['valid_x'])

                    for k in ['valid_x','train_xx','test_x']:
                        sample_set[k] = np.nan_to_num(sample_set[k], nan=-99.9)

                    print(group_code, testing_period, len(sample_set['train_yy_final']))
                    HPOT(space, max_evals=10)  # start hyperopt
                    cv_number += 1
            except Exception as e:
                print(testing_period, e)
                continue
