import datetime as dt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
import numpy as np
import argparse
import pandas as pd
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
from sqlalchemy import create_engine, TIMESTAMP, TEXT, BIGINT, NUMERIC
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

from preprocess.load_data import load_data
from preprocess.ratios_calculations import calc_factor_variables
from preprocess.premium_calculation import calc_premium_all, calc_premium_all_v2
from random_forest import rf_HPOT, rf_space
from results_analysis.write_merged_pred import download_stock_pred
from results_analysis.score_backtest import score_history

from lasso import start_lasso
import itertools
import global_vals

if __name__ == "__main__":

    start_time = dt.datetime.now()

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    # parser.add_argument('--tree_type', default='rf', type=str)
    # parser.add_argument('--use_pca', default=0.6, type=float)
    # parser.add_argument('--group_code', default='USD', type=str)

    parser.add_argument('--objective', default='mse')
    parser.add_argument('--qcut_q', default=0, type=int)  # Default: Low, Mid, High
    parser.add_argument('--mode', default='v2', type=str)
    parser.add_argument('--backtest_period', default=46, type=int)
    parser.add_argument('--n_splits', default=3, type=int)
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--recalc_premium', action='store_true', help='Recalculate ratios & premiums = True')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # --------------------------------------- Schedule for Production --------------------------------
    if not args.debug:
        td = dt.datetime.today()
        if (td.day >= 7):
            print('Not start: Factor model only run on the next day after first Sunday every month! ')
            exit(0)

    # --------------------------------- Rerun Write Premium ------------------------------------------
    if args.recalc_premium:
        calc_factor_variables(price_sample='last_week_avg', fill_method='fill_all', sample_interval='monthly',
                              use_cached=True, save=True)
        if args.mode == 'default':
            calc_premium_all(stock_last_week_avg=True, use_biweekly_stock=False, update=False)
        elif args.mode == 'v2':
            calc_premium_all_v2(use_biweekly_stock=False, stock_last_week_avg=True, save_membership=True, trim_outlier_=False)
        elif args.mode == 'v2_trim':
            calc_premium_all_v2(use_biweekly_stock=False, stock_last_week_avg=True, save_membership=True, trim_outlier_=True)
        else:
            raise ValueError("Invalid mode. Expecting 'default', 'v2', or 'v2_trim' got ", args.mode)

    end_time = dt.datetime.now()
    print('Rerun Premium Time: ', start_time, end_time, end_time-start_time)

    # --------------------------------- Different Configs -----------------------------------------

    group_code_list = ['USD'] # ,
    # group_code_list = pd.read_sql('SELECT DISTINCT currency_code from universe WHERE currency_code IS NOT NULL', global_vals.engine.connect())['currency_code'].to_list()
    tree_type_list = ['rf']
    use_pca_list = [0.4, 0.6, 0.8]

    # create date list of all testing period
    last_test_date = dt.date.today().date() + MonthEnd(-2)  # Default last_test_date is month end of 2 month ago from today
    backtest_period = args.backtest_period
    testing_period_list = [last_test_date + relativedelta(days=1) - i * relativedelta(months=1)
                           - relativedelta(days=1) for i in range(0, backtest_period + 1)]

    # --------------------------------- Prepare Training Set -------------------------------------

    sql_result = vars(args).copy()  # data write to DB TABLE lightgbm_results
    sql_result['name_sql'] = f'{args.mode}_' + dt.datetime.strftime(dt.datetime.now(), '%Y%m%d')
    if args.debug:
        sql_result['name_sql'] += f'_debug'
    sql_result.pop('backtest_period')
    sql_result.pop('n_splits')
    sql_result.pop('recalc_premium')
    sql_result.pop('debug')
    # sql_result.pop('tree_type')
    # sql_result.pop('use_pca')


    data = load_data(use_biweekly_stock=False, stock_last_week_avg=True, mode=args.mode)  # load_data (class) STEP 1
    sql_result['y_type'] = y_type = data.factor_list  # random forest model predict all factor at the same time
    print(f"===== test on y_type", len(y_type), y_type, "=====")

    # --------------------------------- Run Lasso Benchmark -------------------------------------

    # start_lasso(data, testing_period_list, group_code_list, y_type)

    # --------------------------------- Model Training ------------------------------------------
    for i in range(1):
        for group_code, testing_period, tree_type, use_pca in itertools.product(group_code_list, testing_period_list, tree_type_list, use_pca_list):
            sql_result['tree_type'] = tree_type + str(i)
            sql_result['testing_period'] = testing_period
            sql_result['group_code'] = group_code
            sql_result['use_pca'] = use_pca

            data.split_group(group_code)
            # start_lasso(sql_result['testing_period'], sql_result['y_type'], sql_result['group_code'])

            load_data_params = {'qcut_q': args.qcut_q, 'y_type': sql_result['y_type'], 'valid_method': 'chron',
                                'use_median': False, 'use_pca': sql_result['use_pca'], 'n_splits': args.n_splits}
            sample_set, cv = data.split_all(testing_period, **load_data_params)  # load_data (class) STEP 3
            cv_number = 1  # represent which cross-validation sets

            for train_index, valid_index in cv:  # roll over different validation set
                sql_result['cv_number'] = cv_number

                sample_set['valid_x'] = sample_set['train_x'][valid_index]
                sample_set['train_xx'] = sample_set['train_x'][train_index]
                sample_set['valid_y'] = sample_set['train_y'][valid_index]
                sample_set['train_yy'] = sample_set['train_y'][train_index]
                sample_set['valid_y_final'] = sample_set['train_y_final'][valid_index]
                sample_set['train_yy_final'] = sample_set['train_y_final'][train_index]

                sql_result['train_len'] = len(sample_set['train_xx'])  # record length of training/validation sets
                sql_result['valid_len'] = len(sample_set['valid_x'])

                for k in ['valid_x', 'train_xx', 'test_x', 'train_x']:
                    sample_set[k] = np.nan_to_num(sample_set[k], nan=0)

                sql_result['neg_factor'] = ','.join(data.neg_factor)
                rf_HPOT(rf_space, max_evals=10)  # start hyperopt
                cv_number += 1

    # --------------------------------- Results Analysis ------------------------------------------
    download_stock_pred(
            q=1/3,
            model='rf_reg',
            name_sql=sql_result['name_sql'],
            keep_all_history=True,
            save_plot=True,
            save_xls=True,
        )
    score_history()     # calculate score with DROID v2 method & evaluate

    end_time = dt.datetime.now()
    print(start_time, end_time, end_time-start_time)


