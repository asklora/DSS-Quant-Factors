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
from random_forest import rf_HPOT
import global_vals

if __name__ == "__main__":
    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_type', default='extra')
    parser.add_argument('--objective', default='mse')
    parser.add_argument('--qcut_q', default=0, type=int)  # Default: Low, Mid, High
    parser.add_argument('--mode', default='v2_trim', type=str)
    args = parser.parse_args()
    sql_result = vars(args)  # data write to DB TABLE lightgbm_results

    # --------------------------------- Different Config ------------------------------------------

    sql_result['name_sql'] = 'pca_trimnew1'
    use_biweekly_stock = False
    stock_last_week_avg = True
    valid_method = 'chron'
    n_splits = 1
    defined_cut_bins = []
    group_code_list = ['USD']
    # group_code_list = ['HKD']
    use_pca = 0.6
    use_median = False

    # --------------------------------- Define Variables ------------------------------------------

    hpot = {}  # storing data for best trials in each Hyperopt

    # create date list of all testing period
    if use_biweekly_stock:
        last_test_date = dt.datetime(2021, 7, 11)
        backtest_period = 50
        testing_period_list = [last_test_date + relativedelta(days=1) - 2 * i * relativedelta(weeks=2)
                               - relativedelta(days=1) for i in range(0, backtest_period + 1)]
    else:
        last_test_date = dt.date.today() + MonthEnd(-2)  # Default last_test_date is month end of 2 month ago from today
        backtest_period = 46
        testing_period_list = [last_test_date + relativedelta(days=1) - i * relativedelta(months=1)
                               - relativedelta(days=1) for i in range(0, backtest_period + 1)]

    # --------------------------------- Model Training ------------------------------------------

    data = load_data(use_biweekly_stock=use_biweekly_stock, stock_last_week_avg=stock_last_week_avg,
                     mode=args.mode)  # load_data (class) STEP 1
    sql_result['y_type'] = y_type = data.factor_list[:10]  # random forest model predict all factor at the same time
    # sql_result['y_type'] = []       # random forest model predict all factor at the same time
    # other_y = [x for x in data.x_col_dict['factor'] if x not in sql_result['y_type']]
    # other_y = data.factor_list[:13]

    # sql_result['y_type'] = y_type = ['vol_0_30','book_to_price','earnings_yield','market_cap_usd']
    print(f"===== test on y_type", len(y_type), y_type, "=====")

    r_mean = 0

    i = 1
    # for y in other_y:
    #     sql_result['name_sql'] = f'pca_trylessx_{i}'
    #     sql_result['y_type'].append(y)
    # if i>1:
    #     sql_result['y_type'].remove(y)
    # print(sql_result['y_type'])
    # i += 1
    if 1 == 1:
        for tree_type in ['extra']:
            sql_result['tree_type'] = tree_type
            for group_code in group_code_list:
                sql_result['group_code'] = group_code
                data.split_group(group_code)  # load_data (class) STEP 2
                for testing_period in reversed(testing_period_list):
                    sql_result['testing_period'] = testing_period
                    load_data_params = {'qcut_q': args.qcut_q, 'y_type': sql_result['y_type'],
                                        'valid_method': valid_method, 'defined_cut_bins': defined_cut_bins,
                                        'use_median': use_median, 'use_pca': use_pca, 'n_splits': n_splits}
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

                            sql_result['train_len'] = len(
                                sample_set['train_xx'])  # record length of training/validation sets
                            sql_result['valid_len'] = len(sample_set['valid_x'])

                            for k in ['valid_x', 'train_xx', 'test_x']:
                                sample_set[k] = np.nan_to_num(sample_set[k], nan=0)

                            # sample_set['weight'] = np.array(range(len(sample_set['train_xx'])))/len(sample_set['train_xx'])
                            # sample_set['weight'] = np.tanh(sample_set['weight']-0.5)+0.5
                            sample_set['weight'] = np.ones((len(sample_set['train_xx']),))

                            # sql_result['weight'] = pd.cut(sql_result['weight'], bins=12, labels=False)
                            # print(sql_result['weight'])
                            print(data.x_col)
                            sql_result['neg_factor'] = ','.join(data.neg_factor)
                            print(group_code, testing_period, len(sample_set['train_yy_final']))
                            HPOT(space, max_evals=10)  # start hyperopt
                            cv_number += 1
                    except Exception as e:
                        print(testing_period, e)
                        continue
        # r = download_stock_pred(4, sql_result['name_sql'], False, False)
        # if r_mean < (r['max_ret'][0] - r['min_ret'][0]):
        #     r_mean = r['max_ret'][0] - r['min_ret'][0]
        # else:
        #     print(y, sql_result['y_type'])
        #     # if i > 1:
        #     # sql_result['y_type'].remove(y)
        #     sql_result['y_type'].append(y)
        #     print(sql_result['y_type'])
