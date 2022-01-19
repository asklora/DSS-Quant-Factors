import datetime as dt
import logging

import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta

import global_vars
from preprocess.load_data import load_data
from preprocess.calculation_ratio import calc_factor_variables_multi
from preprocess.calculation_premium import calc_premium_all
from random_forest import rf_HPOT
from results_analysis.write_merged_pred import download_stock_pred
from general.report_to_slack import to_slack
from general.sql_process import read_query, read_table, trucncate_table_in_database

from itertools import product
import multiprocessing as mp
from tqdm import tqdm

def mp_rf(*mp_args):
    ''' run random forest on multi-processor '''

    try:
        data, sql_result, i, group_code, testing_period, tree_type, use_pca, y_type = mp_args

        logging.debug(f"===== test on y_type", len(y_type), y_type, "=====")
        sql_result['y_type'] = y_type   # random forest model predict all factor at the same time
        sql_result['tree_type'] = tree_type + str(i)
        sql_result['testing_period'] = testing_period
        sql_result['group_code'] = group_code
        sql_result['use_pca'] = use_pca

        data.split_group(group_code)
        # start_lasso(sql_result['testing_period'], sql_result['y_type'], sql_result['group_code'])

        load_data_params = {'qcut_q': args.qcut_q, 'y_type': sql_result['y_type'], 'valid_method': 'chron',
                            'use_median': False, 'use_pca': sql_result['use_pca'], 'n_splits': args.n_splits}
        testing_period = dt.datetime.combine(testing_period, dt.datetime.min.time())
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
            rf_HPOT(max_evals=(2 if global_vars.debug else 10), sql_result=sql_result, sample_set=sample_set,
                    x_col=data.x_col, y_col=data.y_col, group_index=data.test['group'].to_list()).write_db() # start hyperopt
            cv_number += 1
    except Exception as e:
        to_slack("clair").message_to_slack(f'*** Exception: {testing_period},{use_pca},{y_type}: {e}')

if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    # parser.add_argument('--tree_type', default='rf', type=str)
    # parser.add_argument('--use_pca', default=0.6, type=float)
    # parser.add_argument('--group_code', default='USD', type=str)

    parser.add_argument('--objective', default='mse')
    parser.add_argument('--qcut_q', default=0, type=int)  # Default: Low, Mid, High
    parser.add_argument('--weeks_to_expire', default=1, type=int)
    parser.add_argument('--processes', default=8, type=int)
    parser.add_argument('--backtest_period', default=210, type=int)
    parser.add_argument('--n_splits', default=3, type=int)
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--recalc_ratio', action='store_true', help='Recalculate ratios = True')
    parser.add_argument('--recalc_premium', action='store_true', help='Recalculate premiums = True')
    parser.add_argument('--trim', action='store_true', help='Trim Outlier = True')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    group_code_list = ['USD', 'EUR'] # ,

    # --------------------------------------- Schedule for Production --------------------------------
    def start_on_update(check_interval=60):
        ''' check if data tables finished ingestion -> then start '''
        table_names = ['data_ibes', 'data_macro', 'data_worldscope']
        waiting = True
        while waiting:
            update_time = read_table("ingestion_update_time", global_vars.db_url_alibaba_prod)
            update_time = update_time.loc[update_time['tbl_name'].isin(table_names)]
            if all(update_time['finish']==True) & all(update_time['last_update']>(dt.datetime.today()-relativedelta(days=1))):
                waiting = False
            else:
                logging.info(f'Keep waiting...Check again in {check_interval}s ({dt.datetime.now()})')
                time.sleep(check_interval)
        return True

    if not args.debug:
        # Check 1. if monthly -> only first Sunday every month
        if args.weeks_to_expire >= 4:
            if dt.datetime.today().day>7:
                raise Exception('Not start: Factor model only run on the next day after first Sunday every month! ')
        # Check 2. if all input data df finished update
        # start_on_update()

    # --------------------------------- Rerun Write Premium ------------------------------------------
    weeks_to_expire = args.weeks_to_expire
    if args.recalc_ratio:
        calc_factor_variables_multi(ticker=None, restart=False, processes=args.processes)
    if args.recalc_premium:
        calc_premium_all(weeks_to_expire, processes=args.processes, trim_outlier_=args.trim, all_groups=group_code_list)

    # --------------------------------- Different Configs -----------------------------------------

    # group_code_list = pd.read_sql('SELECT DISTINCT currency_code from universe WHERE currency_code IS NOT NULL', global_vars.engine.connect())['currency_code'].to_list()
    tree_type_list = ['rf']
    use_pca_list = [0.6]
    # use_pca_list = [0.4]

    # create date list of all testing period
    query = f"SELECT DISTINCT trading_day FROM {global_vars.factor_premium_table}"
    last_test_date = read_query(query, db_url=global_vars.db_url_write)
    testing_period_list = sorted(last_test_date['trading_day'])[-args.backtest_period:]
    logging.info(f'Testing period: [{testing_period_list[0]}] --> [{testing_period_list[-1]}]')
    # testing_period_list = [dt.date(2021,4,30)]

    # --------------------------------- Prepare Training Set -------------------------------------

    sql_result = vars(args).copy()  # data write to DB TABLE lightgbm_results
    sql_result['name_sql'] = f'week{weeks_to_expire}_' + dt.datetime.strftime(dt.datetime.now(), '%Y%m%d')
    if args.debug:
        sql_result['name_sql'] += f'_debug'
    sql_result.pop('backtest_period')
    sql_result.pop('n_splits')
    sql_result.pop('recalc_premium')
    sql_result.pop('recalc_ratio')
    sql_result.pop('debug')
    sql_result.pop('processes')
    sql_result.pop('weeks_to_expire')
    # sql_result.pop('use_pca')

    # --------------------------------- Model Training ------------------------------------------

    mode = 'trim' if args.trim else ''
    data = load_data(weeks_to_expire, mode=mode)  # load_data (class) STEP 1

    # y_type_list = [data.x_col_dict['y_col']]
    # y_type_list = list(combinations(set(data.x_col_dict['momentum'])&set(data.x_col_dict['y_col']), 5))
    # y_type_list += list(combinations(set(data.x_col_dict['quality'])&set(data.x_col_dict['y_col']), 5))
    # y_type_list += list(combinations(set(data.x_col_dict['value'])&set(data.x_col_dict['y_col']), 5))

    y_type_list = []
    y_type_list.append(list(set(data.x_col_dict['momentum'])&set(data.factor_list)))
    y_type_list.append(list(set(data.x_col_dict['value'])&set(data.factor_list)))
    y_type_list.append(list(set(data.x_col_dict['quality'])&set(data.factor_list)))

    all_groups = product([data], [sql_result], [1], group_code_list, testing_period_list,
                         tree_type_list, use_pca_list, y_type_list)
    all_groups = [tuple(e) for e in all_groups]

    # Reset results table everytimes
    # trucncate_table_in_database(f"{global_vars.result_pred_table}", global_vars.db_url_write)
    # trucncate_table_in_database( f"{global_vars.feature_importance_table}", global_vars.db_url_write)
    with mp.Pool(processes=args.processes) as pool:
        tqdm(pool.starmap(mp_rf, all_groups), total=len(all_groups))

    # --------------------------------- Results Analysis ------------------------------------------
    download_stock_pred(
            q=1/3,
            model='',
            name_sql=sql_result['name_sql'],
            save_plot=False,
            save_xls=False,
            suffix=weeks_to_expire,
        )

    # score_history(weeks_to_expire)     # calculate score with DROID v2 method & evaluate


