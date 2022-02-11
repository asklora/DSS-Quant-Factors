import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta

from global_vars import *
from preprocess.load_data import load_data
from preprocess.calculation_ratio import calc_factor_variables_multi
from preprocess.calculation_premium import calc_premium_all
from random_forest import rf_HPOT
from results_analysis.calculation_rank import rank_pred
from general.send_slack import to_slack
from general.sql_process import read_query, read_table, trucncate_table_in_database

from itertools import product
import multiprocessing as mp

def mp_rf(*mp_args):
    ''' run random forest on multi-processor '''

    # try:
    if True:
        data, sql_result, i, group_code, testing_period, y_type, tree_type, use_pca, n_splits, valid_method, qcut_q, \
            use_average, down_mkt_pct = mp_args

        logging.debug(f"===== test on y_type [{y_type}] =====")
        sql_result['y_type'] = y_type   # random forest model predict all factor at the same time
        sql_result['tree_type'] = tree_type + str(i)
        sql_result['testing_period'] = testing_period
        sql_result['group_code'] = group_code
        sql_result['use_pca'] = use_pca
        sql_result['n_splits'] = n_splits
        sql_result['valid_method'] = valid_method
        sql_result['qcut_q'] = qcut_q
        sql_result['use_average'] = use_average     # neg_factor use average
        sql_result['down_mkt_pct'] = down_mkt_pct     # neg_factor use average

        data.split_group(group_code)
        # start_lasso(sql_result['testing_period'], sql_result['y_type'], sql_result['group_code'])

        # map y_type name to list of factors
        y_type_query = f"SELECT * FROM {factors_y_type_table}"
        y_type_map = read_query(y_type_query, db_url_read).set_index(["y_type"])["factor_list"].to_dict()
        load_data_params = {'valid_method': sql_result['valid_method'], 'n_splits': sql_result['n_splits'],
                            "output_options": {"y_type": y_type_map[y_type], "qcut_q": sql_result['qcut_q'],
                                               "use_median": sql_result['qcut_q']>0, "defined_cut_bins": [],
                                               "use_average":  sql_result['use_average']},
                            "input_options": {"ar_period": [], "ma3_period": [], "ma12_period": [],
                                              "factor_pca": use_pca, "mi_pca": 0.6}}
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

            # calculate weight for negative / positive index
            sample_set['train_yy_weight'] = np.where(sample_set['train_yy'][:, 0]<0, down_mkt_pct, 1-down_mkt_pct)

            sql_result['neg_factor'] = data.neg_factor
            rf_HPOT(max_evals=(2 if DEBUG else 10), sql_result=sql_result, sample_set=sample_set,
                    x_col=data.x_col, y_col=data.y_col, group_index=data.test['group'].to_list()).write_db() # start hyperopt
            cv_number += 1
    # except Exception as e:
    #     to_slack("clair").message_to_slack(f'*** Exception: {testing_period},{use_pca},{y_type}: {e}')

if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='squared_error')
    parser.add_argument('--weeks_to_expire', default=26, type=int)
    parser.add_argument('--average_days', default=28, type=int)
    parser.add_argument('--processes', default=1, type=int)
    parser.add_argument('--backtest_period', default=210, type=int)
    # parser.add_argument('--n_splits', default=3, type=int)      # validation set partition
    parser.add_argument('--recalc_ratio', action='store_true', help='Recalculate ratios = True')
    parser.add_argument('--recalc_premium', action='store_true', help='Recalculate premiums = True')
    parser.add_argument('--trim', action='store_true', help='Trim Outlier = True')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    group_code_list = ['USD']   # TODO: add EUR

    # --------------------------------------- Schedule for Production --------------------------------

    def start_on_update(check_interval=60, table_names=None):
        ''' check if data tables finished ingestion -> then start '''
        waiting = True
        while waiting:
            update_time = read_table("ingestion_update_time", db_url_alibaba_prod)
            update_time = update_time.loc[update_time['tbl_name'].isin(table_names)]
            if all(update_time['finish']==True) & all(update_time['last_update']>(dt.datetime.today()-relativedelta(days=3))):
                waiting = False
            else:
                logging.debug(f'Keep waiting...Check again in {check_interval}s ({dt.datetime.now()})')
                time.sleep(check_interval)
        to_slack("clair").message_to_slack(f"*[Start Factor]*: week_to_expire=[{args.weeks_to_expire}]\n-> updated {table_names}")
        return True

    if not args.debug:
        # Check 1: if monthly -> only first Sunday every month
        if args.weeks_to_expire >= 4:
            if dt.datetime.today().day>7:
                raise Exception('Not start: Factor model only run on the next day after first Sunday every month! ')
            # Check 2(b): monthly update after weekly update
            start_on_update(table_names=['factor_result_rank'])
        else:
            # Check 2(a): weekly update after input data update
            start_on_update(table_names=['data_ibes', 'data_macro', 'data_worldscope'])

    # --------------------------------- Rerun Write Premium ------------------------------------------
    if args.recalc_ratio:
        calc_factor_variables_multi(ticker=None, restart=False, tri_return_only=False)
    if args.recalc_premium:
        calc_premium_all(weeks_to_expire=args.weeks_to_expire,
                         average_days=args.average_days,
                         weeks_to_offset=min(4, args.weeks_to_expire),
                         trim_outlier_=args.trim,
                         all_groups=group_code_list,
                         processes=args.processes)

    # --------------------------------- Different Configs -----------------------------------------
    # tree_type_list = ['rf', 'extra', 'rf', 'extra', 'rf', 'extra']
    tree_type_list = ['rf']
    use_pca_list = [0.6, 0.4, None]
    n_splits_list = [.1]
    valid_method_list = [2010, 2012, 2014]  # 'chron'
    qcut_q_list = [10]
    use_average_list = [False]
    down_mkt_pct_list = [0.5, 0.6, 0.7]

    # y_type_list = ["all"]
    # y_type_list = ["test"]
    y_type_list = ["value", "quality", "momentum"]
    # y_type_list = ["momentum_top4", "quality_top4", "value_top4"]

    # create date list of all testing period
    query = f"SELECT DISTINCT trading_day FROM {factor_premium_table} " \
            f"WHERE weeks_to_expire={args.weeks_to_expire} AND average_days={args.average_days}"
    testing_period_list_all = read_query(query, db_url=db_url_read)
    testing_period_list = sorted(testing_period_list_all['trading_day'])[-args.backtest_period*args.weeks_to_expire-1::4]
    logging.info(f'Testing period: [{testing_period_list[0]}] --> [{testing_period_list[-1]}] (n=[{len(testing_period_list)}])')

    # --------------------------------- Model Training ------------------------------------------

    sql_result = vars(args).copy()  # data write to DB TABLE lightgbm_results
    sql_result['name_sql'] = f'w{args.weeks_to_expire}_d{args.average_days}_' + dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')
    if args.debug:
        sql_result['name_sql'] += f'_debug'

    mode = 'trim' if args.trim else ''
    data = load_data(args.weeks_to_expire, args.average_days, mode=mode)  # load_data (class) STEP 1

    all_groups = product([data], [sql_result], [1], group_code_list, testing_period_list, y_type_list,
                         tree_type_list, use_pca_list, n_splits_list, valid_method_list, qcut_q_list, use_average_list,
                         down_mkt_pct_list)
    all_groups = [tuple(e) for e in all_groups]

    # Reset results table everytimes
    with mp.Pool(processes=args.processes) as pool:
        pool.starmap(mp_rf, all_groups)

    # --------------------------------- Results Analysis ------------------------------------------

    from results_analysis.analysis_score_backtest import test_score_history
    test_score_history(name_sql=sql_result['name_sql'])

    # rank_pred(args.q, args.weeks_to_expire, args.average_days, name_sql=sql_result['name_sql']).write_to_db()


