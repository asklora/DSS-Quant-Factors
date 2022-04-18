import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta
from itertools import product
import multiprocessing as mp

from global_vars import (
    logging,
    factor_formula_config_train_prod,
    factor_formula_config_eval_prod,
    factors_pillar_defined_table,
    factors_pillar_cluster_table,
    factor_premium_table,
    result_score_table,
    result_pred_table,
    feature_importance_table,
    db_url_alibaba_prod
)
from general.send_slack import to_slack
from preprocess.load_data import load_data
from preprocess.calculation_ratio import calc_factor_variables_multi
from preprocess.calculation_premium import calc_premium_all
from preprocess.calculation_pillar_cluster import calc_pillar_cluster
from random_forest import rf_HPOT
from general.sql_process import (
    read_query,
    read_table,
    upsert_data_to_database,
    migrate_local_save_to_prod
)
from results_analysis.calculation_rank import rank_pred
from results_analysis.calculation_backtest import backtest_score_history
from results_analysis.analysis_score_backtest_eval2 import top2_table_tickers_return


def mp_rf(*args):
    """ run random forest on multi-processor """

    data, sql_result, kwargs = args
    sql_result.update(kwargs)

    logging.debug(f"===== test on pillar: [{sql_result['pillar']}] =====")
    data.split_group(sql_result["train_currency"], sql_result["pred_currency"])
    # start_lasso(sql_result['testing_period'], sql_result['pillar'], sql_result['group_code'])

    stock_df_list = []
    score_df_list = []
    feature_df_list = []
    try:
        load_data_params = {'valid_method': sql_result['_valid_method'], 'n_splits': sql_result['_n_splits'],
                            "output_options": {"pillar": sql_result["factor_list"], "qcut_q": sql_result['_qcut_q'],
                                               "use_median": sql_result['_qcut_q'] > 0, "defined_cut_bins": [],
                                               "use_average": sql_result['_use_average']},
                            "input_options": {"ar_period": [], "ma3_period": [], "ma12_period": [],
                                              "factor_pca": sql_result["_use_pca"], "mi_pca": 0.6}}
        testing_period = dt.datetime.combine(sql_result['testing_period'], dt.datetime.min.time())
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
            sample_set['train_yy_weight'] = np.where(sample_set['train_yy'][:, 0] < 0,
                                                     sql_result["_down_mkt_pct"], 1 - sql_result["_down_mkt_pct"])

        sql_result['neg_factor'] = data.neg_factor
        stock_df, score_df, feature_df = rf_HPOT(max_evals=10, sql_result=sql_result,
                                                 sample_set=sample_set, x_col=data.x_col, y_col=data.y_col,
                                                 group_index=data.test['group'].to_list()).hpot_dfs
        stock_df_list.append(stock_df)
        score_df_list.append(score_df)
        feature_df_list.append(feature_df)
        cv_number += 1

    except Exception as e:
        to_slack("clair").message_to_slack(f"*[factor train] ERROR* on config {sql_result}: {e.args}")
    return stock_df_list, score_df_list, feature_df_list


def write_db(stock_df_all, score_df_all, feature_df_all):
    """ write score/prediction/feature to DB """

    # update results
    try:
        upsert_data_to_database(stock_df_all, result_pred_table, how="append", verbose=-1)
        upsert_data_to_database(score_df_all, result_score_table, how="update", verbose=-1)
        upsert_data_to_database(feature_df_all, feature_importance_table, how="append", verbose=-1)
        return True
    except Exception as e:
        # save to pickle file in local for recovery
        # stock_df_all.to_pickle('cache_stock_df_all.pkl')
        # score_df_all.to_pickle('cache_score_df_all.pkl')
        # feature_df_all.to_pickle('cache_feature_df_all.pkl')

        to_slack("clair").message_to_slack(f"*[Factor] ERROR [FINAL] write to DB*: {e.args}")
        return False


def mp_eval(*args, pred_start_testing_period='2015-09-01', eval_current=False, xlsx_name="ai_score"):
    """ evaluate test results based on name_sql / eval args """

    sql_result, eval_metric, eval_n_configs, eval_backtest_period, eval_removed_subpillar, q = args

    # Step 1: pred -> ranking
    try:
        factor_rank = pd.read_csv(f'fac1tor_rank_{eval_metric}_{eval_n_configs}_{eval_backtest_period}.csv')
    except Exception as exp:
        from time import time
        start = time()
        factor_rank = rank_pred(q, name_sql=sql_result['name_sql'],
                                pred_start_testing_period=pred_start_testing_period,
                                # this period is before (+ weeks_to_expire)
                                eval_current=eval_current,
                                eval_metric=eval_metric,
                                eval_top_config=eval_n_configs,
                                eval_config_select_period=eval_backtest_period,
                                eval_removed_subpillar=eval_removed_subpillar,
                                if_eval_top=sql_result['restart_eval_top'],
                                )
        end = time()
        to_slack("clair").message_to_slack(f"[rank_pred] time: {end - start}")

        # factor_rank = factor_rank.write_to_db()
        factor_rank.to_csv(f'factor_rank_{eval_metric}_{eval_n_configs}_{eval_backtest_period}.csv', index=False)

    # ----------------------- modified factor_rank for testing ----------------------------
    # factor_rank = factor_rank.loc[factor_rank['group'] == 'CNY']
    # -------------------------------------------------------------------------------------

    # Step 2: ranking -> backtest score
    # set name_sql=None i.e. using current backtest table writen by rank_pred
    if sql_result['restart_eval_top']:
        backtest_df = backtest_score_history(factor_rank, sql_result['name_sql'], eval_metric=eval_metric, eval_q=q,
                                             n_config=eval_n_configs, n_backtest_period=eval_backtest_period,
                                             xlsx_name=xlsx_name).return_df


# TODO: change all report to "clair" -> report to factor slack channel
def start_on_update(check_interval=60, table_names=None, report_only=True):
    """ check if data tables finished ingestion -> then start

    Parameters
    ----------
    check_interval (Int): wait x seconds to check again
    table_names (List[Str]): List of table names to check whether update finished within past 3 days)
    report_only (Bool): if True, will report last update date only (i.e. will start training even no update done within 3 days)
    """
    waiting = True
    while waiting:
        update_time = read_table("ingestion_update_time", db_url_alibaba_prod)
        update_time = update_time.loc[update_time['tbl_name'].isin(table_names)]
        if report_only:
            to_slack("clair").df_to_slack("=== Ingestion Table last update === ", update_time.set_index("tbl_name")[['finish', 'last_update']])
            break

        if all(update_time['finish'] == True) & all(
                update_time['last_update'] > (dt.datetime.today() - relativedelta(days=3))):
            waiting = False
        else:
            logging.debug(f'Keep waiting...Check again in {check_interval}s ({dt.datetime.now()})')
            time.sleep(check_interval)

    to_slack("clair").message_to_slack(f" === Start Factor Model for weeks_to_expire=[{args.weeks_to_expire}] === ")
    return True


if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--weeks_to_expire', default=4, type=int)
    parser.add_argument('--backtest_period', default=14, type=int)
    parser.add_argument('--sample_interval', default=4, type=int)
    parser.add_argument('--average_days', default=-7, type=int)
    parser.add_argument('--train_currency', default='USD', type=str)    # TODO: rename group code
    parser.add_argument('--pred_currency', default='USD,EUR', type=str)    # TODO: rename group code
    parser.add_argument('--pillar', default='momentum', type=str)
    parser.add_argument('--recalc_ratio', action='store_true', help='Recalculate ratios = True')
    parser.add_argument('--recalc_premium', action='store_true', help='Recalculate premiums = True')
    parser.add_argument('--recalc_subpillar', action='store_true', help='Recalculate cluster pillar / subpillar = True')
    parser.add_argument('--trim', action='store_true', help='Trim Outlier = True')
    parser.add_argument('--objective', default='squared_error')
    parser.add_argument('--hpot_eval_metric', default='adj_mse_valid')
    parser.add_argument('--processes', default=mp.cpu_count(), type=int)
    parser.add_argument('--eval_q', default='0.33', type=str)
    parser.add_argument('--eval_removed_subpillar', action='store_true', help='if removed subpillar in evaluation')
    parser.add_argument('--eval_top_metric', default='max_ret,net_ret', type=str)
    parser.add_argument('--eval_top_n_configs', default='10,20', type=str)      # TODO: update to ratio
    parser.add_argument('--eval_top_backtest_period', default='12,36', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--restart', default=None, type=str)
    parser.add_argument('--restart_eval', action='store_true', help='restart evaluation only')      # TODO: pass training if True
    parser.add_argument('--restart_eval_top', action='store_true', help='restart evaluation top')   # TODO: pass training if True
    args = parser.parse_args()

    # --------------------------------------- Production / Development --------------------------------------------

    if args.debug:
        train_currency_list = args.train_currency.split(',')
        premium_currency_list = pred_currency_list = args.pred_currency.split(',')
        pillar_list = args.pillar.split(',')

        data_options = {
            "train_currency": args.train_currency.split(','),
            "pred_currency": args.pred_currency.split(','),
            "pillar": args.pillar.split(','),
            "hpot_eval_metric": [args.hpot_eval_metric],
        }
        data_configs = [dict(zip(data_options.keys(), e)) for e in product(*data_options.values())]
    else:
        # Check 1: if monthly -> only first Sunday every month      # TODO: start for production
        # if dt.datetime.today().day > 7:
        #     raise Exception('Not start: Factor model only run on the next day after first Sunday every month! ')

        # Check 2(b): monthly update after weekly update
        start_on_update(table_names=['data_ibes', 'data_macro', 'data_worldscope'], report_only=True)

        data_configs = read_query(f"SELECT train_currency, pred_currency, pillar, hpot_eval_metric "
                                  f"FROM {factor_formula_config_train_prod} "
                                  f"WHERE weeks_to_expire = {args.weeks_to_expire}").to_dict("records")
        premium_currency_list = ["HKD", "CNY", "USD", "EUR"]

        logging.warning(f'Production will ignore args (train_currency, pred_currency, pillar, hpot_eval_metric) '
                        f'and use combination in [{factor_formula_config_train_prod}')

    # ---------------------------------------- Rerun Write Premium -----------------------------------------------

    # TODO: report to production
    # if args.recalc_ratio:
    #     calc_factor_variables_multi(ticker=None,
    #                                 restart=False,
    #                                 tri_return_only=False,
    #                                 processes=args.processes)
    # if args.recalc_premium:
    #     calc_premium_all(weeks_to_expire=args.weeks_to_expire,
    #                      average_days=args.average_days,
    #                      weeks_to_offset=min(4, args.weeks_to_expire),
    #                      trim_outlier_=args.trim,
    #                      all_groups=premium_currency_list,
    #                      processes=args.processes)

    # ---------------------------------------- Different Configs ----------------------------------------------

    load_options = {
        "_tree_type": ['rf'],
        "_use_pca": [0.4, None],
        "_n_splits": [.2],
        "_valid_method": [2010, 2012, 2014],
        "_qcut_q": [0, 10],
        "_use_average": [None],                 # True, False
        "_down_mkt_pct": [0.5, 0.7]
    }
    load_configs = [dict(zip(load_options.keys(), e)) for e in product(*load_options.values())]

    # create date list of all testing period
    query = f"SELECT DISTINCT trading_day FROM {factor_premium_table} WHERE weeks_to_expire={args.weeks_to_expire} " \
            f"AND average_days={args.average_days} ORDER BY trading_day DESC"
    period_list_all = read_query(query)['trading_day'].to_list()
    period_list = period_list_all[:args.sample_interval * args.backtest_period + 1:args.sample_interval]
    logging.info(f"Testing period: [{period_list[0]}] --> [{period_list[-1]}] (n=[{len(period_list)}])")

    # update cluster separation table for any currency with 'cluster' pillar
    cluster_configs = {"_subpillar_trh": [5], "_pillar_trh": [2]}
    if args.recalc_subpillar:
        for c in data_configs:
            if c["pillar"] == "cluster":
                for period in period_list:
                    for subpillar_trh in cluster_configs["_subpillar_trh"]:
                        for pillar_trh in cluster_configs["_pillar_trh"]:
                            calc_pillar_cluster(period, args.weeks_to_expire, c['train_currency'], subpillar_trh, pillar_trh)
        logging.info(f"=== Update Cluster Partition for {cluster_configs} ===")

    # --------------------------------- Model Training ------------------------------------------

    # sql_result = vars(args).copy()  # data write to DB TABLE lightgbm_results
    datetimeNow = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')
    sql_result = {"name_sql": f"w{args.weeks_to_expire}_d{args.average_days}_{datetimeNow}",
                  "objective": args.objective}
    if args.debug:
        sql_result['name_sql'] += f'_debug'

    if not args.restart_eval:
        data = load_data(args.weeks_to_expire, args.average_days, trim=args.trim)  # load_data (class) STEP 1

        # all_groups ignore [cluster_configs] -> fix subpillar for now (change if needed)
        all_groups = product([{**a, **l, **{"testing_period": p}}
                              for a in data_configs for l in load_configs for p in period_list])
        all_groups_df = pd.DataFrame([tuple(e)[0] for e in all_groups])

        # Get factor list by merging pillar tables & configs
        cluster_pillar = read_query(f"SELECT \"group\" as train_currency, testing_period, pillar, factor_list "
                                    f"FROM {factors_pillar_cluster_table}")
        defined_pillar = read_query(f"SELECT pillar, factor_list FROM {factors_pillar_defined_table}")

        all_groups_defined_df = all_groups_df.loc[all_groups_df["pillar"] != "cluster"]
        all_groups_cluster_df = all_groups_df.loc[all_groups_df["pillar"] == "cluster"].drop(columns=["pillar"])

        all_groups_defined_df = all_groups_defined_df.merge(defined_pillar, on=["pillar"], how="left")
        cluster_pillar_pillar = cluster_pillar.loc[cluster_pillar["pillar"].str.startswith("pillar")]
        all_groups_cluster_df = all_groups_cluster_df.merge(cluster_pillar_pillar,
                                                            on=["train_currency", "testing_period"], how="left")
        all_groups_df = all_groups_defined_df.append(all_groups_cluster_df)

        # (restart) filter for failed iteration
        if args.restart:
            local_migrate_status = migrate_local_save_to_prod()  # save local db to cloud
            sql_result["name_sql"] = args.restart

            diff_config_col = [x for x in all_groups_df if x != "factor_list"]
            fin_df = read_query(f"SELECT {', '.join(diff_config_col)}, count(uid) as uid "
                                f"FROM {result_score_table} WHERE name_sql='{args.restart}' "
                                f"GROUP BY {', '.join(diff_config_col)}")
            all_groups_df = all_groups_df.merge(fin_df, how='left', on=diff_config_col).sort_values(by=diff_config_col)
            all_groups_df = all_groups_df.loc[all_groups_df['uid'].isnull(), diff_config_col]
            to_slack("clair").message_to_slack(
                f"=== Restart [{args.restart}]: rest iterations (n={len(all_groups_df)}) ===")

        all_groups = all_groups_df.to_dict("records")
        all_groups = [tuple([data, sql_result, e]) for e in all_groups]

        # multiprocess return result dfs = (stock_df_all, score_df_all, feature_df_all)
        with mp.Pool(processes=args.processes) as pool:
            result_dfs = pool.starmap(mp_rf, all_groups)

        stock_df_all = [e for x in result_dfs for e in x[0]]
        stock_df_all_df = pd.concat(stock_df_all, axis=0)
        score_df_all = [e for x in result_dfs for e in x[1]]
        score_df_all_df = pd.concat(score_df_all, axis=0)
        feature_df_all = [e for x in result_dfs for e in x[2]]
        feature_df_all_df = pd.concat(feature_df_all, axis=0)

        if not args.debug:
            write_db_status = write_db(stock_df_all_df, score_df_all_df, feature_df_all_df)

    # --------------------------------- Results Analysis ------------------------------------------

    subpillar_dict, pillar_dict = False  # TODO: when training retrieve from DB

    sql_result["name_sql"] = args.restart
    try:
        all_eval_groups = product([sql_result],
                                  args.eval_top_metric.split(','),
                                  [int(e) for e in args.eval_top_n_configs.split(',')],
                                  [int(e) for e in args.eval_top_backtest_period.split(',')],
                                  [args.eval_removed_subpillar],
                                  [float(e) for e in args.eval_q.split(',')],
                                  )
        all_eval_groups = [tuple(e) for e in all_eval_groups]
        logging.info(f"=== evaluation iteration: n={len(all_eval_groups)} ===")

        with mp.Pool(processes=args.processes) as pool:
            pool.starmap(mp_eval, all_eval_groups)
    except Exception as e:
        to_slack("clair").message_to_slack(f"ERROR in func mp_eval(): {e}")
