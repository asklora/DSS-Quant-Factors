import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta
from itertools import product
import multiprocessing as mp
import gc

from global_vars import (
    logger,
    LOGGER_LEVEL,
    config_train_table,
    config_eval_table,
    pillar_defined_table,
    pillar_cluster_table,
    factor_premium_table,
    result_score_table,
    result_pred_table,
    feature_importance_table,
    backtest_eval_table,
    backtest_top_table,
    production_rank_table,
    production_rank_history_table,
    db_url_alibaba_prod,
    backtest_eval_dtypes,
    backtest_top_dtypes,
    rank_dtypes,
)
from general.report.send_slack import to_slack
from preprocess.load_data import load_data
from preprocess.calculation_ratio import calc_factor_variables_multi
from preprocess.calculation_premium import calcPremium
from preprocess.calculation_pillar_cluster import calcPillarCluster
from random_forest import rf_HPOT
from general.sql.sql_process import (
    read_query,
    read_table,
    upsert_data_to_database,
    migrate_local_save_to_prod
)
from results_analysis.calculation_rank import calculate_rank_pred

logger = logger(__name__, LOGGER_LEVEL)

logger.info(f" ---> result_score_table: [{result_score_table}]")
logger.info(f" ---> result_pred_table: [{result_pred_table}]")
logger.info(f" ---> production_rank_table: [{production_rank_table}]")
logger.info(f" ---> backtest_eval_table: [{backtest_eval_table}]")
logger.info(f" ---> backtest_top_table: [{backtest_top_table}]")


def mp_rf(*args):
    """ run random forest on multi-processor """

    data, sql_result, kwargs = args
    sql_result.update(kwargs)

    logger.debug(f"===== test on pillar: [{sql_result['pillar']}] =====")
    data.split_train_currency(**sql_result)
    # start_lasso(sql_result['testing_period'], sql_result['pillar'], sql_result['group_code'])

    # stock_df_list = []
    # score_df_list = []
    # feature_df_list = []
    try:
        sample_set, cv = data.split_all(**sql_result)  # load_data (class) STEP 3

        cv_number = 1  # represent which cross-validation sets
        for train_index, valid_index in cv:  # roll over different validation set
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
        # stock_df_list.append(stock_df)
        # score_df_list.append(score_df)
        # feature_df_list.append(feature_df)
        cv_number += 1
    except Exception as e:
        to_slack("clair").message_to_slack(f"*[factor train] ERROR* on config {sql_result}: {e.args}")
    # return stock_df_list, score_df_list, feature_df_list


def write_db(stock_df_all, score_df_all, feature_df_all):
    """ write score/prediction/feature to DB """

    # update results
    try:
        upsert_data_to_database(stock_df_all, result_pred_table, how="append", verbose=-1, dtype=pred_dtypes)
        upsert_data_to_database(score_df_all, result_score_table, how="update", verbose=-1, dtype=score_dtypes)
        upsert_data_to_database(feature_df_all, feature_importance_table, how="append", verbose=-1, dtype=feature_dtypes)
        return True
    except Exception as e:
        # save to pickle file in local for recovery
        stock_df_all.to_pickle('cache_stock_df_all.pkl')
        score_df_all.to_pickle('cache_score_df_all.pkl')
        feature_df_all.to_pickle('cache_feature_df_all.pkl')

        to_slack("clair").message_to_slack(f"*[Factor] ERROR [FINAL] write to DB*: {e.args}")
        return False


def load_train_configs(data_configs, load_configs, period_list, restart=False):
    """ based on data_configs (i.e. train_config Table) & load_configs (i.e. config to select) & period list """

    # all_train_currencys ignore [cluster_configs] -> fix subpillar for now (change if needed)
    all_train_currencys = product([{**a, **l, **{"testing_period": p}}
                          for a in data_configs for l in load_configs for p in period_list])
    all_train_currencys_df = pd.DataFrame([tuple(e)[0] for e in all_train_currencys])

    # Map pillar name to factor list by merging pillar tables & configs
    cluster_pillar = read_query(f"SELECT currency_code as train_currency, testing_period, pillar, factor_list "
                                f"FROM {pillar_cluster_table}")
    defined_pillar = read_query(f"SELECT pillar, factor_list FROM {pillar_defined_table}")

    all_train_currencys_defined_df = all_train_currencys_df.loc[all_train_currencys_df["pillar"] != "cluster"]
    all_train_currencys_cluster_df = all_train_currencys_df.loc[all_train_currencys_df["pillar"] == "cluster"].drop(columns=["pillar"])

    all_train_currencys_defined_df = all_train_currencys_defined_df.merge(defined_pillar, on=["pillar"], how="left")
    cluster_pillar_pillar = cluster_pillar.loc[cluster_pillar["pillar"].str.startswith("pillar")]
    all_train_currencys_cluster_df = all_train_currencys_cluster_df.merge(cluster_pillar_pillar,
                                                        on=["train_currency", "testing_period"], how="left")
    all_train_currencys_df = all_train_currencys_defined_df.append(all_train_currencys_cluster_df)
    all_train_currencys_df["testing_period"] = pd.to_datetime(all_train_currencys_df["testing_period"])

    # Check DB for score table for failed iteration
    if restart:
        diff_config_col = [x for x in all_train_currencys_df if x != "factor_list"]
        fin_df = read_query(f"SELECT {', '.join(diff_config_col)}, count(uid) as uid "
                            f"FROM {result_score_table} WHERE name_sql='{args.restart}' "
                            f"GROUP BY {', '.join(diff_config_col)}")
        all_train_currencys_df = all_train_currencys_df.merge(fin_df, how='left', on=diff_config_col).sort_values(by=diff_config_col)
        all_train_currencys_df = all_train_currencys_df.loc[all_train_currencys_df['uid'].isnull(), diff_config_col + ["factor_list"]]
        to_slack("clair").message_to_slack(f"=== Restart [{args.restart}]: rest iterations (n={len(all_train_currencys_df)}) ===")

    return all_train_currencys_df


if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()

    # define training periods
    parser.add_argument('--weeks_to_expire', default=4, type=int, help='Prediction period length in weeks')
    parser.add_argument('--backtest_period', default=14, type=int, help='Number of backtest period')
    parser.add_argument('--sample_interval', default=4, type=int, help='Number of weeks between two backtest periods')
    parser.add_argument('--processes', default=1, type=int, help='Number of multiprocessing threads')
    args = parser.parse_args()

    # for production: use defined configs
    data_configs = read_query(f"SELECT * FROM {config_train_table} "
                              f"WHERE is_active AND weeks_to_expire = {args.weeks_to_expire}")
    data_configs = data_configs.drop(columns=["is_active", "last_finish"]).to_dict("records")
    assert len(data_configs) > 0    # else no training will be done

    eval_configs = read_query(f"SELECT * FROM {config_eval_table} "
                              f"WHERE is_active AND weeks_to_expire = {args.weeks_to_expire}")
    eval_configs = eval_configs.drop(columns=["is_active"]).to_dict("records")
    assert len(eval_configs) > 0    # else no training will be done

    all_eval_train_currencys = [tuple([e]) for e in eval_configs]

    # with mp.Pool(processes=args.processes) as pool:

    logger.info(f"=== evaluation iteration: n={len(all_eval_train_currencys)} ===")
    rank_cls = calculate_rank_pred(name_sql=sql_result["name_sql"], pred_start_testing_period='2015-09-01',
                                   pass_eval=args.pass_eval,
                                   pass_eval_top=args.pass_eval_top,
                                   fix_config_col=list(data_configs[0].keys()))

    eval_results = []
    for group in all_eval_train_currencys:
        try:
            r = rank_cls.rank_(group)
            eval_results.append(r)
        except Exception as e:
            to_slack().message_to_slack(e)

        # eval_results = pool.starmap(rank_cls.rank_, all_eval_train_currencys)

    # 1. update [backtest_eval_table]
    if not args.pass_eval:
        eval_df = pd.concat([e[0] for e in eval_results], axis=0)  # df for each config evaluation results
        eval_df["_name_sql"] = sql_result["name_sql"]
        eval_df['updated'] = dt.datetime.now()
        eval_primary_key = eval_df.filter(regex="^_").columns.to_list()
        eval_df.to_pickle(f"eval_{sql_result['name_sql']}.pkl")
        upsert_data_to_database(eval_df, backtest_eval_table,
                                primary_key=eval_primary_key, how="update", dtype=backtest_eval_dtypes)

    # 2. update [backtest_top_table]
    if not args.pass_eval_top:
        score_df = pd.concat([e[1] for e in eval_results], axis=0)  # df for all scores
        top_eval_df = rank_cls.score_top_eval_(score_df)  # df for backtest score evalution
        top_eval_df["name_sql"] = sql_result["name_sql"]
        config_col = top_eval_df.filter(regex="^_").columns.to_list()
        primary_key = ["name_sql", "weeks_to_expire", "currency_code", "trading_day", "top_n"]

        if args.debug:  # if debug: write top ticker evaluation to other table
            backtest_top_dtypes = {**backtest_eval_dtypes, **backtest_top_dtypes}
            primary_key += config_col
        else:           # if production: remove fixed config columns
            top_eval_df = top_eval_df.drop(columns=config_col)
        top_eval_df.to_pickle(f"top_eval_{sql_result['name_sql']}.pkl")
        upsert_data_to_database(top_eval_df, backtest_top_table,
                                primary_key=primary_key, how='update', dtype=backtest_top_dtypes)

    # 3. update [production_rank_table]
    if not args.debug:
        select_df = pd.concat([e[2] for e in eval_results], axis=0)  # df for current selected factors
        logger.debug(select_df)
        logger.debug(select_df.columns.to_list())
        select_df = select_df.rename(columns={"_pred_currency": "currency_code",
                                              "_weeks_to_expire": "weeks_to_expire",
                                              "_pillar": "pillar",
                                              "_eval_top_metric": "eval_metric"})
        select_df = select_df[["weeks_to_expire", "currency_code", "trading_day", "pillar", "eval_metric", "max_factor",
                               "min_factor", "max_factor_extra", "min_factor_extra", "max_factor_trh", "min_factor_trh"]]
        select_df["updated"] = dt.datetime.now()
        upsert_data_to_database(select_df, production_rank_table,
                                primary_key=["weeks_to_expire", "currency_code", "pillar"], schema="public",
                                db_url=db_url_alibaba_prod, how='update', dtype=rank_dtypes)    # TODO: always to ali?

        # update [production_rank_history_table]
        upsert_data_to_database(select_df, production_rank_history_table,
                                primary_key=["weeks_to_expire", "currency_code", "pillar", "updated"], schema="public",
                                db_url=db_url_alibaba_prod, how='update', dtype=rank_dtypes)    # TODO: always to ali?