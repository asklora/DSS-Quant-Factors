import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta
from itertools import product
import multiprocessing as mp
import gc
from contextlib import closing
from sqlalchemy import select
import sys
import os
from pathlib import Path

path = Path(os.path.abspath(__file__))
sys.path.append(str(path.parent.parent.parent.absolute()))

from src.load_eval_configs import load_eval_config
from src.evaluate_factor_premium import evalFactor
# from src.calculation_backtest import
# from src.calculation_backtest_score import

from src.load_eval_configs import (
    load_eval_config,
    load_latest_name_sql,
)
from utils import (
    sys_logger,
    read_query,
    upsert_data_to_database,
    recreate_engine,
    models,
)

logger = sys_logger(__name__, "DEBUG")


if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()

    # define training periods
    parser.add_argument('--name_sql', default=None, type=str, help='Batch Name in score data for analysis')
    parser.add_argument('--weeks_to_expire', default=4, type=int, help='Prediction period length in weeks')

    parser.add_argument('--eval_factor', action='store_true', help='Factor premiums evaluation')
    parser.add_argument('--eval_top', action='store_true', help='Top selection from selected factor evaluation')
    parser.add_argument('--eval_select', action='store_true', help='Use iteration selection to overwrite factor selection table')
    parser.add_argument('--processes', default=1, type=int, help='Number of multiprocessing threads')
    args = parser.parse_args()

    if args.name_sql:
        eval_name_sql = args.name_sql
    else:
        eval_name_sql = load_latest_name_sql(args.weeks_to_expire)

    all_groups = load_eval_config(args.weeks_to_expire)

    # 1. update [backtest_eval_table]
    if args.eval_factor:
        rank_cls = evalFactor(name_sql=eval_name_sql, processes=args.processes, all_groups=all_groups).write_db()

    # 2. update [backtest_top_table]
    if args.eval_top:
        score_df = pd.concat([e[1] for e in eval_results], axis=0)  # df for all scores
        top_eval_df = rank_cls.score_top_eval_(score_df)  # df for backtest score evalution
        top_eval_df["name_sql"] = eval_name_sql
        config_col = top_eval_df.filter(regex="^_").columns.to_list()
        primary_key = ["name_sql", "weeks_to_expire", "currency_code", "trading_day", "top_n"]

        top_eval_df.to_pickle(f"top_eval_{eval_name_sql}.pkl")
        upsert_data_to_database(top_eval_df, models.FactorBacktestTop.__tablename__, how='update')

    # 3. update [production_rank_table]
    if args.eval_select:
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
        upsert_data_to_database(select_df, models.FactorResultSelect.__tablename__, how='update')

        # update [production_rank_history_table]
        upsert_data_to_database(select_df, models.FactorResultSelectHistory.__tablename__, how='update')