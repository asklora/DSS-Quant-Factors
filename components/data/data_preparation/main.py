import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta
from itertools import product
import multiprocessing as mp
import gc

from .src.calculation_ratio import calc_factor_variables_multi
from .src.calculation_premium import calcPremium
from .src.calculation_pillar_cluster import calcPillarCluster
from utils import (
    sys_logger,
    read_query,
    models,

)

logger = sys_logger(__name__, "DEBUG")

config_train_table = models.FactorFormulaTrainConfig.__tablename__

all_currency_list = ["HKD", "USD", "CNY", "EUR"]
all_average_days = [7]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # define training periods
    parser.add_argument('--weeks_to_expire', default=4, type=int, help='Prediction period length in weeks')
    parser.add_argument('--sample_interval', default=4, type=int, help='Number of weeks between two backtest periods')

    parser.add_argument('--recalc_ratio', action='store_true', help='Start recalculate ratios')
    parser.add_argument('--recalc_premium', action='store_true', help='Start recalculate premiums')
    parser.add_argument('--recalc_subpillar', action='store_true', help='Start recalculate cluster pillar / subpillar')
    parser.add_argument('--processes', default=4, type=int, help='Multiprocessing')

    args = parser.parse_args()

    # Check 1: if monthly -> only first Sunday every month
    if not args.pass_train:
        if dt.datetime.today().day > 7:
            raise Exception('Not start: Factor model only run on the next day after first Sunday every month! ')

    # ---------------------------------------- Rerun Write Premium -----------------------------------------------

    if args.recalc_ratio:
        # default = update ratios for past 3 months
        logger.info("=== Calculate ratio ===")
        calc_factor_variables_multi(tickers=None,
                                    currency_codes=all_currency_list,
                                    tri_return_only=False,
                                    processes=args.processes)

    if args.recalc_premium:
        logger.info("=== Calculate premium ===")
        premium_data = calcPremium(weeks_to_expire=args.weeks_to_expire,
                                   average_days_list=all_average_days,
                                   weeks_to_offset=min(4, args.sample_interval),
                                   currency_code_list=all_currency_list,
                                   processes=args.processes).write_all()

    if args.recalc_subpillar

        # default = update ratios for past 3 months
        logger.info("=== Calculate cluster premium ===")
        for cur in all_currency_list:
            calcPillarCluster(weeks_to_expire=args.weeks_to_expire,
                              currency_code=cur,
                              sample_interval=args.sample_interval)
