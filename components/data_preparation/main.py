import datetime as dt
import argparse
import os
from pathlib import Path

import sys
import os

path = Path(os.path.abspath(__file__))
sys.path.append(str(path.parent.parent.parent.absolute()))

from src.calculation_ratio import calc_factor_variables_multi
from src.calculation_premium import calcPremium
from src.calculation_pillar_cluster import calcPillarCluster
from utils import (
    sys_logger,
    read_query,
    models,
)

logger = sys_logger(__name__, "DEBUG")


all_currency_list = ["HKD", "USD", "CNY", "EUR"]            # currency covered by factor model (train / prediction)
all_average_days = [-7]                                     # stock return use average of next 7 days (i.e. prediction valid for 7 days)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # define training periods
    parser.add_argument('--weeks_to_expire', default=4, type=int, help='Prediction period length in weeks')
    parser.add_argument('--sample_interval', default=4, type=int, help='Number of weeks between two backtest periods')

    parser.add_argument('--recalc_ratio', action='store_true', help='Start recalculate ratios')
    parser.add_argument('--recalc_premium', action='store_true', help='Start recalculate premiums')
    parser.add_argument('--recalc_subpillar', action='store_true', help='Start recalculate cluster pillar / subpillar')
    parser.add_argument('--processes', default=1, type=int, help='Multiprocessing')

    parser.add_argument('--history', action='store_true', help='Rewrite entire history')
    parser.add_argument('--currency_code', default=None, type=str, help='calculate for certain currency only')

    parser.add_argument('--debug', action='store_true', help='bypass monthly running check')
    args = parser.parse_args()

    # Check 1: if monthly -> only first Sunday every month
    if not ((os.getenv("DEBUG").lower() == "true") or args.debug):
        if dt.datetime.today().day > 7:
            logger.warning('Not start: Factor model only run on the next day after first Sunday every month! ')
            exit(0)

    if args.currency_code:                              # for debugging only
        all_currency_list = [args.currency_code]

    # ---------------------------------------- Rerun Write Premium -----------------------------------------------

    if args.recalc_ratio:
        # default = update ratios for past 3 months
        logger.info("=== Calculate ratio ===")
        calc_factor_variables_multi(tickers=None,
                                    currency_codes=all_currency_list,
                                    tri_return_only=False,
                                    processes=args.processes,
                                    start_date=dt.datetime(1998, 1, 1) if args.history else None)

    if args.recalc_premium:
        # default = update ratios for as long as possible (b/c universe changes will affect the value of premium).
        logger.info("=== Calculate premium ===")
        premium_data = calcPremium(weeks_to_expire=args.weeks_to_expire,
                                   average_days_list=all_average_days,
                                   weeks_to_offset=min(4, args.sample_interval),
                                   currency_code_list=all_currency_list,
                                   processes=args.processes).write_all()

    if args.recalc_subpillar:
        # default = update subpillar for past 3 months
        logger.info("=== Calculate cluster pillar ===")
        calcPillarCluster(weeks_to_expire=args.weeks_to_expire,
                          currency_code_list=all_currency_list,
                          sample_interval=args.sample_interval,
                          processes=args.processes,
                          start_date=dt.datetime(1998, 1, 1) if args.history else None).write_all()
