import datetime as dt
import argparse
import os
import gc
from src.calculation_ratio import calc_factor_variables_multi
from src.calculation_premium import calcPremium
from src.calculation_pillar_cluster import calcPillarCluster

from utils import (
    sys_logger,
    read_query,
    models,
    check_memory,
    dateNow
)
from src.configs import LOGGER_LEVELS
logger = sys_logger(__name__, LOGGER_LEVELS.MAIN)


# currency covered by factor model (train / prediction)
all_currency_list = ["HKD", "USD", "CNY", "EUR"]
# stock return use average of next 7 days (i.e. prediction valid for 7 days)
all_average_days = [-7]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # define training periods
    parser.add_argument('--weeks_to_expire', default=4,
                        type=int, help='Prediction period length in weeks')
    parser.add_argument('--sample_interval', default=4, type=int,
                        help='Number of weeks between two backtest periods')

    parser.add_argument('--recalc_ratio', action='store_true',
                        help='Start recalculate ratios')
    parser.add_argument('--recalc_premium', action='store_true',
                        help='Start recalculate premiums')
    parser.add_argument('--recalc_subpillar', action='store_true',
                        help='Start recalculate cluster pillar / subpillar')
    parser.add_argument('--processes', default=1,
                        type=int, help='Multiprocessing')
    parser.add_argument('--qcut_for_premium', default=0.2, type=float,
                        help='top/bottom percentage for cutting premium quantile')
    parser.add_argument('--pickle', default=False, type=bool,
                        help='load factor_processed_ratio_from_pickle')

    parser.add_argument('--start_date', default='2015-01-01', type=str,
                        help='start_date for calculating factor ratio, factor premium or factor subpillar')
    parser.add_argument('--end_date', default=dateNow(), type=str,
                        help='end date for calculating factor ratio, factor premium or factor subpillar')

    parser.add_argument('--history', action='store_true',
                        help='Rewrite entire history')
    parser.add_argument('--currency_code', default=None,
                        type=str, help='calculate for certain currency only')
    parser.add_argument('--look_back', default=5, type=int,
                        help='lookback period for clustering factors')
    parser.add_argument('--ticker', default=None, type=str,
                        help='ticker for recalc_ratio')
    parser.add_argument('--revert_premium', default=False,
                        type=str, help='revert_premium_according_to_smb_positive')

    parser.add_argument('--debug', action='store_true',
                        help='bypass monthly running check')
    args = parser.parse_args()

    # Check 1: if monthly -> only first Sunday every month
    if not ((os.getenv("DEBUG").lower() == "true") or args.debug):
        if dt.datetime.today().day > 7:
            logger.warning(
                'Not start: Factor model only run on the next day after first Sunday every month! ')
            exit(0)
    # breakpoint()
    if args.currency_code:                              # for debugging only
        all_currency_list = [args.currency_code]

    # -------------------------- Rerun Write Premium ---------------------------

    if args.recalc_ratio:
        # default = update ratios for past 3 months
        logger.info("=== Calculate ratio ===")
        calc_factor_variables_multi(tickers=None if type(args.ticker) == type(None) else [args.ticker],
                                    currency_codes=all_currency_list,
                                    tri_return_only=False,
                                    processes=args.processes,
                                    start_date=dt.datetime(1998, 1, 1) if args.history else dt.datetime.strptime(args.start_date, '%Y-%m-%d'))
        del calc_factor_variables_multi
        gc.collect()

    logger.info("Done Ratio calculation")
    check_memory(logger=logger)

    if args.recalc_premium:
        # default = update ratios for as long as possible (b/c universe changes will affect the value of premium).
        logger.info("=== Calculate premium ===")
        premium_data = calcPremium(weeks_to_expire=args.weeks_to_expire,
                                   average_days_list=all_average_days,
                                   weeks_to_offset=min(
                                       4, args.sample_interval),
                                   currency_code_list=[args.currency_code],
                                   processes=args.processes, percent_for_qcut=args.qcut_for_premium).write_all(start_date=args.start_date, end_date=args.end_date)
    check_memory(logger=logger)

    if args.recalc_subpillar:
        # default = update subpillar for past 3 months
        logger.info("=== Calculate cluster pillar ===")
        calcPillarCluster(weeks_to_expire=args.weeks_to_expire,
                          currency_code_list=all_currency_list,
                          sample_interval=args.sample_interval,
                          processes=args.processes,
                          start_date=dt.datetime(1998, 1, 1) if args.history else dt.datetime.strptime(args.start_date, '%Y-%m-%d').date(), end_date=dt.datetime.strptime(args.end_date, '%Y-%m-%d').date(), lookback=args.look_back).write_all()
    check_memory(logger=logger)

    # how to calculate subpillar: start_date = '2010-01-01' ---> database will show start_date = '2010-01-01', with lookback = ....
