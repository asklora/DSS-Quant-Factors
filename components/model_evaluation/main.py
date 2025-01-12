import os
import datetime as dt
import argparse
from src.evaluate_factor_premium import EvalFactor
from src.evaluate_top_selection import EvalTop
from src.load_eval_configs import load_latest_name_sql
from utils import (
    sys_logger,
)

logger = sys_logger(__name__, "DEBUG")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # define training periods
    parser.add_argument('--name_sql', default=None, type=str,
                        help='Batch Name in score data for analysis')
    parser.add_argument('--weeks_to_expire', default=4, type=int,
                        help='Prediction period length in weeks')
    parser.add_argument('--eval_factor', action='store_true',
                        help='Factor premiums evaluation')
    parser.add_argument('--eval_top', action='store_true',
                        help='Top selection from selected factor evaluation')
    parser.add_argument('--eval_select', action='store_true',
                        help='Use iteration selection to overwrite factor '
                             'selection table')
    parser.add_argument('--processes', default=1, type=int,
                        help='Number of multiprocessing threads')
    parser.add_argument('--debug', action='store_true',
                        help='bypass monthly running check')
    args = parser.parse_args()

    if not ((os.getenv("DEBUG").lower() == "true") or args.debug):
        if dt.datetime.today().day > 7:
            logger.warning(
                'Not start: Factor model only run on the next day '
                'after first Sunday every month! ')
            exit(0)

    if args.name_sql:
        eval_name_sql = args.name_sql
    else:
        eval_name_sql = load_latest_name_sql(args.weeks_to_expire)

    # 1. update [FactorBacktestEval]
    if args.eval_factor:
        eval_df = EvalFactor(name_sql=eval_name_sql,
                             processes=args.processes).write_db()
    else:
        eval_df = None

    if args.eval_top or args.eval_select:
        cls = EvalTop(name_sql=eval_name_sql,
                      processes=args.processes,
                      eval_df=eval_df)

        # 2. update [FactorBacktestTop]
        if args.eval_top:
            score_df = cls.write_top_select_eval()

        # 3. update [FactorResultSelect/History]
        if args.eval_select:
            select_df = cls.write_latest_select()
