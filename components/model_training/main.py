import datetime as dt
import numpy as np
import argparse
from functools import partial
import multiprocessing as mp
import pandas as pd
from contextlib import closing
from src.load_data import combineData, loadData
from src.load_train_configs import loadTrainConfig
from src.random_forest import rf_HPOT
import os
from utils import (
    to_slack,
    read_query,
    read_table,
    upsert_data_to_database,
    models,
    recreate_engine,
    sys_logger,
    err2slack,
    dateNow,
    timestampNow
)

logger = sys_logger(__name__, "DEBUG")


def write_args_finished(kwargs):
    """
    write finish timing to database for sqlalchemy
    """
    tbl = models.FactorFormulaTrainConfig
    df = pd.Series(kwargs).to_frame().T.filter([x.name for x in tbl.__table__.columns])
    if kwargs["pillar"][:6] == "pillar":
        df["pillar"] = "cluster"
    df = df.assign(finished=timestampNow(), id=0)
    upsert_data_to_database(data=df, table=tbl.__tablename__, how="update")


@err2slack("factor")
def start(*args, sql_result: dict = None, raw_df: pd.DataFrame = None):
    """
    run random forest on multi-processor
    """
    try: 
        kwargs, = args
        sql_result.update(kwargs)

        logger.debug(f"===== test on pillar: [{sql_result['pillar']}] =====")

        data = loadData(**sql_result)
        sample_sets, neg_factor, cut_bins = data.split_all(raw_df)
        sql_result['neg_factor'] = neg_factor

        for sample_set in sample_sets:
            hpot_cls = rf_HPOT(max_evals=10, sql_result=sql_result, **sql_result)
            hpot_cls.train_and_write(sample_set=sample_set)

        if not os.getenv("DEBUG").lower == "true":
            write_args_finished(kwargs)

        return True
    except Exception as e:
        # logger.debug(f"During training for {sql_result['pillar']}, on testing period {sql_result['testing_period']}\n
        #                 and factor list {sql_result['factor_list']}")
        pass


if __name__ == "__main__":

    # --------------------------------- Parser ------------------------------------------

    parser = argparse.ArgumentParser()

    # define training periods
    parser.add_argument('--weeks_to_expire', default=4, type=int, help='Prediction period length in weeks')
    parser.add_argument('--backtest_period', default=14, type=int, help='Number of backtest period')
    parser.add_argument('--sample_interval', default=4, type=int, help='Number of weeks between two backtest periods')
    parser.add_argument('--processes', default=1, type=int, help='Number of multiprocessing threads')

    parser.add_argument('--restart', default=None, type=str, help='uid for to restart iteration')
    parser.add_argument('--currency_code', default=None, type=str, help='calculate for certain currency only')

    parser.add_argument('--look_back', default=5, type=int, help='look back year for loading clustered features') 

    parser.add_argument('--debug', action='store_true', help='bypass monthly running check')
    args = parser.parse_args()

    # --------------------------------------- Production / Development --------------------------------------------

    if not ((os.getenv("DEBUG").lower() == "true") or args.debug):
        if dt.datetime.today().day > 7:
            logger.warning('Not start: Factor model only run on the next day after first Sunday every month! ')
            exit(0)

    # sql_result = data write to score TABLE
    datetimeNow = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')
    sql_result = {"name_sql": f"w{args.weeks_to_expire}_{datetimeNow}"}

    if args.restart:
        sql_result['name_sql'] = args.restart
        args.weeks_to_expire = int(sql_result['name_sql'].split('_')[0][1:])

    # --------------------------------- Model Training ------------------------------------------

    with closing(mp.Pool(processes=args.processes, initializer=recreate_engine)) as pool:

        raw_df = combineData(weeks_to_expire=args.weeks_to_expire,
                             sample_interval=args.sample_interval,
                             backtest_period=args.backtest_period,
                             currency_code=None,                        # raw_df should get all
                             restart=args.restart).get_raw_data()

        all_groups = loadTrainConfig(weeks_to_expire=args.weeks_to_expire,
                                     sample_interval=args.sample_interval,
                                     backtest_period=args.backtest_period,
                                     restart=args.restart,
                                     currency_code=args.currency_code,look_back=args.look_back).get_all_groups() # look_back for clustered features
        # breakpoint()
        # breakpoint()
        pool.starmap(partial(start, raw_df=raw_df, sql_result=sql_result.copy()), all_groups)           # training will write to DB right after training


