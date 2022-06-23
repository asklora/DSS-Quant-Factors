import datetime as dt
import numpy as np
import argparse
from functools import partial
import multiprocessing as mp
import pandas as pd

import sys
import os
from pathlib import Path

path = Path(os.path.abspath(__file__))
sys.path.append(str(path.parent.parent.parent.absolute()))

from src.load_data import combineData, loadData
from src.load_train_configs import loadTrainConfig
from src.random_forest import rf_HPOT

from utils import (
    to_slack,
    read_query,
    read_table,
    upsert_data_to_database,
    models,
    sys_logger,
    err2slack,
    dateNow,
)

logger = sys_logger(__name__, "DEBUG")


class modelRun:
    """ run random forest on multi-processor """

    def __init__(self, sql_result):
        self.sql_result = sql_result

    def start(self, *args, raw_df=None):

        kwargs, = args
        self.sql_result.update(kwargs)

        logger.debug(f"===== test on pillar: [{self.sql_result['pillar']}] =====")

        data = loadData(**self.sql_result)
        sample_set, cv, neg_factor, cut_bins = data.split_all(raw_df)
        x_col = sample_set["train_x"].columns.to_list()
        y_col = sample_set["train_y"].columns.to_list()
        group_index = sample_set["test_x"].index.get_level_values("group").to_list()

        self.sql_result['neg_factor'] = neg_factor

        for train_index, valid_index in cv:  # roll over different validation set
            self._sample_set_split_valid(sample_set, train_index, valid_index)
            self._sql_result_record_len(sample_set)
            self._sample_set_x_fillna(sample_set)
            self._calc_neg_sample_weight(sample_set)

            rf_HPOT(max_evals=10,
                    sql_result=self.sql_result,
                    sample_set=sample_set,
                    x_col=x_col,
                    y_col=y_col,
                    group_index=group_index).write_db()

    def _sample_set_split_valid(self, sample_set, train_index, valid_index):
        """
        split training / validation dataframe;
        and convert pd.DataFrame to np.array.
        """

        train_index = pd.MultiIndex.from_tuples(set(train_index) & set(sample_set["train_y"].dropna(how='any').index))     # remove nan Y
        valid_index = pd.MultiIndex.from_tuples(set(valid_index) & set(sample_set["train_y"].dropna(how='any').index))

        sample_set['train_xx'] = sample_set['train_x'].loc[train_index].values
        sample_set['train_yy'] = sample_set['train_y'].loc[train_index].values
        sample_set['train_yy_final'] = sample_set['train_y_final'].loc[train_index].values

        sample_set['valid_x'] = sample_set['train_x'].loc[valid_index].values
        sample_set['valid_y'] = sample_set['train_y'].loc[valid_index].values
        sample_set['valid_y_final'] = sample_set['train_y_final'].loc[valid_index].values

        sample_set['test_x'] = sample_set['test_x'].values
        sample_set['test_y'] = sample_set['test_y'].values
        sample_set['test_y_final'] = sample_set['test_y_final'].values
        return True

    def _sql_result_record_len(self, sample_set):
        sql_result['train_len'] = len(sample_set['train_xx'])  # record length of training/validation sets
        sql_result['valid_len'] = len(sample_set['valid_x'])
        return True

    def _sample_set_x_fillna(self, sample_set):
        for k in ['valid_x', 'train_xx', 'test_x', 'train_x']:
            sample_set[k] = np.nan_to_num(sample_set[k], nan = 0)
        return True

    def _calc_neg_sample_weight(self, sample_set):
        sample_set['train_yy_weight'] = np.where(sample_set['train_yy'][:, 0] < 0,
                                                 self.sql_result["down_mkt_pct"],
                                                 1 - self.sql_result["down_mkt_pct"])
        return True


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
    args = parser.parse_args()

    # --------------------------------------- Production / Development --------------------------------------------

    if os.getenv("DEBUG").lower() != "true":
        if dt.datetime.today().day > 7:
            raise Exception('Not start: Factor model only run on the next day after first Sunday every month! ')

    # sql_result = data write to score TABLE
    datetimeNow = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')
    sql_result = {"name_sql": f"w{args.weeks_to_expire}_{datetimeNow}"}

    if args.restart:
        sql_result['name_sql'] = args.restart
        args.weeks_to_expire = int(sql_result['name_sql'].split('_')[0][1:])

    # --------------------------------- Model Training ------------------------------------------

    with mp.Pool(processes=args.processes) as pool:

        all_groups = loadTrainConfig(weeks_to_expire=args.weeks_to_expire,
                                     sample_interval=args.sample_interval,
                                     backtest_period=args.backtest_period,
                                     restart=args.restart,
                                     currency_code=args.currency_code).get_all_groups()
        raw_df = combineData(weeks_to_expire=args.weeks_to_expire,
                             sample_interval=args.sample_interval,
                             backtest_period=args.backtest_period,
                             restart=args.restart).get_raw_data()

        model_run_cls = modelRun(sql_result=sql_result)
        pool.starmap(partial(model_run_cls.start, raw_df=raw_df), all_groups)           # training will write to DB right after training


