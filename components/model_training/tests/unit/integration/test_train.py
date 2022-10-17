from components.model_training.src.load_data import combineData, loadData
from components.model_training.src.load_train_configs import loadTrainConfig
from components.model_training.main import start
import os
import datetime as dt
import numpy as np
import argparse
from functools import partial
import multiprocessing as mp
import pandas as pd
from contextlib import closing
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

processes = 1
weeks_to_expire = 1
sample_interval = 4
backtest_period = 1
restart = False
currency_code = "USD"


def test_rf_train():
    sql_result = {"name_sql": "test1"}

    raw_df = combineData(weeks_to_expire=weeks_to_expire,
                         sample_interval=sample_interval,
                         backtest_period=backtest_period,
                         currency_code=None,  # raw_df should get all
                         restart='w4_20221014150807').get_raw_data()

    all_groups = loadTrainConfig(weeks_to_expire=weeks_to_expire,
                                 sample_interval=sample_interval,
                                 backtest_period=backtest_period,
                                 restart='w4_20221014150807',
                                 currency_code=currency_code) \
        .get_all_groups()

    # test on first group
    start(all_groups[0][0], raw_df=raw_df, sql_result=sql_result.copy())

