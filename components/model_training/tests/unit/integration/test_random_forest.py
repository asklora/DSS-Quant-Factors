import pandas as pd
from pandas import Timestamp
import numpy as np
from typing import Dict
from components.model_training.src.random_forest import rf_HPOT

hpot_cls = rf_HPOT(
    max_evals=10,
    down_mkt_pct=0.7,
    tree_type='rf',
    objective='squared_error',
    sql_result={"name_sql": "test_name_sql", "uid": "test_uid",
                "train_currency": "USD"},
    hpot_eval_metric='mse_valid'
)


def test_rf_train():
    from components.model_training.src.load_data import combineData, loadData

    main_df = combineData(weeks_to_expire=8, sample_interval=4,
                          backtest_period=14).get_raw_data()
    sample_testing_period = pd.to_datetime(main_df["testing_period"].to_list()[-5])

    test_factor_list = ["roic", "book_to_price", "earnings_1yr",
                        "earnings_yield", "ni_to_cfo", "sales_to_price", "roe",
                        "ebitda_to_ev"]
    test_y_qcut = 10
    cls = loadData(
        weeks_to_expire=8,
        train_currency='USD',
        pred_currency='USD,EUR',
        testing_period=sample_testing_period,
        average_days=-7,
        factor_list=test_factor_list,
        y_qcut=test_y_qcut,
        factor_reverse=True,
        factor_pca=0.6,
        valid_pct=0.2,
        valid_method=2010,
    )
    sample_set, neg_factor, cut_bins = cls.split_all(main_df)

    hpot_cls.train_and_write(sample_set=sample_set[0])
