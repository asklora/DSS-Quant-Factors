import pandas as pd
from pandas import Timestamp
import numpy as np
from typing import Dict


def df_n_curr_n_time():
    return pd.DataFrame(
        [[0, -1, 3], [1, -1, 5], [2, 0, 2], [-1, 0, 1], [-2, 4, 4],
         [0, -1, 3], [1, -1, 5], [2, 0, 2], [-1, 0, 1], [-2, 4, 4]],
        index=[
            ('USD', Timestamp('2001-01-28 00:00:00')),
            ('USD', Timestamp('2001-02-25 00:00:00')),
            ('USD', Timestamp('2001-03-25 00:00:00')),
            ('USD', Timestamp('2001-04-22 00:00:00')),
            ('USD', Timestamp('2001-05-20 00:00:00')),
            ('EUR', Timestamp('2001-01-28 00:00:00')),
            ('EUR', Timestamp('2001-02-25 00:00:00')),
            ('EUR', Timestamp('2001-03-25 00:00:00')),
            ('EUR', Timestamp('2001-04-22 00:00:00')),
            ('EUR', Timestamp('2001-05-20 00:00:00')),
        ],
        columns=["roic", "book_to_price", "ebitda_ev"]
    )


def df_1_curr_n_time():
    return pd.DataFrame(
        [[0, -1], [1, -1], [2, 0], [-1, 0], [-2, 4]],           # Don't change -> use to test sample weight
        index=[
            ('EUR', Timestamp('2001-01-28 00:00:00')),
            ('EUR', Timestamp('2001-02-25 00:00:00')),
            ('EUR', Timestamp('2001-03-25 00:00:00')),
            ('EUR', Timestamp('2001-04-22 00:00:00')),
            ('EUR', Timestamp('2001-05-20 00:00:00')),
        ],
        columns=["roic", "book_to_price"]
    )


def df_n_curr_1_time():
    return pd.DataFrame(
        [[0, -1], [1, -1], [2, 0], [-1, 0]],
        index=pd.MultiIndex.from_tuples([
            ('EUR', Timestamp('2001-01-28 00:00:00')),
            ('CNY', Timestamp('2001-01-28 00:00:00')),
            ('USD', Timestamp('2001-01-28 00:00:00')),
            ('HKD', Timestamp('2001-01-28 00:00:00')),
        ], names=["group", "testing_period"]),
        columns=["roic", "book_to_price"]
    )


def hpot_cls():
    from components.model_training.src.random_forest import rf_HPOT
    return rf_HPOT(max_evals=10,
                   down_mkt_pct=0,
                   tree_type='rf',
                   objective='squared_error',
                   sql_result={"uid": "test_uid"},
                   hpot_eval_metric='mse_valid'
                   )


def test_rf_HPOT__sample_weight():

    train_y = df_1_curr_n_time()

    sample_weight = hpot_cls()._rf_HPOT__sample_weight(train_y=train_y)

    assert all(sample_weight == np.array([0.4, 0.6, 0.6, 0.4, 0.6]))


def test_rf_HPOT__to_sql_prediction():

    test_y = df_n_curr_1_time()
    test_y_pred = test_y.copy() * 2

    sql_pred = hpot_cls()._rf_HPOT__to_sql_prediction(test_y=test_y, test_y_pred=test_y_pred)
    assert set(sql_pred.columns) == {"currency_code", "factor_name", "uid", "actual", "pred"}

    drop_dup_sql_pred = sql_pred.drop_duplicates(subset=["currency_code", "factor_name", "uid"])
    assert len(sql_pred) == len(drop_dup_sql_pred)


def test_get_timestamp_now_str():
    from components.model_training.src.random_forest import get_timestamp_now_str
    x = get_timestamp_now_str(n_suffix=4)

    assert len(x) == 24

    import multiprocessing as mp

    all_groups = [4 for _ in range(50)]
    with mp.Pool(processes=50) as pool:
        results = pool.map(get_timestamp_now_str, all_groups)           # training will write to DB right after training

    assert len(set(results)) == 50


def test_adj_mse_score():
    from components.model_training.src.random_forest import adj_mse_score

    adj_mse = adj_mse_score(
        actual=df_n_curr_n_time().values,
        pred=df_n_curr_n_time().applymap(lambda x: np.random.random()).values
    )

    assert adj_mse > 0
    assert isinstance(adj_mse, float)


def test_rf_HPOT__calc_pred_eval_scores_multifactor():

    sample_set = {
        "train_y": df_n_curr_n_time(),
        "train_y_pred": df_n_curr_n_time() + 1,
        "valid_y": df_n_curr_n_time(),
        "valid_y_pred": df_n_curr_n_time() * 5,
        "test_y": df_n_curr_1_time(),
        "test_y_pred": df_n_curr_1_time() * 4 - 1,
    }

    result = hpot_cls()._rf_HPOT__calc_pred_eval_scores(sample_set=sample_set)

    assert len(result) == 12
    assert type(result) == dict
    assert all(isinstance(x, str) for x in list(result.keys()))
    assert all(isinstance(x, float) for x in list(result.values()))


def test_rf_HPOT__calc_pred_eval_scores_singlefactor():

    sample_set = {
        "train_y": df_n_curr_n_time().iloc[:, [0]],
        "train_y_pred": df_n_curr_n_time().iloc[:, [0]] + 1,
        "valid_y": df_n_curr_n_time().iloc[:, [0]],
        "valid_y_pred": df_n_curr_n_time().iloc[:, [0]] * 5,
        "test_y": df_n_curr_1_time().iloc[:, [0]],
        "test_y_pred": df_n_curr_1_time().iloc[:, [0]] * 4 - 1,
    }

    result = hpot_cls()._rf_HPOT__calc_pred_eval_scores(sample_set=sample_set)

    assert len(result) == 12
    assert type(result) == dict
    assert all(isinstance(x, str) for x in list(result.keys()))
    assert all(isinstance(x, float) for x in list(result.values()))


def test_rf_HPOT__write_score_db():

    cls = hpot_cls()

    test_y = df_n_curr_1_time()
    test_y_pred = test_y.copy() * 2

    cls.hpot["best_stock_df"] = cls._rf_HPOT__to_sql_prediction(test_y=test_y, test_y_pred=test_y_pred)

    result = cls._rf_HPOT__write_prediction_db()


def test_rf_HPOT_split_all():
    from components.model_training.src.load_data import combineData, loadData

    main_df = combineData(weeks_to_expire=8, sample_interval=4, backtest_period=14).get_raw_data()
    sample_testing_period = pd.to_datetime(main_df["testing_period"].to_list()[-5])

    test_factor_list = ["roic", "book_to_price", "earnings_1yr", "earnings_yield", "ni_to_cfo",
                        "sales_to_price", "roe", "ebitda_to_ev"]
    test_y_qcut = 10
    cls = loadData(
        weeks_to_expire=8,
        train_currency='USD,EUR',
        pred_currency='USD,EUR',
        testing_period=sample_testing_period,
        average_days=-7,
        factor_list=test_factor_list,
        y_qcut=test_y_qcut,
        factor_reverse=True,
        factor_pca=0.6,
        valid_pct=0.2,
        valid_method=2010
    )
    sample_set, neg_factor, cut_bins = cls.split_all(main_df)

    hpot_cls().train_and_write(sample_set=sample_set[0])

