import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from utils import dateNow


combineData_kwargs = dict(weeks_to_expire=4, sample_interval=4, backtest_period=14, currency_code="EUR", restart=None)


# def test_combineData_testing_period_list():
#     from components.model_training.src.load_data import calcTestingPeriod
#     lst = calcTestingPeriod(**combineData_kwargs)._testing_period_list
#
#     assert len(lst) > 1
#     assert min(lst) < dt.datetime(1998, 1, 1)
#     assert max(lst).date() == pd.date_range(end=dateNow(), freq="W-SUN", periods=2)[0].date()
#
#
# def test_cleanMacros_download_index():
#     from components.model_training.src.load_data import cleanMacros
#     df = cleanMacros(**combineData_kwargs)._download_index_return()
#
#     assert len(df) > 1
#     assert set(df.columns.to_list()) == {"field", "trading_day", "value"}
#     assert all(["stock_return" in x for x in df["field"].unique()])
#
#
# def test_cleanMacros_download_fred():
#     from components.model_training.src.load_data import cleanMacros
#     df = cleanMacros(**combineData_kwargs)._download_fred()
#
#     assert len(df) > 1
#     assert set(df.columns.to_list()) == {"field", "trading_day", "value"}
#
#
# def test_cleanMacros_download_macro():
#     from components.model_training.src.load_data import cleanMacros
#     df = cleanMacros(**combineData_kwargs)._download_clean_macro()
#
#     df_is_nan = df.loc[df["value"].isnull()]
#
#     assert len(df) > 1
#     assert set(df.columns.to_list()) == {"field", "trading_day", "value"}
#     assert df_is_nan["trading_day"].max() < dt.datetime(2000, 1, 1)
#
#
# def test_cleanMacros_download_vix():
#     from components.model_training.src.load_data import cleanMacros
#     df = cleanMacros(**combineData_kwargs)._download_vix()
#
#     assert len(df) > 1
#     assert set(df.columns.to_list()) == {"field", "trading_day", "value"}
#
#
# def test_cleanMacros():
#     from components.model_training.src.load_data import cleanMacros
#     df = cleanMacros(**combineData_kwargs).get_all_macros()
#
#     assert len(df) > 1
#     assert df["trading_day"].isnull().sum() == 0
#     assert all(df.notnull().sum() > 0)
#     assert len(df.columns) > 4
#
#
# def test_combineData_download_premium():
#     from components.model_training.src.load_data import combineData
#     df = combineData(**combineData_kwargs)._download_premium()
#
#     assert len(df) > 1
#     assert {"group", "testing_period", "average_days"}.issubset(set(df.columns.to_list()))
#     assert len(df["group"].unique()) > 1
#
#
# def test_combineData_resample_macros_to_testing_period():
#     from components.model_training.src.load_data import combineData, cleanMacros
#     df = cleanMacros(**combineData_kwargs).get_all_macros()
#
#     cls = combineData(**combineData_kwargs)
#     df_resample = cls._resample_macros_to_testing_period(df)
#
#     assert len(df_resample) > 1
#     assert all(df_resample.notnull().sum() > 0)
#     assert "testing_period" in df_resample.columns.to_list()
#
#
# def test_combineData_remove_high_missing_samples():
#     from components.model_training.src.load_data import combineData
#     cls = combineData(**combineData_kwargs)
#     df = cls._download_premium()
#     df_missing_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
#     clean_df = cls._remove_high_missing_samples(df, trh=0.5)
#     clean_df_missing_rate = clean_df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
#
#     assert len(clean_df) > 1
#     assert len(clean_df) <= len(df)
#     assert df_missing_rate >= clean_df_missing_rate
#     assert clean_df_missing_rate < 0.5
#
#
# def test_combineData_get_raw_data():
#     from components.model_training.src.load_data import combineData
#     cls = combineData(**combineData_kwargs)
#     df = cls.get_raw_data()
#
#     diff = df.groupby(["group"])["testing_period"].diff()
#     diff = diff.loc[diff.notnull()]
#
#     assert len(df) > 1
#     assert "fred_data" in df.columns.to_list()
#     assert len(set(diff.values)) == 1               # equal interval consecutive testing_period = NaN / weeks_to_expire
#     assert set(df["group"].unique()) == {"HKD", "CNY", "USD", "EUR"}
#     assert len(set(df["group"].to_list()[-2:])) == 1
#     assert df["group"].to_list()[-2:] != df["group"].to_list()[:2]
#
#
# def test_loadData_factor_reverse_on_lasso():
#     from components.model_training.src.load_data import combineData, loadData
#     main_df = combineData(weeks_to_expire=8, sample_interval=4, backtest_period=14).get_raw_data()
#     sample_testing_period = pd.to_datetime(main_df["testing_period"].to_list()[-5])
#
#     cls = loadData(
#         weeks_to_expire=4,
#         train_currency='USD,EUR',
#         pred_currency='USD,EUR',
#         testing_period=sample_testing_period,
#         average_days=-7,
#         factor_list=["roic", "book_to_price", "earnings_1yr", "earnings_yield", "ni_to_cfo", "sales_to_price", "roe", "ebitda_to_ev"],
#         y_qcut=1,
#         factor_reverse=1,
#         factor_pca=0.6,
#         valid_pct=0.2,
#         valid_method=2010
#     )
#     sample_df = cls._filter_sample(main_df)
#     assert set(cls.train_currency).issubset(sample_df.index.get_level_values("group").unique())
#
#     neg_factor = cls._factor_reverse_on_lasso(sample_df)
#     assert type(neg_factor) == list
#
#
# def test_loadData__test_sample_all_testing_period():
#     from components.model_training.src.load_data import combineData, loadData
#     from components.model_training.src.load_train_configs import loadTrainConfig
#
#     testing_period_list = loadTrainConfig(weeks_to_expire=8,
#                                           sample_interval=4,
#                                           backtest_period=14)._testing_period_list
#
#     main_df = combineData(weeks_to_expire=8, sample_interval=4, backtest_period=14).get_raw_data()
#
#     test_factor_list = ["roic", "book_to_price", "earnings_1yr", "earnings_yield", "ni_to_cfo", "sales_to_price", "roe",
#                         "ebitda_to_ev"]
#     test_y_qcut = 10
#
#     for t in reversed(testing_period_list):
#         cls = loadData(
#             weeks_to_expire=8,
#             train_currency='USD,EUR',
#             pred_currency='USD,EUR',
#             testing_period=t - relativedelta(weeks=8),
#             average_days=-7,
#             factor_list=test_factor_list,
#             y_qcut=test_y_qcut,
#             factor_reverse=1,
#             factor_pca=0.6,
#             valid_pct=0.2,
#             valid_method=2010
#         )
#         sample_df = cls._filter_sample(main_df)
#         df_y = cls._loadData__y_convert_testing_period(sample_df=sample_df)
#         df_test = cls._loadData__test_sample(df_y)
#         assert len(df_test) > 0
#
#
# def test_loadData_get_y():
#     from components.model_training.src.load_data import combineData, loadData
#     main_df = combineData(weeks_to_expire=8, sample_interval=4, backtest_period=14).get_raw_data()
#     sample_testing_period = pd.to_datetime(main_df["testing_period"].to_list()[-5])
#
#     test_factor_list = ["roic", "book_to_price", "earnings_1yr", "earnings_yield", "ni_to_cfo", "sales_to_price", "roe", "ebitda_to_ev"]
#     test_y_qcut = 10
#     cls = loadData(
#         weeks_to_expire=8,
#         train_currency='USD,EUR',
#         pred_currency='USD,EUR',
#         testing_period=sample_testing_period,
#         average_days=-7,
#         factor_list=test_factor_list,
#         y_qcut=test_y_qcut,
#         factor_reverse=1,
#         factor_pca=0.6,
#         valid_pct=0.2,
#         valid_method=2010
#     )
#     sample_df = cls._filter_sample(main_df)
#     assert set(cls.train_currency).issubset(sample_df.index.get_level_values("group").unique())
#
#     df_train_cut, df_test_cut, df_train, df_test, cut_bins = cls._get_y(sample_df)
#     after_cut_unique_value = set(df_train_cut.stack().dropna(how="any").values)
#
#     assert set(df_train_cut.columns.to_list()) == set(test_factor_list)
#     assert len(after_cut_unique_value) == test_y_qcut
#
#     if cls.convert_y_use_median:
#         assert not after_cut_unique_value.issubset(set(range(10)))
#
#
# def test_loadData_get_x():
#     from components.model_training.src.load_data import combineData, loadData
#     main_df = combineData(weeks_to_expire=8, sample_interval=4, backtest_period=14).get_raw_data()
#     sample_testing_period = pd.to_datetime(main_df["testing_period"].to_list()[-5])
#
#     test_factor_list = ["roic", "book_to_price", "earnings_1yr", "earnings_yield", "ni_to_cfo",
#                         "sales_to_price", "roe", "ebitda_to_ev"]
#     test_y_qcut = 10
#     cls = loadData(
#         weeks_to_expire=8,
#         train_currency='EUR',
#         pred_currency='USD,EUR',
#         testing_period=sample_testing_period,
#         average_days=-7,
#         factor_list=test_factor_list,
#         y_qcut=test_y_qcut,
#         factor_reverse=0,
#         factor_pca=0.6,
#         valid_pct=0.2,
#         valid_method=2010
#     )
#     sample_df = cls._filter_sample(main_df)
#     assert set(cls.train_currency) == set(sample_df.index.get_level_values("group").unique())
#
#     df_train_pca, df_test_pca = cls._get_x(sample_df)
#     assert len(df_train_pca) > 0


def test_loadData_split_all():
    from components.model_training.src.load_data import combineData, loadData
    main_df = combineData(weeks_to_expire=8, sample_interval=4, backtest_period=14).get_raw_data()
    sample_testing_period = pd.to_datetime(main_df["testing_period"].to_list()[-1])

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

    assert all(sample_set[0]['train_x'].index == sample_set[0]['train_y'].index)
    assert all(sample_set[0]['valid_x'].index == sample_set[0]['valid_y'].index)
    assert all(sample_set[0]['test_x'].index == sample_set[0]['test_y'].index)
