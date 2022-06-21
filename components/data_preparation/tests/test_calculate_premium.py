import pandas as pd
from utils import dateNow


def test_download_pivot_ratios():
    from components.data_preparation.src.calculation_premium import calcPremium
    df = calcPremium(weeks_to_expire=8, average_days_list=[-7], weeks_to_offset=4, processes=1, currency_code_list=["USD"]
                     )._download_pivot_ratios()

    assert len(df) > 0
    assert df["testing_period"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]


def test_clean_download_pivot_table():
    from components.data_preparation.src.calculation_premium import calcPremium

    cls = calcPremium(weeks_to_expire=8, average_days_list=[-7], weeks_to_offset=4, currency_code_list=["USD"])
    df = cls._download_pivot_ratios()
    factor_df = cls._filter_factor_df("USD", "fwd_ey", "stock_return_y_w8_d-7", ratio_df=df)

    assert len(factor_df) > 0
    assert factor_df["testing_period"].max() == pd.date_range(end=dateNow(), periods=9, freq='W-Sun')[0]


def test_filter_factor_df():
    from components.data_preparation.src.calculation_premium import calcPremium

    cls = calcPremium(weeks_to_expire=8, average_days_list=[-7], weeks_to_offset=4, currency_code_list=["USD"])
    df = cls._download_pivot_ratios()
    factor_df = cls._filter_factor_df("USD", "stock_return_r1_0", "stock_return_y_w8_d-7", ratio_df=df)

    assert len(factor_df) > 0
    assert factor_df["testing_period"].max() == pd.date_range(end=dateNow(), periods=9, freq='W-Sun')[0]

    recent_df = factor_df.loc[factor_df["testing_period"] == factor_df["testing_period"].max()]
    assert len(recent_df) >= cls.min_group_size


def test_calcPremium():
    from components.data_preparation.src.calculation_premium import calcPremium
    df = calcPremium(weeks_to_expire=8, average_days_list=[-7], weeks_to_offset=4, currency_code_list=["USD"]).write_all()

    assert len(df) > 0
    assert df["testing_period"].max() == pd.date_range(end=dateNow(), periods=9, freq='W-Sun')[0]

