import pandas as pd
from utils import dateNow


# def test_download_pivot_ratios():
#     from components.data_preparation.src.calculation_premium import CalcPremium
#     df = CalcPremium(weeks_to_expire=8, average_days_list=[-7], weeks_to_offset=4, processes=1, currency_code_list=["USD"]
#                      )._download_pivot_ratios()
#
#     assert len(df) > 0
#     assert df["testing_period"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]
#
#
# def test_clean_download_pivot_table():
#     from components.data_preparation.src.calculation_premium import CalcPremium
#
#     cls = CalcPremium(weeks_to_expire=8, average_days_list=[-7], weeks_to_offset=4, currency_code_list=["USD"])
#     df = cls._download_pivot_ratios()
#     factor_df = cls._filter_factor_df("USD", "fwd_ey", "stock_return_y_w8_d-7", ratio_df=df)
#
#     assert len(factor_df) > 0
#     assert factor_df["testing_period"].max() == pd.date_range(end=dateNow(), periods=9, freq='W-Sun')[0]
#
#
# def test_filter_factor_df():
#     from components.data_preparation.src.calculation_premium import CalcPremium
#
#     cls = CalcPremium(weeks_to_expire=8, average_days_list=[-7], weeks_to_offset=4, currency_code_list=["USD"])
#     df = cls._download_pivot_ratios()
#     factor_df = cls._filter_factor_df("USD", "stock_return_r1_0", "stock_return_y_w8_d-7", ratio_df=df)
#
#     assert len(factor_df) > 0
#     assert factor_df["testing_period"].max() == pd.date_range(end=dateNow(), periods=9, freq='W-Sun')[0]
#
#     recent_df = factor_df.loc[factor_df["testing_period"] == factor_df["testing_period"].max()]
#     assert len(recent_df) >= cls.min_group_size
#
#
# def test_CalcPremium_single_factor():
#     from components.data_preparation.src.calculation_premium import CalcPremium
#     df = CalcPremium(weeks_to_expire=8, average_days_list=[-7], weeks_to_offset=4,
#                      currency_code_list=["EUR"], factor_list=["gross_margin"]).write_all()
#
#     assert len(df) > 0
#
#     from utils import backdate_by_day
#     end_date = pd.date_range(end=pd.date_range(end=backdate_by_day(1), periods=1, freq='W-MON')[0], periods=2, freq='8W-SUN')[0]
#     assert df["testing_period"].max() == end_date

def test_CalcPremium_one_field():
    from components.data_preparation.src.calculation_premium import CalcPremium
    df = CalcPremium(weeks_to_expire=26, average_days_list=[-7],
                     weeks_to_offset=1, currency_code_list=["EUR"],
                     factor_list=["market_cap_usd"]).write_all()

    assert len(df) > 0
    assert df["testing_period"].max() == pd.date_range(
        end=dateNow(), periods=9, freq='W-Sun')[0]


def test_CalcPremium():
    from components.data_preparation.src.calculation_premium import CalcPremium
    df = CalcPremium(weeks_to_expire=8, average_days_list=[-7],
                     weeks_to_offset=4, currency_code_list=["EUR"]).write_all()

    assert len(df) > 0
    assert df["testing_period"].max() == pd.date_range(
        end=dateNow(), periods=9, freq='W-Sun')[0]

