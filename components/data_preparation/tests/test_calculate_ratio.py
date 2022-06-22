import pandas as pd
from utils import backdate_by_day, str_to_date, dateNow


def test_get_tri():
    from components.data_preparation.src.calculation_ratio import get_tri
    df = get_tri(ticker=["AAPL.O"], start_date=str_to_date(backdate_by_day(5)))

    assert len(df) > 0
    assert df["trading_day"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Fri')[0]


def test_get_daily_fx_rate_df():
    from components.data_preparation.src.calculation_ratio import get_daily_fx_rate_df
    df = get_daily_fx_rate_df()

    assert len(df) > 0


def test_cleanStockReturn__get_consecutive_tri():
    from components.data_preparation.src.calculation_ratio import cleanStockReturn
    from utils import backdate_by_day, str_to_date

    calc = cleanStockReturn(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc._get_consecutive_tri(ticker=["ICF"])

    assert len(df) > 0
    assert df["trading_day"].max() > pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]


def test_cleanStockReturn__calc_rs_vol():
    from components.data_preparation.src.calculation_ratio import cleanStockReturn
    from utils import backdate_by_day, str_to_date

    calc = cleanStockReturn(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc._get_consecutive_tri(ticker=["ICF"])
    df_vol = calc._calc_rs_vol(df)

    assert len(df_vol) > 0
    assert "vol_0_30" in df_vol.columns.to_list()
    assert df["trading_day"].max() > pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]


def test_cleanStockReturn__calc_avg_day_tri():
    from components.data_preparation.src.calculation_ratio import cleanStockReturn
    from utils import backdate_by_day, str_to_date

    calc = cleanStockReturn(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc._get_consecutive_tri(ticker=["AAPL.O"])
    df = calc._calc_avg_day_tri(df)

    df_exist = df.loc[df[f'tri_avg_-7d'].notnull()]
    assert len(df_exist) > 0
    assert df["trading_day"].max() > pd.date_range(end=dateNow(), periods=2, freq='W-Sun')[0]


def test_cleanStockReturn_get_tri():
    from components.data_preparation.src.calculation_ratio import cleanStockReturn
    from utils import backdate_by_day, str_to_date

    calc = cleanStockReturn(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_tri_all_return(ticker=["AAPL.O"])

    df_y_exist = df.loc[df["stock_return_y_w4_d-7"].notnull()]
    assert len(df_y_exist) > 0
    assert df_y_exist["trading_day"].max() == pd.date_range(end=dateNow(), periods=5, freq='W-Sun')[0]


def test_calcStockReturn_get_all():
    from components.data_preparation.src.calculation_ratio import cleanStockReturn

    calc = cleanStockReturn(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_tri_all(ticker=["0005.HK"])

    df_y_exist = df.loc[df["stock_return_y_w4_d-7"].notnull()]
    assert len(df_y_exist) > 0
    assert df_y_exist["trading_day"].max() == pd.date_range(end=dateNow(), periods=5, freq='W-Sun')[0]


def test_cleanWorldscope_get_worldscope():
    from components.data_preparation.src.calculation_ratio import cleanWorldscope

    calc = cleanWorldscope(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_worldscope(ticker=["0700.HK"])

    assert len(df) > 0


def test_cleanIBES_get_ibes():
    from components.data_preparation.src.calculation_ratio import cleanIBES

    calc = cleanIBES(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_ibes(ticker=["0700.HK"])

    assert len(df) > 0


def test_combineData():
    from components.data_preparation.src.calculation_ratio import combineData

    calc = combineData(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_all(ticker=["0700.HK"])

    df_y_exist = df.loc[df["stock_return_y_w4_d-7"].notnull()]
    assert len(df_y_exist) > 0
    assert df_y_exist["trading_day"].max() == pd.date_range(end=dateNow(), periods=5, freq='W-Sun')[0]


def test_calc_factor_variables_get_all():
    from components.data_preparation.src.calculation_ratio import calcRatio
    from datetime import datetime

    calc_ratio_cls = calcRatio(start_date=datetime(2021, 1, 1, 0, 0, 0),
                               end_date=datetime.now(),
                               tri_return_only=False)
    df = calc_ratio_cls.get(('0700.HK', ))

    assert len(df) > 0
    assert df["trading_day"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]


def test_calc_factor_variables_tri_return_only():
    from components.data_preparation.src.calculation_ratio import calcRatio
    from datetime import datetime

    calc_ratio_cls = calcRatio(start_date=datetime(2021, 1, 1, 0, 0, 0),
                               end_date=datetime.now(),
                               tri_return_only=True)
    df = calc_ratio_cls.get(('0700.HK',))

    assert len(df) > 0
    assert df["trading_day"].max() == pd.date_range(end=dateNow(), periods=5, freq='W-Sun')[0]
    assert all(['stock_return_y' in x for x in df["field"].unique()])


def test_calc_factor_variables_index():
    from components.data_preparation.src.calculation_ratio import calcRatio
    from datetime import datetime

    calc_ratio_cls = calcRatio(start_date=datetime(2021, 1, 1, 0, 0, 0),
                               end_date=datetime.now(),
                               tri_return_only=False)
    df = calc_ratio_cls.get(('.SPX',))

    assert len(df) > 0
    assert df["trading_day"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]
    assert "stock_return_r1_0" in df["field"].unique()
    assert "vol_0_30" in df["field"].unique()


def test_calc_factor_variables_etf():
    from components.data_preparation.src.calculation_ratio import calcRatio
    from datetime import datetime

    calc_ratio_cls = calcRatio(start_date=datetime(2021, 1, 1, 0, 0, 0),
                               end_date=datetime.now(),
                               tri_return_only=False)
    df = calc_ratio_cls.get(('RODM.K',))

    assert len(df) > 0
    assert df["trading_day"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]
    assert "stock_return_r1_0" in df["field"].unique()
    assert "vol_0_30" in df["field"].unique()


def test_calc_factor_variables_multi():
    from components.data_preparation.src.calculation_ratio import calc_factor_variables_multi
    df = calc_factor_variables_multi(tickers=[".SXXGR"], processes=1)
    # df = calc_factor_variables_multi(tickers=["LIVN.O"], processes=1)

    assert len(df) > 0
    assert df["trading_day"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]


# tests ratio calculation missing rate
# test_missing(df, formula[['name','field_num','field_denom']], ingestion_cols)

# def test_missing(df_org, formula, ingestion_cols):
#     for group in ['USD']:
#         df = df_org.loc[df_org['currency_code'] == group]
#         writer = pd.ExcelWriter(f'missing_by_ticker_{group}.xlsx')
#
#         df = df.groupby('ticker').apply(lambda x: x.notnull().sum())
#         df.to_excel(writer, sheet_name='by ticker')
#
#         df_miss = df[ingestion_cols].unstack()
#         df_miss = df_miss.loc[df_miss == 0].reset_index()
#         df_miss.to_excel(writer, sheet_name='all_missing', index=False)
#         df_miss.to_csv(f'dsws_missing_ingestion_{group}.csv')
#
#         df_sum = pd.DataFrame(df.sum(0))
#         df_sum_df = df_sum.merge(formula, left_index=True, right_on=['name'], how='left')
#         for i in ['field_num', 'field_denom']:
#             df_sum_df = df_sum_df.merge(df_sum, left_on=[i], how='left', right_index=True)
#         df_sum_df.to_excel(writer, sheet_name='count', index=False)
#         df_sum.to_excel(writer, sheet_name='count_lst')
#
#         writer.save()