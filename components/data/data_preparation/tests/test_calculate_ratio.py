def test_get_tri():
    from components.data.data_preparation.src.calculation_ratio import get_tri
    from utils import backdate_by_day, str_to_date
    df = get_tri(ticker=["AAPL.O"], start_date=str_to_date(backdate_by_day(5)))

    assert len(df) > 0


def test_get_daily_fx_rate_df():
    from components.data.data_preparation.src.calculation_ratio import get_daily_fx_rate_df
    df = get_daily_fx_rate_df()

    assert len(df) > 0


def test_cleanStockReturn_get_tri():
    from components.data.data_preparation.src.calculation_ratio import cleanStockReturn
    from utils import backdate_by_day, str_to_date

    calc = cleanStockReturn(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_tri_return(ticker=["AAPL.O"])

    assert len(df) > 0


def test_calcStockReturn_get_all():
    from components.data.data_preparation.src.calculation_ratio import cleanStockReturn
    from utils import backdate_by_day, str_to_date

    calc = cleanStockReturn(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_tri_all(ticker=["0005.HK"])

    assert len(df) > 0


def test_cleanWorldscope_get_worldscope():
    from components.data.data_preparation.src.calculation_ratio import cleanWorldscope
    from utils import backdate_by_day, str_to_date

    calc = cleanWorldscope(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_worldscope(ticker=["0700.HK"])

    assert len(df) > 0


def test_cleanIBES_get_ibes():
    from components.data.data_preparation.src.calculation_ratio import cleanIBES
    from utils import backdate_by_day, str_to_date

    calc = cleanIBES(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_ibes(ticker=["0700.HK"])

    assert len(df) > 0


def test_combineData():
    from components.data.data_preparation.src.calculation_ratio import combineData
    from utils import backdate_by_day, str_to_date

    calc = combineData(start_date=str_to_date(backdate_by_day(0)), end_date=str_to_date(backdate_by_day(0)))
    df = calc.get_all(ticker=["0700.HK"])

    assert len(df) > 0


def test_calc_factor_variables_get_all():
    from components.data.data_preparation.src.calculation_ratio import calcRatio
    from utils import backdate_by_month, str_to_date

    calc_ratio_cls = calcRatio(start_date=str_to_date(backdate_by_month(6)),
                               end_date=str_to_date(backdate_by_month(0)),
                               tri_return_only=False)
    df = calc_ratio_cls.get(('0700.HK', ))

    assert len(df) > 0


def test_calc_factor_variables_tri_return_only():
    from components.data.data_preparation.src.calculation_ratio import calcRatio
    from utils import backdate_by_month, str_to_date

    calc_ratio_cls = calcRatio(start_date=str_to_date(backdate_by_month(6)),
                               end_date=str_to_date(backdate_by_month(0)),
                               tri_return_only=True)
    df = calc_ratio_cls.get(('0700.HK',))

    assert len(df) > 0


def test_calc_factor_variables_index():
    from components.data.data_preparation.src.calculation_ratio import calcRatio
    from utils import backdate_by_month, str_to_date

    calc_ratio_cls = calcRatio(start_date=str_to_date(backdate_by_month(6)),
                               end_date=str_to_date(backdate_by_month(0)),
                               tri_return_only=True)
    df = calc_ratio_cls.get(('0700.HK',))

    assert len(df) > 0


# tests ratio calculation missing rate
# test_missing(df, formula[['name','field_num','field_denom']], ingestion_cols)

def test_calc_factor_variables_multi():
    from components.data.data_preparation.src.calculation_ratio import calc_factor_variables_multi
    df, db_table_name = calc_factor_variables_multi(tickers=["AAPL.O"])

    assert len(df) > 0