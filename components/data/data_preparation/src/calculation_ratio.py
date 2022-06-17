import datetime as dt
import numpy as np
import pandas as pd
from sqlalchemy import select, text
import multiprocessing as mp
from contextlib import suppress
from functools import partial
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, QuarterEnd
from typing import List
from contextlib import closing

from utils import (
    models,
    upsert_data_to_database,
    read_table,
    read_query,
    recreate_engine,
    to_slack,
    err2slack,
    sys_logger
)

logger = sys_logger(__name__, "INFO")

stock_data_table_tri = models.DataTri.__table__.schema + '.' + models.DataTri.__table__.name
stock_data_table_ohlcv = models.DataOhlcv.__table__.schema + '.' + models.DataOhlcv.__table__.name
universe_table = models.Universe.__table__.schema + '.' + models.Universe.__table__.name
latest_mktcap_data_table = models.LatestMktcap.__table__.schema + '.' + models.LatestMktcap.__table__.name

worldscope_data_table = models.DataWorldscope.__table__.schema + '.' + models.DataWorldscope.__table__.name
ibes_data_table = models.DataIbes.__table__.schema + '.' + models.DataIbes.__table__.name
currency_history_table = "currency_price_history"       # TODO: change to ORM

factors_formula_table = models.FormulaRatio.__table__.schema + '.' + models.FormulaRatio.__table__.name
ingestion_name_table = "ingestion_name"     # TODO: change to ORM


# ----------------------------------------- FX Conversion --------------------------------------------

def get_universe_map(col):
    """
    use Universe data to map static to tickers
    """

    df = read_query(f"SELECT ticker, {col} FROM {universe_table}")
    mapping = df.set_index("ticker")[col].to_dict()
    return mapping


def fill_all_day(df, id_col="ticker", date_col="trading_day"):
    """
    Fill all the weekends between first / last day and fill NaN
    """

    daily = pd.date_range(df[date_col].min(), df[date_col].max(), freq='D')
    indexes = pd.MultiIndex.from_product([df[id_col].unique(), daily], names=[id_col, date_col])

    # Insert weekend/before first trading date to df
    result = df.set_index([id_col, date_col]).reindex(indexes).reset_index()

    return result


def get_daily_fx_rate_df():
    """ Get daily FX -> USD rate and RENAME for easier merge """

    fx = read_table(currency_history_table)
    fx['last_date'] = pd.to_datetime(fx['last_date'])

    fx = fill_all_day(fx, id_col="currency_code", date_col='last_date')
    fx.loc[fx["currency_code"] == "USD", "last_price"] = 1
    fx['last_price'] = fx.sort_values(by=["currency_code", "last_date"]).groupby('currency_code')['last_price'].ffill()

    fx = fx.rename(columns={"currency_code": "_currency", "last_date": "trading_day", "last_price": "fx_rate"})

    return fx


def get_ingest_non_ratio_col():
    """
    no fx conversion for ratio items
    """

    ingestion_source = read_table(ingestion_name_table)
    ingest_non_ratio_col = ingestion_source.loc[ingestion_source['non_ratio'], "our_name"].to_list()

    return ingest_non_ratio_col


class convertFxData:
    """
    Convert all columns to USD for factor calculation (DSWS, WORLDSCOPE, IBES using different currency)
    """

    currency_code_ws_map = get_universe_map("currency_code_ws")
    currency_code_ibes_map = get_universe_map("currency_code_ibes")
    fx = get_daily_fx_rate_df()
    ingest_non_ratio_col = get_ingest_non_ratio_col()

    def convert_close(self, df):
        df["_currency"] = df["currency_code"]
        df = df.merge(self.fx, on=['_currency', 'trading_day'], how='inner')
        convert_cols = ["close"]
        df[convert_cols] = df[convert_cols].div(df['fx_rate'], axis="index")

        return df.drop(columns=["_currency", "fx_rate"])

    def convert_worldscope(self, df):
        """
        worldscope conversion based on FX rate as at reporting period_end
        """

        df["_currency"] = df["ticker"].map(self.currency_code_ws_map)
        df = df.merge(self.fx, left_on=['_currency', '_period_end'], right_on=['_currency', 'trading_day'], how='inner')
        convert_cols = df.filter(self.ingest_non_ratio_col).columns.to_list()
        df[convert_cols] = df[convert_cols].div(df['fx_rate'], axis="index")

        return df.drop(columns=["_currency", "fx_rate"])

    def convert_ibes(self, df):
        df["_currency"] = df["ticker"].map(self.currency_code_ibes_map)
        df = df.merge(self.fx, on=['_currency', 'trading_day'], how='inner')
        convert_cols = df.filter(self.ingest_non_ratio_col).columns.to_list()
        df[convert_cols] = df[convert_cols].div(df['fx_rate'], axis="index")

        return df.drop(columns=["_currency", "fx_rate"])


# ----------------------------------------- Calculate Stock Ralated Factors --------------------------------------------


def get_tri(ticker=None, start_date=None):
    """
    get stock price data from data_dss & data_dsws
    """

    tri_start_date = (start_date - relativedelta(years=2)).strftime("%Y-%m-%d")

    query = text(
        f"SELECT T.ticker, T.trading_day, currency_code, total_return_index as tri, open, high, low, close, volume, M.value as market_cap "
        f"FROM {stock_data_table_tri} T "
        f"INNER JOIN {stock_data_table_ohlcv} C ON (T.ticker = C.ticker) AND (T.trading_day = C.trading_day) "
        f"INNER JOIN {universe_table} U ON T.ticker = U.ticker "
        f"INNER JOIN {latest_mktcap_data_table} M ON T.ticker = M.ticker "
        f"WHERE T.ticker in {tuple(ticker)} AND T.trading_day>='{tri_start_date}' "
        f"ORDER BY T.ticker, T.trading_day".replace(",)", ")"))
    tri = read_query(query)

    tri['trading_day'] = pd.to_datetime(tri['trading_day'])
    return tri


def get_rogers_satchell(tri, list_of_start_end, days_in_year=256):
    """
    Calculate roger satchell volatility:
    daily = average over period from start to end: Log(High/Open)*Log(High/Close)+Log(Low/Open)*Log(Open/Close)
    annualized = sqrt(daily*256)
    """

    open_data, high_data, low_data, close_data = tri['open'].values, tri['high'].values, tri['low'].values, tri[
        'close'].values

    # Calculate daily volatility
    hc_ratio = np.divide(high_data, close_data)
    log_hc_ratio = np.log(hc_ratio.astype(float))
    ho_ratio = np.divide(high_data, open_data)
    log_ho_ratio = np.log(ho_ratio.astype(float))
    lo_ratio = np.divide(low_data, open_data)
    log_lo_ratio = np.log(lo_ratio.astype(float))
    lc_ratio = np.divide(low_data, close_data)
    log_lc_ratio = np.log(lc_ratio.astype(float))

    input1 = np.multiply(log_hc_ratio, log_ho_ratio)
    input2 = np.multiply(log_lo_ratio, log_lc_ratio)
    sum_ = np.add(input1, input2)

    # Calculate annualize volatility
    for l in list_of_start_end:
        start, end = l[0], l[1]
        name_col = f'vol_{start}_{end}'
        tri[name_col] = sum_
        tri[name_col] = tri.groupby('ticker')[name_col].rolling(end - start, min_periods=1).mean().reset_index(drop=1)
        tri[name_col] = tri[name_col].apply(lambda x: np.sqrt(x * days_in_year))
        tri[name_col] = tri[name_col].shift(start)
        tri.loc[tri.groupby('ticker').head(end - 1).index, name_col] = np.nan  # y-1 ~ y0

    return tri


def get_skew(tri):
    """ Calculate past 1yr daily return skewness """

    tri["skew"] = tri['tri'] / tri.groupby('ticker')['tri'].shift(
        1) - 1  # update tri to 1d before (i.e. all stock ret up to 1d before)
    tri = tri.sort_values(by=['ticker', 'trading_day'])
    tri['skew'] = tri["skew"].rolling(365, min_periods=1).skew()
    tri.loc[tri.groupby('ticker').head(364).index, 'skew'] = np.nan  # y-1 ~ y0

    return tri


def resample_to_weekly(df, date_col):
    """ Resample to weekly stock tri """
    monthly = pd.date_range(min(df[date_col].to_list()), max(df[date_col].to_list()), freq='W-SUN')
    df = df.loc[df[date_col].isin(monthly)]
    return df


class cleanStockReturn:
    """
    Calcualte monthly stock return

    Parameters
    ----------
    ticker / restart / tri_return_only: refer to calc_factor_variables_multi()
    stock_return_map (Dict):
        dictionary of {"forward_weeks": [list of "average_days"]}
        "forward_weeks": number of weeks look into future for premium calculation (i.e. return in the next N weeks)
        "average_days": days in calculate rolling average tri for future returns calculation

    Returns
    -------
    DataFrame for price related returns / variables
    """

    stock_return_map = {4: [-7], 8: [-7], 26: [-7, -28], 52: [-7, -28]}
    rs_vol_start_end = [[0, 30]]  # , [30, 90], [90, 182]
    drop_col = set()
    ffill_col = set()

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def get_tri_return(self, ticker):
        self.drop_col = {'tri', 'open', 'high', 'low', 'close', 'volume', 'market_cap'}
        self.ffill_col = {'currency_code'}

        tri = self._get_consecutive_tri(ticker)
        tri = self._calc_avg_day_tri(tri)
        tri = self._ffill_set_columns(tri)
        tri = resample_to_weekly(tri, date_col='trading_day')   # Resample to weekly stock tri
        tri = self._calc_future_stock_returns(tri)              # as Y

        return self._clean_tri(tri)

    def get_tri_all(self, ticker):
        self.drop_col = {'tri', 'open', 'high', 'low'}
        self.ffill_col = {'currency_code', 'market_cap', 'tri', 'close', 'volume'}

        tri = self._get_consecutive_tri(ticker)
        tri = get_skew(tri)
        tri = self._calc_rs_vol(tri)
        tri = self._calc_avg_day_volume(tri)
        tri = self._calc_avg_day_tri(tri)
        tri = self._ffill_set_columns(tri)
        tri = resample_to_weekly(tri, date_col='trading_day')   # Resample to weekly stock tri
        tri = self._convert_market_cap_close(tri)
        tri = self._calc_future_stock_returns(tri)              # as Y
        tri = self._calc_historic_stock_returns(tri)

        tri = convertFxData().convert_close(tri)

        return self._clean_tri(tri)

    def _get_consecutive_tri(self, ticker):
        tri = get_tri(ticker, self.start_date)
        tri = tri.replace(0, np.nan)  # Remove all 0 since total_return_index not supposed to be 0
        tri = fill_all_day(tri)  # Add NaN record of tri for weekends
        tri = tri.sort_values(['ticker', 'trading_day'])
        return tri

    def _calc_rs_vol(self, tri):
        """
        Calculate RS volatility for 3-month & 6-month~2-month (before ffill)
        """

        logger.debug(f'Calculate RS volatility')
        tri = get_rogers_satchell(tri, self.rs_vol_start_end)
        self.ffill_col.update([f'vol_{l[0]}_{l[1]}' for l in self.rs_vol_start_end])

        return tri

    def _calc_avg_day_volume(self, tri):
        """
        resample tri using last week average as the proxy for monthly tri
        """

        logger.debug(f'Stock volume using last 7 days average / 91 days average ')
        tri[['volume']] = tri.groupby("ticker")[['volume']].rolling(7, min_periods=1).mean().reset_index(drop=1)
        tri['volume_3m'] = tri.groupby("ticker")['volume'].rolling(91, min_periods=1).mean().values
        tri['volume'] = tri['volume'] / tri['volume_3m']
        self.drop_col.add('volume_3m')

        return tri

    def _calc_avg_day_tri(self, tri):
        """
        calculate rolling average tri (before forward fill tri)
        """

        logger.debug(f'Stock tri rolling average ')
        avg_day_options = set([i for x in self.stock_return_map.values() for i in x if i != 1] + [7])
        for i in avg_day_options:
            if i > 0:
                tri[f'tri_avg_{i}d'] = tri.groupby("ticker")['tri'].rolling(i, min_periods=1).mean().reset_index(drop=1)
            else:
                tri[f'tri_avg_{i}d'] = tri.groupby("ticker")['tri'].rolling(-i, min_periods=1).mean().reset_index(drop=1)
                tri[f'tri_avg_{i}d'] = tri.groupby("ticker")[f'tri_avg_{i}d'].shift(i)
            self.drop_col.add(f'tri_avg_{i}d')

        return tri

    def _ffill_set_columns(self, tri):
        """
        Fill forward (-> holidays/weekends) + backward (<- first trading price)
        """

        tri.update(tri.groupby('ticker')[list(self.ffill_col)].fillna(method='ffill'))
        return tri

    def _convert_market_cap_close(self, tri):
        """
        update market_cap & close refer to tri for each period
        """

        tri['trading_day'] = pd.to_datetime(tri['trading_day'])
        anchor_idx = tri.dropna(subset=['tri']).groupby('ticker').trading_day.idxmax()
        tri.loc[anchor_idx, 'anchor_tri'] = tri['tri']
        tri['anchor_tri'] = tri.groupby("ticker")['anchor_tri'].ffill().bfill()
        tri['market_cap'] = tri['market_cap'] / tri['anchor_tri'] * tri['tri']
        tri['close'] = tri['close'] / tri['anchor_tri'] * tri['tri']
        self.drop_col.add('anchor_tri')

        return tri

    def _calc_future_stock_returns(self, tri):
        logger.debug(f'Calculate future stock returns')
        tri['tri_avg_1d'] = tri['tri']
        for fwd_week, avg_days in self.stock_return_map.items():
            for d in avg_days:
                tri["tri_y"] = tri.groupby('ticker')[f'tri_avg_{d}d'].shift(-fwd_week)
                tri[f"stock_return_y_w{fwd_week}_d{d}"] = (tri["tri_y"] / tri[f'tri_avg_{d}d']) ** (4 / fwd_week) - 1
        self.drop_col.update(['tri_y', 'tri_avg_1d'])
        return tri

    def _calc_historic_stock_returns(self, tri):

        logger.debug(f'Calculate past stock returns')
        tri["tri_1wb"] = tri.groupby('ticker')['tri_avg_1d'].shift(1)
        tri["tri_2wb"] = tri.groupby('ticker')['tri_avg_7d'].shift(2)
        tri["tri_1mb"] = tri.groupby('ticker')['tri_avg_7d'].shift(4)
        tri["tri_2mb"] = tri.groupby('ticker')['tri_avg_7d'].shift(8)
        tri['tri_6mb'] = tri.groupby('ticker')['tri_avg_7d'].shift(26)
        tri['tri_7mb'] = tri.groupby('ticker')['tri_avg_7d'].shift(30)
        tri['tri_12mb'] = tri.groupby('ticker')['tri_avg_7d'].shift(52)
        self.drop_col.update(['tri_1wb', 'tri_2wb', 'tri_1mb', 'tri_2mb', 'tri_6mb', 'tri_7mb', 'tri_12mb'])

        tri["stock_return_ww1_0"] = (tri["tri"] / tri["tri_1wb"]) - 1
        tri["stock_return_ww2_1"] = (tri["tri_1wb"] / tri["tri_2wb"]) - 1
        tri["stock_return_ww4_2"] = (tri["tri_2wb"] / tri["tri_1mb"]) - 1

        tri["stock_return_r1_0"] = (tri["tri"] / tri["tri_1mb"]) - 1
        tri["stock_return_r6_2"] = (tri["tri_2mb"] / tri["tri_6mb"]) - 1
        tri["stock_return_r12_7"] = (tri["tri_7mb"] / tri["tri_12mb"]) - 1

        return tri

    def _clean_tri(self, tri):
        tri = tri.drop(columns=list(self.drop_col))
        tri['trading_day'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')
        check_duplicates(tri)
        return tri


# -------------------------------------------- Calculate Fundamental Ratios --------------------------------------------

def drop_dup(df, col='trading_day'):
    ''' drop duplicate records for same identifier & fiscal period, keep the most complete records '''

    logger.debug(f'Drop duplicates in {worldscope_data_table} ')

    df['count'] = pd.isnull(df).sum(1)  # count the missing in each records (row)
    df = df.sort_values(['count']).drop_duplicates(subset=['ticker', col], keep='first')
    return df.drop('count', axis=1)


def check_duplicates(df):
    """
    check if dataframe has duplicated records on (ticker + trading_day)
    """

    df_duplicated = df.loc[df.duplicated(subset=['trading_day', 'ticker'], keep=False)]
    if len(df_duplicated) > 0:
        raise ValueError(f'Duplicate records founded: {df_duplicated}')


class cleanWorldscope:

    fiscal_year_end_map = get_universe_map("fiscal_year_end")

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def get_worldscope(self, ticker):
        ws = self._download_worldscope(ticker)
        ws = self._fill_missing_ws(ws)
        ws = self._convert_trading_day(ws)
        ws = drop_dup(ws)                       # drop duplicate and retain the most complete record
        check_duplicates(ws)
        return ws

    def _download_worldscope(self, ticker):
        ws_start_date = (self.start_date - relativedelta(years=2)).strftime("%Y-%m-%d")
        query_ws = f"SELECT * FROM {worldscope_data_table} " \
                   f"WHERE ticker in {tuple(ticker)} AND trading_day>='{ws_start_date}' ".replace(",)", ")")
        ws = read_query(query_ws)
        ws = ws.pivot(index=["ticker", "trading_day"], columns=["field"], values="value").reset_index()

        ws['trading_day'] = pd.to_datetime(ws['trading_day'], format='%Y-%m-%d')
        return ws

    def _fill_missing_ws(self, ws):
        """
        fill in missing values by calculating with existing data
        """

        logger.debug(f'Fill missing in {worldscope_data_table} ')
        with suppress(Exception):
            ws['net_debt'] = ws['net_debt'].fillna(ws['debt'] - ws['cash'])
        with suppress(Exception):
            ws['ttm_ebit'] = ws['ttm_ebit'].fillna(ws['ttm_pretax_income'] + ws['ttm_interest'])
        with suppress(Exception):
            ws['ttm_ebitda'] = ws['ttm_ebitda'].fillna(ws['ttm_ebit'] + ws['ttm_dda'])
        with suppress(Exception):
            ws['current_asset'] = ws['current_asset'].fillna(ws['total_asset'] - ws['ppe_net'])

        return ws

    def _convert_trading_day(self, ws):
        """
        map icb_sector, member_ric, trading_day -> last_year_end for each identifier + frequency_number * 3m
        """

        logger.debug(f'Update trading_day as actual/estimate reporting date ')
        ws = self.__get_quarter_year_from_org_trading_day(ws)
        ws = self.__get_clean_fiscal_year_end(ws)
        ws = self.__get_actual_period_end(ws)
        ws = convertFxData().convert_worldscope(ws)
        ws = self.__get_estimated_report_date(ws)

        drop_col = ws.filter(regex="^_").columns.to_list()
        ws = ws.drop(columns=drop_col)

        return ws

    def __get_quarter_year_from_org_trading_day(self, ws):
        ws["trading_day"] = pd.to_datetime(ws["trading_day"], format='%Y-%m-%d')
        ws["_year"] = pd.DatetimeIndex(ws["trading_day"]).year
        ws["_frequency_number"] = np.ceil(pd.DatetimeIndex(ws["trading_day"]).month / 3)
        ws = ws.drop(columns=["trading_day"])
        return ws

    def __get_clean_fiscal_year_end(self, ws):
        ws['_fiscal_year_end'] = ws["ticker"].map(self.fiscal_year_end_map)
        ws['_fiscal_year_end'] = ws['_fiscal_year_end'].replace("NA", np.nan)
        ws['_fiscal_year_end'] = (pd.to_datetime(ws['_fiscal_year_end'], format='%b') + MonthEnd(0)).dt.strftime('%m%d')
        return ws

    def __get_actual_period_end(self, ws):
        """
        find last fiscal year end for each company (ticker) and actual period_end (in terms of quarter end)
        """

        ws['_last_year_end'] = (ws['_year'].astype(int) - 1).astype(str) + ws['_fiscal_year_end']
        ws['_last_year_end'] = pd.to_datetime(ws['_last_year_end'], format='%Y%m%d')
        ws["_period_end"] = ws[['_last_year_end', "_frequency_number"]].apply(lambda x: x[0] + MonthEnd(x[1] * 3), axis=1)

        return ws

    def __get_estimated_report_date(self, ws):
        """
        report_date if not exist -> use period_end + 1Q
        """

        ws = ws.rename(columns={"report_date": "_report_date"})
        ws["_estimated_report_date"] = ws['_period_end'] + QuarterEnd(1)
        ws['_report_date'] = pd.to_datetime(ws['_report_date'], format='%Y%m%d')
        ws['_report_date'] = ws['_report_date'].fillna(ws["_estimated_report_date"])
        ws['trading_day'] = ws['_report_date'].mask(ws['_report_date'] < ws['_period_end'],
                                                    ws['_report_date'] + QuarterEnd(-1))
        return ws


class cleanIBES:

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def get_ibes(self, ticker):
        ibes = self._download_ibes(ticker)
        ibes = convertFxData().convert_ibes(ibes)
        check_duplicates(ibes)

        return ibes

    def _download_ibes(self, ticker):
        ws_start_date = (self.start_date - relativedelta(years=2)).strftime("%Y-%m-%d")
        query_ibes = f"SELECT * FROM {ibes_data_table} " \
                     f"WHERE ticker in {tuple(ticker)} AND trading_day>='{ws_start_date}' ".replace(",)", ")")
        ibes = read_query(query_ibes)
        ibes = ibes.pivot(index=["ticker", "trading_day"], columns=["field"], values="value").reset_index()

        ibes['trading_day'] = pd.to_datetime(ibes['trading_day'], format='%Y-%m-%d')
        return ibes


class cleanData(cleanStockReturn, cleanWorldscope, cleanIBES):
    """
    get clean data from different sources / tables
    """
    pass


def fill_all_given_date(result, ref):
    """
    Fill all the date based on given date_df (e.g. tri) to align for biweekly / monthly sampling
    """

    # Construct indexes for all date / ticker used in ref (reference dataframe)
    result['trading_day'] = pd.to_datetime(result['trading_day'], format='%Y-%m-%d')
    date_list = ref['trading_day'].unique()
    ticker_list = ref['ticker'].unique()
    indexes = pd.MultiIndex.from_product([ticker_list, date_list], names=['ticker', 'trading_day']).to_frame(
        index=False, name=['ticker', 'trading_day'])
    logger.debug(f"Fill for {len(ref['ticker'].unique())} ticker, {len(date_list)} date")

    # Insert weekend/before first trading date to df
    indexes['trading_day'] = pd.to_datetime(indexes['trading_day'])
    result = result.merge(indexes, on=['ticker', 'trading_day'], how='outer')
    result = result.sort_values(by=['ticker', 'trading_day'], ascending=True)
    result.update(result.groupby(['ticker']).fillna(method='ffill'))  # fill forward for date

    result = result.loc[(result['trading_day'].isin(date_list)) & (result['ticker'].isin(ticker_list))]
    result = result.drop_duplicates(subset=['trading_day', 'ticker'], keep='last')  # remove ibes duplicates

    return result


# def combine_stock_factor_data(ticker, start_date, end_date, tri_return_only):
    # """ This part do the following:
    #     1. import all data from DB refer to other functions
    #     2. combined stock_return, worldscope, ibes, macroeconomic tables """
    #
    # # 1. Stock return/volatility/volume
    # tri, stocks_col = calc_stock_return(ticker, start_date, end_date, tri_return_only)
    # tri['trading_day'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')
    # # check_duplicates(tri, 'tri')
    #
    # if tri_return_only:
    #     return tri, stocks_col
    # elif (len(ticker) == 1) and (ticker[0][0] == '.'):
    #     logger.warning(f"index [{ticker}] calculate stock_return ratios only")
    #     tri = pd.melt(tri, id_vars=['ticker', "trading_day"], value_vars=stocks_col, var_name="field",
    #                   value_name="value").dropna(subset=["value"])
    #     return tri, stocks_col
    # else:
    #     # 2. Fundamental financial data - from Worldscope
    #     # 3. Consensus forecasts - from I/B/E/S
    #     # 4. Universe
    #     ws, ibes, universe = download_clean_worldscope_ibes(ticker, start_date, end_date)
    #
    #     # align worldscope / ibes data with stock return date (monthly/biweekly)
    #     ws = fill_all_given_date(ws, tri)
    #     ibes = fill_all_given_date(ibes, tri)
    #
    #     check_duplicates(ws, 'worldscope')  # check if worldscope/ibes has duplicated records on ticker + trading_day
    #     check_duplicates(ibes, 'ibes')
    #
    #     # Use 6-digit ICB code in industry groups
    #     universe['industry_code'] = universe['industry_code'].replace('NA', np.nan).dropna().astype(int).astype(str). \
    #         replace({'10102010': '101021', '10102015': '101022', '10102020': '101023', '10102030': '101024',
    #                  '10102035': '101024'})  # split industry 101020 - software (100+ samples)
    #     universe['industry_code'] = universe['industry_code'].astype(str).str[:6]
    #
    #     # Combine all data for table (1) - (6) above
    #     logger.debug(f'Merge all dataframes ')
    #     df = pd.merge(tri, ws, on=['ticker', 'trading_day'], how='left', suffixes=('', '_ws'))
    #     df = df.merge(ibes, on=['ticker', 'trading_day'], how='left', suffixes=('', '_ibes'))
    #     df = df.sort_values(by=['ticker', 'trading_day'])
    #
    #     # Update close price to adjusted value
    #     def adjust_close(df):
    #         """ using market cap to adjust close price for stock split, ..."""
    #
    #         logger.debug(f'Adjust closing price with market cap ')
    #
    #         df = df[['ticker', 'trading_day', 'market_cap', 'close']].dropna(how='any')
    #         df['market_cap_latest'] = df.groupby(['ticker'])['market_cap'].transform('last')
    #         df['close_latest'] = df.groupby(['ticker'])['close'].transform('last')
    #         df['close'] = df['market_cap'] / df['market_cap_latest'] * df['close_latest']
    #
    #         return df[['ticker', 'trading_day', 'close']]
    #
    #     df.update(adjust_close(df))
    #
    #     # Forward fill for fundamental data
    #     cols = df.select_dtypes('float').columns.to_list()
    #     cols = [x for x in cols if not x.startswith("stock_return_y")]  # for stock_return_y -> no ffill
    #     df.update(df.groupby(['ticker'])[cols].fillna(method='ffill'))
    #     df = resample_to_weekly(df, date_col='trading_day')  # Resample to monthly stock tri
    #     df = df.merge(universe, on=['ticker'], how='left',
    #                   suffixes=('_old', ''))  # label industry_code, currency_code for each ticker
    #     check_duplicates(df, 'final')
    #
    #     return df, stocks_col

def get_industry_code_map():
    """ Use 6-digit ICB code in industry groups """

    df = read_query(f"SELECT ticker, industry_code FROM {universe_table}")

    df['industry_code'] = df['industry_code'].replace('NA', np.nan).dropna().astype(int).astype(str)
    df = df.replace({'10102010': '101021', '10102015': '101022', '10102020': '101023', '10102030': '101024',
                     '10102035': '101024'})             # split industry 101020 - software (100+ samples)
    df['industry_code'] = df['industry_code'].astype(str).str[:6]

    return df.set_index("ticker")["industry_code"].to_dict()


class combineData:

    industry_code_map = get_industry_code_map()

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        pass

    def get_tri(self, ticker):
        """
        Get tri_return only
        """

        tri = cleanData(self.start_date, self.end_date).get_tri_return(ticker)
        return tri

    def get_all(self, ticker):

        data_cls = cleanData(self.start_date, self.end_date)

        tri = data_cls.get_tri_all(ticker).set_index(["ticker", "trading_day"])
        ws = data_cls.get_worldscope(ticker).set_index(["ticker", "trading_day"])
        ibes = data_cls.get_ibes(ticker).set_index(["ticker", "trading_day"])

        df = self._merge_data(tri, ws, ibes)
        df = self._ffill_fundamental(df)
        df = df.reindex(tri.index)

        return df.reset_index()

    def _merge_data(self, tri, ws, ibes):
        logger.debug(f'Merge all dataframes')
        df = tri.merge(ws, left_index=True, right_index=True, how='outer', suffixes=('', '_ws'))
        df = df.merge(ibes, left_index=True, right_index=True, how='outer', suffixes=('', '_ibes'))
        df = df.sort_values(by=['ticker', 'trading_day'])

        return df

    def _ffill_fundamental(self, df):
        """
        TRI is weekly, but IBES / Worldscope is monthly / quarter. Forward fill to make up for missing dates.
        """

        cols = df.select_dtypes('float').columns.to_list()
        cols = [x for x in cols if not x.startswith("stock_return_y")]  # for stock_return_y -> no ffill
        df[cols] = df.groupby(['ticker'])[cols].ffill()

        return df


class calcRatio:
    """ Calculate all factor used referring to DB ratio table """

    formula = read_query(select(models.FormulaRatio).where(models.FormulaRatio.is_active == True))
    etf_list = read_query(select(models.Universe.ticker).where(models.Universe.is_etf == True))["ticker"].to_list()

    def __init__(self, start_date: dt.datetime = None, end_date: dt.datetime = None, tri_return_only: bool = False):
        self.start_date = start_date
        self.end_date = end_date
        self.tri_return_only = tri_return_only
        self.raw_data = combineData(self.start_date, self.end_date)

    def _error_state(self, ticker):
        status_df = pd.DataFrame({"trading_day": dt.datetime.now(), "field": "status", "value": 0, "ticker": ticker},
                                 index=[0])
        return status_df

    @err2slack("clair")
    def get(self, *args):
        ticker = args[0]

        logger.info(f'=== (n={len(ticker)}) Calculate ratio for {ticker}  ===')

        if self.tri_return_only:
            df = self.raw_data.get_tri(ticker)
        elif all([x[0] == '.' for x in ticker]):        # index
            df = self.raw_data.get_tri(ticker)
        elif all([x in self.etf_list for x in ticker]):
            df = self.raw_data.get_tri(ticker)
        else:
            df = self.raw_data.get_all(ticker)
            df = self._calc_add_minus_fields(df)
            df = self._calculate_keep_ratios(df)
            df = self._calculate_ts_ratios(df)
            df = self._calculate_divide_ratios(df)
            df = self._clean_missing_ratio_records(df)

        df = self._clean_missing_y_records(df)
        df = self._reformat_df(df)

        return df

    def _calc_add_minus_fields(self, df):
        """
        Prepare for field requires add/minus
        """

        add_minus_fields = self.formula[['field_num', 'field_denom']].dropna(how='any').to_numpy().flatten()
        add_minus_fields = [i for i in list(set(add_minus_fields)) if any(['-' in i, '+' in i, '*' in i])]

        for i in add_minus_fields:
            x = [op.strip() for op in i.split()]
            if x[0] in "*+-": raise Exception("Invalid formula")
            n = 1
            try:
                temp = df[x[0]].copy()
            except:
                temp = np.empty((len(df), 1))
            while n < len(x):
                try:
                    if x[n] == '+':
                        temp += df[x[n + 1]].replace(np.nan, 0)
                    elif x[n] == '-':
                        temp -= df[x[n + 1]].replace(np.nan, 0)
                    elif x[n] == '*':
                        temp *= df[x[n + 1]]
                    else:
                        raise Exception(f"Unexpected operand/operator: {x[n]}")
                except Exception as e:
                    logger.warning(f"[Warning] add_minus_fields not calculate: {add_minus_fields}: {e}")
                n += 2
            df[i] = temp

        return df

    def _calculate_keep_ratios(self, df):
        """
        Ratios Keep original values = just rename
        """

        keep_original_mask = self.formula['field_denom'].isnull() & self.formula['field_num'].notnull()
        logger.debug(f'Calculate keep ratio')
        for new_name, old_name in self.formula.loc[keep_original_mask, ['name', 'field_num']].to_numpy():
            try:
                df[new_name] = df[old_name]
            except Exception as e:
                logger.warning(f"[Warning] Factor ratio [{new_name}] not calculate: {e}")
        return df

    def _calculate_ts_ratios(self, df, period_yr = 52, period_q = 12):
        """
        Time series ratios (Calculate 1m change first)
        """

        logger.debug(f'Calculate time-series ratio')
        for r in self.formula.loc[self.formula['field_num'] == self.formula['field_denom'], ['name', 'field_denom']].to_dict(
                orient='records'):  # minus calculation for ratios
            try:
                if r['name'][-2:] == 'yr':
                    df[r['name']] = df[r['field_denom']] / df[r['field_denom']].shift(period_yr) - 1
                    df.loc[df.groupby('ticker').head(period_yr).index, r['name']] = np.nan
                elif r['name'][-1] == 'q':
                    df[r['name']] = df[r['field_denom']] / df[r['field_denom']].shift(period_q) - 1
                    df.loc[df.groupby('ticker').head(period_q).index, r['name']] = np.nan
            except Exception as e:
                logger.warning(f"[Warning] Factor ratio [{r['name']}] not calculate: {e}")
        return df

    def _calculate_divide_ratios(self, df):
        """ Divide ratios = field A / field B """

        logger.debug(f'Calculate dividing ratios ')
        for r in self.formula.dropna(how='any', axis=0).loc[(self.formula['field_num'] != self.formula['field_denom'])].to_dict(
                orient='records'):  # minus calculation for ratios
            try:
                df[r['name']] = df[r['field_num']] / df[r['field_denom']].replace(0, np.nan)
            except Exception as e:
                logger.warning(f"[Warning] Factor ratio [{r['name']}] not calculate: {e}")
        return df

    def _clean_missing_ratio_records(self, df):
        """
        drop records without any ratios
        """

        dropna_col = set(df.columns.to_list()) & set(self.formula['name'].to_list())
        df = df.dropna(subset=list(dropna_col), how='all')
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _clean_missing_y_records(self, df):
        """
        drop records before stock price return is first available;
        latest records will also have missing stock_return, but we will keep the records.
        """

        y_col = [x for x in df.columns.to_list() if x.startswith("stock_return_y")]
        df[[x + '_ffill' for x in y_col]] = df.groupby('ticker')[y_col].ffill()
        df = df.dropna(subset=[x + '_ffill' for x in y_col], how='any')
        df = df.filter(['ticker', 'trading_day'] + y_col + self.formula['name'].to_list())

        return df

    def _reformat_df(self, df):
        """
        filter for time period needed & melt table into (ticker, trading_day, field, value) format
        """

        df = df.loc[(df["trading_day"] >= self.start_date) & (df["trading_day"] <= self.end_date)]
        df = pd.melt(df, id_vars=['ticker', "trading_day"], var_name="field", value_name="value").dropna(
            subset=["value"])
        return df


def calc_factor_variables_multi(tickers: List[str] = None, currency_codes: List[str] = None,
                                start_date: dt.datetime = None, end_date: dt.datetime = None,
                                tri_return_only: bool = False, processes: int = 1):
    """
    Calculate weekly ratios for all factors

    Parameters
    ----------
    tickers (List[Str], Optional):
        tickers to calculate variables (default=None, i.e., calculate for all active tickers).
    currency_codes (List[Str], Optional):
        tickers in which currency to calculate variables (default=None).
    start_date / end_date (Datetime, Optional):
        time period to update ratio table (default = past 3 month).
    tri_return_only (Bool, Optional, default=False):
        if True, only write for 'stock_return_y_%' columns; this will also force restart=True.
    processes (Int, Optional, default=1):
        number of processes in multiprocessing.
    """

    # get list of active tickers to calculate ratios
    conditions = ["is_active"]
    if type(tickers) != type(None):
        conditions.append(f"ticker in {tuple(tickers)}")
    if type(currency_codes) != type(None):
        conditions.append(f"currency_code in {tuple(currency_codes)}")
    ticker_query = f"SELECT ticker FROM universe WHERE {' AND '.join(conditions)}".replace(",)", ")")
    tickers = read_query(ticker_query)["ticker"].to_list()

    # define start_date / end_date for AI score
    if type(end_date) == type(None):
        end_date = dt.datetime.now()
    if type(start_date) == type(None):
        start_date = end_date - relativedelta(months=3)

    # multiprocessing
    tickers = [tuple([[e]]) for e in tickers]
    with closing(mp.Pool(processes=processes, initializer=recreate_engine)) as pool:
        calc_ratio_cls = calcRatio(start_date, end_date, tri_return_only)
        df = pool.starmap(calc_ratio_cls.get, tickers)
    df = pd.concat([x for x in df if type(x) != type(None)], axis=0)
    # df = calc_factor_variables(tickers, restart=restart, tri_return_only=tri_return_only)

    # save calculated ratios to DB (remove truncate -> everything update)
    db_table_name = models.PreprocessRatio.__tablename__
    upsert_data_to_database(df, db_table_name, how="update")

    return df, db_table_name