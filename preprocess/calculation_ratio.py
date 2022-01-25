import logging
import datetime as dt
import numpy as np
import pandas as pd
from sqlalchemy import text
from global_vars import *
import multiprocessing as mp
from contextlib import suppress

from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, QuarterEnd
from general.sql_process import upsert_data_to_database, read_table, read_query, uid_maker, trucncate_table_in_database
from general.report_to_slack import to_slack

# ----------------------------------------- Calculate Stock Ralated Factors --------------------------------------------

def get_tri(ticker=None, restart=True):
    ''' get stock price data from data_dss & data_dsws '''

    if restart:
        start_date = dt.datetime(1998,1,1)
    else:   # if not restart only from 1yr ago
        start_date = (dt.datetime.today() - relativedelta(months=6)).strftime("%Y-%m-%d")

    query = text(f"SELECT T.ticker, T.trading_day, currency_code, total_return_index as tri, open, high, low, close, volume "
                 f"FROM {stock_data_table_tri} T "
                 f"INNER JOIN {stock_data_table_ohlc} C ON T.uid = C.uid "
                 f"INNER JOIN {universe_table} U ON T.ticker = U.ticker "
                 f"WHERE T.ticker in {tuple(ticker)} AND T.trading_day>='{start_date}' "
                 f"ORDER BY T.ticker, T.trading_day".replace(",)",")"))
    tri = read_query(query, db_url_read)

    query2 = f"SELECT * FROM {anchor_table_mkt_cap} " \
             f"WHERE field='mkt_cap' AND ticker in {tuple(ticker)} AND trading_day>='{start_date}' " \
             f"ORDER BY ticker, trading_day".replace(",)",")")
    market_cap_anchor = read_query(query2, db_url_read)
    market_cap_anchor = market_cap_anchor.pivot(index=["ticker","trading_day"], columns=["field"], values="value").reset_index()

    return tri, market_cap_anchor

def fill_all_day(result, date_col="trading_day"):
    ''' Fill all the weekends between first / last day and fill NaN'''

    # Construct indexes for all day between first/last day * all ticker used
    df = result[["ticker", date_col]].copy()
    df.trading_day = pd.to_datetime(df[date_col])
    result.trading_day = pd.to_datetime(result[date_col])
    df = df.sort_values(by=[date_col], ascending=True)
    daily = pd.date_range(df.iloc[0, 1], df.iloc[-1, 1]+relativedelta(days=6), freq='D')
    indexes = pd.MultiIndex.from_product([df['ticker'].unique(), daily], names=['ticker', date_col])

    # Insert weekend/before first trading date to df
    df = df.set_index(['ticker', date_col]).reindex(indexes).reset_index()
    df = df.sort_values(by=['ticker', date_col], ascending=True)
    result = df.merge(result, how="left", on=["ticker", date_col])

    return result

def get_rogers_satchell(tri, list_of_start_end, days_in_year=256):
    ''' Calculate roger satchell volatility:
        daily = average over period from start to end: Log(High/Open)*Log(High/Close)+Log(Low/Open)*Log(Open/Close)
        annualized = sqrt(daily*256)
    '''

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
    ''' Calculate past 1yr daily return skewness '''

    tri["skew"] = tri['tri']/tri.groupby('ticker')['tri'].shift(1)-1       # update tri to 1d before (i.e. all stock ret up to 1d before)
    tri = tri.sort_values(by=['ticker','trading_day'])
    tri['skew'] = tri["skew"].rolling(365, min_periods=1).skew()
    tri.loc[tri.groupby('ticker').head(364).index, 'skew'] = np.nan  # y-1 ~ y0

    return tri

def resample_to_weekly(df, date_col):
    ''' Resample to bi-weekly stock tri '''
    monthly = pd.date_range(min(df[date_col].to_list()), max(df[date_col].to_list()),  freq='W')
    df = df.loc[df[date_col].isin(monthly)]
    return df

def calc_stock_return(ticker, restart):
    ''' Calcualte monthly stock return '''

    tri, market_cap_anchor = get_tri(ticker, restart)
    market_cap_anchor = market_cap_anchor.loc[market_cap_anchor['ticker'].isin(tri['ticker'].unique())]

    # merge stock return from DSS & from EIKON (i.e. longer history)
    tri['trading_day'] = pd.to_datetime(tri['trading_day'])

    # x = tri.loc[tri['ticker']=='AAPL.O'].sort_values(by=['trading_day'], ascending=False)
    tri = tri.replace(0, np.nan)  # Remove all 0 since total_return_index not supposed to be 0
    tri = fill_all_day(tri)  # Add NaN record of tri for weekends
    tri = tri.sort_values(['ticker','trading_day'])

    logging.info(f'Calculate skewness ')
    tri = get_skew(tri)    # Calculate past 1 year skewness

    # Calculate RS volatility for 3-month & 6-month~2-month (before ffill)
    logging.info(f'Calculate RS volatility ')
    list_of_start_end = [[0, 30]] # , [30, 90], [90, 182]
    tri = get_rogers_satchell(tri, list_of_start_end)
    tri = tri.drop(['open', 'high', 'low'], axis=1)

    # resample tri using last week average as the proxy for monthly tri
    logging.info(f'Stock volume/tri using last 7 days average ')
    tri[['volume', 'tri_7d_avg']] = tri.groupby("ticker")[['volume', 'tri']].rolling(7, min_periods=1).mean().reset_index(drop=1)
    tri['volume_3m'] = tri.groupby("ticker")['volume'].rolling(91, min_periods=1).mean().values
    tri['volume'] = tri['volume'] / tri['volume_3m']
    tri.to_csv('test_tri_aapl_before_resample.csv')

    # Fill forward (-> holidays/weekends) + backward (<- first trading price)
    cols = ['tri', 'close','volume'] + [f'vol_{l[0]}_{l[1]}' for l in list_of_start_end]
    tri.update(tri.groupby('ticker')[cols].fillna(method='ffill'))

    logging.info(f'Sample weekly interval ')
    tri = resample_to_weekly(tri, date_col='trading_day')  # Resample to weekly stock tri

    # update market_cap/market_cap_usd refer to tri for each period
    market_cap_anchor = market_cap_anchor.set_index('ticker')['mkt_cap'].to_dict()      # use mkt_cap from fundamental score
    tri['trading_day'] = pd.to_datetime(tri['trading_day'])
    anchor_idx = tri.dropna(subset=['tri']).groupby('ticker').trading_day.idxmax()
    tri.loc[anchor_idx, 'market_cap'] = tri.loc[anchor_idx, 'ticker'].map(market_cap_anchor)
    tri.loc[tri['market_cap'].notnull(), 'anchor_tri'] = tri.loc[tri['market_cap'].notnull(), 'tri']
    tri[['anchor_tri','market_cap']] = tri.groupby('ticker')[['anchor_tri','market_cap']].apply(lambda x: x.ffill().bfill())
    tri['market_cap'] = tri['market_cap']/tri['anchor_tri']*tri['tri']
    tri = tri.drop(['anchor_tri'], axis=1)

    # Calculate monthly return (Y) + R6,2 + R12,7
    logging.info(f'Calculate stock returns ')
    for rolling_period in [1, 4]:
        if rolling_period==1:
            y_base_col = 'tri'
        elif rolling_period==4:
            y_base_col = 'tri_7d_avg'
        tri["tri_y"] = tri.groupby('ticker')[y_base_col].shift(-rolling_period)
        tri[f"stock_return_y_{rolling_period}week"] = (tri["tri_y"] / tri[y_base_col]) - 1
        tri[f"stock_return_y_{rolling_period}week"] = tri[f"stock_return_y_{rolling_period}week"]*4/rolling_period
    # tri.to_csv('test_tri_aapl.csv')
    tri["tri_1wb"] = tri.groupby('ticker')['tri'].shift(1)
    tri["tri_2wb"] = tri.groupby('ticker')['tri_7d_avg'].shift(2)
    tri["tri_1mb"] = tri.groupby('ticker')['tri_7d_avg'].shift(4)
    tri["tri_2mb"] = tri.groupby('ticker')['tri_7d_avg'].shift(8)
    tri['tri_6mb'] = tri.groupby('ticker')['tri_7d_avg'].shift(26)
    tri['tri_7mb'] = tri.groupby('ticker')['tri_7d_avg'].shift(30)
    tri['tri_12mb'] = tri.groupby('ticker')['tri_7d_avg'].shift(52)
    drop_col = ['tri_1wb', 'tri_2wb', 'tri_1mb', 'tri_2mb', 'tri_6mb', 'tri_7mb', 'tri_12mb']

    tri["stock_return_ww1_0"] = (tri["tri"] / tri["tri_1wb"]) - 1
    tri["stock_return_ww2_1"] = (tri["tri_1wb"] / tri["tri_2wb"]) - 1
    tri["stock_return_ww4_2"] = (tri["tri_2wb"] / tri["tri_1mb"]) - 1

    tri["stock_return_r1_0"] = (tri["tri"] / tri["tri_1mb"]) - 1
    tri["stock_return_r6_2"] = (tri["tri_2mb"] / tri["tri_6mb"]) - 1
    tri["stock_return_r12_7"] = (tri["tri_7mb"] / tri["tri_12mb"]) - 1

    tri = tri.drop(['tri', 'tri_y', 'tri_7d_avg'] + drop_col, axis=1)
    stock_col = tri.select_dtypes('float').columns  # all numeric columns

    return tri, stock_col

# -------------------------------------------- Calculate Fundamental Ratios --------------------------------------------

def download_clean_worldscope_ibes(ticker, restart):
    ''' download all data for factor calculate & LGBM input (except for stock return) '''

    if restart:
        start_date = dt.datetime(1998,1,1)
    else:   # if not restart only from 1yr ago
        start_date = (dt.datetime.today() - relativedelta(years=2)).strftime("%Y-%m-%d")

    query_ws = f"select * from {worldscope_quarter_summary_table} " \
               f"WHERE ticker in {tuple(ticker)} AND trading_day>='{start_date}' ".replace(",)",")")
    ws = read_query(query_ws, db_url_read)
    ws = ws.pivot(index=["ticker","trading_day"], columns=["field"], values="value").reset_index()

    query_ibes = f"SELECT * FROM {ibes_data_table} " \
                 f"WHERE ticker in {tuple(ticker)} AND trading_day>='{start_date}' ".replace(",)",")")
    ibes = read_query(query_ibes, db_url_read)
    ibes = ibes.pivot(index=["ticker","trading_day"], columns=["field"], values="value").reset_index()

    query_universe = f"SELECT ticker, currency_code, industry_code FROM {universe_table} WHERE ticker in {tuple(ticker)}".replace(",)",")")
    universe = read_query(query_universe, db_url_read)

    def fill_missing_ws(ws):
        ''' fill in missing values by calculating with existing data '''

        logging.info(f'Fill missing in {worldscope_quarter_summary_table} ')
        with suppress(Exception):
            ws['net_debt'] = ws['net_debt'].fillna(ws['debt'] - ws['cash'])
        with suppress(Exception):
            ws['ttm_ebit'] = ws['ttm_ebit'].fillna(ws['ttm_pretax_income'] + ws['ttm_interest'])
        with suppress(Exception):
            ws['ttm_ebitda'] = ws['ttm_ebitda'].fillna(ws['ttm_ebit'] + ws['ttm_dda'])
        with suppress(Exception):
            ws['current_asset'] = ws['current_asset'].fillna(ws['total_asset'] - ws['ppe_net'])
        return ws

    ws = drop_dup(ws)  # drop duplicate and retain the most complete record
    ws = fill_missing_ws(ws)        # selectively fill some missing fields
    ws = update_trading_day(ws)      # correct timestamp for worldscope data (i.e. trading_day)

    # label trading_day with month end of trading_day (update_date)
    ws['trading_day'] = pd.to_datetime(ws['trading_day'], format='%Y-%m-%d')

    return ws, ibes, universe

def check_duplicates(df, name=''):
    df1 = df.drop_duplicates(subset=['trading_day','ticker'])
    if df.shape != df1.shape:
        raise ValueError(f'{name} duplicate records: {df.shape[0] - df1.shape[0]}')

def update_trading_day(ws=None):
    ''' map icb_sector, member_ric, trading_day -> last_year_end for each identifier + frequency_number * 3m '''

    logging.info(f'Update trading_day in {worldscope_quarter_summary_table} ')

    query_universe = f"SELECT ticker, fiscal_year_end FROM {universe_table}"
    universe = read_query(query_universe, db_url_read)

    ws = pd.merge(ws, universe, on='ticker', how='left')   # map static information for each company

    ws["trading_day"] = pd.to_datetime(ws["trading_day"], format='%Y-%m-%d')
    ws['report_date'] = pd.to_datetime(ws['report_date'], format='%Y%m%d')
    ws['fiscal_year_end'] = (pd.to_datetime(ws['fiscal_year_end'], format='%b') + MonthEnd(0)).dt.strftime('%m%d')
    ws["year"] = pd.DatetimeIndex(ws["trading_day"]).year
    ws["frequency_number"] = np.ceil(pd.DatetimeIndex(ws["trading_day"]).month/3)

    # find last fiscal year end for each company (ticker)
    ws['last_year_end'] = (ws['year'].astype(int)-1).astype(str) + ws['fiscal_year_end']
    ws['last_year_end'] = pd.to_datetime(ws['last_year_end'], format='%Y%m%d')

    # find actual period_end (in terms of quarter end)
    ws["trading_day"] = ws[['last_year_end', "frequency_number"]].apply(lambda x: x[0] + MonthEnd(x[1]*3), axis=1)

    # trading_day = report_date (if not exist -> use trading_day(i.e. period_end) + 1Q)
    ws['trading_day'] = ws['trading_day'].mask(ws['report_date'] < ws['trading_day'], ws['report_date'] + QuarterEnd(-1))
    ws['report_date'] = ws['report_date'].fillna(ws['trading_day'] + QuarterEnd(1))
    ws['trading_day'] = ws['report_date']
    ws = drop_dup(ws)  # drop duplicate and retain the most complete record

    return ws.drop(['last_year_end', 'fiscal_year_end', 'year', 'frequency_number', 'report_date'], axis=1)

def fill_all_given_date(result, ref):
    ''' Fill all the date based on given date_df (e.g. tri) to align for biweekly / monthly sampling '''

    # Construct indexes for all date / ticker used in ref (reference dataframe)
    result['trading_day'] = pd.to_datetime(result['trading_day'], format='%Y-%m-%d')
    date_list = ref['trading_day'].unique()
    ticker_list = ref['ticker'].unique()
    indexes = pd.MultiIndex.from_product([ticker_list, date_list], names=['ticker', 'trading_day']).to_frame(index=False, name=['ticker', 'trading_day'])
    logging.info(f"Fill for {len(ref['ticker'].unique())} ticker, {len(date_list)} date")

    # Insert weekend/before first trading date to df
    indexes['trading_day'] = pd.to_datetime(indexes['trading_day'])
    result = result.merge(indexes, on=['ticker', 'trading_day'], how='outer')
    result = result.sort_values(by=['ticker', 'trading_day'], ascending=True)
    result.update(result.groupby(['ticker']).fillna(method='ffill'))        # fill forward for date

    result = result.loc[(result['trading_day'].isin(date_list)) & (result['ticker'].isin(ticker_list))]
    result = result.drop_duplicates(subset=['trading_day','ticker'], keep='last')   # remove ibes duplicates

    return result

def drop_dup(df, col='trading_day'):
    ''' drop duplicate records for same identifier & fiscal period, keep the most complete records '''

    logging.info(f'Drop duplicates in {worldscope_quarter_summary_table} ')

    df['count'] = pd.isnull(df).sum(1)  # count the missing in each records (row)
    df = df.sort_values(['count']).drop_duplicates(subset=['ticker', col], keep='first')
    return df.drop('count', axis=1)

def combine_stock_factor_data(ticker, restart):
    ''' This part do the following:
        1. import all data from DB refer to other functions
        2. combined stock_return, worldscope, ibes, macroeconomic tables '''

    # 1. Stock return/volatility/volume
    tri, stocks_col = calc_stock_return(ticker, restart)
    tri['trading_day'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')
    check_duplicates(tri, 'tri')

    # if ticker[0]=='.':  # if ticker=index, write stock return to DB first -> later report error and stop process due to lack of worldscope data
    #     stock_return_col = tri.filter(regex='^stock_return_').columns.to_list()
    #     tri = pd.melt(tri, id_vars=['ticker', "trading_day"], value_vars=stock_return_col, var_name="field",
    #                   value_name="value").dropna(subset=["value"])
    #     tri = uid_maker(tri, primary_key=['ticker', "trading_day", "field"])
    #
    #     # save calculated ratios to DB
    #     db_table_name = processed_ratio_table
    #     upsert_data_to_database(tri, db_table_name, primary_key=["uid"], db_url=db_url_write, how="append")

    # x = tri.loc[tri['ticker']=='TSLA.O'].sort_values(by=['trading_day'], ascending=False).head(100)

    # 2. Fundamental financial data - from Worldscope
    # 3. Consensus forecasts - from I/B/E/S
    # 4. Universe
    ws, ibes, universe = download_clean_worldscope_ibes(ticker, restart)

    # align worldscope / ibes data with stock return date (monthly/biweekly)
    ws = fill_all_given_date(ws, tri)
    ibes = fill_all_given_date(ibes, tri)

    check_duplicates(ws, 'worldscope')  # check if worldscope/ibes has duplicated records on ticker + trading_day
    check_duplicates(ibes, 'ibes')

    # Use 6-digit ICB code in industry groups
    universe['industry_code'] = universe['industry_code'].replace('NA',np.nan).dropna().astype(int).astype(str).\
        replace({'10102010':'101021','10102015':'101022','10102020':'101023','10102030':'101024','10102035':'101024'})   # split industry 101020 - software (100+ samples)
    universe['industry_code'] = universe['industry_code'].astype(str).str[:6]

    # Combine all data for table (1) - (6) above
    logging.info(f'Merge all dataframes ')
    df = pd.merge(tri, ws, on=['ticker', 'trading_day'], how='left', suffixes=('','_ws'))
    df = df.merge(ibes, on=['ticker', 'trading_day'], how='left', suffixes=('','_ibes'))
    df = df.sort_values(by=['ticker', 'trading_day'])

    # Update close price to adjusted value
    def adjust_close(df):
        ''' using market cap to adjust close price for stock split, ...'''

        logging.info(f'Adjust closing price with market cap ')

        df = df[['ticker','trading_day','market_cap','close']].dropna(how='any')
        df['market_cap_latest'] = df.groupby(['ticker'])['market_cap'].transform('last')
        df['close_latest'] = df.groupby(['ticker'])['close'].transform('last')
        df['close'] = df['market_cap'] / df['market_cap_latest'] * df['close_latest']

        return df[['ticker','trading_day','close']]

    df.update(adjust_close(df))

    # Forward fill for fundamental data
    cols = df.select_dtypes('float').columns.to_list()
    cols = [x for x in cols if not x.startswith("stock_return_y")]     # for stock_return_y -> no ffill
    df.update(df.groupby(['ticker'])[cols].fillna(method='ffill'))
    df = resample_to_weekly(df, date_col='trading_day')  # Resample to monthly stock tri
    df = df.merge(universe, on=['ticker'], how='left', suffixes=('_old', ''))      # label industry_code, currency_code for each ticker
    check_duplicates(df, 'final')
    return df, stocks_col

def calc_fx_conversion(df):
    """ Convert all columns to USD for factor calculation (DSS, WORLDSCOPE, IBES using different currency) """

    org_cols = df.columns.to_list()     # record original columns for columns to return

    curr_code_query = f"SELECT ticker, currency_code_ibes, currency_code_ws FROM {universe_table}"
    curr_code = read_query(curr_code_query, db_url_read)

    # combine download fx data from eikon & daily currency price ingestion
    fx = read_table(eikon_fx_table, db_url_read)
    fx2_query = f"SELECT currency_code as ticker, last_price as fx_rate, last_date as trading_day FROM {currency_history_table}"
    fx2 = read_query(fx2_query, db_url_read)
    fx['trading_day'] = pd.to_datetime(fx['trading_day']).dt.tz_localize(None)
    fx2['trading_day'] = pd.to_datetime(fx2['trading_day'])
    fx = fx.append(fx2).drop_duplicates(subset=['ticker', 'trading_day'], keep='last')

    ingestion_source = read_table(ingestion_name_table, db_url_read)

    df = df.merge(curr_code, on='ticker', how='inner')
    df = df.dropna(subset=['currency_code_ibes', 'currency_code_ws', 'currency_code'], how='any')   # remove ETF / index / some B-share -> tickers will not be recommended

    # map fx rate for conversion for each ticker
    fx = fx.drop_duplicates(subset=['ticker','trading_day'])
    fx = fill_all_day(fx, date_col='trading_day')
    fx['fx_rate'] = fx.groupby('ticker')['fx_rate'].ffill().bfill()
    fx['trading_day'] = fx['trading_day'].dt.strftime("%Y-%m-%d")
    fx = fx.set_index(['ticker', 'trading_day'])['fx_rate'].to_dict()

    currency_code_cols = ['currency_code', 'currency_code_ibes', 'currency_code_ws']
    fx_cols = ['fx_dss', 'fx_ibes', 'fx_ws']
    df['trading_day'] = pd.to_datetime(df['trading_day']).dt.strftime("%Y-%m-%d")
    for cur_col, fx_col in zip(currency_code_cols, fx_cols):
        df = df.set_index([cur_col, 'trading_day'])
        df['index'] = df.index.to_numpy()
        df[fx_col] = df['index'].map(fx)
        df = df.reset_index()

    df['trading_day'] = pd.to_datetime(df['trading_day'])
    ingestion_source = ingestion_source.loc[ingestion_source['non_ratio']]     # no fx conversion for ratio items

    for name, g in ingestion_source.groupby(['source']):        # convert for ibes / ws
        cols = list(set(g['our_name'].to_list()) & set(df.columns.to_list()))
        logging.info(f'[{name}] source data with fx conversion: {cols}')
        df[cols] = df[cols].div(df[f'fx_{name}'], axis="index")

    df[['close','market_cap']] = df[['close','market_cap']].div(df['fx_dss'], axis="index")  # convert close price
    df['market_cap_usd'] = df['market_cap']
    return df[org_cols]

def calc_factor_variables(ticker, restart):
    ''' Calculate all factor used referring to DB ratio table '''

    logging.info(f'=== (n={len(ticker)}) Calculate ratio for {ticker}  ===')
    error_universe = []
    try:
        df, stocks_col = combine_stock_factor_data(ticker, restart)

        formula = read_table(formula_factors_table_prod, db_url_read)
        formula = formula.loc[formula['is_active']]

        logging.info(f'Calculate all factors in {formula_factors_table_prod}')

        # Foreign exchange conversion on absolute value items
        df = calc_fx_conversion(df)
        ingestion_cols = df.columns.to_list()

        # Prepare for field requires add/minus
        add_minus_fields = formula[['field_num', 'field_denom']].dropna(how='any').to_numpy().flatten()
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
                    logging.warning(f"[Warning] add_minus_fields not calculate: {add_minus_fields}: {e}")
                n += 2
            df[i] = temp

        # a) Keep original values
        keep_original_mask = formula['field_denom'].isnull() & formula['field_num'].notnull()
        for new_name, old_name in formula.loc[keep_original_mask, ['name','field_num']].to_numpy():
            logging.info(f'Calculating: {new_name}')
            try:
                df[new_name] = df[old_name]
            except Exception as e:
                logging.warning(f"[Warning] Factor ratio [{new_name}] not calculate: {e}")

        # b) Time series ratios (Calculate 1m change first)
        logging.info(f'Calculate time-series ratio ')
        period_yr = 52
        period_q = 12
        for r in formula.loc[formula['field_num'] == formula['field_denom'], ['name', 'field_denom']].to_dict(
                orient='records'):  # minus calculation for ratios
            logging.info(f"Calculating: {r['name']}")
            try:
                if r['name'][-2:] == 'yr':
                    df[r['name']] = df[r['field_denom']] / df[r['field_denom']].shift(period_yr) - 1
                    df.loc[df.groupby('ticker').head(period_yr).index, r['name']] = np.nan
                elif r['name'][-1] == 'q':
                    df[r['name']] = df[r['field_denom']] / df[r['field_denom']].shift(period_q) - 1
                    df.loc[df.groupby('ticker').head(period_q).index, r['name']] = np.nan
            except Exception as e:
                logging.warning(f"[Warning] Factor ratio [{r['name']}] not calculate: {e}")

        # c) Divide ratios
        logging.info(f'Calculate dividing ratios ')
        for r in formula.dropna(how='any', axis=0).loc[(formula['field_num'] != formula['field_denom'])].to_dict(
                orient='records'):  # minus calculation for ratios
            try:
                logging.info(f"Calculating: {r['name']}")
                df[r['name']] = df[r['field_num']] / df[r['field_denom']].replace(0, np.nan)
            except Exception as e:
                logging.warning(f"[Warning] Factor ratio [{r['name']}] not calculate: {e}")

        # drop records with no stock_return_y & any ratios
        dropna_col = set(df.columns.to_list()) & set(formula['name'].to_list())
        df = df.dropna(subset=list(dropna_col), how='all')
        df = df.replace([np.inf, -np.inf], np.nan)

        # test ratio calculation missing rate
        # test_missing(df, formula[['name','field_num','field_denom']], ingestion_cols)

        y_col = [x for x in df.columns.to_list() if x.startswith("stock_return_y")]
        df[[x+'_ffill' for x in y_col]] = df.groupby('ticker')[y_col].ffill()
        df = df.dropna(subset=[x+'_ffill' for x in y_col], how='any')
        df = df.filter(['ticker', 'trading_day'] + y_col + formula['name'].to_list())
        df = pd.melt(df, id_vars=['ticker', "trading_day"], var_name="field", value_name="value").dropna(subset=["value"])
        df = uid_maker(df, primary_key=['ticker', "trading_day", "field"])

        if not restart:
            df = df.loc[df["trading_day"]>(dt.datetime.today()-relativedelta(months=3))]

        # save calculated ratios to DB
        db_table_name = processed_ratio_table
        if restart:
            trucncate_table_in_database(f"{processed_ratio_table}", db_url_write)
            upsert_data_to_database(df, db_table_name, primary_key=["uid"], db_url=db_url_write, how="append")
        else:
            upsert_data_to_database(df, db_table_name, primary_key=["uid"], db_url=db_url_write, how="update")
    except Exception as e:
        error_msg = f"===  ERROR IN Getting Data == {e}"
        to_slack("clair").message_to_slack(error_msg)
        error_universe.append(ticker)

def calc_factor_variables_multi(ticker=None, currency=None, restart=True):
    ''' Calculate weekly ratios for all factors

    Parameters
    ----------
    ticker (Str, default=None):
        tickers to calculate variables (default calculate for all active tickers)
    currency (Str, default=None):
        tickers in which currency to calculate variables
    restart (Bool, default=False):
        if True, calculate variables for all trading_period, AND [rewrite] entire factor_processed_ratio table;
        if False, calculate variables for the most recent 3 months, AND [update] factor_processed_ratio table.
            Get From Table "data_tri" for recent 6 months data, and "data_worldscope"/"data_ibes" for recent 2 years.
    '''

    if ticker:
        tickers = [ticker]
    elif currency:
        tickers = read_query(f"SELECT ticker FROM universe WHERE is_active AND currency_code='{currency}'")["ticker"].to_list()
    else:
        tickers = read_query(f"SELECT ticker FROM universe WHERE is_active")["ticker"].to_list()

    # tickers_exist = read_query(f"SELECT distinct ticker FROM {processed_ratio_table}", db_url_write)
    # tickers = list(set(tickers)-set(tickers_exist))

    calc_factor_variables(tickers, restart)
    # tickers = [tuple([e]) for e in tickers]
    # with mp.Pool(processes=processes) as pool:
    #     pool.starmap(calc_factor_variables, tickers)

def test_missing(df_org, formula, ingestion_cols):

    for group in ['USD']:
        df = df_org.loc[df_org['currency_code']==group]
        writer = pd.ExcelWriter(f'missing_by_ticker_{group}.xlsx')

        df = df.groupby('ticker').apply(lambda x: x.notnull().sum())
        df.to_excel(writer, sheet_name='by ticker')

        df_miss = df[ingestion_cols].unstack()
        df_miss = df_miss.loc[df_miss==0].reset_index()
        df_miss.to_excel(writer, sheet_name='all_missing', index=False)
        df_miss.to_csv(f'dsws_missing_ingestion_{group}.csv')

        df_sum = pd.DataFrame(df.sum(0))
        df_sum_df = df_sum.merge(formula, left_index=True, right_on=['name'], how='left')
        for i in ['field_num', 'field_denom']:
            df_sum_df = df_sum_df.merge(df_sum, left_on=[i], how='left', right_index=True)
        df_sum_df.to_excel(writer, sheet_name='count', index=False)
        df_sum.to_excel(writer, sheet_name='count_lst')

        writer.save()

if __name__ == "__main__":

    restart = False
    calc_factor_variables_multi(ticker=None, restart=restart)


