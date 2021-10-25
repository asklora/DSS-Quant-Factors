import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, QuarterEnd
from scipy.stats import skew
from utils import record_table_update_time

# ----------------------------------------- Calculate Stock Ralated Factors --------------------------------------------

# cur = 'USD'
def get_tri(save=True, currency=None, ticker=None):
    if not isinstance(save, bool):
        raise Exception("Parameter 'save' must be a bool")
    # if not isinstance(currency, str):
    #     raise Exception("Parameter 'currency' must be a str")

    conditions = ["True"]
    if currency:
        conditions.append(f"currency_code = '{currency}'")
    if ticker:
        conditions.append(f"T.ticker = '{ticker}'")

    with global_vals.engine.connect() as conn_droid, global_vals.engine_ali.connect() as conn_ali:
        print(f'#################################################################################################')
        print(f'      ------------------------> Download stock data from {global_vals.stock_data_table_tri}')
        query = text(f"SELECT T.ticker, T.trading_day, currency_code, total_return_index as tri, open, high, low, close, volume "
                     f"FROM {global_vals.stock_data_table_tri} T "
                     f"INNER JOIN {global_vals.stock_data_table_ohlc} C ON T.dsws_id = C.dss_id "
                     f"INNER JOIN {global_vals.dl_value_universe_table} U ON T.ticker = U.ticker "
                     f"WHERE {' AND '.join(conditions)}")
        tri = pd.read_sql(query, con=conn_droid, chunksize=10000)
        tri = pd.concat(tri, axis=0, ignore_index=True)

        print(f'      ------------------------> Download stock data from {global_vals.eikon_price_table}/{global_vals.fundamental_score_mkt_cap}')
        eikon_price = pd.read_sql(f"SELECT * FROM {global_vals.eikon_price_table} ORDER BY ticker, trading_day", conn_ali, chunksize=10000)
        eikon_price = pd.concat(eikon_price, axis=0, ignore_index=True)
        market_cap_anchor = pd.read_sql(f'SELECT ticker, mkt_cap FROM {global_vals.fundamental_score_mkt_cap}', conn_droid)
        if save:
            tri.to_csv('cache_tri.csv', index=False)
            eikon_price.to_csv('cache_eikon_price.csv', index=False)
            market_cap_anchor.to_csv('cache_market_cap_anchor.csv', index=False)
    global_vals.engine.dispose()
    global_vals.engine_ali.dispose()
    return tri, eikon_price, market_cap_anchor

def fill_all_day(result, date_col="trading_day"):
    ''' Fill all the weekends between first / last day and fill NaN'''

    # Construct indexes for all day between first/last day * all ticker used
    df = result[["ticker", date_col]].copy()
    df.trading_day = pd.to_datetime(df[date_col])
    result.trading_day = pd.to_datetime(result[date_col])
    df = df.sort_values(by=[date_col], ascending=True)
    daily = pd.date_range(df.iloc[0, 1], df.iloc[-1, 1], freq='D')
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

def resample_to_monthly(df, date_col):
    ''' Resample to monthly stock tri '''
    monthly = pd.date_range(min(df[date_col].to_list()), max(df[date_col].to_list()), freq='M')
    df = df.loc[df[date_col].isin(monthly)]
    return df

def resample_to_biweekly(df, date_col):
    ''' Resample to bi-weekly stock tri '''
    monthly = pd.date_range(min(df[date_col].to_list()), max(df[date_col].to_list()),  freq='2W')
    df = df.loc[df[date_col].isin(monthly)]
    return df

def resample_to_weekly(df, date_col):
    ''' Resample to bi-weekly stock tri '''
    monthly = pd.date_range(min(df[date_col].to_list()), max(df[date_col].to_list()),  freq='W')
    df = df.loc[df[date_col].isin(monthly)]
    return df

def calc_stock_return(price_sample, sample_interval, rolling_period, use_cached, save, currency, ticker):
    ''' Calcualte monthly stock return '''

    if use_cached:
        try:
            tri = pd.read_csv('cache_tri.csv')
            eikon_price = pd.read_csv('cache_eikon_price.csv')
            market_cap_anchor = pd.read_csv('cache_market_cap_anchor.csv')
        except Exception as e:
            print(e)
            tri, eikon_price, market_cap_anchor = get_tri(save, currency, ticker)
    else:
        tri, eikon_price, market_cap_anchor = get_tri(save, currency, ticker)

    if currency:        # if only calculate single currency
        tri = tri.loc[tri['currency_code']==currency]
    if ticker:
        tri = tri.loc[tri['ticker']==ticker]
    eikon_price = eikon_price.loc[eikon_price['ticker'].isin(tri['ticker'].unique())]
    market_cap_anchor = market_cap_anchor.loc[market_cap_anchor['ticker'].isin(tri['ticker'].unique())]

    # merge stock return from DSS & from EIKON (i.e. longer history)
    tri['trading_day'] = pd.to_datetime(tri['trading_day'])
    eikon_price['trading_day'] = pd.to_datetime(eikon_price['trading_day'])

    # find first tri from DSS as anchor
    tri_first = tri.dropna(subset=['tri']).sort_values(by=['trading_day']).groupby(['ticker']).first().reset_index()
    tri_first['anchor_tri'] = tri_first['tri']
    eikon_price['close'] = eikon_price['close'].fillna(eikon_price[['high','low']].mean(axis=1))
    eikon_price = eikon_price.merge(tri_first[['ticker','trading_day','anchor_tri']], on=['ticker','trading_day'], how='left')

    # find anchor close price (adj.)
    eikon_price = eikon_price.sort_values(['ticker','trading_day'])
    eikon_price.loc[eikon_price['anchor_tri'].notnull(), 'anchor_close'] = eikon_price.loc[eikon_price['anchor_tri'].notnull(), 'close']
    eikon_price[['anchor_close','anchor_tri']] = eikon_price.groupby('ticker')[['anchor_close','anchor_tri']].bfill()

    # calculate tri based on EIKON close price data
    eikon_price['tri'] = eikon_price['close']/eikon_price['anchor_close']*eikon_price['anchor_tri']

    # merge DSS & EIKON data
    tri = tri.merge(eikon_price, on=['ticker','trading_day'], how='outer', suffixes=['','_eikon']).sort_values(by=['ticker','trading_day'])
    value_col = ['open','high','low','close','tri','volume']
    for col in value_col:
        tri[col] = tri[col].fillna(tri[col+'_eikon'])  # update missing tri (i.e. prior history) with EIKON tri calculated
    tri = tri[['ticker','trading_day'] + value_col]

    # x = tri.loc[tri['ticker']=='AAPL.O'].sort_values(by=['trading_day'], ascending=False)

    tri = tri.replace(0, np.nan)  # Remove all 0 since total_return_index not supposed to be 0
    tri = fill_all_day(tri)  # Add NaN record of tri for weekends
    tri = tri.sort_values(['ticker','trading_day'])

    print(f'      ------------------------> Calculate skewness ')
    tri = get_skew(tri)    # Calculate past 1 year skewness

    # Calculate RS volatility for 3-month & 6-month~2-month (before ffill)
    print(f'      ------------------------> Calculate RS volatility ')
    list_of_start_end = [[0, 30]] # , [30, 90], [90, 182]
    tri = get_rogers_satchell(tri, list_of_start_end)
    tri = tri.drop(['open', 'high', 'low'], axis=1)

    # resample tri using last week average as the proxy for monthly tri
    print(f'      ------------------------> Stock price using [{price_sample}] ')
    tri[['tri','volume']] = tri.groupby("ticker")[['tri','volume']].rolling(7, min_periods=1).mean().reset_index(drop=1)
    tri['volume_3m'] = tri.groupby("ticker")['volume'].rolling(91, min_periods=1).mean().values
    tri['volume'] = tri['volume'] / tri['volume_3m']

    # Fill forward (-> holidays/weekends) + backward (<- first trading price)
    cols = ['tri', 'close','volume'] + [f'vol_{l[0]}_{l[1]}' for l in list_of_start_end]
    tri.update(tri.groupby('ticker')[cols].fillna(method='ffill'))

    print(f'      ------------------------> Sample interval using [{sample_interval}] ')
    if sample_interval == 'monthly':
        tri = resample_to_monthly(tri, date_col='trading_day')  # Resample to monthly stock tri
    # elif sample_interval == 'biweekly':
    #     tri = resample_to_biweekly(tri, date_col='trading_day')  # Resample to bi-weekly stock tri
    elif sample_interval == 'weekly':
        tri = resample_to_weekly(tri, date_col='trading_day')  # Resample to weekly stock tri
    else:
        raise ValueError("Invalid sample_interval method. Expecting 'monthly' or 'weekly' got ", sample_interval)

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
    print(f'      ------------------------> Calculate stock returns ')
    if sample_interval == 'monthly':
        tri["tri_y"] = tri.groupby('ticker')['tri'].shift(-rolling_period)
        tri["stock_return_y"] = (tri["tri_y"] / tri["tri"]) - 1
        tri["stock_return_y"] = tri["stock_return_y"]/rolling_period

        tri["tri_1mb"] = tri.groupby('ticker')['tri'].shift(1)
        tri["tri_2mb"] = tri.groupby('ticker')['tri'].shift(2)
        tri['tri_6mb'] = tri.groupby('ticker')['tri'].shift(6)
        tri['tri_7mb'] = tri.groupby('ticker')['tri'].shift(7)
        tri['tri_12mb'] = tri.groupby('ticker')['tri'].shift(12)
        drop_col = ['tri_1mb', 'tri_2mb', 'tri_6mb', 'tri_7mb', 'tri_12mb']
    elif sample_interval == 'weekly':
        tri["tri_y"] = tri.groupby('ticker')['tri'].shift(-rolling_period)
        tri["stock_return_y"] = (tri["tri_y"] / tri["tri"]) - 1
        tri["stock_return_y"] = tri["stock_return_y"]*4/rolling_period
        tri["tri_1wb"] = tri.groupby('ticker')['tri'].shift(1)
        tri["tri_2wb"] = tri.groupby('ticker')['tri'].shift(2)
        tri["tri_1mb"] = tri.groupby('ticker')['tri'].shift(4)
        tri["tri_2mb"] = tri.groupby('ticker')['tri'].shift(8)
        tri['tri_6mb'] = tri.groupby('ticker')['tri'].shift(26)
        tri['tri_7mb'] = tri.groupby('ticker')['tri'].shift(30)
        tri['tri_12mb'] = tri.groupby('ticker')['tri'].shift(52)
        drop_col = ['tri_1wb', 'tri_2wb', 'tri_1mb', 'tri_2mb', 'tri_6mb', 'tri_7mb', 'tri_12mb']

        tri["stock_return_ww1_0"] = (tri["tri"] / tri["tri_1wb"]) - 1
        tri["stock_return_ww2_1"] = (tri["tri_1wb"] / tri["tri_2wb"]) - 1
        tri["stock_return_ww4_2"] = (tri["tri_2wb"] / tri["tri_1mb"]) - 1

    tri["stock_return_r1_0"] = (tri["tri"] / tri["tri_1mb"]) - 1
    tri["stock_return_r6_2"] = (tri["tri_2mb"] / tri["tri_6mb"]) - 1
    tri["stock_return_r12_7"] = (tri["tri_7mb"] / tri["tri_12mb"]) - 1

    tri = tri.drop(['tri', 'tri_y'] + drop_col, axis=1)
    stock_col = tri.select_dtypes('float').columns  # all numeric columns

    if save:
        tri.to_csv('cache_tri_ratio.csv', index=False)

    return tri, stock_col

# -------------------------------------------- Calculate Fundamental Ratios --------------------------------------------

def download_clean_worldscope_ibes(save, currency, ticker):
    ''' download all data for factor calculate & LGBM input (except for stock return) '''

    conditions = ["ticker is not null"]
    if currency:
        raise ValueError("Error: Currency only available for dsws filter")
    if ticker:
        conditions.append(f"ticker = '{ticker}'")

    with global_vals.engine.connect() as conn:
        print(f'#################################################################################################')
        query_ws = f"select * from {global_vals.worldscope_quarter_summary_table} WHERE {' AND '.join(conditions)}"
        query_ibes = f"SELECT * FROM {global_vals.ibes_data_table} WHERE {' AND '.join(conditions)}"
        print(f'      ------------------------> Download worldscope data from {global_vals.worldscope_quarter_summary_table}')
        ws = pd.read_sql(query_ws, conn, chunksize=10000)  # quarterly records
        ws = pd.concat(ws, axis=0, ignore_index=True)
        print(f'      ------------------------> Download ibes data from {global_vals.ibes_data_table}')
        ibes = pd.read_sql(query_ibes, conn, chunksize=10000)  # ibes_data
        ibes = pd.concat(ibes, axis=0, ignore_index=True)
        universe = pd.read_sql(f"SELECT ticker, currency_code, icb_code FROM {global_vals.dl_value_universe_table}", conn, chunksize=10000)
        universe = pd.concat(universe, axis=0, ignore_index=True)
    global_vals.engine.dispose()

    ws = drop_dup(ws)  # drop duplicate and retain the most complete record

    def fill_missing_ws(ws):
        ''' fill in missing values by calculating with existing data '''

        print(f'      ------------------------> Fill missing in {global_vals.worldscope_quarter_summary_table} ')

        ws['net_debt'] = ws['net_debt'].fillna(ws['debt'] - ws['cash'])  # Net debt = total debt - C&CE
        ws['ttm_ebit'] = ws['ttm_ebit'].fillna(
            ws['ttm_pretax_income'] + ws['ttm_interest'])  # TTM EBIT = TTM Pretax Income + TTM Interest Exp.
        ws['ttm_ebitda'] = ws['ttm_ebitda'].fillna(ws['ttm_ebit'] + ws['ttm_dda'])  # TTM EBITDA = TTM EBIT + TTM DDA
        ws['current_asset'] = ws['current_asset'].fillna(ws['total_asset'] - ws['ppe_net']) # fill missing for current assets
        return ws

    ws = fill_missing_ws(ws)        # selectively fill some missing fields
    ws = update_period_end(ws)      # correct timestamp for worldscope data (i.e. period_end)

    # label period_end with month end of trading_day (update_date)
    ws['period_end'] = pd.to_datetime(ws['period_end'], format='%Y-%m-%d')

    if save:
        ws.to_csv('cache_ws.csv', index=False)
        ibes.to_csv('cache_ibes.csv', index=False)
        universe.to_csv('cache_universe.csv', index=False)

    return ws, ibes, universe

def check_duplicates(df, name=''):
    df1 = df.drop_duplicates(subset=['period_end','ticker'])
    if df.shape != df1.shape:
        raise ValueError(f'{name} duplicate records: {df.shape[0] - df1.shape[0]}')

def update_period_end(ws=None):
    ''' map icb_sector, member_ric, period_end -> last_year_end for each identifier + frequency_number * 3m '''

    print(f'      ------------------------> Update period_end in {global_vals.worldscope_quarter_summary_table} ')

    with global_vals.engine.connect() as conn, global_vals.engine_ali.connect() as conn_ali:
        universe = pd.read_sql(f'SELECT ticker, fiscal_year_end FROM {global_vals.dl_value_universe_table}', conn, chunksize=10000)
        universe = pd.concat(universe, axis=0, ignore_index=True)
        eikon_report_date = pd.read_sql(f'SELECT * FROM {global_vals.eikon_other_table}_date', conn_ali, chunksize=10000)
        eikon_report_date = pd.concat(eikon_report_date, axis=0, ignore_index=True)
    global_vals.engine.dispose()
    global_vals.engine_ali.dispose()

    ws = ws.dropna(subset=['year'])
    ws = pd.merge(ws, universe, on='ticker', how='left')   # map static information for each company

    # ws = ws.loc[ws['fiscal_year_end'].isin(['MAR','JUN','SEP','DEC'])]      # select identifier with correct year end

    ws['period_end'] = pd.to_datetime(ws['period_end'], format='%Y-%m-%d')
    ws['report_date'] = pd.to_datetime(ws['report_date'], format='%Y-%m-%d')
    ws['fiscal_year_end'] = (pd.to_datetime(ws['fiscal_year_end'], format='%b') + MonthEnd(0)).dt.strftime('%m%d')

    # find last fiscal year end for each company (ticker)
    ws['last_year_end'] = (ws['year'].astype(int)-1).astype(str) + ws['fiscal_year_end']
    ws['last_year_end'] = pd.to_datetime(ws['last_year_end'], format='%Y%m%d')

    ws = ws.merge(eikon_report_date, on=['ticker', 'period_end'], suffixes=('', '_ek'), how='left')

    ws['report_date'] = ws['report_date'].fillna(ws['report_date_ek'])
    ws['period_end'] = ws['period_end'].mask(ws['report_date'] < ws['period_end'], ws['report_date'] + QuarterEnd(-1))
    ws['report_date'] = ws['report_date'].fillna(ws['period_end'] + QuarterEnd(1))

    ws['period_end'] = ws['report_date']
    ws = drop_dup(ws)  # drop duplicate and retain the most complete record

    return ws.drop(['last_year_end','fiscal_year_end','year','frequency_number','fiscal_quarter_end','report_date','report_date_ek'], axis=1)

def fill_all_given_date(result, ref):
    ''' Fill all the date based on given date_df (e.g. tri) to align for biweekly / monthly sampling '''

    # Construct indexes for all date / ticker used in ref (reference dataframe)
    try:
        result['period_end'] = result['trading_day']  # rename trading to period_end for later merge with other df
        result = result.drop(['trading_day'], axis=1)
    except Exception as e:
        print("No need to convert column name 'trading_day' to 'period_end'", e)

    result['period_end'] = pd.to_datetime(result['period_end'], format='%Y-%m-%d')
    date_list = ref['trading_day'].unique()
    ticker_list = ref['ticker'].unique()
    indexes = pd.MultiIndex.from_product([ticker_list, date_list],
                                         names=['ticker', 'period_end']).to_frame(index=False, name=['ticker', 'period_end'])
    print(f"      ------------------------> Fill for {len(ref['ticker'].unique())} ticker, {len(date_list)} date")

    # Insert weekend/before first trading date to df
    indexes['period_end'] = pd.to_datetime(indexes['period_end'])
    result = result.merge(indexes, on=['ticker', 'period_end'], how='outer')
    result = result.sort_values(by=['ticker', 'period_end'], ascending=True)
    # x = result.loc[result['ticker']=='AAPL.O']
    result.update(result.groupby(['ticker']).fillna(method='ffill'))        # fill forward for date
    # x = result.loc[result['ticker']=='AAPL.O']

    result = result.loc[(result['period_end'].isin(date_list)) & (result['ticker'].isin(ticker_list))]
    result = result.drop_duplicates(subset=['period_end','ticker'], keep='last')   # remove ibes duplicates

    return result

def drop_dup(df, col='period_end'):
    ''' drop duplicate records for same identifier & fiscal period, keep the most complete records '''

    print(f'      ------------------------> Drop duplicates in {global_vals.worldscope_quarter_summary_table} ')

    df['count'] = pd.isnull(df).sum(1)  # count the missing in each records (row)
    df = df.sort_values(['count']).drop_duplicates(subset=['ticker', col], keep='first')
    return df.drop('count', axis=1)

def combine_stock_factor_data(price_sample, fill_method, sample_interval, rolling_period, use_cached, save, currency, ticker):
    ''' This part do the following:
        1. import all data from DB refer to other functions
        2. combined stock_return, worldscope, ibes, macroeconomic tables '''

    # 1. Stock return/volatility/volume
    if use_cached:
        try:
            tri = pd.read_csv('cache_tri_ratio.csv')
            stocks_col = tri.select_dtypes("float").columns
        except Exception as e:
            print(e)
            tri, stocks_col = calc_stock_return(price_sample, sample_interval, rolling_period, use_cached, save, currency, ticker)
    else:
        tri, stocks_col = calc_stock_return(price_sample, sample_interval, rolling_period, use_cached, save, currency, ticker)

    tri['period_end'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')
    check_duplicates(tri, 'tri')

    # x = tri.loc[tri['ticker']=='TSLA.O'].sort_values(by=['period_end'], ascending=False).head(100)

    # 2. Fundamental financial data - from Worldscope
    # 3. Consensus forecasts - from I/B/E/S
    # 4. Universe
    if use_cached:
        try:
            ws = pd.read_csv('cache_ws.csv')
            ibes = pd.read_csv('cache_ibes.csv')
            universe = pd.read_csv('cache_universe.csv')
        except Exception as e:
            print(e)
            ws, ibes, universe = download_clean_worldscope_ibes(save, currency, ticker)
    else:
        ws, ibes, universe = download_clean_worldscope_ibes(save, currency, ticker)

    # align worldscope / ibes data with stock return date (monthly/biweekly)
    ws = fill_all_given_date(ws, tri)
    ibes = fill_all_given_date(ibes, tri)

    check_duplicates(ws, 'worldscope')  # check if worldscope/ibes has duplicated records on ticker + period_end
    check_duplicates(ibes, 'ibes')

    # Use 6-digit ICB code in industry groups
    universe['icb_code'] = universe['icb_code'].replace('NA',np.nan).dropna().astype(int).astype(str).\
        replace({'10102010':'101021','10102015':'101022','10102020':'101023','10102030':'101024','10102035':'101024'})   # split industry 101020 - software (100+ samples)
    # print(universe['icb_code'].unique())
    universe['icb_code'] = universe['icb_code'].astype(str).str[:6]

    # Combine all data for table (1) - (6) above
    print(f'      ------------------------> Merge all dataframes ')
    df = pd.merge(tri.drop("trading_day", axis=1), ws, on=['ticker', 'period_end'], how='left', suffixes=('','_ws'))
    df = df.merge(ibes, on=['ticker', 'period_end'], how='left', suffixes=('','_ibes'))
    df = df.sort_values(by=['ticker', 'period_end'])

    # Update close price to adjusted value
    def adjust_close(df):
        ''' using market cap to adjust close price for stock split, ...'''

        print(f'      ------------------------> Adjust closing price with market cap ')

        df = df[['ticker','period_end','market_cap','close']].dropna(how='any')
        df['market_cap_latest'] = df.groupby(['ticker'])['market_cap'].transform('last')
        df['close_latest'] = df.groupby(['ticker'])['close'].transform('last')
        df['close'] = df['market_cap'] / df['market_cap_latest'] * df['close_latest']

        return df[['ticker','period_end','close']]

    df.update(adjust_close(df))

    # Forward fill for fundamental data
    cols = df.select_dtypes('float').columns.to_list()
    cols = list(set(cols) - {'stock_return_y'})     # for stock_return_y -> no ffill
    if fill_method == 'fill_all':           # e.g. Quarterly June -> Monthly July/Aug
        df.update(df.groupby(['ticker'])[cols].fillna(method='ffill'))
    elif fill_method == 'fill_monthly':     # e.g. only 1 June -> 30 June
        df.update(df.groupby(['ticker', 'period_end'])[cols].fillna(method='ffill'))
    else:
        raise ValueError("Invalid fill_method. Expecting 'fill_all' or 'fill_monthly' got ", fill_method)

    if sample_interval == 'monthly':
        df = resample_to_monthly(df, date_col='period_end')  # Resample to monthly stock tri
    elif sample_interval == 'weekly':
        df = resample_to_weekly(df, date_col='period_end')  # Resample to monthly stock tri
    else:
        raise ValueError("Invalid sample_interval method. Expecting 'monthly' or 'weekly' got ", sample_interval)

    df = df.merge(universe, on=['ticker'], how='left')      # label icb_code, currency_code for each ticker

    if save:
        df.to_csv('cache_all_data.csv')  # for debug
        pd.DataFrame(stocks_col).to_csv('cache_stocks_col.csv', index=False)  # for debug

    check_duplicates(df, 'final')
    return df, stocks_col

def calc_fx_conversion(df):
    """ Convert all columns to USD for factor calculation (DSS, WORLDSCOPE, IBES using different currency) """

    org_cols = df.columns.to_list()     # record original columns for columns to return

    with global_vals.engine.connect() as conn, global_vals.engine_ali.connect() as conn_ali:
        curr_code = pd.read_sql(f"SELECT ticker, currency_code_ibes, currency_code_ws FROM {global_vals.dl_value_universe_table}", conn)     # map ibes/ws currency for each ticker
        fx = pd.read_sql(f"SELECT * FROM {global_vals.eikon_other_table}_fx", conn_ali)
        fx2 = pd.read_sql(f"SELECT currency_code as ticker, last_price as fx_rate, last_date as period_end "
                          f"FROM {global_vals.currency_history_table}", conn)
        fx = fx.append(fx2).drop_duplicates(subset=['ticker','period_end'], keep='last')
        ingestion_source = pd.read_sql(f"SELECT * FROM ingestion_name", conn_ali)
    global_vals.engine.dispose()
    global_vals.engine_ali.dispose()

    df = df.merge(curr_code, on='ticker', how='inner')
    df = df.dropna(subset=['currency_code_ibes', 'currency_code_ws', 'currency_code'], how='any')   # remove ETF / index / some B-share -> tickers will not be recommended

    # map fx rate for conversion for each ticker
    fx = fx.drop_duplicates(subset=['ticker','period_end'])
    fx = fill_all_day(fx, date_col='period_end')
    fx['fx_rate'] = fx.groupby('ticker')['fx_rate'].ffill().bfill()
    fx['period_end'] = fx['period_end'].dt.strftime("%Y-%m-%d")
    fx = fx.set_index(['ticker', 'period_end'])['fx_rate'].to_dict()

    currency_code_cols = ['currency_code', 'currency_code_ibes', 'currency_code_ws']
    fx_cols = ['fx_dss', 'fx_ibes', 'fx_ws']
    df['period_end'] = pd.to_datetime(df['period_end']).dt.strftime("%Y-%m-%d")
    for cur_col, fx_col in zip(currency_code_cols, fx_cols):
        df = df.set_index([cur_col, 'period_end'])
        df['index'] = df.index.to_numpy()
        df[fx_col] = df['index'].map(fx)
        df = df.reset_index()

    df['period_end'] = pd.to_datetime(df['period_end'])
    ingestion_source = ingestion_source.loc[ingestion_source['non_ratio']]     # no fx conversion for ratio items

    for name, g in ingestion_source.groupby(['source']):        # convert for ibes / ws
        cols = list(set(g['our_name'].to_list()) & set(df.columns.to_list()))
        print(f'----> [{name}] source data with fx conversion: ', cols)
        df[cols] = df[cols].div(df[f'fx_{name}'], axis="index")

    df[['close','market_cap']] = df[['close','market_cap']].div(df['fx_dss'], axis="index")  # convert close price
    df['market_cap_usd'] = df['market_cap']
    return df[org_cols]

def calc_factor_variables(price_sample='last_day', fill_method='fill_all', sample_interval='monthly', rolling_period=1,
                          use_cached=False, save=True, currency=None, ticker=None):
    ''' Calculate all factor used referring to DB ratio table '''

    # if update:  # update for the latest month (not using cachec & not save locally)
    #     use_cached = False
    #     save = False

    if use_cached:
        try:
            df = pd.read_csv('cache_all_data.csv', dtype={"icb_code": str})
            stocks_col = pd.read_csv('cache_stocks_col.csv').iloc[:, 0].to_list()
        except Exception as e:
            print(e)
            df, stocks_col = combine_stock_factor_data(price_sample, fill_method, sample_interval, rolling_period, use_cached, save, currency, ticker)
    else:
        df, stocks_col = combine_stock_factor_data(price_sample, fill_method, sample_interval, rolling_period, use_cached, save, currency, ticker)

    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table} WHERE x_col', conn, chunksize=10000)  # ratio calculation used
        formula = pd.concat(formula, axis=0, ignore_index=True)
    global_vals.engine_ali.dispose()

    print(f'#################################################################################################')
    print(f'      ------------------------> Calculate all factors in {global_vals.formula_factors_table}')

    # Foreign exchange conversion on absolute value items
    df = calc_fx_conversion(df)

    ingestion_cols = df.columns.to_list()

    # Prepare for field requires add/minus
    add_minus_fields = formula[['field_num', 'field_denom']].dropna(how='any').to_numpy().flatten()
    add_minus_fields = [i for i in list(set(add_minus_fields)) if any(['-' in i, '+' in i, '*' in i])]

    for i in add_minus_fields:
        x = [op.strip() for op in i.split()]
        if x[0] in "*+-": raise Exception("Invalid formula")
        temp = df[x[0]].copy()
        n = 1
        while n < len(x):
            if x[n] == '+':
                temp += df[x[n + 1]].replace(np.nan, 0)
            elif x[n] == '-':
                temp -= df[x[n + 1]].replace(np.nan, 0)
            elif x[n] == '*':
                temp *= df[x[n + 1]]
            else:
                raise Exception(f"Unexpected operand/operator: {x[n]}")
            n += 2
        df[i] = temp

    # a) Keep original values
    keep_original_mask = formula['field_denom'].isnull() & formula['field_num'].notnull()
    new_name = formula.loc[keep_original_mask, 'name'].to_list()
    old_name = formula.loc[keep_original_mask, 'field_num'].to_list()
    df[new_name] = df[old_name]

    # b) Time series ratios (Calculate 1m change first)
    print(f'      ------------------------> Calculate time-series ratio ')
    if sample_interval == 'monthly':
        period_yr = 12
        period_q = 3
    elif sample_interval == 'weekly':
        period_yr = 52
        period_q = 12

    for r in formula.loc[formula['field_num'] == formula['field_denom'], ['name', 'field_denom']].to_dict(
            orient='records'):  # minus calculation for ratios
        print('Calculating:', r['name'])
        if r['name'][-2:] == 'yr':
            df[r['name']] = df[r['field_denom']] / df[r['field_denom']].shift(period_yr) - 1
            df.loc[df.groupby('ticker').head(period_yr).index, r['name']] = np.nan
        elif r['name'][-1] == 'q':
            df[r['name']] = df[r['field_denom']] / df[r['field_denom']].shift(period_q) - 1
            df.loc[df.groupby('ticker').head(period_q).index, r['name']] = np.nan

    # c) Divide ratios
    print(f'      ------------------------> Calculate dividing ratios ')
    for r in formula.dropna(how='any', axis=0).loc[(formula['field_num'] != formula['field_denom'])].to_dict(
            orient='records'):  # minus calculation for ratios
        print('Calculating:', r['name'])
        df[r['name']] = df[r['field_num']] / df[r['field_denom']].replace(0, np.nan)

    # drop records with no stock_return_y & any ratios
    dropna_col = set(df.columns.to_list()) & set(['stock_return_y']+formula['name'].to_list())
    df = df.dropna(subset=list(dropna_col), how='all')
    df = df.replace([np.inf, -np.inf], np.nan)

    # test ratio calculation missing rate
    test_missing(df, formula[['name','field_num','field_denom']], ingestion_cols)
    print(f'      ------------------------> Save missing Excel')

    db_table_name = global_vals.processed_ratio_table
    if sample_interval == 'weekly':
        db_table_name += f'_weekly{rolling_period}'
    elif price_sample == 'last_week_avg':
        db_table_name += f'_monthly{rolling_period}'

    df['stock_return_y_ffill'] = df.groupby('ticker')['stock_return_y'].ffill()
    df = df.dropna(subset=['stock_return_y_ffill'])

    # save calculated ratios to DB
    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 1000}
        filter_col = set(df.columns.to_list()) & set(['ticker','period_end','currency_code','icb_code', 'stock_return_y']+formula['name'].to_list())
        ddf = df[list(filter_col)]
        ddf['peroid_end'] = pd.to_datetime(ddf['period_end'])
        ddf.to_sql(db_table_name, **extra)
        record_table_update_time(db_table_name, conn)
        print(f'      ------------------------> Finish writing {db_table_name} table ', ddf.shape)
    global_vals.engine_ali.dispose()
    return df, stocks_col, formula

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
        # print(df)

if __name__ == "__main__":
    # update_period_end()
    calc_factor_variables(price_sample='last_week_avg',
                          fill_method='fill_all',
                          sample_interval='weekly',
                          rolling_period=1,
                          use_cached=False,
                          save=True,
                          ticker='AAPL.O',
                          currency=None)
