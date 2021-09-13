import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from scipy.stats import skew

# ----------------------------------------- Calculate Stock Ralated Factors --------------------------------------------

def get_tri(save=True, currency=None):
    if not isinstance(save, bool):
        raise Exception("Parameter 'save' must be a bool")
    # if not isinstance(currency, str):
    #     raise Exception("Parameter 'currency' must be a str")

    with global_vals.engine.connect() as conn_droid, global_vals.engine_ali.connect() as conn_ali:
        print(f'#################################################################################################')
        print(f'      ------------------------> Download stock data from {global_vals.stock_data_table_tri}')
        conditions = ["True"]
        if currency:
            conditions.append(f"currency_code = '{currency}'")
        query = text(f"SELECT T.ticker, T.trading_day, total_return_index as tri, open, high, low, close, volume "
                     f"FROM {global_vals.stock_data_table_tri} T "
                     f"INNER JOIN {global_vals.stock_data_table_ohlc} C ON T.dsws_id = C.dss_id "
                     f"WHERE {' AND '.join(conditions)}")
        tri = pd.read_sql(query, con=conn_droid, chunksize=10000)
        tri = pd.concat(tri, axis=0, ignore_index=True)

        print(f'      ------------------------> Download stock data from {global_vals.eikon_price_table}/{global_vals.eikon_mktcap_table}')
        eikon_price = pd.read_sql(f"SELECT * FROM {global_vals.eikon_price_table} ORDER BY ticker, trading_day", conn_ali, chunksize=10000)
        eikon_price = pd.concat(eikon_price, axis=0, ignore_index=True)
        market_cap_anchor = pd.read_sql(f'SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trading_day DESC) '
                                        f'rank FROM {global_vals.eikon_mktcap_table}) a WHERE a.rank = 1;', conn_ali).iloc[:,:-1]
        if save:
            tri.to_csv('cache_tri.csv', index=False)
            eikon_price.to_csv('cache_eikon_price.csv', index=False)
            market_cap_anchor.to_csv('cache_market_cap_anchor.csv', index=False)
    global_vals.engine.dispose()
    global_vals.engine_ali.dispose()
    return tri, eikon_price, market_cap_anchor

def fill_all_day(result):
    ''' Fill all the weekends between first / last day and fill NaN'''

    # Construct indexes for all day between first/last day * all ticker used
    df = result[["ticker", "trading_day"]].copy()
    df.trading_day = pd.to_datetime(df['trading_day'])
    result.trading_day = pd.to_datetime(result['trading_day'])
    df = df.sort_values(by=['trading_day'], ascending=True)
    daily = pd.date_range(df.iloc[0, 1], df.iloc[-1, 1], freq='D')
    indexes = pd.MultiIndex.from_product([df['ticker'].unique(), daily], names=['ticker', 'trading_day'])

    # Insert weekend/before first trading date to df
    df = df.set_index(['ticker', 'trading_day']).reindex(indexes).reset_index()
    df = df.sort_values(by=['ticker', 'trading_day'], ascending=True)
    result = df.merge(result, how="left", on=["ticker", "trading_day"])

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

def calc_stock_return(price_sample, sample_interval, use_cached, save):
    ''' Calcualte monthly stock return '''

    if use_cached:
        try:
            tri = pd.read_csv('cache_tri.csv', low_memory=False)
            eikon_price = pd.read_csv('cache_eikon_price.csv', low_memory=False)
            market_cap_anchor = pd.read_csv('cache_market_cap_anchor.csv', low_memory=False)
        except Exception as e:
            print(e)
            tri, eikon_price, market_cap_anchor = get_tri(save=save)
    else:
        tri, eikon_price, market_cap_anchor = get_tri(save=save)

    # merge stock return from DSS & from EIKON (i.e. longer history)
    tri['trading_day'] = pd.to_datetime(tri['trading_day'])
    eikon_price['trading_day'] = pd.to_datetime(eikon_price['trading_day'])
    market_cap_anchor['trading_day'] = pd.to_datetime(market_cap_anchor['trading_day'], format='%Y-%m-%d')

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
    list_of_start_end = [[0, 30], [30, 90], [90, 182]]
    tri = get_rogers_satchell(tri, list_of_start_end)
    tri = tri.drop(['open', 'high', 'low'], axis=1)

    # resample tri using last week average as the proxy for monthly tri
    print(f'      ------------------------> Stock price using [{price_sample}] ')
    tri[['tri','volume']] = tri.groupby("ticker")[['tri','volume']].rolling(7, min_periods=1).mean().reset_index(drop=1)

    # Fill forward (-> holidays/weekends) + backward (<- first trading price)
    cols = ['tri', 'close','volume'] + [f'vol_{l[0]}_{l[1]}' for l in list_of_start_end]
    tri.update(tri.groupby('ticker')[cols].fillna(method='ffill'))

    print(f'      ------------------------> Sample interval using [{sample_interval}] ')
    if sample_interval == 'monthly':
        tri = resample_to_monthly(tri, date_col='trading_day')  # Resample to monthly stock tri
    elif sample_interval == 'biweekly':
        tri = resample_to_biweekly(tri, date_col='trading_day')  # Resample to bi-weekly stock tri
    else:
        raise ValueError("Invalid sample_interval method. Expecting 'monthly' or 'biweekly' got ", sample_interval)

    # update market_cap/market_cap_usd refer to tri for each period
    tri = tri.merge(market_cap_anchor, on=['ticker','trading_day'], how='left')
    tri.loc[tri['market_cap'].notnull(), 'anchor_tri'] = tri.loc[tri['market_cap'].notnull(), 'tri']
    tri[['anchor_tri','market_cap','market_cap_usd']] = tri.groupby('ticker')[['anchor_tri','market_cap','market_cap_usd']].apply(lambda x: x.ffill().bfill())
    tri['market_cap'] = tri['market_cap']/tri['anchor_tri']*tri['tri']
    tri['market_cap_usd'] = tri['market_cap_usd']/tri['anchor_tri']*tri['tri']
    tri = tri.drop(['anchor_tri'], axis=1)

    # Calculate monthly return (Y) + R6,2 + R12,7
    print(f'      ------------------------> Calculate stock returns ')
    tri["tri_1ma"] = tri.groupby('ticker')['tri'].shift(-1)
    if sample_interval == 'monthly':
        tri["tri_1mb"] = tri.groupby('ticker')['tri'].shift(1)
        tri['tri_6mb'] = tri.groupby('ticker')['tri'].shift(6)
        tri['tri_12mb'] = tri.groupby('ticker')['tri'].shift(12)
    elif sample_interval == 'biweekly':
        tri["tri_1mb"] = tri.groupby('ticker')['tri'].shift(2)
        tri['tri_6mb'] = tri.groupby('ticker')['tri'].shift(12)
        tri['tri_12mb'] = tri.groupby('ticker')['tri'].shift(24)

    tri["stock_return_y"] = (tri["tri_1ma"] / tri["tri"]) - 1
    tri["stock_return_r1_0"] = (tri["tri"] / tri["tri_1mb"]) - 1
    tri["stock_return_r6_2"] = (tri["tri_1mb"] / tri["tri_6mb"]) - 1
    tri["stock_return_r12_7"] = (tri["tri_6mb"] / tri["tri_12mb"]) - 1

    tri = tri.drop(['tri', 'tri_1ma', 'tri_1mb', 'tri_6mb', 'tri_12mb'], axis=1)
    stock_col = tri.select_dtypes('float').columns  # all numeric columns

    if save:
        tri.to_csv('cache_tri_ratio.csv', index=False)

    return tri, stock_col

# -------------------------------------------- Calculate Fundamental Ratios --------------------------------------------

def update_period_end(ws):
    ''' map icb_sector, member_ric, period_end -> last_year_end for each identifier + frequency_number * 3m '''

    print(f'      ------------------------> Update period_end in {global_vals.worldscope_quarter_summary_table} ')

    with global_vals.engine.connect() as conn:
        universe = pd.read_sql(f'SELECT ticker, fiscal_year_end FROM {global_vals.dl_value_universe_table}', conn, chunksize=10000)
        universe = pd.concat(universe, axis=0, ignore_index=True)
    global_vals.engine.dispose()

    ws = ws.dropna(subset=['year'])
    ws = pd.merge(ws, universe, on='ticker', how='left')   # map static information for each company

    ws = ws.loc[ws['fiscal_year_end'].isin(['MAR','JUN','SEP','DEC'])]      # select identifier with correct year end

    ws['period_end'] = pd.to_datetime(ws['period_end'], format='%Y-%m-%d')
    ws['report_date'] = pd.to_datetime(ws['report_date'], format='%Y-%m-%d')

    # dataframe for original period - report_date matching
    ws_report_date_remap = ws[['ticker','period_end','report_date']]
    ws = ws.drop(['report_date'], axis=1)

    # find last fiscal year end for each company (ticker)
    ws['fiscal_year_end'] = ws['fiscal_year_end'].replace(['MAR','JUN','SEP','DEC'], ['0331','0630','0930','1231'])
    ws['last_year_end'] = (ws['year'].astype(int)- 1).astype(str) + ws['fiscal_year_end']
    ws['last_year_end'] = pd.to_datetime(ws['last_year_end'], format='%Y%m%d')

    # find period_end for each record (row)
    ws['period_end'] = ws.apply(lambda x: x['last_year_end'] + MonthEnd(x['frequency_number']*3), axis=1)

    # Update report_date with the updated period_end
    ws = ws.merge(ws_report_date_remap, on=['ticker','period_end'])
    ws['report_date'] = ws['report_date'].mask(ws['report_date']<ws['period_end'], np.nan) + MonthEnd(0)
    ws['report_date'] = ws['report_date'].mask(ws['frequency_number']!=4, ws['report_date'].fillna(ws['period_end'] + MonthEnd(3)))
    ws['report_date'] = ws['report_date'].mask(ws['frequency_number']==4, ws['report_date'].fillna(ws['period_end'] + MonthEnd(3)))     # assume all report issued within 3 months

    ws['period_end'] = ws['report_date']        # align worldscope data with stock return using report date
    ws = ws.loc[ws[global_vals.date_column]<dt.datetime.today()]

    return ws.drop(['last_year_end','fiscal_year_end','year','frequency_number','fiscal_quarter_end','report_date'], axis=1)

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

def download_clean_worldscope_ibes(save):
    ''' download all data for factor calculate & LGBM input (except for stock return) '''

    with global_vals.engine.connect() as conn:
        print(f'#################################################################################################')
        query_ws = f'select * from {global_vals.worldscope_quarter_summary_table} WHERE ticker is not null'
        query_ibes = f'SELECT * FROM {global_vals.ibes_data_table}'
        print(f'      ------------------------> Download worldscope data from {global_vals.worldscope_quarter_summary_table}')
        ws = pd.read_sql(query_ws, conn, chunksize=10000)  # quarterly records
        ws = pd.concat(ws, axis=0, ignore_index=True)
        print(f'      ------------------------> Download ibes data from {global_vals.ibes_data_table}')
        ibes = pd.read_sql(query_ibes, conn, chunksize=10000)  # ibes_data
        ibes = pd.concat(ibes, axis=0, ignore_index=True)
        universe = pd.read_sql(f"SELECT ticker, currency_code, icb_code FROM {global_vals.dl_value_universe_table}", conn, chunksize=10000)
        universe = pd.concat(universe, axis=0, ignore_index=True)
    global_vals.engine.dispose()

    def drop_dup(df):
        ''' drop duplicate records for same identifier & fiscal period, keep the most complete records '''

        print(f'      ------------------------> Drop duplicates in {global_vals.worldscope_quarter_summary_table} ')

        df['count'] = pd.isnull(df).sum(1)  # count the missing in each records (row)
        df = df.sort_values(['count']).drop_duplicates(subset=['ticker', 'period_end'], keep='first')
        return df.drop('count', axis=1)

    ws = drop_dup(ws)  # drop duplicate and retain the most complete record

    def fill_missing_ws(ws):
        ''' fill in missing values by calculating with existing data '''

        print(f'      ------------------------> Fill missing in {global_vals.worldscope_quarter_summary_table} ')

        ws['fn_18199'] = ws['fn_18199'].fillna(ws['fn_3255'] - ws['fn_2001'])  # Net debt = total debt - C&CE
        ws['fn_18308'] = ws['fn_18308'].fillna(
            ws['fn_18271'] + ws['fn_18269'])  # TTM EBIT = TTM Pretax Income + TTM Interest Exp.
        ws['fn_18309'] = ws['fn_18309'].fillna(ws['fn_18308'] + ws['fn_18313'])  # TTM EBITDA = TTM EBIT + TTM DDA

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

def combine_stock_factor_data(price_sample='last_day', fill_method='fill_all', sample_interval='monthly',
                              use_cached=False, save=True, update=False):
    ''' This part do the following:
        1. import all data from DB refer to other functions
        2. combined stock_return, worldscope, ibes, macroeconomic tables '''

    # 1. Stock return/volatility/volume
    if use_cached:
        try:
            tri = pd.read_csv('cache_tri_ratio.csv', low_memory=False)
            stocks_col = tri.select_dtypes("float").columns
        except Exception as e:
            print(e)
            tri, stocks_col = calc_stock_return(price_sample, sample_interval, use_cached, save)
    else:
        tri, stocks_col = calc_stock_return(price_sample, sample_interval, use_cached, save)

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
            ws, ibes, universe = download_clean_worldscope_ibes(save)
    else:
        ws, ibes, universe = download_clean_worldscope_ibes(save)

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
    df = pd.merge(tri.drop("trading_day", axis=1), ws, on=['ticker', 'period_end'], how='left')
    df = df.merge(ibes, on=['ticker', 'period_end'], how='left')
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
    if fill_method == 'fill_all':           # e.g. Quarterly June -> Monthly July/Aug
        df.update(df.groupby(['ticker'])[cols].fillna(method='ffill'))
    elif fill_method == 'fill_monthly':     # e.g. only 1 June -> 30 June
        df.update(df.groupby(['ticker', 'period_end'])[cols].fillna(method='ffill'))
    else:
        raise ValueError("Invalid fill_method. Expecting 'fill_all' or 'fill_monthly' got ", fill_method)

    if sample_interval == 'monthly':
        df = resample_to_monthly(df, date_col='period_end')  # Resample to monthly stock tri
    elif sample_interval == 'biweekly':
        df = resample_to_biweekly(df, date_col='period_end')  # Resample to monthly stock tri
    else:
        raise ValueError("Invalid sample_interval method. Expecting 'monthly' or 'biweekly' got ", sample_interval)

    df = df.merge(universe, on=['ticker'], how='left')      # label icb_code, currency_code for each ticker

    if save:
        df.to_csv('cache_all_data.csv')  # for debug
        pd.DataFrame(stocks_col).to_csv('cache_stocks_col.csv', index=False)  # for debug

    check_duplicates(df, 'final')
    return df, stocks_col

def calc_factor_variables(price_sample='last_day', fill_method='fill_all', sample_interval='monthly',
                          use_cached=False, save=True, update=True):
    ''' Calculate all factor used referring to DB ratio table '''

    if update:  # update for the latest month (not using cachec & not save locally)
        use_cached = False
        save = False

    if use_cached:
        try:
            df = pd.read_csv('cache_all_data.csv', low_memory=False, dtype={"icb_code": str})
            stocks_col = pd.read_csv('cache_stocks_col.csv', low_memory=False).iloc[:, 0].to_list()
        except Exception as e:
            print(e)
            df, stocks_col = combine_stock_factor_data(price_sample, fill_method, sample_interval, use_cached, save)
    else:
        df, stocks_col = combine_stock_factor_data(price_sample, fill_method, sample_interval, use_cached, save, update)

    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table} WHERE x_col', conn, chunksize=10000)  # ratio calculation used
        formula = pd.concat(formula, axis=0, ignore_index=True)
    global_vals.engine_ali.dispose()

    print(f'#################################################################################################')
    print(f'      ------------------------> Calculate all factors in {global_vals.formula_factors_table}')

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
    elif sample_interval == 'biweekly':
        period_yr = 24
        period_q = 6

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
    df = df.dropna(subset=['stock_return_y']+formula['name'].to_list(), how='all')

    db_table_name = global_vals.processed_ratio_table
    if sample_interval == 'biweekly':
        db_table_name += '_biweekly'
    elif price_sample == 'last_week_avg':
        db_table_name += '_weekavg'

    # save calculated ratios to DB
    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 100000}
        ddf = df[['ticker','period_end','currency_code','icb_code', 'stock_return_y']+formula['name'].to_list()].dropna(subset=['stock_return_y'])
        ddf['peroid_end'] = pd.to_datetime(ddf['period_end'])
        ddf.to_sql(db_table_name, **extra)
        print(f'      ------------------------> Finish writing {db_table_name} table ', ddf.shape)
    return df, stocks_col, formula


if __name__ == "__main__":

    calc_factor_variables(price_sample='last_week_avg', fill_method='fill_all', sample_interval='monthly',
                          use_cached=True, save=False)
