import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, QuarterEnd
from scipy.stats import skew

##########################################################################################
################# Calculate monthly stock return/volatility ##############################

def get_tri(engine, save=True):
    with engine.connect() as conn:
        query = text(f"""SELECT ticker, trading_day, total_return_index as tri, open, high, low, close, volume 
        FROM {global_vals.stock_data_table}
        """)
        tri = pd.read_sql(query, con=conn)
        if save:
            tri.to_csv('data_tri.csv', index=False)
    engine.dispose()
    return tri

def FillAllDay(result):
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
    sum = np.add(input1, input2)

    # Calculate annualize volatility
    for l in list_of_start_end:
        start, end = l[0], l[1]
        name_col = f'vol_{start}_{end}'
        tri[name_col] = pd.Series(sum).rolling(end - start, min_periods=1).mean()
        tri[name_col] = tri[name_col].apply(lambda x: np.sqrt(x * days_in_year))
        tri[name_col] = tri[name_col].shift(start)
        tri.loc[tri.groupby('ticker').head(end - 1).index, name_col] = np.nan  # y-1 ~ y0

    return tri

def get_skew(tri):
    ''' Calculate past 1yr daily return skewness '''

    tri["tri_1d"] = tri['tri']/tri.groupby('ticker')['tri'].shift(1)
    tri['skew'] = tri["tri_1d"].rolling(365, min_periods=1).skew()
    tri.loc[tri.groupby('ticker').head(364).index, 'skew'] = np.nan  # y-1 ~ y0

    return tri

def resample_to_monthly(df, date_col):
    ''' Resample to monthly stock tri '''
    monthly = pd.date_range(min(df[date_col].to_list()), max(df[date_col].to_list()), freq='M')
    df = df.loc[df[date_col].isin(monthly)]
    return df

def resample_to_biweekly(df, date_col):
    ''' Resample to monthly stock tri '''
    monthly = pd.date_range(min(df[date_col].to_list()), max(df[date_col].to_list()),  freq='2W')
    df = df.loc[df[date_col].isin(monthly)]
    return df

def calc_stock_return(price_sample, sample_interval, use_cached, save):
    ''' Calcualte monthly stock return '''

    engine = global_vals.engine

    if use_cached:
        try:
            tri = pd.read_csv('data_tri.csv')
        except Exception as e:
            print(e)
            print(f'#################################################################################################')
            print(f'#      ------------------------> Download stock data from {global_vals.stock_data_table}')
            tri = get_tri(engine, save=save)
    else:
        print(f'#################################################################################################')
        print(f'      ------------------------> Download stock data from {global_vals.stock_data_table}')
        tri = get_tri(engine, save=save)

    tri = tri.replace(0, np.nan)  # Remove all 0 since total_return_index not supposed to be 0

    tri = FillAllDay(tri)  # Add NaN record of tri for weekends

    print(f'      ------------------------> Calculate skewness ')
    tri = get_skew(tri)    # Calculate past 1 year skewness

    # Calculate RS volatility for 3-month & 6-month~2-month (before ffill)
    print(f'      ------------------------> Calculate RS volatility ')
    list_of_start_end = [[0, 30], [30, 90], [90, 182]]
    tri = get_rogers_satchell(tri, list_of_start_end)
    tri = tri.drop(['open', 'high', 'low'], axis=1)

    # resample tri using last week average as the proxy for monthly tri
    print(f'      ------------------------> Stock price using {price_sample} ')
    if price_sample == 'last_week_avg':
        tri['tri'] = tri['tri'].rolling(7, min_periods=1).mean()
        tri.loc[tri.groupby('ticker').head(6).index, ['tri']] = np.nan
    elif price_sample == 'last_day':
        pass
    else:
        raise ValueError("Invalid price_sample method. Expecting 'last_week_avg' or 'last_day' got ", price_sample)

    # Fill forward (-> holidays/weekends) + backward (<- first trading price)
    cols = ['tri', 'close'] + [f'vol_{l[0]}_{l[1]}' for l in list_of_start_end]
    tri.update(tri.groupby('ticker')[cols].fillna(method='ffill'))

    print(f'      ------------------------> Sample interval using {sample_interval} ')
    if sample_interval == 'monthly':
        tri = resample_to_monthly(tri, date_col='trading_day')  # Resample to monthly stock tri
    elif sample_interval == 'biweekly':
        tri = resample_to_biweekly(tri, date_col='trading_day')  # Resample to bi-weekly stock tri
    else:
        raise ValueError("Invalid sample_interval method. Expecting 'monthly' or 'biweekly' got ", sample_interval)

    # Calculate monthly return (Y) + R6,2 + R12,7
    print(f'      ------------------------> Calculate stock returns ')
    tri["tri_1ma"] = tri.groupby('ticker')['tri'].shift(-1)
    tri["tri_1mb"] = tri.groupby('ticker')['tri'].shift(1)
    tri['tri_6mb'] = tri.groupby('ticker')['tri'].shift(6)
    tri['tri_12mb'] = tri.groupby('ticker')['tri'].shift(12)

    tri["stock_return_y"] = (tri["tri_1ma"] / tri["tri"]) - 1
    tri["stock_return_r1_0"] = (tri["tri"] / tri["tri_1mb"]) - 1
    tri["stock_return_r6_2"] = (tri["tri_1mb"] / tri["tri_6mb"]) - 1
    tri["stock_return_r12_7"] = (tri["tri_6mb"] / tri["tri_12mb"]) - 1

    tri = tri.dropna(subset=['stock_return_y'])
    tri = tri.drop(['tri', 'tri_1ma', 'tri_1mb', 'tri_6mb', 'tri_12mb'], axis=1)

    stock_col = tri.select_dtypes('float').columns  # all numeric columns

    return tri, stock_col

##########################################################################################
################################## Calculate factors #####################################

def download_clean_macros():
    ''' download macros data from DB and preprocess: convert some to yoy format '''

    print(f'#################################################################################################')
    print(f'      ------------------------> Download macro data from {global_vals.macro_data_table}')

    with global_vals.engine.connect() as conn:
        macros = pd.read_sql(f'SELECT * FROM {global_vals.macro_data_table} WHERE period_end IS NOT NULL', conn)
    global_vals.engine.dispose()

    macros['trading_day'] = pd.to_datetime(macros['trading_day'], format='%Y-%m-%d')

    yoy_col = macros.select_dtypes('float').columns[macros.select_dtypes('float').mean(axis=0) > 100]  # convert YoY
    num_col = macros.select_dtypes('float').columns.to_list()  # all numeric columns

    macros[yoy_col] = (macros[yoy_col] / macros[yoy_col].shift(12)).sub(1)  # convert yoy_col to YoY

    return macros.drop(['period_end'], axis=1), num_col

def update_period_end(ws):
    ''' map icb_sector, member_ric, period_end -> last_year_end for each identifier + frequency_number * 3m '''

    print(f'      ------------------------> Update period_end in {global_vals.worldscope_quarter_summary_table} ')

    with global_vals.engine.connect() as conn:
        universe = pd.read_sql(f'SELECT ticker, fiscal_year_end FROM {global_vals.dl_value_universe_table}', conn)
    global_vals.engine.dispose()

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
    ws[global_vals.date_column] = ws.apply(lambda x: x['last_year_end'] + pd.offsets.MonthEnd(x['frequency_number']*3), axis=1)

    # Update report_date with the updated period_end
    ws = ws.merge(ws_report_date_remap, on=['ticker','period_end'])
    ws['report_date'] = ws['report_date'].mask(ws['report_date']<ws['period_end'], np.nan)

    return ws.drop(['last_year_end','fiscal_year_end','year','frequency_number','fiscal_quarter_end'], axis=1)

def download_clean_worldscope_ibes():
    ''' download all data for factor calculate & LGBM input (except for stock return) '''

    with global_vals.engine.connect() as conn:
        print(f'#################################################################################################')
        print(f'      ------------------------> Download worldscope data from {global_vals.worldscope_quarter_summary_table}')
        ws = pd.read_sql(f'select * from {global_vals.worldscope_quarter_summary_table} WHERE ticker is not null', conn)  # quarterly records
        print(f'      ------------------------> Download ibes data from {global_vals.ibes_data_table}')
        ibes = pd.read_sql(f'SELECT * FROM {global_vals.ibes_data_table}', conn)  # ibes_data
        universe = pd.read_sql(f"SELECT ticker, currency_code, icb_code FROM {global_vals.dl_value_universe_table}",
                               conn)
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

    return ws, ibes, universe

def count_sample_number(tri):
    ''' count number of samples for each period & each indstry / currency
        -> Select to use 6-digit code = on average 37 samples '''

    c1 = tri.groupby(['trading_day', 'icb_code']).count()['stock_return_y'].unstack(level=1)
    print(c1.mean().mean())
    c1.to_csv('c1.csv', index=False)

    c2 = tri.groupby(['trading_day', 'currency_code']).count()['stock_return_y'].unstack(level=1)
    pd.concat([c1, c2], axis=1).to_csv('number_of_ticker_per_ind_curr.csv')
    print(c1.mean().mean())
    exit(0)

def combine_stock_factor_data(price_sample, sample_interval, fill_method, use_cached, save):
    ''' This part do the following:
        1. import all data from DB refer to other functions
        2. combined stock_return, worldscope, ibes, macroeconomic tables '''

    # 1. Stock return/volatility/volume(?)
    if use_cached:
        tri = pd.read_csv('data_tri_final.csv')
        stocks_col = tri.select_dtypes("float").columns
    else:
        tri, stocks_col = calc_stock_return(price_sample, sample_interval, use_cached, save)
        if save:
            tri.to_csv('data_tri_final.csv', index=False)

    tri['period_end'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')

    # 2. Fundamental financial data - from Worldscope
    # 3. Consensus forecasts - from I/B/E/S
    # 4. Universe
    ws, ibes, universe = download_clean_worldscope_ibes()
    ws['period_end'] = pd.to_datetime(ws['period_end'], format='%Y-%m-%d')
    ibes['period_end'] = pd.to_datetime(ibes['trading_day'], format='%Y-%m-%d')

    # 5. Local file for Market Cap (to be uploaded) - from Eikon
    with global_vals.engine.connect() as conn:
        market_cap = pd.read_sql(f'SELECT * FROM {global_vals.eikon_mktcap_table}', conn)
    global_vals.engine.dispose()
    market_cap['period_end'] = pd.to_datetime(market_cap['trading_day'], format='%Y-%m-%d')

    # 6. Macroeconomic variables - from Datastream
    macros, macros_col = download_clean_macros()
    macros["period_end"] = macros['trading_day'] + MonthEnd(0)

    # Use 6-digit ICB code in industry groups
    universe['icb_code'] = universe['icb_code'].astype(str).str[:6]

    # Combine all data for table (1) - (6) above
    print(f'      ------------------------> Merge all dataframes ')

    df = pd.merge(tri.drop("trading_day", axis=1), ws, on=['ticker', 'period_end'], how='outer')
    df = df.merge(ibes.drop("trading_day", axis=1), on=['ticker', 'period_end'], how='outer')
    df = df.merge(market_cap.drop("trading_day", axis=1), on=['ticker', 'period_end'], how='outer')
    df = df.merge(macros.drop("trading_day", axis=1), on=['period_end'], how='outer')

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

    return df, stocks_col, macros_col

def calc_factor_variables(price_sample='last_day', fill_method='fill_all', sample_interval='month',
                          use_cached=False, save=True):
    ''' Calculate all factor used referring to DB ratio table '''

    if use_cached:
        df = pd.read_csv('all_data.csv')
        stocks_col = pd.read_csv('stocks_col.csv').iloc[:,0].to_list()
        macros_col = pd.read_csv('macros_col.csv').iloc[:,0].to_list()
    else:
        df, stocks_col, macros_col = combine_stock_factor_data(price_sample, sample_interval, fill_method, use_cached, save)
        if save:
            df.to_csv('all_data.csv') # for debug
            pd.DataFrame(stocks_col).to_csv('stocks_col.csv', index=False)  # for debug
            pd.DataFrame(macros_col).to_csv('macros_col.csv', index=False)  # for debug

    with global_vals.engine.connect() as conn:
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)  # ratio calculation used
    global_vals.engine.dispose()

    print(f'#################################################################################################')
    print(f'      ------------------------> Calculate all factors in {global_vals.formula_factors_table}')

    # Prepare for field requires add/minus
    add_minus_fields = formula[['field_num', 'field_denom']].dropna(how='any').to_numpy().flatten()
    add_minus_fields = [i for i in list(set(add_minus_fields)) if any(['-' in i, '+' in i])]

    for i in add_minus_fields:
        x = [op.strip() for op in i.split()]
        if x[0] in "+-": raise Exception("Invalid formula")
        temp = df[x[0]].copy()
        n = 1
        while n < len(x):
            if x[n] == '+':
                temp += df[x[n+1]]
            elif x[n] == '-':
                temp -= df[x[n+1]]
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
    for r in formula.loc[formula['field_num'] == formula['field_denom'], ['name', 'field_denom']].to_dict(
            orient='records'):  # minus calculation for ratios
        if r['name'][-2:] == 'yr':
            df[r['name']] = df[r['field_denom']] / df[r['field_denom']].shift(12) - 1
            df.loc[df.groupby('ticker').head(12).index, r['name']] = np.nan
        elif r['name'][-1] == 'q':
            df[r['name']] = df[r['field_denom']] / df[r['field_denom']].shift(3) - 1
            df.loc[df.groupby('ticker').head(3).index, r['name']] = np.nan

    # c) Divide ratios
    print(f'      ------------------------> Calculate dividing ratios ')
    for r in formula.dropna(how='any', axis=0).loc[(formula['field_num'] != formula['field_denom'])].to_dict(
            orient='records'):  # minus calculation for ratios
        df[r['name']] = df[r['field_num']] / df[r['field_denom']]

    # df.to_csv('all ratio debug.csv')
    # debug_filter = ~df["ticker"].str.startswith(".")
    # debug_filter &= df["currency_code"].notnull()
    # tmp = df[debug_filter].copy()
    # tmp_ticker_grouped = tmp[["currency_code", "ticker"]].drop_duplicates().groupby("currency_code")["ticker"]
    # min_num_tickers = tmp_ticker_grouped.count().min()
    # target_tickers = tmp_ticker_grouped.sample(min(min_num_tickers, 20), replace=False).tolist()
    # tmp = tmp[tmp["ticker"].isin(target_tickers)]
    # tmp.to_csv('all ratio debug.csv')

    return df, stocks_col, macros_col, formula


if __name__ == "__main__":

    # calc_stock_return(price_sample='last_week_avg', sample_interval='month')
    # download_clean_macros()
    # df = combine_stock_factor_data()
    # print(df.describe())
    calc_factor_variables(price_sample='last_day', fill_method='fill_all', sample_interval='monthly',
                          use_cached=True, save=True)