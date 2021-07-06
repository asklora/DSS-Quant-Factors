import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta

##########################################################################################
############################ Calculate monthly stock return ##############################

def FillAllDay(result):
    ''' Fill all the weekends between first / last day and fill NaN'''

    # Construct indexes for all day between first/last day * all ticker used
    df = result[["ticker", "trading_day"]]
    df.trading_day = pd.to_datetime(df['trading_day'])
    result.trading_day = pd.to_datetime(result['trading_day'])
    df = df.sort_values(by=['trading_day'], ascending=True)
    daily = pd.date_range(df.iloc[0,1], df.iloc[-1,1], freq='D')
    indexes = pd.MultiIndex.from_product([df['ticker'].unique(), daily], names=['ticker', 'trading_day'])

    # Insert weekend/before first trading date to df
    df = df.set_index(['ticker', 'trading_day']).reindex(indexes).reset_index()
    df = df.sort_values(by=['ticker', 'trading_day'], ascending=True)
    result = df.merge(result, how="left", on=["ticker", "trading_day"])

    return result

def get_rogers_satchell(tri, lookback=90, days_in_year=256):
    ''' Calculate roger satchell volatility '''

    open_data, high_data, low_data, close_data = tri['open'].values, tri['high'].values, tri['low'].values, tri['close'].values

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

    def moving_average(x, n=lookback):
        ''' calculate rolling average which skips NaN records '''
        a = np.ma.masked_array(x,np.isnan(x))
        ret = np.cumsum(a.filled(0))
        ret[n:] = ret[n:] - ret[:-n]
        counts = np.cumsum(~a.mask)
        counts[n:] = counts[n:] - counts[:-n]
        ret[~a.mask] /= counts[~a.mask]
        ret[a.mask] = np.nan
        return ret

    rogers_satchell_var = moving_average(sum, lookback)

    tri['vol'] = np.sqrt(rogers_satchell_var * days_in_year)
    tri.loc[tri.groupby('ticker').head(lookback-1).index, 'vol'] = np.nan  # y-1 ~ y0

    return tri

def resample_to_monthly(df, date_col):
    ''' Resample to monthly stock tri '''
    monthly = pd.date_range(min(df[date_col].to_list()), max(df[date_col].to_list()), freq='M')
    df = df.loc[df[date_col].isin(monthly)]
    return df

def calc_stock_return():
    ''' Calcualte monthly stock return '''

    tri = pd.read_csv('data_tri.csv')

    # Get tri for all ticker in universe from Database
    # engine = global_vals.engine
    # with engine.connect() as conn:
    #     query = text(f"SELECT ticker, trading_day, total_return_index as tri, open, high, low, close, day_status FROM {global_vals.stock_data_table}")
    #     tri = pd.read_sql(query, con=conn)
    #     tri.to_csv('data_tri.csv', index=False)
    # engine.dispose()

    tri = tri.replace(0, np.nan)    # Remove all 0 since total_return_index not supposed to be 0

    tri = FillAllDay(tri)      # Add NaN record of tri for weekends

    # Calculate RS volatility with 365 days lookback period (before ffill)
    tri = get_rogers_satchell(tri, lookback=365)

    # Fill forward (-> holidays/weekends) + backward (<- first trading price)
    cols = ['tri', 'close', 'vol']
    tri.update(tri.groupby('ticker')[cols].fillna(method='ffill'))

    tri = resample_to_monthly(tri, date_col='trading_day')  # Resample to monthly stock tri

    # Calculate monthly return (Y) + R6,2 + R12,7
    tri["tri_1ma"] = tri.groupby('ticker')['tri'].shift(-1)
    tri["tri_1mb"] = tri.groupby('ticker')['tri'].shift(1)
    tri['tri_6mb'] = tri.groupby('ticker')['tri'].shift(6)
    tri['tri_12mb'] = tri.groupby('ticker')['tri'].shift(12)

    tri["stock_return_y"] = (tri["tri_1ma"] / tri["tri"]) - 1
    tri["stock_return_r10"] = (tri["tri"] / tri["tri_1mb"]) - 1
    tri["stock_return_r62"] = (tri["tri_1mb"] / tri["tri_6mb"]) - 1
    tri["stock_return_r127"] = (tri["tri_6mb"] / tri["tri_12mb"]) - 1

    tri = tri.dropna(subset=['stock_return_y'])
    tri = tri.drop(['tri', 'tri_1ma', 'tri_1mb', 'tri_6mb','tri_12mb'], axis=1)

    tri.to_csv('data_tri_final.csv', index=False)

    return tri

##########################################################################################
################################## Calculate factors #####################################

def download_clean_macros():
    ''' download macros data from DB and preprocess: convert some to yoy format'''

    with global_vals.engine.connect() as conn:
        macros = pd.read_sql(f'SELECT * FROM {global_vals.macro_data_table} WHERE period_end IS NOT NULL', conn)
    global_vals.engine.dispose()

    # macros = macros.sort_values(by=[global_vals.date_column, 'trading_day']).drop_duplicates(
    #     subset=[global_vals.date_column], keep='last')    # Keep most recently monthly data

    macros['trading_day'] = pd.to_datetime(macros['trading_day'], format='%Y-%m-%d')
    yoy_col = macros.select_dtypes('float').columns[macros.mean(axis=0) > 100]     # convert YoY
    macros[yoy_col] = (macros[yoy_col]/macros[yoy_col].shift(4)).sub(1)     # convert yoy_col to YoY

    return macros.drop(['period_end'], axis=1)

def download_clean_worldscope_ibes():
    ''' download all data for factor calculate & LGBM input (except for stock return) '''

    with global_vals.engine.connect() as conn:
        ws = pd.read_sql(f'select ticker, period_end, fn_18264 from {global_vals.worldscope_quarter_summary_table} WHERE ticker is not null', conn)   # quarterly records
        ibes = pd.read_sql(f'SELECT * FROM {global_vals.ibes_data_table}', conn)           # ibes_data
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)   # ratio calculation used
        universe = pd.read_sql(f"SELECT ticker, currency_code, icb_code FROM {global_vals.stock_data_table}", conn)
    global_vals.engine.dispose()

    def drop_dup(df):
        ''' drop duplicate records for same identifier & fiscal period, keep the most complete records '''

        df['count'] = pd.isnull(df).sum(1)  # count the missing in each records (row)
        df = df.sort_values(['count']).drop_duplicates(subset=['ticker', 'period_end'], keep='first')
        return df.drop('count', 1)

    ws = drop_dup(ws)       # drop duplicate and retain the most complete record

    def fill_missing_ws(ws):
        ''' fill in missing values by calculating with existing data '''

        ws['fn_18199'] = ws['fn_18199'].fillna(ws['fn_3255'] - ws['fn_2001']) # Net debt = total debt - C&CE
        ws['fn_18308'] = ws['fn_18308'].fillna(ws['fn_18271'] + ws['fn_18269']) # TTM EBIT = TTM Pretax Income + TTM Interest Exp.
        ws['fn_18309'] = ws['fn_18309'].fillna(ws['fn_18308'] + ws['fn_18313']) # TTM EBITDA = TTM EBIT + TTM DDA

        return ws

    ws = fill_missing_ws(ws)        # selectively fill some missing fields

    return ws, ibes, universe, formula

def count_sample_number(tri):
    ''' count number of samples for each period & each indstry / currency
        -> Select to use 6-digit code = on average 37 samples '''

    c1 = tri.groupby(['trading_day','icb_code']).count()['stock_return_y'].unstack(level=1)
    print(c1.mean().mean())
    c1.to_csv('c1.csv', index=False)

    c2 = tri.groupby(['trading_day','currency_code']).count()['stock_return_y'].unstack(level=1)
    pd.concat([c1, c2], axis=1).to_csv('number_of_ticker_per_ind_curr.csv')
    print(c1.mean().mean())
    exit(0)

def combine_stock_factor_data():        #### Change to combine by report_date
    ''' This part do the following:
        1. import all data from DB refer to other functions
        2. combined stock_return, worldscope, ibes, macroeconomic tables '''

    # 1. Stock return/volatility/volume(?)
    # tri = calc_stock_return()
    tri = pd.read_csv('data_tri_final.csv')
    tri['trading_day'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')

    # 2. Fundamental financial data - Worldscope
    # 3. Consensus forecasts - I/B/E/S
    # 4. Universe
    ws, ibes, universe, formula = download_clean_worldscope_ibes()
    ws['period_end'] = pd.to_datetime(ws['period_end'], format='%Y-%m-%d')
    ibes['trading_day'] = pd.to_datetime(ibes['trading_day'], format='%Y-%m-%d')

    # 5. Local file for Market Cap (to be uploaded)
    market_cap = pd.read_csv('mktcap.csv')

    # Use 6-digit ICB code in industry groups
    universe['icb_code'] = universe['icb_code'].astype(str).str[:6]

    # Combine all data for table (1) - (5) above
    df = pd.merge(tri, ws, left_on=['ticker','trading_day'], right_on=['ticker', 'period_end'], how='outer')
    df = df.merge(ibes, on=['ticker','trading_day'], how='outer')
    df = df.merge(universe, on=['ticker'], how='outer')
    df = df.merge(market_cap, on=['ticker','trading_day'], how='outer')

    # Forward fill for fundamental data (e.g. Quarterly June -> Monthly July/Aug)
    df = df.sort_values(by=['ticker','trading_day'])
    cols = df.select_dtypes('float').columns.to_list()
    df.update(df.groupby('ticker')[cols].fillna(method='ffill'))

    df = resample_to_monthly(df, date_col='trading_day')  # Resample to monthly stock tri

    return df


def calc_factor_variables():
    ''' Calculate all factor used referring to DB ratio table '''
    ##############################################Start Here
    # 1. Combine all df
    # 2. Calculate roic_num/denom/mcap
    # +++ Calculate vol
    # 3. Calculate ratios refer to formula table (to be uploaded)
    # 4. adjust code to calculate return for multiple columns

    # Field for ROIC calculations
    ws['roic_num'] = ws['fn_18309'] - ws['fn_18311']
    ws['roic_demon'] = ws['fn_18309'] - ws['fn_18311']
    ws['roic_num'] = ws['fn_18309'] - ws['fn_18311']


    # Calculate all factors
    df['earning_yield'] = df['fn_18264']/df['close']    ### Change to using formula table
    print(df.describe())

    def calc_monthly_premium_within_group(g): ### Change to for loop #
        g = g.dropna(subset = ['earning_yield'])

        if len(g) > 65:     # If industry(6) sample size is large
            g['earning_yield_cut'] = pd.qcut(g['earning_yield'], q=[0, 0.2, 0.8, 1], retbins=False, labels=False)
            return g.loc[g['earning_yield_cut']==0, 'stock_return_y'].mean() - g.loc[g['earning_yield_cut']==2, 'stock_return_y'].mean()
        elif len(g) < 10:   # If industry(6) sample size is small
            return np.nan
        else:
            g['earning_yield_cut'] = pd.qcut(g['earning_yield'], q=[0, 0.3, 0.7, 1], retbins=False, labels=False)
            return g.loc[g['earning_yield_cut'] == 0, 'stock_return_y'].mean() - g.loc[g['earning_yield_cut'] == 2, 'stock_return_y'].mean()

    results = df.groupby(['trading_day','icb_code']).apply(calc_monthly_premium_within_group)
    print(results)

    return df

if __name__ == "__main__":
    calc_stock_return()
    # df = combine_stock_factor_data()
    # print(df.describe())
    combine_stock_factor_data()