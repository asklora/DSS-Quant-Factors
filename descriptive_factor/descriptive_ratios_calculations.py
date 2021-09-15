import numpy as np
import pandas as pd
import scipy.stats as stats
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from preprocess.ratios_calculations import check_duplicates, fill_all_day, update_period_end
from scipy.stats import skew

# ----------------------------------------- Calculate Stock Ralated Factors --------------------------------------------
def get_tri(save=True, conditions=None):
    ''' get stock related data from DSS & DSWS '''

    print(f'      ------------------------> Download stock data from {global_vals.stock_data_table_tri}')
    with global_vals.engine.connect() as conn_droid, global_vals.engine_ali.connect() as conn_ali:
        query_tri = f"SELECT C.ticker, trading_day, total_return_index as tri, open, high, low, close, volume "
        query_tri += f"FROM {global_vals.stock_data_table_ohlc} C "
        query_tri += f"INNER JOIN (SELECT dsws_id, total_return_index FROM {global_vals.stock_data_table_tri}) T ON T.dsws_id = C.dss_id "
        query_tri += f"INNER JOIN (SELECT ticker, currency_code FROM {global_vals.dl_value_universe_table}) U ON U.ticker = C.ticker "
        query_tri += f"WHERE {' AND '.join(conditions)}"
        tri = pd.read_sql(query_tri, con=conn_droid, chunksize=10000)
        tri = pd.concat(tri, axis=0, ignore_index=True)

        query_mkt = f"SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trading_day DESC) rank "
        query_mkt += f"FROM {global_vals.eikon_mktcap_table}) a WHERE a.rank = 1;"
        market_cap_anchor = pd.read_sql(query_mkt, conn_ali).iloc[:,:-1]
    global_vals.engine.dispose()
    global_vals.engine_ali.dispose()

    # update market_cap/market_cap_usd/tri_adjusted_close refer to tri for each period
    tri = tri.merge(market_cap_anchor, on=['ticker','trading_day'], how='left')
    tri.loc[tri['market_cap'].notnull(), 'anchor_tri'] = tri.loc[tri['market_cap'].notnull(), 'tri']
    tri[['anchor_tri','market_cap','market_cap_usd']] = tri.groupby('ticker')[['anchor_tri','market_cap','market_cap_usd']].apply(lambda x: x.ffill().bfill())
    tri[['anchor_tri']] = tri['anchor_tri']/tri['tri']
    tri[['market_cap','market_cap_usd','tac']] = tri[['market_cap','market_cap_usd','close']].div(tri['anchor_tri'], axis=0)
    tri = tri.drop(['anchor_tri'], axis=1)

    if save:
        tri.to_csv('dcache_tri.csv', index=False)

    return tri

def get_worldscope(save=True, conditions=None):
    ''' get fundamental data related data from Worldscope '''

    ws_conditions = ["C.ticker IS NOT NULL"]
    if conditions:
        ws_conditions.extend(conditions)

    with global_vals.engine.connect() as conn:
        print(f'      ------------------------> Download worldscope data from {global_vals.worldscope_quarter_summary_table}')
        query_ws = f'SELECT C.*, U.currency_code, U.icb_code FROM {global_vals.worldscope_quarter_summary_table} C '
        query_ws += f"INNER JOIN (SELECT ticker, currency_code, icb_code FROM {global_vals.dl_value_universe_table}) U ON U.ticker = C.ticker "
        query_ws += f"WHERE {' AND '.join(ws_conditions)}"
        ws = pd.read_sql(query_ws, conn, chunksize=10000)  # quarterly records
        ws = pd.concat(ws, axis=0, ignore_index=True)
        query_ibes = f'SELECT ticker, trading_day, eps1tr12 FROM {global_vals.ibes_data_table} WHERE ticker IS NOT NULL'
        ibes = pd.read_sql(query_ibes, conn, chunksize=10000)  # ibes_data
        ibes = pd.concat(ibes, axis=0, ignore_index=True)
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
        ws['fn_18308'] = ws['fn_18308'].fillna(ws['fn_18271'] + ws['fn_18269'])  # TTM EBIT = TTM Pretax Income + TTM Interest Exp.
        ws['fn_18309'] = ws['fn_18309'].fillna(ws['fn_18308'] + ws['fn_18313'])  # TTM EBITDA = TTM EBIT + TTM DDA
        return ws

    ws = fill_missing_ws(ws)        # selectively fill some missing fields
    ws = update_period_end(ws)      # correct timestamp for worldscope data (i.e. period_end)

    # label period_end with month end of trading_day (update_date)
    ws['trading_day'] = pd.to_datetime(ws['period_end'], format='%Y-%m-%d')
    ws = ws.drop(['period_end'], axis=1)
    ws['currency_code'] = ws['currency_code'].replace(['EUR','GBP','USD','HKD','CNY','KRW'], list(np.arange(6)))
    ws['icb_code'] = pd.to_numeric(ws['icb_code'], errors='coerce')

    if save:
        ws.to_csv('dcache_ws.csv', index=False)
        ibes.to_csv('dcache_ibes.csv', index=False)

    return ws, ibes

class combine_tri_worldscope:
    ''' combine tri & worldscope raw data '''

    def __init__(self, use_cached, save, currency=None, ticker=None):
        conditions = ["True"]
        if currency:
            conditions.append("currency_code in ({})".format(','.join(['\''+x+'\'' for x in currency])))
        if ticker:
            conditions.append("C.ticker in ({})".format(','.join(['\''+x+'\'' for x in ticker])))
    
        # 1. Stock return/volatility/volume
        if use_cached:
            try:
                tri = pd.read_csv('dcache_tri.csv', low_memory=False)
            except Exception as e:
                print(e)
                tri = get_tri(save, conditions)
        else:
            tri = get_tri(save, conditions)
    
        # 2. Fundamental financial data - from Worldscope
        # 3. Consensus forecasts - from I/B/E/S
        # 4. Universe
        if use_cached:
            try:
                ws = pd.read_csv('dcache_ws.csv')
                ibes = pd.read_csv('dcache_ibes.csv')
            except Exception as e:
                print(e)
                ws, ibes = get_worldscope(save, conditions)
        else:
            ws, ibes = get_worldscope(save, conditions)
    
        # change to datetime
        tri['trading_day'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')
        ws['trading_day'] = pd.to_datetime(ws['trading_day'], format='%Y-%m-%d')
        ibes['trading_day'] = pd.to_datetime(ibes['trading_day'], format='%Y-%m-%d')
    
        # Merge all table and convert to daily basis
        tri = fill_all_day(tri)
        tri[['volume_1w']] = tri.dropna(subset=['ticker']).groupby(['ticker'])['volume'].rolling(7, min_periods=1).mean().values
        tri[['volume_3m']] = tri.dropna(subset=['ticker']).groupby(['ticker'])['volume'].rolling(90, min_periods=1).mean().values
        tri[['volume_1w3m']] = (tri['volume_1w']/tri['volume_3m']).values
        tri[['market_cap','market_cap_usd','tac']] = tri.groupby(['ticker'])[['market_cap','market_cap_usd','tac']].apply(pd.DataFrame.interpolate, limit_direction='forward')
        self.df = pd.merge(tri, ws, on=['ticker', 'trading_day'], how='left')
        self.df = pd.merge(self.df, ibes, on=['ticker', 'trading_day'], how='left')

        self.df, self.mom_factor, self.nonmom_factor, self.change_factor, self.avg_factor = calc_factor_variables(self.df)
        self.df = self.df.filter(tri.columns.to_list()+self.nonmom_factor+['currency_code','icb_code'])
    
        # Fill NaN in fundamental records with interpolate + tri with ffill
        self.df[self.nonmom_factor] = self.df.groupby(['ticker'])[self.nonmom_factor].apply(pd.DataFrame.interpolate, limit_direction='forward')
        self.df['tri_fillna'] = self.df.groupby(['ticker'])['tri'].ffill()
        self.df[['currency_code','icb_code']] = self.df.groupby(['ticker'])[['currency_code','icb_code']].ffill().bfill()

    def calculate_final_results(self, interval=7):
        ''' calculate period average / change / skew / tri '''

        arr = reshape_by_interval(self.df.copy(), interval)
        cols = self.df.columns.to_list()
    
        # create label array (ticker, trading_day - period_end)
        arr_label = arr[:,:,-1, [cols.index('ticker'), cols.index('trading_day')]]
        arr_curind = arr[:,:,-1, [cols.index('currency_code'), cols.index('icb_code')]]
    
        # factor for changes (change over time) + factor for average (point in time)
        change_idx = [cols.index(x) for x in self.change_factor]
        average_idx = [cols.index(x) for x in self.avg_factor]
    
        arr_change = get_avg_change(arr[:,:,:,change_idx])[:,:,0,:]     # calculate average change between two period
        arr_average = get_average(arr[:,:,:,average_idx])[:,:,0,:]      # calculate average in this period
        arr_skew = get_skew(np.expand_dims(arr[:,:,:,cols.index('tri')], 3))    # calculate tri skew
        arr_skew = np.expand_dims(arr_skew, 2)
        arr_vol = get_rogers_satchell(arr[:,:,:,cols.index('open'):(cols.index('close')+1)])    # calculate tri volatility
    
        # concat all array & columns name
        final_arr = np.concatenate((arr_label, arr_vol, arr_skew, arr_change, arr_average, arr_curind), axis=2)
        self.final_col = ['ticker','trading_day', 'vol', 'skew'] + ['change_'+x for x in self.change_factor] + \
                         ['avg_'+x for x in self.avg_factor] + ['currency_code', 'icb_code']
    
        return final_arr

    def get_results(self, list_of_interval=[7, 14, 30, 91, 182, 365]):
        ''' calculate all arr for different period '''

        arr_dict = {}
        for i in list_of_interval:
            arr = self.calculate_final_results(i)
            arr = np.reshape(arr, (arr.shape[0]*arr.shape[1], arr.shape[2]))
            arr_dict[i] = pd.DataFrame(arr, columns=self.final_col)
            print(arr_dict[i])
        return   arr_dict

def reshape_by_interval(df, interval=7):
    ''' reshape 2D sample -> (ticker, #period, interval_length, factor) '''

    # define length as int
    num_ticker   = int(len(df['ticker'].unique()))
    num_day      = int(len(df['trading_day'].unique()))
    num_interval = int(interval)
    num_period   = int(np.ceil(num_day/num_interval))
    num_factor   = int(df.shape[1])

    arr = np.reshape(df.values, (num_ticker, num_day, num_factor), order='C')       # reshape to (ticker, period_end, interval)
    arr = np.pad(arr, ((0,0), (int(num_period*num_interval-num_day),0), (0,0)), mode='constant', constant_values=np.nan)   # fill for dates
    arr = np.reshape(arr, (num_ticker, num_period, num_interval, num_factor), order='C')    # reshape to different interval partition

    return arr

def get_skew(tri):
    ''' Calculate return skewness '''

    returns = get_change(tri)[:,:,:,0]
    log_returns = np.log(returns)
    result = stats.skew(log_returns, axis=-1, nan_policy='omit')
    return result

def get_rogers_satchell(arr, days_in_year=256):
    ''' Calculate return volatility '''

    arr = arr.astype(np.float)
    open_data, high_data, low_data, close_data = arr[:,:,:,0], arr[:,:,:,1], arr[:,:,:,2], arr[:,:,:,3]

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
    rogers_satchell_var = np.nanmean(sum, axis=2, keepdims=True)
    result = np.sqrt(rogers_satchell_var * days_in_year)

    return result

def get_change(arr):
    ''' Calculate daily change'''
    arr = arr.astype(np.float)
    arr_roll = np.roll(arr, shift=1, axis=1)
    arr_roll[:,0,:,:] = np.nan
    return arr / arr_roll - 1

def get_average(arr):
    ''' Calculate daily change'''
    arr = arr.astype(np.float)
    return np.nanmean(arr, axis=2, keepdims=True)

def get_avg_change(arr):
    ''' Calculate period change with the average of change on last (1/4) of total period '''

    arr = arr.astype(np.float)
    avg_period = round(arr.shape[2]/4, 0)
    sample_arr = arr[:, :, int(arr.shape[2]-avg_period):, :]
    arr_mean = np.nanmean(sample_arr, axis=2, keepdims=True)
    arr_roll = np.roll(arr_mean, shift=1, axis=1)
    arr_roll[:,0,:,:] = np.nan

    return arr_mean / arr_roll - 1

# -------------------------------------------- Calculate Fundamental Ratios --------------------------------------------

def calc_factor_variables(df):
    ''' Calculate all factor used referring to DB ratio table '''

    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}_descriptive', conn)  # ratio calculation used
    global_vals.engine_ali.dispose()

    print(f'      ------------------------> Calculate all factors in {global_vals.formula_factors_table}')
    # Prepare for field requires add/minus
    add_minus_fields = formula[['field_num', 'field_denom']].to_numpy().flatten()
    add_minus_fields = [i for i in filter(None, list(set(add_minus_fields))) if any(['-' in i, '+' in i, '*' in i])]

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

    # c) Divide ratios
    print(f'      ------------------------> Calculate dividing ratios ')
    for r in formula.dropna(subset=['field_num','field_denom'], how='any').loc[(formula['field_num'] != formula['field_denom'])].to_dict(
            orient='records'):  # minus calculation for ratios
        print('Calculating:', r['name'])
        df[r['name']] = df[r['field_num']] / df[r['field_denom']].replace(0, np.nan)

    mom_factor = formula.loc[formula['pillar']=='momentum', 'name'].to_list()
    nonmom_factor = formula.loc[formula['pillar']!='momentum', 'name'].to_list()
    change_factor = formula.loc[formula['period_change'], 'name'].to_list()
    avg_factor = formula.loc[formula['period_average'], 'name'].to_list()

    return df, mom_factor, nonmom_factor, change_factor, avg_factor

if __name__ == "__main__":
    # dict = combine_tri_worldscope(False, False, ticker=['AAPL.O','0992.HK']).get_results()
    # print(dict.keys())

    get_worldscope(True)