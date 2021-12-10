import numpy as np
import pandas as pd
import scipy.stats as stats
import global_vars
from factor_model_premium.preprocess.calculation_ratio import fill_all_day, update_period_end


# ----------------------------------------- Calculate Stock Ralated Factors --------------------------------------------
def get_tri(save=True, conditions=None):
    ''' get stock related data from DSS & DSWS '''

    print(f'      ------------------------> Download stock data from {global_vars.stock_data_table_tri}')
    conditions.append("U.is_active")
    with global_vars.engine.connect() as conn_droid:
        query_tri = f"SELECT C.ticker, trading_day, total_return_index as tri, open, high, low, close, volume "
        query_tri += f"FROM {global_vars.stock_data_table_ohlc} C "
        query_tri += f"INNER JOIN (SELECT dsws_id, total_return_index FROM {global_vars.stock_data_table_tri}) T ON T.dsws_id = C.dss_id "
        query_tri += f"INNER JOIN (SELECT ticker, is_active, currency_code FROM {global_vars.universe_table}) U ON U.ticker = C.ticker "
        query_tri += f"WHERE {' AND '.join(conditions)} "
        query_tri += f"ORDER BY C.ticker, trading_day"
        tri = pd.read_sql(query_tri, con=conn_droid, chunksize=10000)
        tri = pd.concat(tri, axis=0, ignore_index=True)
        mkt_cap_anchor = pd.read_sql(f'SELECT ticker, mkt_cap FROM {global_vars.anchor_table_mkt_cap}', conn_droid)
    global_vars.engine.dispose()

    # update mkt_cap/market_cap_usd/tri_adjusted_close refer to tri for each period
    mkt_cap_anchor = mkt_cap_anchor.set_index('ticker')['mkt_cap'].to_dict()      # use mkt_cap from fundamental score
    tri['trading_day'] = pd.to_datetime(tri['trading_day'])
    anchor_idx = tri.groupby('ticker').tail(1).index
    tri.loc[anchor_idx, 'mkt_cap'] = tri.loc[anchor_idx, 'ticker'].map(mkt_cap_anchor)
    tri.loc[tri['mkt_cap'].notnull(), 'anchor_tri'] = tri.loc[tri['mkt_cap'].notnull(), 'tri']
    tri[['anchor_tri', 'mkt_cap']] = tri.groupby('ticker')[['anchor_tri','mkt_cap']].ffill().bfill()
    tri['anchor_tri'] = tri['tri'].values/tri['anchor_tri'].values
    tri[['mkt_cap','tac']] = tri[['mkt_cap','close']].multiply(tri['anchor_tri'], axis=0)

    tri = tri.drop(['anchor_tri'], axis=1)

    if save:
        tri.to_csv('dcache_tri.csv', index=False)

    return tri

def get_worldscope(save=True, conditions=None):
    ''' get fundamental data related data from Worldscope '''

    ws_conditions = ["C.ticker IS NOT NULL"]
    if conditions:
        ws_conditions.extend(conditions)
    with global_vars.engine.connect() as conn:
        print(f'      ------------------------> Download worldscope data from {global_vars.worldscope_quarter_summary_table}')
        query_ws = f'SELECT C.*, U.currency_code, U.industry_code FROM {global_vars.worldscope_quarter_summary_table} C '
        query_ws += f"INNER JOIN (SELECT ticker, currency_code, is_active, industry_code FROM {global_vars.universe_table}) U ON U.ticker = C.ticker "
        query_ws += f"WHERE {' AND '.join(ws_conditions)} "
        ws = pd.read_sql(query_ws, conn, chunksize=10000)  # quarterly records
        ws = pd.concat(ws, axis=0, ignore_index=True)
        query_ibes = f'SELECT ticker, trading_day, eps1tr12 FROM {global_vars.ibes_data_table} WHERE ticker IS NOT NULL'
        ibes = pd.read_sql(query_ibes, conn, chunksize=10000)  # ibes_data
        ibes = pd.concat(ibes, axis=0, ignore_index=True)
    global_vars.engine.dispose()

    def drop_dup(df):
        ''' drop duplicate records for same identifier & fiscal period, keep the most complete records '''

        print(f'      ------------------------> Drop duplicates in {global_vars.worldscope_quarter_summary_table} ')
        df['count'] = pd.isnull(df).sum(1)  # count the missing in each records (row)
        df = df.sort_values(['count']).drop_duplicates(subset=['ticker', 'period_end'], keep='first')
        return df.drop('count', axis=1)

    ws = drop_dup(ws)  # drop duplicate and retain the most complete record

    def fill_missing_ws(ws):
        ''' fill in missing values by calculating with existing data '''

        print(f'      ------------------------> Fill missing in {global_vars.worldscope_quarter_summary_table} ')
        ws['net_debt'] = ws['net_debt'].fillna(ws['debt'] - ws['cash'])  # Net debt = total debt - C&CE
        ws['ttm_ebit'] = ws['ttm_ebit'].fillna(
            ws['ttm_pretax_income'] + ws['ttm_interest'])  # TTM EBIT = TTM Pretax Income + TTM Interest Exp.
        ws['ttm_ebitda'] = ws['ttm_ebitda'].fillna(ws['ttm_ebit'] + ws['ttm_dda'])  # TTM EBITDA = TTM EBIT + TTM DDA
        ws['current_asset'] = ws['current_asset'].fillna(ws['total_asset'] - ws['ppe_net']) # fill missing for current assets
        return ws

    ws = fill_missing_ws(ws)        # selectively fill some missing fields
    ws = update_period_end(ws)      # correct timestamp for worldscope data (i.e. period_end)

    # label period_end with month end of trading_day (update_date)
    ws['trading_day'] = pd.to_datetime(ws['period_end'], format='%Y-%m-%d')
    ws = ws.drop(['period_end'], axis=1)
    # ws['currency_code'] = ws['currency_code'].replace(['EUR','GBP','USD','HKD','CNY','KRW'], list(np.arange(6)))
    ws['industry_code'] = pd.to_numeric(ws['industry_code'], errors='coerce')

    if save:
        ws.to_csv('dcache_ws.csv', index=False)
        ibes.to_csv('dcache_ibes.csv', index=False)

    return ws, ibes

class combine_tri_worldscope:
    ''' combine tri & worldscope raw data '''

    def __init__(self, use_cached, save=True, currency=None, ticker=None):

        conditions = ["True"]
        self.save = save

        if currency:
            conditions.append("currency_code in ({})".format(','.join(['\''+x+'\'' for x in currency])))
        if ticker:
            conditions.append("C.ticker in ({})".format(','.join(['\''+x+'\'' for x in ticker])))
    
        # 1. Stock return/volatility/volume
        if use_cached:
            try:
                tri = pd.read_csv(f'dcache_tri_{currency[0]}.csv', low_memory=False)
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
                ws = pd.read_csv(f'dcache_ws_{currency[0]}.csv')
                ibes = pd.read_csv(f'dcache_ibes_{currency[0]}.csv')
            except Exception as e:
                print(e)
                ws, ibes = get_worldscope(save, conditions)
        else:
            ws, ibes = get_worldscope(save, conditions)
    
        # change to datetime
        tri['trading_day'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')
        ws['trading_day'] = pd.to_datetime(ws['trading_day'], format='%Y-%m-%d')
        ws = ws.drop(columns=['mkt_cap'])
        ibes['trading_day'] = pd.to_datetime(ibes['trading_day'], format='%Y-%m-%d')
    
        # Merge all table and convert to daily basis
        tri = fill_all_day(tri)
        tri['volume_1w'] = tri.dropna(subset=['ticker']).groupby(['ticker'])['volume'].rolling(7, min_periods=1).mean().values
        tri['volume_3m'] = tri.dropna(subset=['ticker']).groupby(['ticker'])['volume'].rolling(90, min_periods=1).mean().values
        tri['volume_1w3m'] = (tri['volume_1w']/tri['volume_3m']).values
        # tri[['mkt_cap','tac']] = tri.groupby(['ticker'])[['mkt_cap','tac']].apply(pd.DataFrame.interpolate, limit_direction='forward')
        tri[['mkt_cap','tac']] = tri.groupby(['ticker'])[['mkt_cap','tac']].ffill()
        self.df = pd.merge(tri, ws, on=['ticker', 'trading_day'], how='left')
        self.df = pd.merge(self.df, ibes, on=['ticker', 'trading_day'], how='left')

        ws_col = ws.select_dtypes(float).columns.to_list()
        ibes_col = ibes.select_dtypes(float).columns.to_list()
        # self.df[ws_col+ibes_col] = self.df.groupby(['ticker'])[ws_col+ibes_col].apply(pd.DataFrame.interpolate,
        #                                                                               limit_direction='forward', limit_area='inside')
        self.df[ws_col+ibes_col] = self.df.groupby(['ticker'])[ws_col+ibes_col].ffill()

        self.df, self.mom_factor, self.nonmom_factor, self.change_factor, self.avg_factor = calc_factor_variables(self.df)
        self.df = self.df.filter(tri.columns.to_list()+self.nonmom_factor+['currency_code','industry_code'])
    
        # Fill NaN in fundamental records with interpolate + tri with
        self.df['tri_fillna'] = self.df.groupby(['ticker'])['tri'].ffill()
        self.df[['currency_code','industry_code']] = self.df.groupby(['ticker'])[['currency_code','industry_code']].ffill().bfill()

    def calculate_final_results(self, interval=7):
        ''' calculate period average / change / skew / tri '''

        cols = self.df.columns.to_list()

        if interval < 7:
            df = self.df.sort_values(['trading_day']).copy()
            df[cols[2:]] = df.groupby('ticker')[cols[2:]].ffill()
            arr = reshape_by_interval(df.ffill(), interval)

            change_fac = set(self.change_factor) & set(self.mom_factor)
            avg_fac = set(self.avg_factor) & set(self.mom_factor)
            change_idx = [cols.index(x) for x in change_fac]
            average_idx = [cols.index(x) for x in avg_fac]

            arr_label = arr[:, :, -1, [cols.index('ticker'), cols.index('trading_day')]]
            arr_change = get_change(arr[:,:,:,change_idx])[:,:,0,:]     # calculate average change between two period
            arr_average = get_average(arr[:,:,:,average_idx], self.avg_factor)[:,:,0,:]      # calculate average in this period

            final_arr = np.concatenate((arr_label, arr_change, arr_average), axis=2)
            self.final_col = ['ticker', 'trading_day']+['change_' + x for x in change_fac]+['avg_' + x for x in avg_fac]

        else:
            change_idx = [cols.index(x) for x in self.change_factor]
            average_idx = [cols.index(x) for x in self.avg_factor]
            arr = reshape_by_interval(self.df.copy(), interval)

            # create label array (ticker, trading_day - period_end)
            arr_label = arr[:, :, -1, [cols.index('ticker'), cols.index('trading_day')]]
            arr_curind = arr[:, :, -1, [cols.index('currency_code'), cols.index('industry_code')]]

            # factor for changes (change over time) + factor for average (point in time)
            arr_change = get_avg_change(arr[:,:,:,change_idx])[:,:,0,:]     # calculate average change between two period
            arr_average = get_average(arr[:,:,:,average_idx], self.avg_factor)[:,:,0,:]      # calculate average in this period
            arr_skew = get_skew(np.expand_dims(arr[:,:,:,cols.index('tri')], 3))    # calculate tri skew
            arr_skew = np.expand_dims(arr_skew, 2)
            arr_tri_momentum = arr_change[:,1:,self.change_factor.index('tri_fillna')]/arr_change[:,:-1,self.change_factor.index('tri_fillna')]
            arr_tri_momentum = np.pad(arr_tri_momentum, ((0,0), (1,0)), mode='constant', constant_values=np.nan)
            arr_tri_momentum = np.expand_dims(arr_tri_momentum, 2)
            arr_vol = get_rogers_satchell(arr[:,:,:,cols.index('open'):(cols.index('close')+1)])    # calculate tri volatility

            # concat all array & columns name
            final_arr = np.concatenate((arr_label, arr_vol, arr_skew, arr_tri_momentum, arr_change, arr_average, arr_curind), axis=2)
            self.final_col = ['ticker','trading_day', 'vol', 'skew','ret_momentum'] + ['change_'+x for x in self.change_factor] + \
                             ['avg_'+x for x in self.avg_factor] + ['currency_code', 'industry_code']
    
        return final_arr

    def read_cache(self, list_of_interval):
        ''' read cache csv if use_cache = True '''

        for i in list_of_interval:
            df = pd.read_csv(f'dcache_sample_{i}.csv')
        return df

    def get_results(self, list_of_interval=[7, 14, 30, 91, 182, 365]):
        ''' calculate all arr for different period '''

        arr_dict = {}
        for i in list_of_interval:
            arr = self.calculate_final_results(i)
            arr = np.reshape(arr, (arr.shape[0]*arr.shape[1], arr.shape[2]))
            df = pd.DataFrame(arr, columns=self.final_col)
            df = df.replace([-np.inf, np.inf], [np.nan, np.nan])
            x = df.iloc[:,2:].std()
            print('std:', x.to_dict())
            use_col = list(x[x>0.001].index)
            df[use_col] = df.groupby(['ticker'])[use_col].ffill()
            arr_dict[i] = df[['ticker','trading_day']+use_col]
            print(arr_dict[i])

            if self.save:
                arr_dict[i].dropna(subset=['change_tri_fillna']).to_csv(f'dcache_sample_{i}.csv', index=False)
                arr_dict[i] = pd.read_csv(f'dcache_sample_{i}.csv')

        return arr_dict

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

def get_change(arr, n=1):
    ''' Calculate daily change'''
    arr = arr.astype(np.float)
    arr_roll = np.roll(arr, shift=n, axis=1)
    arr_roll[:,:n,:,:] = np.nan
    return arr / arr_roll - 1

def get_average(arr, avg_factor):
    ''' Calculate daily change'''
    arr = arr.astype(np.float)
    return np.nanmean(arr, axis=2, keepdims=True)

def get_avg_change(arr, n=1):
    ''' Calculate period change with the average of change on last (1/4) of total period '''

    arr = arr.astype(np.float)
    avg_period = round(arr.shape[2]/4, 0)
    sample_arr = arr[:, :, int(arr.shape[2]-avg_period):, :]
    arr_mean = np.nanmean(sample_arr, axis=2, keepdims=True)
    arr_roll = np.roll(arr_mean, shift=n, axis=1)
    arr_roll[:,:n,:,:] = np.nan

    return arr_mean / arr_roll - 1

# -------------------------------------------- Calculate Fundamental Ratios --------------------------------------------

def calc_fx_conversion(df):
    """ Convert all columns to USD for factor calculation (DSS, WORLDSCOPE, IBES using different currency) """

    org_cols = df.columns.to_list()     # record original columns for columns to return

    with global_vars.engine.connect() as conn, global_vars.engine_ali_prod.connect() as conn_ali:
        curr_code = pd.read_sql(f"SELECT ticker, currency_code_ibes, currency_code_ws FROM {global_vars.universe_table}", conn)     # map ibes/ws currency for each ticker
        fx = pd.read_sql(f"SELECT * FROM {global_vars.eikon_fx_table}", conn_ali)
        fx2 = pd.read_sql(f"SELECT currency_code as ticker, last_price as fx_rate, last_date as period_end "
                          f"FROM {global_vars.currency_history_table}", conn)
        fx['period_end'] = pd.to_datetime(fx['period_end']).dt.tz_localize(None)
        fx = fx.append(fx2).drop_duplicates(subset=['ticker','period_end'], keep='last')
        ingestion_source = pd.read_sql(f"SELECT * FROM ingestion_name", conn_ali)
    global_vars.engine.dispose()
    global_vars.engine_ali_prod.dispose()

    df = df.merge(curr_code, on='ticker', how='left')
    # df = df.dropna(subset=['currency_code_ibes', 'currency_code_ws', 'currency_code'], how='any')   # remove ETF / index / some B-share -> tickers will not be recommended

    # map fx rate for conversion for each ticker
    fx = fx.drop_duplicates(subset=['ticker','period_end'], keep='last')
    fx = fill_all_day(fx, date_col='period_end')
    fx['fx_rate'] = fx.groupby('ticker')['fx_rate'].ffill().bfill()
    fx['period_end'] = fx['period_end'].dt.strftime("%Y-%m-%d")
    fx = fx.set_index(['ticker', 'period_end'])['fx_rate'].to_dict()

    currency_code_cols = ['currency_code', 'currency_code_ibes', 'currency_code_ws']
    fx_cols = ['fx_dss', 'fx_ibes', 'fx_ws']
    for cur_col, fx_col in zip(currency_code_cols, fx_cols):
        df['trading_day_str'] = df['trading_day'].dt.strftime('%Y-%m-%d')
        df = df.set_index([cur_col, 'trading_day_str'])
        df['index'] = df.index.to_numpy()
        df[fx_col] = df['index'].map(fx)
        df = df.reset_index()
    df[fx_cols] = df.groupby(['ticker'])[fx_cols].ffill()

    ingestion_source = ingestion_source.loc[ingestion_source['non_ratio']]     # no fx conversion for ratio items

    for name, g in ingestion_source.groupby(['source']):        # convert for ibes / ws
        cols = list(set(g['our_name'].to_list()) & set(df.columns.to_list()))
        print(f'----> [{name}] source data with fx conversion: ', cols)
        df[cols] = df[cols].div(df[f'fx_{name}'], axis="index")

    df[['close','mkt_cap']] = df[['close','mkt_cap']].div(df['fx_dss'], axis="index")  # convert close price

    return df[org_cols]

def calc_factor_variables(df):
    ''' Calculate all factor used referring to DB ratio table '''

    with global_vars.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT * FROM {descriptive_formula_factors_table}', conn)  # ratio calculation used
    global_vars.engine_ali.dispose()

    # Foreign exchange conversion on absolute value items
    df = calc_fx_conversion(df)

    print(f'      ------------------------> Calculate all factors in {global_vars.formula_factors_table_prod}')
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

    # fillna for after last reporting
    fillna_col = df.filter(regex='^fn_').columns.to_list() + ['eps1tr12']
    df[fillna_col] = df.groupby(['ticker'])[fillna_col].ffill()

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
    dict = combine_tri_worldscope(use_cached=False, save=True, ticker=None, currency=['HKD', 'USD']).get_results(list_of_interval=[1, 7, 30, 91, 365])
    print(dict.keys())

    # get_worldscope(True)