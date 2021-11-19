import numpy as np
import pandas as pd
import scipy.stats as stats
from sqlalchemy import text
import global_vars
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import quantile_transform, scale
from preprocess.calculation_ratio import check_duplicates, fill_all_day, update_period_end
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from utils_sql import upsert_data_to_database
from s_dbw import S_Dbw

def back_by_month(n=12, str=True):
    ''' return date in string for (n) months ago '''
    if str:
        return (dt.datetime.now() - relativedelta(months=n)).strftime("%Y-%m-%d")
    else:
        return dt.datetime.now() - relativedelta(months=n)

# ----------------------------------------- Calculate Stock Ralated Factors --------------------------------------------
def get_tri(conditions=None):
    ''' get stock related data from DSS & DSWS '''

    print(f'      ------------------------> Download stock data from {global_vals.stock_data_table_tri}')
    tri_conditions = ["U.is_active"]
    if conditions:
        tri_conditions.extend(conditions)
    tri_conditions.append(f"trading_day > '{back_by_month(24)}'")
    with global_vals.engine.connect() as conn_droid:
        query_tri = f"SELECT C.ticker, trading_day, total_return_index as tri, open, high, low, close, volume "
        query_tri += f"FROM {global_vals.stock_data_table_ohlc} C "
        query_tri += f"INNER JOIN (SELECT dsws_id, total_return_index FROM {global_vals.stock_data_table_tri}) T ON T.dsws_id = C.dss_id "
        query_tri += f"INNER JOIN (SELECT ticker, is_active, currency_code FROM {global_vals.dl_value_universe_table}) U ON U.ticker = C.ticker "
        query_tri += f"WHERE {' AND '.join(tri_conditions)} "
        query_tri += f"ORDER BY C.ticker, trading_day"
        tri = pd.read_sql(query_tri, con=conn_droid, chunksize=10000)
        tri = pd.concat(tri, axis=0, ignore_index=True)
        mkt_cap_anchor = pd.read_sql(f'SELECT ticker, mkt_cap FROM {global_vals.fundamental_score_mkt_cap}', conn_droid)
    global_vals.engine.dispose()

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

    return tri

def get_worldscope(conditions=None):
    ''' get fundamental data related data from Worldscope '''

    ws_conditions = ["C.ticker IS NOT NULL"]
    conditions.append(f"period_end > '{back_by_month(24)}'")
    if conditions:
        ws_conditions.extend(conditions)
    with global_vals.engine.connect() as conn:
        print(f'      ------------------------> Download worldscope data from {global_vals.worldscope_quarter_summary_table}')
        query_ws = f'SELECT C.*, U.currency_code, U.icb_code FROM {global_vals.worldscope_quarter_summary_table} C '
        query_ws += f"INNER JOIN (SELECT ticker, currency_code, is_active, icb_code FROM {global_vals.dl_value_universe_table}) U ON U.ticker = C.ticker "
        query_ws += f"WHERE {' AND '.join(ws_conditions)} "
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
    ws['icb_code'] = pd.to_numeric(ws['icb_code'], errors='coerce')

    return ws, ibes

def combine_tri_worldscope(currency=None, ticker=None):
    ''' combine tri & worldscope raw data '''

    conditions = ["True"]
    if currency:
        conditions.append("currency_code in ({})".format(','.join(['\''+x+'\'' for x in currency])))
    if ticker:
        conditions.append("C.ticker in ({})".format(','.join(['\''+x+'\'' for x in ticker])))

    # 1. Stock return/volatility/volume
    tri = get_tri(conditions)

    # 2. Fundamental financial data - from Worldscope
    # 3. Consensus forecasts - from I/B/E/S
    # 4. Universe
    ws, ibes = get_worldscope(conditions)

    # change to datetime
    tri['trading_day'] = pd.to_datetime(tri['trading_day'], format='%Y-%m-%d')
    ws['trading_day'] = pd.to_datetime(ws['trading_day'], format='%Y-%m-%d')
    ws = ws.drop(columns=['mkt_cap'])
    ibes['trading_day'] = pd.to_datetime(ibes['trading_day'], format='%Y-%m-%d')

    # Merge all table and convert to daily basis
    tri = fill_all_day(tri)
    tri[['mkt_cap','tac']] = tri.groupby(['ticker'])[['mkt_cap','tac']].ffill()
    df = pd.merge(tri, ws, on=['ticker', 'trading_day'], how='left')
    df = pd.merge(df, ibes, on=['ticker', 'trading_day'], how='left')

    ws_col = ws.select_dtypes(float).columns.to_list()
    ibes_col = ibes.select_dtypes(float).columns.to_list()
    df[ws_col+ibes_col] = df.groupby(['ticker'])[ws_col+ibes_col].ffill()

    df, mom_factor, nonmom_factor, change_factor, avg_factor = calc_factor_variables(df)
    df = df.filter(tri.columns.to_list()+nonmom_factor+['currency_code','icb_code'])

    # Fill NaN in fundamental records with interpolate + tri
    df = fill_all_day(df, date_col='trading_day').sort_values(by=["ticker","trading_day"])
    df['tri_fillna'] = df.groupby(['ticker'])['tri'].ffill()
    df[['currency_code','icb_code']] = df.groupby(['ticker'])[['currency_code','icb_code']].ffill().bfill()

    # vol + factor for changes (change over time) + factor for average (point in time)
    df = get_rogers_satchell(df, list_of_start_end=[[0,7], [0,91]])    # calculate tri volatility
    df[['1w_avg_'+x for x in avg_factor]] = get_average(df, avg_factor, 7)
    df[['3m_avg_'+x for x in avg_factor]] = get_average(df, avg_factor, 91)
    df[['1w_chg_'+x for x in change_factor]] = get_avg_change(df, change_factor, 7)
    df[['3m_chg_'+x for x in change_factor]] = get_avg_change(df, change_factor, 91)
    return df

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

def get_average(df, cols, interval=91):
    ''' Calculate daily change'''
    return df.groupby(["ticker"])[cols].rolling(interval, min_periods=1).mean().values

def get_avg_change(df, cols, interval=1):
    ''' Calculate period change with the average of change on last (1/4) of total period '''
    avg_period = int(round(interval/4, 0))
    df[["avg_"+x for x in cols]] = df.groupby(["ticker"])[cols].rolling(avg_period, min_periods=1).mean().values
    return df[["avg_"+x for x in cols]].values / df.groupby(["ticker"])[["avg_"+x for x in cols]].shift(interval).values

# -------------------------------------------- Calculate Fundamental Ratios --------------------------------------------

def calc_fx_conversion(df):
    """ Convert all columns to USD for factor calculation (DSS, WORLDSCOPE, IBES using different currency) """

    org_cols = df.columns.to_list()     # record original columns for columns to return

    with global_vals.engine.connect() as conn, global_vals.engine_ali_prod.connect() as conn_ali:
        curr_code = pd.read_sql(f"SELECT ticker, currency_code_ibes, currency_code_ws FROM {global_vals.dl_value_universe_table}", conn)     # map ibes/ws currency for each ticker
        fx = pd.read_sql(f"SELECT * FROM {global_vals.eikon_other_table}_fx", conn_ali)
        fx2 = pd.read_sql(f"SELECT currency_code as ticker, last_price as fx_rate, last_date as period_end "
                          f"FROM {global_vals.currency_history_table}", conn)
        fx['period_end'] = pd.to_datetime(fx['period_end']).dt.tz_localize(None)
        fx = fx.append(fx2).drop_duplicates(subset=['ticker','period_end'], keep='last')
        ingestion_source = pd.read_sql(f"SELECT * FROM ingestion_name", conn_ali)
    global_vals.engine.dispose()
    global_vals.engine_ali_prod.dispose()

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

    df[['open','high','low','close','mkt_cap']] = df[['open','high','low','close','mkt_cap']].div(df['fx_dss'], axis="index")  # convert close price

    return df[org_cols]

def calc_factor_variables(df):
    ''' Calculate all factor used referring to DB ratio table '''

    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}_descriptive', conn)  # ratio calculation used
    global_vals.engine_ali.dispose()

    # Foreign exchange conversion on absolute value items
    df = calc_fx_conversion(df)

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

# -------------------------------------------- Clean to Clustering Format --------------------------------------------

def get_cluster_dimensions(ticker=None, currency=None):
    ''' calculate pillar factors for clustering '''
    df = combine_tri_worldscope(ticker=ticker, currency=currency)
    cols = df.select_dtypes(float).columns.to_list()

    df = df.loc[df['trading_day']>back_by_month(2, str=False)]
    df = df.loc[df['ticker'].str[0] != '.']
    df_cluster = df[["ticker", "trading_day","currency_code"]].copy()

    # technical pillar
    tech_factor_1w = ["vol_0_7", "1w_avg_volume", "1w_chg_tri_fillna"]  # 1w factors (ST)
    tech_factor_3m = ["vol_0_91", "3m_avg_volume"]  # 3m factors (LT)
    tech_factor_newname = ["tech_1w_volatility", "tech_1w_volume", "tech_1w_return"]
    tech_factor_newname += ["tech_3m_volatility", "tech_3m_volume"]
    df_cluster[tech_factor_newname] = trim_outlier_std(df.groupby(["ticker"])[tech_factor_1w + tech_factor_3m].ffill().fillna(0))
    _, df_cluster["tech_1w_cluster"] = cluster_hierarchical(df_cluster[["tech_1w_volatility", "tech_1w_volume", "tech_1w_return"]].values)
    _, df_cluster["tech_3m_cluster"] = cluster_hierarchical(df_cluster[["tech_3m_volatility", "tech_3m_volume"]].values)

    # id pillar
    id_factor = ["1w_avg_mkt_cap", "icb_code", "3m_avg_mkt_cap"]
    id_factor_newname = ["id_1w_size", "id_icb"]
    id_factor_newname += ["id_3m_size"]
    df_cluster[id_factor_newname] = trim_outlier_std(df.groupby(["ticker"])[id_factor].ffill().fillna(0))
    _, df_cluster["id_1w_cluster"] = cluster_hierarchical(df_cluster[["id_1w_size", "id_icb"]].values)
    _, df_cluster["id_3m_cluster"] = cluster_hierarchical(df_cluster[["id_3m_size", "id_icb"]].values)

    # fundamental pillar
    funda_factor = [x for x in cols if x[:2]=='3m']
    funda_factor = list(set(funda_factor) - set(tech_factor_1w+tech_factor_3m+["3m_chg_tri_fillna"]+id_factor))
    funda_factor_newname = ["funda_3m_pc1", "funda_3m_pc2", "funda_3m_pc3"]
    X = df.groupby(["ticker"])[funda_factor].ffill().fillna(0).replace([np.inf, -np.inf], 0)
    X = trim_outlier_std(X)
    df_cluster[funda_factor_newname] = PCA(n_components=3).fit_transform(X)
    _, df_cluster["funda_3m_cluster"] = cluster_hierarchical(df_cluster[funda_factor_newname].values)

    df_cluster = upsert_data_to_database(df_cluster, "cluster_factor", primary_key=["ticker","trading_day"],
                                         db_url=global_vals.db_url_alibaba, how="replace")

    return df_cluster

def trim_outlier_std(df):
    ''' trim outlier on testing sets '''

    def trim_scaler(x):
        x = x.values
        return quantile_transform(np.reshape(x, (x.shape[0], 1)), output_distribution='normal', n_quantiles=1000)[:,0]

    cols = df.select_dtypes(float).columns.to_list()
    for col in cols:
        print(col, df[col].isnull().sum())
        if col != 'icb_code':
            x = trim_scaler(df[col])
        else:
            x = df[col].values
        x = scale(x.T)
        df[col] = x
    return df

def cluster_hierarchical(X, distance_threshold=None, n_clusters=20):
    kwargs = {'distance_threshold': distance_threshold, 'linkage': 'complete', 'n_clusters': n_clusters}
    kwargs['affinity'] = 'euclidean'
    model = AgglomerativeClustering(**kwargs).fit(X)
    y = model.labels_
    score = S_Dbw(X, y)
    print("score: ", score)
    return score, y

if __name__ == "__main__":
    get_cluster_dimensions(ticker=None, currency=['HKD', 'USD'])