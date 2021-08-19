import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy import create_engine
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess.ratios_calculations import calc_factor_variables
from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION
from sqlalchemy.dialects.postgresql import INTEGER
from utils import remove_tables_with_suffix
import multiprocessing as mp
import time

icb_num = 6

# define dtypes for final_member_df & final_results_df when writing to DB
mem_dtypes = dict(
    group=TEXT,
    ticker=TEXT,
    period_end=DATE,
    factor_name=TEXT,
    quantile_group=INTEGER,
    last_update=TIMESTAMP
)

results_dtypes = dict(
    group=TEXT,
    period_end=DATE,
    factor_name=TEXT,
    stock_return_y=DOUBLE_PRECISION,
    last_update=TIMESTAMP
)

to_sql_params = {'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 8192}

def trim_outlier(df, prc=0):
    ''' assign a max value for the 99% percentile to replace inf'''

    df_nan = df.replace([np.inf, -np.inf], np.nan)
    pmax = df_nan.quantile(q=(1 - prc))
    pmin = df_nan.quantile(q=prc)
    df = df.mask(df > pmax, pmax)
    df = df.mask(df < pmin, pmin)

    return df

def calc_group_premium_fama(name, g, factor_list):
    ''' calculate factor premium with avg(top group monthly return) - avg(bottom group monthly return) '''

    cut_col = [x + '_cut' for x in factor_list]

    g['stock_return_y'] = trim_outlier(g['stock_return_y'], prc=.05)
    # print(g['stock_return_y'].describe())

    # factor_list = ['market_cap_usd']

    premium = {}
    num_data = g[factor_list].notnull().sum(axis=0)
    for factor_id, f in enumerate(factor_list):
        if num_data[factor_id] < 3:   # If group sample size is too small -> factor = NaN
            continue
        elif num_data[factor_id] > 65: # If group sample size is large -> using long/short top/bottom 20%
            prc_list = [0, 0.2, 0.8, 1]
        else:               # otherwise -> using long/short top/bottom 30%
            prc_list = [0, 0.3, 0.7, 1]

        bins = g[f].quantile(prc_list).to_list()
        bins[0] -= 1e-8

        isinf_mask = np.isinf(g[f])
        if isinf_mask.any():
            g.loc[isinf_mask, f] = np.nan_to_num(g.loc[isinf_mask, f])

        bins = g[f].quantile(prc_list).tolist()
        bin_edges_is_dup = (np.diff(bins) == 0)
        try:
            if bin_edges_is_dup.sum() > 1:    # in case like bins = [0,0,0,..] -> premium = np.nan
                continue
            elif bin_edges_is_dup[0]:   # e.g. [0,0,3,8] -> use 0 as "L", equal % of data from the top as "H"
                prc = g[f].to_list().count(bins[0]) / num_data[factor_id] + 1e-8
                g[f'{f}_cut'] = pd.qcut(g[f], q=[0, prc, 1 - prc, 1], retbins=False, labels=[0,1,2])
            elif bin_edges_is_dup[1]:   # e.g. [-1,0,0,8] -> <0 as "L", >0 as "H"
                g[f'{f}_cut'] = g[f]
                g[f'{f}_cut'] = g[f'{f}_cut'].mask(g[f]<0, -1)
                g[f'{f}_cut'] = g[f'{f}_cut'].mask(g[f]>0, 1)
                g[f'{f}_cut'] += 2
                g[f'{f}_cut'] = g[f'{f}_cut'].astype(int)
            elif bin_edges_is_dup[2]:   # e.g. [-2,-1,0,0] -> use 0 as "H", equal % of data from the bottom as "L"
                prc = g[f].to_list().count(bins[-1]) / num_data[factor_id] + 1e-8
                g[f'{f}_cut'] = pd.qcut(g[f], q=[0, 1 - prc, prc, 1], retbins=False, labels=[0,1,2])
            else:                       # others using 20% / 30%
                g[f'{f}_cut'] = pd.cut(g[f], bins=bins, include_lowest=True, retbins=False, labels=[0,1,2])
        except Exception as e:
            print(name, f, e)
            continue

        # g=g.sort_values(by=[f])
        # x1 = g.loc[g[f'{f}_cut'] == 0]
        # m1 = x1.mean()
        # x2 = g.loc[g[f'{f}_cut'] == 2]
        # m2 = x2.mean()
        premium[f] = g.loc[g[f'{f}_cut'] == 0, 'stock_return_y'].mean()-g.loc[g[f'{f}_cut'] == 2, 'stock_return_y'].mean()

    return premium, g.filter(['ticker','period_end']+cut_col)

def calc_group_premium_msci():
    '''  calculate factor premium = inverse of factor value * returns '''
    return 1

def get_premium_data(use_biweekly_stock=False, stock_last_week_avg=False, update=False):
    ''' calculate factor premium for different configurations:
        1. monthly sample + using last day price
        2. biweekly sample + using last day price
        3. monthly sample + using average price of last week
    '''

    if use_biweekly_stock and stock_last_week_avg:
        raise ValueError("Expecting 'use_biweekly_stock' or 'stock_last_week_avg' is TRUE. Got both is TRUE")

    # Read stock_return / ratio table
    print(f'#################################################################################################')
    print(f'      ------------------------> Download ratio data from DB')
    with global_vals.engine_ali.connect() as conn:
        if use_biweekly_stock:
            print(f'      ------------------------> Use biweekly ratios')
            df = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}_biweekly", conn)
        elif stock_last_week_avg:
            print(f'      ------------------------> Replace stock return with last week average returns')
            df = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}_weekavg", conn)
        else:
            df = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}", conn)
        formula = pd.read_sql(f"SELECT * FROM {global_vals.formula_factors_table}", conn)

    global_vals.engine_ali.dispose()

    df = df.dropna(subset=['stock_return_y','ticker'])       # remove records without next month return -> not used to calculate factor premium
    df = df.loc[~df['ticker'].str.startswith('.')]   # remove index e.g. ".SPX" from factor calculation

    factor_list = formula['name'].to_list()                           # factor = all variabales

    return df, factor_list

def calc_premium_all(use_biweekly_stock=False, stock_last_week_avg=False, save_membership=False, update=False):
    ''' calculate factor premium for each currency_code / icb_code(6-digit) for each month '''

    df, factor_list = get_premium_data(use_biweekly_stock, stock_last_week_avg, update=False)

    # Calculate premium for currency / industry partition
    all_member_df = []      # record member_df *2 (cur + ind)
    all_results_df = []     # record results_df *2

    group_list = ['currency_code', 'icb_code']

    if icb_num != 6:
        df['icb_code'] = df['icb_code'].str[:icb_num]
        group_list = ['icb_code']

    # factor_list = ['vol_0_30']

    print(f'#################################################################################################')
    for i in group_list:
        print(f'      ------------------------> Start calculate factor premium - [{i}] Partition')
        member_g_list = []
        results = {}
        target_cols = factor_list + ['ticker', 'period_end', i, 'stock_return_y']
        for name, g in df[target_cols].groupby(['period_end', i]):
            # if name[0]!=dt.datetime(2019,12,31):
            #     continue
            # if name[1] != 'USD':
            #     continue
            results[name] = {}
            results[name], member_g = calc_group_premium_fama(name, g, factor_list)
            member_g['group'] = name[1]
            results[name]['len'] = len(member_g)
            member_g_list.append(member_g)

        member_df = pd.concat(member_g_list, axis=0)
        results_df = pd.DataFrame(results).transpose().reset_index(drop=False)
        results_df.columns = ['period_end', 'group'] + results_df.columns.to_list()[2:]

        all_member_df.append(member_df)
        all_results_df.append(results_df)

    final_member_df = pd.concat(all_member_df, axis=0)
    final_results_df = pd.concat(all_results_df, axis=0)

    print(f'#################################################################################################')
    # define dtypes for final_member_df & final_results_df when writing to DB
    mem_dtypes = {}
    for i in list(final_member_df.columns):
        mem_dtypes[i] = DOUBLE_PRECISION
    mem_dtypes['period_end'] = DATE
    mem_dtypes['group']=TEXT
    mem_dtypes['ticker']=TEXT

    results_dtypes = {}
    for i in list(final_results_df.columns):
        results_dtypes[i] = DOUBLE_PRECISION
    results_dtypes['period_end'] = DATE
    results_dtypes['group']=TEXT

    # get name of DB TABLE based on config
    factor_table = global_vals.factor_premium_table
    member_table = global_vals.membership_table
    if stock_last_week_avg:
        factor_table += '_weekavg'
        member_table += '_weekavg'
    elif use_biweekly_stock:
        factor_table += '_biweekly'
        member_table += '_biweekly'

    if icb_num != 6:
        factor_table += str(icb_num)
        member_table += str(icb_num)

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':1000}
        final_results_df.to_sql(factor_table, **extra, dtype=results_dtypes)
        print(f'      ------------------------> Finish writing factor premium table ')
        if save_membership:
            final_member_df.to_sql(member_table, **extra, dtype=mem_dtypes)
            print(f'      ------------------------> Finish writing factor membership table ')
    global_vals.engine_ali.dispose()

def write_local_csv_to_db():

    final_member_df = pd.read_csv('membership.csv', low_memory=False)
    final_results_df = pd.read_csv('factor_premium.csv')

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':1000}
        final_results_df.to_sql(global_vals.factor_premium_table, **extra)
        final_member_df.to_sql(global_vals.membership_table, **extra)
    global_vals.engine_ali.dispose()


def insert_prem_and_membership_for_group(*args):
    def qcut(series):
        try:
            series_fillinf = series.replace([-np.inf, np.inf], np.nan)
            nonnull_count = series_fillinf.notnull().sum()
            if nonnull_count < 3:
                return series.map(lambda _: np.nan)
            elif nonnull_count > 65:
                prc = [0, 0.2, 0.8, 1]
            else:
                prc = [0, 0.3, 0.7, 1]
            
            q = pd.qcut(series_fillinf, prc, duplicates='drop')

            n_cat = len(q.cat.categories)
            if n_cat == 3:
                q.cat.categories = range(3)
            elif n_cat == 2:
                q.cat.categories = [0, 2]
            else:
                return series.map(lambda _: np.nan)
            q[series_fillinf == np.inf] = 2
            q[series_fillinf == -np.inf] = 0
            return q
        except ValueError as e:
            print(e)
            return series.map(lambda _: np.nan)

    gp_type, group, tbl_suffix, factor_list = args
    print(f'      ------------------------> Start calculating factor premium - [{group}] Partition')

    thread_engine_ali = create_engine(global_vals.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")

    try:
        with thread_engine_ali.connect() as conn:
            if gp_type == 'curr':
                df = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}{tbl_suffix} WHERE currency_code = '{group}';", conn)
            else:
                df = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}{tbl_suffix} WHERE SUBSTR(icb_code, 1, {icb_num}) = '{group}';", conn)

        df = df.dropna(subset=['stock_return_y','ticker'])

        if len(df) == 0:
            raise Exception(f"Either stock_return_y or ticker in group '{group}' is all missing")

        df = df.loc[~df['ticker'].str.startswith('.')].copy()
        df = df.melt(
            id_vars=['ticker', 'period_end', 'stock_return_y'],
            value_vars=factor_list,
            var_name='factor_name',
            value_name='factor_value')
        df['quantile_group'] = df.groupby(['period_end', 'factor_name'])['factor_value'].transform(qcut)
        df = df.dropna(subset=['quantile_group']).copy()
        df['quantile_group'] = df['quantile_group'].astype(int)

        quantile_mean_returns = df.groupby(['period_end', 'factor_name', 'quantile_group'])['stock_return_y'].mean().reset_index()

        # Calculate small minus big
        prem = quantile_mean_returns.pivot(['period_end', 'factor_name'], ['quantile_group']).droplevel(0, axis=1)
        prem = (prem[0] - prem[2]).dropna().rename('premium').reset_index()
        membership = df[['ticker', 'period_end', 'factor_name', 'quantile_group']].reset_index(drop=True)

        prem['group'] = group
        membership['group'] = group
        
        prem['last_update'] = last_update
        membership['last_update'] = last_update
        
        print(f'      ------------------------> Start writing membership and factor premium - [{group}] Partition')
        with thread_engine_ali.connect() as conn:
            membership.sort_values(by=['group', 'ticker', 'period_end', 'factor_name']).to_sql(f"{global_vals.membership_table}{tbl_suffix}{tbl_suffix_extra}", con=conn, dtype=mem_dtypes, **to_sql_params)
            prem.sort_values(by=['group', 'period_end', 'factor_name']).to_sql(f"{global_vals.factor_premium_table}{tbl_suffix}{tbl_suffix_extra}", con=conn, dtype=results_dtypes, **to_sql_params)
    except Exception as e:
        print(e)
        thread_engine_ali.dispose()
        return False
    thread_engine_ali.dispose()
    return True

def calc_premium_all_v2(use_biweekly_stock=False, stock_last_week_avg=False, save_membership=False, update=False):

    ''' calculate factor premium for different configurations:
        1. monthly sample + using last day price
        2. biweekly sample + using last day price
        3. monthly sample + using average price of last week
    '''

    if use_biweekly_stock and stock_last_week_avg:
        raise ValueError("Expecting 'use_biweekly_stock' or 'stock_last_week_avg' is TRUE. Got both is TRUE")

    # Read stock_return / ratio table
    print(f'#################################################################################################')
    print(f'      ------------------------> Download ratio data from DB')

    if use_biweekly_stock:
        tbl_suffix = '_biweekly'
        print(f'      ------------------------> Use biweekly ratios')
    elif stock_last_week_avg:
        tbl_suffix = '_weekavg'
        print(f'      ------------------------> Replace stock return with last week average returns')
    else:
        tbl_suffix = ''

    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f"SELECT * FROM {global_vals.formula_factors_table}", conn)
        factor_list = formula['name'].to_list()                           # factor = all variabales
                
        all_curr = pd.read_sql(f"SELECT DISTINCT currency_code from {global_vals.processed_ratio_table}{tbl_suffix} WHERE currency_code IS NOT NULL;", conn).values.flatten().tolist()
        all_icb = pd.read_sql(f"SELECT DISTINCT icb_code from {global_vals.processed_ratio_table}{tbl_suffix} WHERE icb_code IS NOT NULL AND icb_code != 'nan';", conn).values.flatten().tolist()
        
    if icb_num != 6:
        all_icb = [icb[:icb_num] for icb in all_icb]
        all_icb = list(set(all_icb))
        all_curr = []

    all_groups = [('curr', curr) for curr in all_curr] + [('icb', icb) for icb in all_icb]

    print(f'      ------------------------> {" -> ".join([group for _, group in all_groups])}')

    with mp.Pool(processes=3) as pool:
        # all_groups = [('curr', 'KRW')]
        res = pool.starmap(insert_prem_and_membership_for_group, [(*x, tbl_suffix, factor_list) for x in all_groups])
    
    return res


if __name__ == "__main__":

    last_update = datetime.now()
    tbl_suffix_extra = '_v2'

    start = datetime.now()

    remove_tables_with_suffix(global_vals.engine_ali, tbl_suffix_extra)
    # calc_premium_all(stock_last_week_avg=True, use_biweekly_stock=False, save_membership=True)
    calc_premium_all_v2(use_biweekly_stock=False, stock_last_week_avg=True, save_membership=True)

    end = datetime.now()

    print(f'Time elapsed: {(end - start).total_seconds():.2f} s')
    # write_local_csv_to_db()
