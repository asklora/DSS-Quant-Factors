import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess.ratios_calculations import calc_factor_variables
from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION

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

        premium[f] = g.loc[g[f'{f}_cut'] == 0, 'stock_return_y'].mean()-g.loc[g[f'{f}_cut'] == 2, 'stock_return_y'].mean()

    return premium, g.filter(['ticker','period_end']+cut_col)

def calc_group_premium_msci():
    '''  calculate factor premium = inverse of factor value * returns '''
    exit(0)
    return 1

def calc_premium_all(use_biweekly_stock=False, stock_last_week_avg=False):
    ''' calculate factor premium for each currency_code / icb_code(6-digit) for each month '''

    # df, stocks_col, formula = calc_factor_variables(price_sample='last_day', fill_method='fill_all',
    #                                                 sample_interval='monthly', use_cached=True, save=True)
    if use_biweekly_stock and stock_last_week_avg:
        raise ValueError("Expecting 'use_biweekly_stock' or 'stock_last_week_avg' is TRUE. Got both is TRUE")

    # Read stock_return / ratio table
    print(f'#################################################################################################')
    print(f'      ------------------------> Download ratio data from DB')
    with global_vals.engine_ali.connect() as conn:
        if use_biweekly_stock:
            print(f'      ------------------------> Use biweekly ratios')
            df = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}_biweekly", conn)
        else:
            df = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}", conn)
        formula = pd.read_sql(f"SELECT * FROM {global_vals.formula_factors_table}", conn)
        if stock_last_week_avg:
            print(f'      ------------------------> Replace stock return with last week average returns')
            df_stock_avg = pd.read_sql(f"SELECT * FROM {global_vals.processed_stock_table}", conn)
            df['period_end'] = pd.to_datetime(df['period_end'])
            df = df.merge(df_stock_avg, on=['ticker', 'period_end'], suffixes=['_org',''])
    global_vals.engine_ali.dispose()

    df = df.dropna(subset=['stock_return_y','ticker'])       # remove records without next month return -> not used to calculate factor premium
    df = df.loc[~df['ticker'].str.startswith('.')]   # remove index e.g. ".SPX" from factor calculation

    factor_list = formula['name'].to_list()                           # factor = all variabales
    # factor_list = ['book_to_price']

    # df = df.loc[(df['currency_code']=='USD')&(df['period_end']=='2020-10-31')]

    # Calculate premium for currency partition
    print(f'#################################################################################################')
    print(f'      ------------------------> Start calculate factor premium - Currency Partition')
    member_g_list = []
    results = {}
    target_cols = factor_list + ['ticker', 'period_end', 'currency_code', 'stock_return_y']
    for name, g in df[target_cols].groupby(['period_end', 'currency_code']):
        results[name] = {}
        results[name], member_g = calc_group_premium_fama(name, g, factor_list)
        member_g['group'] = name[1]
        results[name]['len'] = len(member_g)
        member_g_list.append(member_g)

    member_df = pd.concat(member_g_list, axis=0)
    results_df = pd.DataFrame(results).transpose().reset_index(drop=False)

    results_df.columns = ['period_end','group'] + results_df.columns.to_list()[2:]

    # member_df.to_csv('membership_curr.csv', index=False)
    # results_df.to_csv('factor_premium_curr.csv')

    # Calculate premium for industry partition
    print(f'      ------------------------> Start calculate factor premium - Industry Partition')
    member_g_list = []
    results = {}
    target_cols = factor_list + ['ticker', 'period_end', 'icb_code', 'stock_return_y']
    for name, g in df[target_cols].groupby(['period_end', 'icb_code']):
        results[name] = {}
        results[name], member_g = calc_group_premium_fama(name, g, factor_list)
        results[name]['len'] = len(member_g)
        member_g['group'] = name[1]
        member_g_list.append(member_g)

    member_df_1 = pd.concat(member_g_list, axis=0)
    results_df_1 = pd.DataFrame(results).transpose().reset_index(drop=False)
    results_df_1.columns = ['period_end','group'] + results_df_1.columns.to_list()[2:]

    final_member_df = pd.concat([member_df, member_df_1], axis=0)
    final_results_df = pd.concat([results_df, results_df_1], axis=0)

    final_member_df.to_csv('membership.csv', index=False)
    final_results_df.to_csv('factor_premium.csv', index=False)

    final_member_df = pd.read_csv('membership.csv', low_memory=False)
    final_results_df = pd.read_csv('factor_premium.csv')

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

    factor_table = global_vals.factor_premium_table
    member_table = global_vals.membership_table
    if stock_last_week_avg:
        factor_table += '_weekavg'
        member_table += '_weekavg'
    elif use_biweekly_stock:
        factor_table += '_biweekly'
        member_table += '_biweekly'

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':1000}
        final_results_df.to_sql(factor_table, **extra, dtype=results_dtypes)
        print(f'      ------------------------> Finish writing factor premium table ')
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

if __name__=="__main__":
    calc_premium_all(stock_last_week_avg=False, use_biweekly_stock=True)
    calc_premium_all(stock_last_week_avg=False, use_biweekly_stock=False)
    calc_premium_all(stock_last_week_avg=True, use_biweekly_stock=False)

    # write_local_csv_to_db()
