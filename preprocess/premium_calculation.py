import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess.ratios_calculations import calc_factor_variables
from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION

def calc_group_premium_fama(name, g, factor_list):
    ''' calculate factor premium with avg(top group monthly return) - avg(bottom group monthly return) '''

    cut_col = [x + '_cut' for x in factor_list]

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
                bins = pd.IntervalIndex.from_tuples([(g[f].min(), 0), (0, 0), (0, g[f].max())])
                g[f'{f}_cut'] = pd.cut(g[f], bins=bins, include_lowest=True, retbins=False, labels=[0,1,2])
            elif bin_edges_is_dup[2]:   # e.g. [-2,-1,0,0] -> use 0 as "H", equal % of data from the bottom as "L"
                prc = g[f].to_list().count(bins[-1]) / num_data[factor_id] + 1e-8
                g[f'{f}_cut'] = pd.qcut(g[f], q=[0, 1 - prc, prc, 1], retbins=False, labels=[0,1,2])
            else:                       # others using 20% / 30%
                g[f'{f}_cut'] = pd.cut(g[f], bins=bins, include_lowest=True, retbins=False, labels=[0,1,2])
        except Exception as e:
            print(name, f, e)
            continue

        s = list(set(g[f'{f}_cut'].dropna().to_list()))
        for i in s:
            if i not in [0, 1, 2]:
                print(s)
                continue

        premium[f] = g.loc[g[f'{f}_cut'] == 0, 'stock_return_y'].mean()-g.loc[g[f'{f}_cut'] == 2, 'stock_return_y'].mean()

    return premium, g.filter(['ticker','period_end']+cut_col)

def calc_group_premium_msci():
    '''  calculate factor premium = inverse of factor value * returns '''
    exit(0)
    return 1

def calc_premium_all():
    ''' calculate factor premium for each currency_code / icb_code(6-digit) for each month '''

    # df, stocks_col, macros_col, formula = calc_factor_variables(price_sample='last_day', fill_method='fill_all',
    #                                                           sample_interval='monthly', use_cached=True, save=True)
    #
    # df = df.dropna(subset=['stock_return_y','ticker'])       # remove records without next month return -> not used to calculate factor premium
    # df = df.loc[~df['ticker'].str.startswith('.')]   # remove index e.g. ".SPX" from factor calculation
    # df = df.iloc[:1000,:]
    #
    # df.to_csv('premium_debug.csv', index=False)

    df = pd.read_csv('premium_debug.csv')
    factor_list = list(df.columns)[-24:]

    # print(set(df['icb_code'].to_list()))
    # print(set(df['currency_code'].to_list()))
    # df = df.drop_duplicates(['ticker','period_end'])
    # print(df.shape)

    # factor_list = formula.loc[formula['factors'], 'name'].to_list()     # factor = factor variables
    # factor_list = formula['name'].to_list()                           # factor = all variabales

    # factor_list = ['earnings_yield']

    # Calculate premium for currency partition
    print(f'#################################################################################################')
    print(f'      ------------------------> Calculate factor premium - Currency Partition')
    member_g_list = []
    results = {}
    target_cols = factor_list + ['ticker', 'period_end', 'currency_code', 'stock_return_y']
    for name, g in df[target_cols].groupby(['period_end', 'currency_code']):
        results[name], member_g = calc_group_premium_fama(name, g, factor_list)
        member_g['group'] = name[1]
        member_g_list.append(member_g)

    member_df = pd.concat(member_g_list, axis=0)
    results_df = pd.DataFrame(results).transpose().reset_index(drop=False)

    results_df.columns = ['period_end','group'] + results_df.columns.to_list()[2:]

    # member_df.to_csv('membership_curr.csv', index=False)
    # results_df.to_csv('factor_premium_curr.csv')

    # Calculate premium for industry partition
    print(f'      ------------------------> Calculate factor premium - Industry Partition')
    member_g_list = []
    results = {}
    target_cols = factor_list + ['ticker', 'period_end', 'icb_code', 'stock_return_y']
    for name, g in df[target_cols].groupby(['period_end', 'icb_code']):
        results[name], member_g = calc_group_premium_fama(name, g, factor_list)
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

    try:
        with global_vals.engine.connect() as conn:
            extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':1000}
            final_results_df.to_sql(global_vals.factor_premium_table, **extra, dtype=results_dtypes)
            print(f'      ------------------------> Finish writing factor premium table ')
            final_member_df.to_sql(global_vals.membership_table, **extra, dtype=mem_dtypes)
            print(f'      ------------------------> Finish writing factor membership table ')
        global_vals.engine.dispose()
    except Exception as e:
        print(e)
        write_local_csv_to_db()

def write_local_csv_to_db():

    final_member_df = pd.read_csv('membership.csv', low_memory=False)
    final_results_df = pd.read_csv('factor_premium.csv')

    with global_vals.engine.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':1000}
        final_results_df.to_sql(global_vals.factor_premium_table, **extra)
        final_member_df.to_sql(global_vals.membership_table, **extra)
    global_vals.engine.dispose()


if __name__=="__main__":
    calc_premium_all()
    # write_local_csv_to_db()
