import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess.ratios_calculations import calc_factor_variables
from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION
from preprocess.premium_calculation import trim_outlier
from functools import partial
from pandas.tseries.offsets import MonthEnd

def combine_like_premium(name, g, factor_list):
    ''' calculate combined average with avg(top group monthly, ratios) - avg(bottom group monthly, ratios) '''

    cut_col = [x + '_cut' for x in factor_list]

    # factor_list = ['market_cap_usd']

    premium = {}
    num_data = g[factor_list].notnull().sum(axis=0)
    for factor_id, f in enumerate(factor_list):
        g[f] = trim_outlier(g[f], prc=.05)

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
                g[f'{f}_cut'] = pd.qcut(g[f], q=[0, prc, 1 - prc, 1], retbins=False, labels=False, duplicates='drop')
                premium[f] = g.loc[g[f'{f}_cut'] == 2, f].mean() - g[f].min()
            elif bin_edges_is_dup[1]:   # e.g. [-1,0,0,8] -> <0 as "L", >0 as "H"
                g[f'{f}_cut'] = g[f]
                g[f'{f}_cut'] = g[f'{f}_cut'].mask(g[f]<0, -1)
                g[f'{f}_cut'] = g[f'{f}_cut'].mask(g[f]>0, 1)
                g[f'{f}_cut'] += 2
                g[f'{f}_cut'] = g[f'{f}_cut'].astype(int)
                premium[f] = g.loc[g[f'{f}_cut'] == 2, f].mean() - g.loc[g[f'{f}_cut'] == 0, f].mean()
            elif bin_edges_is_dup[2]:   # e.g. [-2,-1,0,0] -> use 0 as "H", equal % of data from the bottom as "L"
                prc = g[f].to_list().count(bins[2]) / num_data[factor_id] + 1e-8
                g[f'{f}_cut'] = pd.qcut(g[f], q=[0, 1 - prc, prc, 1], retbins=False, labels=False, duplicates='drop')
                premium[f] = g[f].max() - g.loc[g[f'{f}_cut'] == 0, f].mean()
            else:                       # others using 20% / 30%
                g[f'{f}_cut'] = pd.cut(g[f], bins=bins, include_lowest=True, retbins=False, labels=[0,1,2])
                premium[f] = g.loc[g[f'{f}_cut'] == 2, f].mean() - g.loc[g[f'{f}_cut'] == 0, f].mean()
        except Exception as e:
            print(name, f, e)
            continue

    return premium

def combine_mean(name, g, factor_list):
    return g[factor_list].mean()

def combine_median(name, g, factor_list):
    return g[factor_list].median()

def method_combine(method, name, g, factor_list):
    if method == 'mean':
        return combine_mean(name=name, g=g, factor_list=factor_list)
    elif method == 'median':
        return combine_median(name=name, g=g, factor_list=factor_list)
    elif method == 'premium':
        return combine_like_premium(name=name, g=g, factor_list=factor_list)

def calc_group_ratio():
    ''' calculate combined group ratios for each currency_code / icb_code(6-digit) for each month '''

    # Read stock_return / ratio table
    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}", conn)
        formula = pd.read_sql(f"SELECT * FROM {global_vals.formula_factors_table}", conn)
    global_vals.engine_ali.dispose()

    df = df.dropna(subset=['stock_return_y','ticker'])       # remove records without next month return -> not used to calculate factor premium
    df = df.loc[~df['ticker'].str.startswith('.')]   # remove index e.g. ".SPX" from factor calculation
    factor_list = formula['name'].to_list()                           # factor = all variabales

    for method in ['mean','median','premium']:      # also for 'mean','median'

        # Calculate premium for currency partition
        print(f'#################################################################################################')
        print(f'      ------------------------> Calculate group average - Currency Partition')
        results = {}
        target_cols = factor_list + ['ticker', 'period_end', 'currency_code', 'stock_return_y']
        for name, g in df[target_cols].groupby(['period_end', 'currency_code']):
            results[name] = {}
            results[name] = method_combine(method, name, g, factor_list)

        results_df = pd.DataFrame(results).transpose().reset_index(drop=False)
        results_df.columns = ['period_end','group'] + results_df.columns.to_list()[2:]

        # Calculate premium for industry partition
        print(f'      ------------------------> Calculate group average - Industry Partition')
        results = {}
        target_cols = factor_list + ['ticker', 'period_end', 'icb_code', 'stock_return_y']
        for name, g in df[target_cols].groupby(['period_end', 'icb_code']):
            results[name] = {}
            results[name] = method_combine(method, name, g, factor_list)

        results_df_1 = pd.DataFrame(results).transpose().reset_index(drop=False)
        results_df_1.columns = ['period_end','group'] + results_df_1.columns.to_list()[2:]

        final_results_df = pd.concat([results_df, results_df_1], axis=0)
        final_results_df['method'] = method

        results_dtypes = {}
        for i in list(final_results_df.columns):
            results_dtypes[i] = DOUBLE_PRECISION
        results_dtypes['period_end'] = DATE
        results_dtypes['group']=TEXT
        results_dtypes['method']=TEXT

        final_results_df['period_end'] = final_results_df['period_end'] + MonthEnd(1)
        try:
            with global_vals.engine_ali.connect() as conn:
                extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize':1000}
                final_results_df.to_sql(global_vals.processed_group_ratio_table, **extra, dtype=results_dtypes)
                print(f'      ------------------------> Finish writing factor premium table ')
            global_vals.engine_ali.dispose()
        except Exception as e:
            print(e)

if __name__=="__main__":
    calc_group_ratio()
    # write_local_csv_to_db()
