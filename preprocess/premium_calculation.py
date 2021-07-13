import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess.ratios_calculations import calc_factor_variables

def calc_group_premium_fama(name, g, factor_list):
    ''' calculate factor premium with avg(top group monthly return) - avg(bottom group monthly return) '''

    cut_col = [x + '_cut' for x in factor_list]

    premium = {}
    for f in factor_list:
        num_data = g[f].notnull().sum()    # count # of non-NaN value

        if num_data < 3:   # If group sample size is too small -> factor = NaN
            continue
        elif num_data > 65: # If group sample size is large -> using long/short top/bottom 20%
            prc_list = [0, 0.2, 0.8, 1]
        else:               # otherwise -> using long/short top/bottom 30%
            prc_list = [0, 0.3, 0.7, 1]

        bins = g[f].quantile(prc_list).to_list()
        bins[0] -= 1e-8

        bins = g[f].quantile(prc_list).fillna(np.inf).to_list()
        bin_edges_is_dup = (np.diff(bins) == 0)
        try:
            if bin_edges_is_dup.sum() > 1:    # in case like bins = [0,0,0,..] -> premium = np.nan
                continue
            elif bin_edges_is_dup[0]:   # e.g. [0,0,3,8] -> premium = np.nan
                prc = g[f].to_list().count(0) / num_data + 1e-8
                prc = (prc if prc < .5 else 1. - prc)
                g[f'{f}_cut'] = pd.qcut(g[f], q=[0, prc, 1 - prc, 1], retbins=False, labels=False)
            elif bin_edges_is_dup[1]:
                bins = pd.IntervalIndex.from_tuples([(g[f].min(), 0), (0, 0), (0, g[f].max())])
                g[f'{f}_cut'] = pd.cut(g[f], bins=bins, include_lowest=True, retbins=False, labels=False)
            elif bin_edges_is_dup[2]:
                prc = g[f].to_list().count(0) / num_data + 1e-8
                prc = (prc if prc < .5 else 1. - prc)
                g[f'{f}_cut'] = pd.qcut(g[f], q=[0, prc, 1 - prc, 1], retbins=False, labels=False)
            else:
                g[f'{f}_cut'] = pd.cut(g[f], bins=bins, include_lowest=True, retbins=False, labels=False)
        except Exception as e:
            print(name, f, e)
            continue

        # print(g.loc[g[f'{f}_cut']==0, 'stock_return_y'])
        premium[f] = g.loc[g[f'{f}_cut'] == 0, 'stock_return_y'].mean()-g.loc[g[f'{f}_cut'] == 2, 'stock_return_y'].mean()

    return premium, g.filter(['ticker','period_end']+cut_col)

def calc_group_premium_msci():
    '''  calculate factor premium = inverse of factor value * returns '''
    exit(0)
    return 1

def calc_premium_all():
    ''' calculate factor premium for each currency_code / icb_code(6-digit) for each month '''

    df, stocks_col, macros_col, formula = calc_factor_variables(price_sample='last_day', fill_method='fill_all',
                                                              sample_interval='monthly', use_cached=True, save=True)

    df = df.loc[~df['ticker'].str.startswith('.')]   # remove index e.g. ".SPX" from factor calculation

    df = df.dropna(subset=['stock_return_y'])       # remove records without next month return -> not used to calculate factor premium

    # factor_list = formula.loc[formula['factors'], 'name'].to_list()     # factor = factor variables
    factor_list = formula['name'].to_list()                           # factor = all variabales

    # Calculate premium for currency partition
    print(f'################## Calculate factor premium - Currency Partition ######################')
    member_g_list = []
    results = {}
    for name, g in df.groupby(['period_end', 'currency_code']):
        results[name], member_g = calc_group_premium_fama(name, g, factor_list)
        member_g_list.append(member_g)

    member_df = pd.concat(member_g_list, axis=0)
    results_df = pd.DataFrame(results).transpose()

    member_df.to_csv('membership_curr.csv', index=False)            # Change to upload to DB
    results_df.to_csv('factor_premium_curr.csv')

    # Calculate premium for industry partition
    print(f'################## Calculate factor premium - Industry Partition ######################')
    member_g_list = []
    results = {}
    for name, g in df.groupby(['period_end', 'icb_code']):
        results[name], member_g = calc_group_premium_fama(name, g, factor_list)
        member_g_list.append(member_g)

    member_df = pd.concat(member_g_list, axis=0)
    results_df = pd.DataFrame(results).transpose()

    member_df.to_csv('membership_ind.csv', index=False)
    results_df.to_csv('factor_premium_ind.csv')

if __name__=="__main__":
    calc_premium_all()
