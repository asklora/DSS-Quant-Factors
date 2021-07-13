import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess.ratios_calculations import calc_factor_variables

def calc_group_premium_fama(g, factor_list):
    ''' calculate factor premium with avg(top group monthly return) - avg(bottom group monthly return) '''

    def select_prc(l):
        if l > 65:  # If group sample size is large
            return [0, 0.2, 0.8, 1]
        else:
            return [0, 0.3, 0.7, 1]

    premium = {}
    for f in factor_list:
        # print(g)
        if len(g) < 10:
            continue
        prc_list = select_prc(g[f].notnull().sum())
        bins = g[f].quantile(prc_list).to_list()
        # print(f, bins)
        if (np.isnan(bins[0])) or bins.count(0)>2:
            continue
        elif bins[0] == bins[1] == 0:
            prc = g[f].to_list().count(0) / g[f].notnull().sum() + 1e-8
            g[f'{f}_cut'] = pd.qcut(g[f], q=[0, prc, 1-prc, 1], retbins=False, labels=False)
        elif bins[1] == bins[2] == 0:
            bins = pd.IntervalIndex.from_tuples([(g[f].min(), 0), (0,0), (0, g[f].max())])
            g[f'{f}_cut'] = pd.cut(g[f], bins, retbins=False, labels=False)
        elif bins[2] == bins[3] == 0:
            prc = g[f].to_list().count(0) / g[f].notnull().sum() + 1e-8
            g[f'{f}_cut'] = pd.qcut(g[f], q=[0, 1-prc, prc, 1], retbins=False, labels=False)
        else:
            g[f'{f}_cut'] = pd.cut(g[f], bins=bins, retbins=False, labels=False)

        # print(g.loc[g[f'{f}_cut']==0, 'stock_return_y'])
        premium[f] = g.loc[g[f'{f}_cut'] == 0, 'stock_return_y'].mean()-g.loc[g[f'{f}_cut'] == 2, 'stock_return_y'].mean()

    print(premium)
    cut_col = [x+'_cut' for x in factor_list]
    return premium, g.filter(['ticker','period_end']+cut_col

def calc_group_premium_msci():
    '''  calculate factor premium = inverse of factor value * returns '''



def calc_premium_all():
    ''' calculate factor premium for each currency_code / icb_code(6-digit) for each month '''
    df, stock_col, macros_col, formula = calc_factor_variables()

    df = df.dropna(subset=['stock_return_y'])       # remove records without next month return -> not used to calculate factor premium

    factor_list = formula.loc[formula['factors'], 'name'].to_list()     # factor = factor variables
    # factor_list = formula['name'].to_list()                           # factor = all variabales

    # Calculate premium for currency partition
    member_g_list = []
    results = {}
    for name, g in df.groupby(['period_end', 'currency_code']):
        results[name], member_g = calc_group_premium_fama(g, factor_list)
        member_g_list.append(member_g)

    member_df = pd.concat(member_g_list, axis=0)
    results_df = pd.DataFrame(results).transpose()

    member_df.to_csv('membership_curr.csv', index=False)            # Change to upload to DB
    results_df.to_csv('factor_premium_curr.csv', index=False)


    # Calculate premium for industry partition
    member_g_list = []
    results = {}
    for name, g in df.groupby(['period_end', 'icb_code']):
        results[name], member_g = calc_group_premium_fama(g, factor_list)
        member_g_list.append(member_g)

    member_df = pd.concat(member_g_list, axis=0)
    results_df = pd.DataFrame(results).transpose()

    member_df.to_csv('membership_ind.csv', index=False)
    results_df.to_csv('factor_premium_ind.csv', index=False)

    #######################
    #1. reshape results
    #2. membership table
    #3. MSCI weighting

if __name__=="__main__":
    calc_premium_all()
