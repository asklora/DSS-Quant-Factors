import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess.ratios_calculations import calc_factor_variables

def calc_monthly_premium_within_group(g, factor_list):
    ''' calculate factor premium with avg(top group monthly return) - avg(bottom group monthly return) '''

    def select_prc(l):
        if l > 65:  # If group sample size is large
            return [0, 0.2, 0.8, 1]
        elif l < 4:  # If group sample size is small
            return np.nan
        else:
            return [0, 0.3, 0.7, 1]

    premium = {}
    for f in factor_list:
        prc = select_prc(g[f].notnull().sum())
        bins = g[f].quantile(prc)
        if bins.isnull().sum() > 0:
            continue
        elif bins[0] == bins[1] == 0:
            prc = g[f].to_list().count(0) / g[f].notnull().sum() + 1e-8
            g[f'{f}_cut'] = pd.qcut(g[f], q=[0, prc, 1 - prc, 1], retbins=False, labels=False)
        elif bins[1] == bins[2] == 0:
            g[f'{f}_cut'] = pd.cut(g[f], [g[f].min(),0,0,g[f].max()], retbins=False, labels=False)
        elif bins[2] == bins[3] == 0:
            prc = g[f].to_list().count(0) / g[f].notnull().sum() + 1e-8
            g[f'{f}_cut'] = pd.qcut(g[f], q=[0, 1-prc, prc, 1], retbins=False, labels=False)
        else:
            print(f'ERROR on {f}, available value {g[f].notnull().sum()}/{len(g)}')
            continue  # Update
        premium[f] = g.loc[g[f'{f}_cut'] == 0, 'stock_return_y'].mean() - g.loc[
            g[f'{f}_cut'] == 2, 'stock_return_y'].mean()

    print(premium)
    return premium

def calc_premium_all():
    df, stock_col, macros_col, formula = calc_factor_variables()

    factor_list = formula.loc[formula['factors'], 'name'].to_list()
    # factor_list = formula['name'].to_list()

    results = df.groupby(['icb_code']).apply(lambda x: calc_monthly_premium_within_group(x, factor_list))

    results = df.groupby(['trading_day', 'icb_code']).apply(lambda x: calc_monthly_premium_within_group(x, factor_list))
    print(results)

if __name__=="__main__":
    calc_premium_all()
