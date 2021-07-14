import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, QuarterEnd


if __name__ == "__main__":

    market_cap = pd.read_csv('mktcap.csv')
    market_cap1 = pd.read_csv('mktcap_usd.csv')
    market_cap1.columns = ['ticker','market_cap_usd', 'trading_day']
    df = pd.merge(market_cap, market_cap1, on=['ticker', 'trading_day'])
    # df.to_csv('mktcap_final.csv', index=False)

    with global_vals.engine.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi'}
        df.to_sql('data_eikon_factor_mktcap', **extra)
    global_vals.engine.dispose()

    exit(1)




    mem_curr = pd.read_csv('membership_curr.csv')
    factor_curr = pd.read_csv('factor_premium_curr.csv')
    mem_ind = pd.read_csv('membership_ind.csv')
    factor_ind = pd.read_csv('factor_premium_ind.csv')

    factor_curr.columns = ['period_end','group'] + factor_curr.columns.to_list()[2:]

    mem = pd.concat([mem_ind, mem_curr], axis=0)

    f = pd.concat([factor_ind, factor_curr], axis=0)

    print(mem)
    print(f)

    exit(0)


    with global_vals.engine.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi'}
        final_best_pred.to_sql(global_vals.final_eps_pred_results_table, **extra)
    global_vals.engine.dispose()



    ratio = pd.read_csv('all ratio debug.csv', usecols=['ticker', 'period_end', 'earnigs_yield', 'stock_return_y', 'currency_code'])
    ratio = ratio.loc[(ratio['currency_code']=='USD')&(ratio['period_end']=='2020-10-31')]
    # ratio.to_csv('test_p_debug.csv')

    comp = list(set(ratio['ticker'].to_list()))

    mem = pd.read_csv('membership_curr.csv', usecols=['ticker', 'period_end', 'earnigs_yield_cut'])
    mem = mem.loc[(mem['ticker'].isin(comp))&(mem['period_end']=='2020-10-31')]
    # mem.to_csv('test_curr_debug.csv')

    df = ratio.merge(mem, on=['ticker']).sort_values(by=['earnigs_yield'])

    df.to_csv('test_curr_debug.csv')
