import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, QuarterEnd

if __name__ == "__main__":
    ratio = pd.read_csv('all ratio debug.csv', usecols=['ticker', 'period_end', 'earnigs_yield', 'stock_return_y', 'currency_code'])
    ratio = ratio.loc[(ratio['currency_code']=='USD')&(ratio['period_end']=='2020-10-31')]
    # ratio.to_csv('test_p_debug.csv')

    comp = list(set(ratio['ticker'].to_list()))

    mem = pd.read_csv('membership_curr.csv', usecols=['ticker', 'period_end', 'earnigs_yield_cut'])
    mem = mem.loc[(mem['ticker'].isin(comp))&(mem['period_end']=='2020-10-31')]
    # mem.to_csv('test_curr_debug.csv')

    df = ratio.merge(mem, on=['ticker']).sort_values(by=['earnigs_yield'])

    df.to_csv('test_curr_debug.csv')
