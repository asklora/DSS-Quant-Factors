import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from global_vars import *
from general.sql_process import read_query
import matplotlib.pyplot as plt

def stock_return_hist(currency='USD', weeks_to_expire='%%', average_days='%%', test_period=[5, 10, 20]):
    ''' analyze the distribution in [histogram] of stock return over past test_period(e.g. [5, 10, 20]) years

    Parameters
    ----------
    currency (Str):
        stock returns in which currency to calculate return distribution
    weeks_to_expire (Int, Optional):
        stock returns for given weeks_to_expire
    average_days (Int, Optional):
        stock returns for given average_days
    test_period (List[Str]):
        calculate distributions of past n years returns
    '''

    query = f"SELECT * FROM {processed_ratio_table} WHERE field like 'stock_return_y_w{weeks_to_expire}_d{average_days}'" \
            f" AND ticker in (SELECT ticker FROM universe WHERE currency_code='{currency}')"
    df = read_query(query, db_url_read)
    df['trading_day'] = pd.to_datetime(df['trading_day'])

    num_group = len(df['field'].unique())
    num_period = len(test_period)

    fig, ax = plt.subplots(nrows=num_group, ncols=num_period, figsize=(5*num_period, 5*num_group))

    r = 0
    c = 0
    for name, g in df.groupby('field'):
        for t in test_period:
            g_period = g.loc[g['trading_day'] >= (dt.datetime.now() - relativedelta(years=t))]
            if num_group==1:
                current_ax = ax[c]
            else:
                current_ax = ax[(r, c)]
            current_ax.hist(g_period['value'], bins=1000)
            current_ax.set_xlim((-.5,.5))
            if r==0:
                current_ax.set_xlabel(t)
            if c==0:
                current_ax.set_ylabel(name)
            c+=1
        r+=1
    plt.show()

def stock_return_boxplot(currency='USD', weeks_to_expire='%%', average_days='%%', test_period=[30]):
    ''' analyze the distribution in [boxplot] of stock return over past test_period(e.g. [5, 10, 20]) years

    Parameters - Same above
    '''

    # query = f"SELECT * FROM {processed_ratio_table} WHERE field like 'stock_return_y_w{weeks_to_expire}_d{average_days}'" \
    #         f" AND ticker in (SELECT ticker FROM universe WHERE currency_code='{currency}')"
    # df = read_query(query, db_url_read)
    # df.to_csv('stock_return_y_ratio.csv', index=False)

    df = pd.read_csv('stock_return_y_ratio.csv')
    df['trading_day'] = pd.to_datetime(df['trading_day'])
    df['field'] = df['field'].str[15:]
    des = df.groupby('field').agg(['min', 'mean', 'median', 'max', 'std'])
    print(des)

    fig, ax = plt.subplots(nrows=len(test_period), ncols=1, figsize=(10, 8*len(test_period)))
    c = 0
    for t in test_period:
        df_period = df.loc[df['trading_day'] >= (dt.datetime.now() - relativedelta(years=t))]
        d = {k:v.tolist() for k, v in tuple(df_period.groupby('field')['value'])}
        if len(test_period)==1:
            current_ax = ax
        else:
            current_ax = ax[c]
        current_ax.boxplot(d.values())
        current_ax.set_xticklabels(d.keys())
        current_ax.set_ylabel(t)
        current_ax.axhline(y=0, color='r', linestyle='-')
        # current_ax.set_ylim((-1, 1))
        current_ax.set_ylim((-.5, .5))
        c+=1
    plt.show()

if __name__ == '__main__':
    # stock_return_hist(weeks_to_expire=4, average_days=7)
    stock_return_boxplot()