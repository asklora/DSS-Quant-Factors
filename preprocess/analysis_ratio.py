import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

import global_vars
from global_vars import processed_ratio_table
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
    df = read_query(query)
    df['trading_day'] = pd.to_datetime(df['trading_day'])

    num_group = len(df['field'].unique())
    num_period = len(test_period)

    fig, ax = plt.subplots(nrows=num_group, ncols=num_period, figsize=(5 * num_period, 5 * num_group))

    r = 0
    c = 0
    for name, g in df.groupby('field'):
        for t in test_period:
            g_period = g.loc[g['trading_day'] >= (dt.datetime.now() - relativedelta(years=t))]
            if num_group == 1:
                current_ax = ax[c]
            else:
                current_ax = ax[(r, c)]
            current_ax.hist(g_period['value'], bins=1000)
            current_ax.set_xlim((-.5, .5))
            if r == 0:
                current_ax.set_xlabel(t)
            if c == 0:
                current_ax.set_ylabel(name)
            c += 1
        r += 1
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

    fig, ax = plt.subplots(nrows=len(test_period), ncols=1, figsize=(10, 8 * len(test_period)))
    c = 0
    for t in test_period:
        df_period = df.loc[df['trading_day'] >= (dt.datetime.now() - relativedelta(years=t))]
        d = {k: v.tolist() for k, v in tuple(df_period.groupby('field')['value'])}
        if len(test_period) == 1:
            current_ax = ax
        else:
            current_ax = ax[c]
        current_ax.boxplot(d.values())
        current_ax.set_xticklabels(d.keys())
        current_ax.set_ylabel(t)
        current_ax.axhline(y=0, color='r', linestyle='-')
        # current_ax.set_ylim((-1, 1))
        current_ax.set_ylim((-.5, .5))
        c += 1
    plt.show()


from sklearn.preprocessing import scale
from sklearn.cluster import FeatureAgglomeration
from scipy.cluster.hierarchy import dendrogram, cophenet
from scipy.spatial.distance import pdist, squareform
from collections import Counter


class ratio_cluster:

    def __init__(self, start_date='2000-01-01', currency_code='USD'):

        # conditions = [f"ticker in (SELECT ticker FROM universe WHERE currency_code='{currency_code}')"]
        # if start_date:
        #     conditions.append(f"trading_day > '{start_date}'")
        # query = f"SELECT * FROM {processed_ratio_table} WHERE {' AND '.join(conditions)}"
        # df = read_query(query)
        # df.to_pickle('cache_factor_ratio.pkl')

        df = pd.read_pickle('cache_factor_ratio.pkl')

        df['trading_day'] = pd.to_datetime(df['trading_day'])
        df = df.set_index(['trading_day', 'ticker', 'field'])['value'].unstack()
        df = df.fillna(0)

        df_year = df.index.get_level_values('trading_day').to_series().dt.year

        X = scale(df)
        # Y = {"since 2000": pdist(X.T), "since 2016": pdist(X[df_year > 2016].T)}
        # for year in np.sort(df_year.unique()):
        #     X_year = X[df_year == year]
        #     Y[year] = pdist(X_year.T)

        cop_coef = {}
        for i in [5]:
            start_year = 2016 - i
            X_sample = X[(df_year < 2016) & (df_year >= start_year)]
            cop_coef[start_year] = {}

            for l in ['average']:     # ['ward', 'average', 'complete', 'single']
                agglo = FeatureAgglomeration(n_clusters=3, distance_threshold=None, linkage=l)
                agglo.fit(X_sample)
                print(Counter(agglo.labels_))

                plt.plot(agglo.distances_)
                plt.show()

                Z = ratio_cluster.__linkage(agglo)

                # cop_coef[l] = {}
                # for k, v in Y.items():
                #     cop_coef[start_year][k], _ = cophenet(Z, v)
                #     print(l, k, cop_coef)

        cop_coef_df = pd.DataFrame(cop_coef)
        cop_coef_df.to_csv(f'temp.csv')

            # ratio_cluster.plot_dendrogram(Z, png_name=f"{start_date}_{l}.png", labels=df.columns.to_list())

    @staticmethod
    def __linkage(model):
        """ Create linkage metrix from model """

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        return linkage_matrix

    @staticmethod
    def plot_dendrogram(linkage_matrix, png_name, **kwargs):
        """ Plot the corresponding dendrogram """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        dendrogram(linkage_matrix, orientation='left', ax=ax, **kwargs)
        plt.tight_layout()
        plt.savefig(png_name)


if __name__ == '__main__':
    # stock_return_hist(weeks_to_expire=4, average_days=7)
    # stock_return_boxplot()
    ratio_cluster()