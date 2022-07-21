import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

# from global_vars import processed_ratio_table
# from general.sql.sql_process import read_query
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

    num_train_currency = len(df['field'].unique())
    num_period = len(test_period)

    fig, ax = plt.subplots(nrows=num_train_currency, ncols=num_period, figsize=(5 * num_period, 5 * num_train_currency))

    r = 0
    c = 0
    for name, g in df.groupby('field'):
        for t in test_period:
            g_period = g.loc[g['trading_day'] >= (dt.datetime.now() - relativedelta(years=t))]
            if num_train_currency == 1:
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
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
from typing import List
from sqlalchemy import select, and_

from utils import  (
    models,
    date_minus_,
    dateNow,
    read_query
)


class ratio_cluster:

    save_cached = True

    def __init__(self, end_date=dateNow(), lookback: int = 5, currency_code_list: List[str] = ('USD',),
                 ticker_list: List[str] = None, png_name: str = None):
        self.end_date = end_date
        self.lookback = lookback
        self.currency_code_list = currency_code_list
        self.ticker_list = ticker_list
        self.png_name = png_name

    def _get_ratio(self):
        """
        Load Data: load save pickle or download for preprocessed ratios
        """

        if self.save_cached:
            try:
                df = self.__load_cache_ratio()
            except Exception as e:
                df = self.__download_ratio()
                df.to_pickle('cached_cluster_factor_ratio.pkl')
        else:
            df = self.__download_ratio()

        return df

    def __load_cache_ratio(self):
        """
        Load Data: load save pickle
        """
        df = pd.read_pickle('cached_cluster_factor_ratio.pkl')
        df = df.loc[
            (df.index.get_level_values("currency_code").isin(self.currency_code_list)) &
            (df.index.get_level_values("trading_day") <= pd.to_datetime(self.end_date)) &
            (df.index.get_level_values("trading_day") > pd.to_datetime(date_minus_(self.end_date, years=self.lookback)))
        ]

        if type(self.ticker_list) != type(None):
            df = df.loc[df.index.get_level_values("ticker").isin(self.ticker_list)]

        if len(df) == 0:
            raise Exception("Local cached not includes requested trading_day / ticker.")

        return df

    def __download_ratio(self):
        """
        Load Data: download from Table factor_preprocessed_ratios
        """
        conditions = [
            models.Universe.currency_code.in_(self.currency_code_list),
            models.Universe.is_active,
            models.FactorPreprocessRatio.trading_day <= self.end_date,
            models.FactorPreprocessRatio.trading_day > date_minus_(self.end_date, years=self.lookback),
        ]

        if type(self.ticker_list) != type(None):
            conditions.append(models.FactorPreprocessRatio.ticker.in_(self.ticker_list))

        query = select(*models.FactorPreprocessRatio.__table__.columns, models.Universe.currency_code)\
            .join(models.Universe)\
            .where(and_(*conditions))
        df = read_query(query)
        df = df.pivot(index=["ticker", "trading_day", "currency_code"], columns=["field"], values='value')
        df = df.fillna(0)

        return df

    def get(self):
        # df = pd.read_pickle('cache_factor_ratio.pkl')

        df = self._get_ratio()

        X = scale(df)

        # cop_coef = {}

        for l in ['average']:     # ['ward', 'average', 'complete', 'single']
            agglo = FeatureAgglomeration(n_clusters=None, distance_threshold=0, linkage=l)
            agglo.fit(X)
            # print(Counter(agglo.labels_))

            # plt.plot(agglo.distances_)
            # plt.show()

            Z = self.__linkage(agglo)
            self.plot_dendrogram(Z, labels=df.columns.to_list())

            # cop_coef[l] = {}
            # for k, v in Y.items():
            #     cop_coef[start_year][k], _ = cophenet(Z, v)
            #     print(l, k, cop_coef)

        # cop_coef_df = pd.DataFrame(cop_coef)
        # cop_coef_df.to_csv(f'temp.csv')

    def __linkage(self, model):
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

        distance = np.arange(model.children_.shape[0])
        linkage_matrix = np.column_stack(
            [model.children_, distance, counts]
        ).astype(float)

        return linkage_matrix

    def plot_dendrogram(self, linkage_matrix, **kwargs):
        """ Plot the corresponding dendrogram """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        dendrogram(linkage_matrix, orientation='left', ax=ax, **kwargs)
        plt.tight_layout()

        if type(self.png_name) == type(None):
            plt.show()
        else:
            plt.title(self.png_name)
            plt.savefig(self.png_name + ".png")

    def sample_ratio_rank(self):
        """
        Calculate: rank% for selected samples + selected factors by comparing two dendrogram (universe vs sample)
        """

        sample_ticker = self.ticker_list

        self.ticker_list = None
        df = self._get_ratio()

        df = df.reset_index().sort_values(by="trading_day").groupby(["ticker"]).last()

        df_rank = df.rank()[["market_cap_usd", "inv_turnover", "cash_ratio", "ca_turnover"]]
        df_rank = df_rank / df_rank.max(axis=0)

        df_rank_sample = df_rank.loc[df.index.get_level_values("ticker").isin(sample_ticker)]
        return df_rank_sample


if __name__ == '__main__':
    # stock_return_hist(weeks_to_expire=4, average_days=7)
    # stock_return_boxplot()

    ratio_cluster(currency_code_list=["USD"],
                  png_name="usd_all").get()

    # ratio_cluster(currency_code_list=["USD"],
    #               ticker_list=["ARKF.K", "TSLA.O", "F", "GILD.O", "AAPL.O", "MA", "INTC.O", "C"],
    #               png_name="usd_innovative").get()

    # ratio_cluster(currency_code_list=["USD"],
    #               ticker_list=["ARKF.K", "TSLA.O", "F", "GILD.O", "AAPL.O", "MA", "INTC.O", "C"],
    #               png_name="usd_innovative").sample_ratio_rank()

    # ratio_cluster(currency_code_list=["USD"],
    #               ticker_list=["AMD.O", "THC", "BA", "AAPL.O", "ETSY.O"],
    #               png_name="usd_revenue").get()
    #
    # ratio_cluster(currency_code_list=["USD"],
    #               ticker_list=["AMD.O", "THC", "BA", "AAPL.O", "ETSY.O"],
    #               png_name="usd_revenue").sample_ratio_rank()