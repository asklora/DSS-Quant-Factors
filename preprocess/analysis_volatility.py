import logger

import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
import itertools
from general.send_slack import to_slack

from global_vars import *
from general.sql_process import read_query, upsert_data_to_database, trucncate_table_in_database, uid_maker

from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION
from sqlalchemy.sql.sqltypes import BOOLEAN

import seaborn as sns
import matplotlib.pyplot as plt

class vol_analysis:

    def __init__(self, ticker=None, currency=None, vol_col='vol_0_30'):

        self.vol_col = vol_col  # vol_0_30 / skew for evaluation

        try:
            self.ratio = pd.read_csv(f'cached_vol_ret_{vol_col}.csv').dropna(how='any')
        except:
            self.ratio = self.read_ratio(ticker, currency)
            self.ratio.to_csv(f'cached_vol_ret_{vol_col}.csv', index=False)

        self.prem = self.read_premium(currency)

        self.corr_vol()
        self.dist_vol()

    def read_ratio(self, ticker, currency):
        ''' read vol / return ratios from factor_processed_ratio Table '''
        logger.info(f"=== Get vol/ret ratios from {processed_ratio_table} ===")

        conditions = ["is_active", f"(field='{self.vol_col}' or field like 'stock_return_%%')"]
        if ticker:
            conditions.append(f"u.ticker in {tuple(ticker)}")
        if currency:
            conditions.append(f"u.currency_code in {tuple(currency)}")

        ratio_query = f"SELECT r.*, u.currency_code " \
                      f"FROM {processed_ratio_table} r " \
                      f"INNER JOIN universe u ON r.ticker=u.ticker " \
                      f"WHERE {' AND '.join(conditions)}".replace(",)", ")")

        df = read_query(ratio_query, db_url_read)
        df = df.pivot(index=["ticker", "trading_day", "currency_code"], columns=["field"], values='value').reset_index()
        df.columns = [x[13:] if x.startswith('stock_return_') else x for x in df]

        df['year'] = pd.to_datetime(df['trading_day']).dt.year
        df = df.sort_values(by=['year', 'currency_code'])

        # filter df
        df = df.loc[~df['ticker'].str.startswith('.')].copy()
        return df

    def read_premium(self, currency):
        ''' read vol / return ratios from factor_processed_ratio Table '''
        logger.info(f"=== Get vol/ret ratios from {factor_premium_table} ===")

        conditions = [f"field='{self.vol_col}'"]
        if currency:
            conditions.append(f"\"group\" in {tuple(currency)}")

        ratio_query = f"SELECT * FROM {factor_premium_table} " \
                      f"WHERE {' AND '.join(conditions)}".replace(",)", ")")

        df = read_query(ratio_query, db_url_read).rename(columns={'group': 'currency_code'})
        df['year'] = pd.to_datetime(df['trading_day']).dt.year
        df = df.sort_values(by=['year', 'currency_code'])
        return df

    def corr_vol(self):
        ''' evaluate the relationship between volatility & return '''

        rlist = ['y_w26_d7', 'y_w8_d7', 'y_w4_d7', 'y_w1_d1', 'ww1_0', 'ww2_1', 'ww4_2', 'r1_0', 'r6_2', 'r12_7']
        rlist = list(reversed(rlist))

        # corr (vol -> ret) by [group, year]
        corr1 = self.ratio.groupby(['currency_code']).corr()[self.vol_col].unstack().drop(columns=[self.vol_col]).transpose()

        fig = plt.figure(figsize=(20, 5), dpi=120, constrained_layout=True)
        k = 1
        for cur, g in self.ratio.groupby(['currency_code']):
            corr2 = g.dropna(how='any').groupby(['year']).corr()[self.vol_col].unstack().filter(rlist)
            ax = fig.add_subplot(1, 4, k)
            sns.heatmap(corr2, ax=ax, cmap='coolwarm', vmin=-0.3, vmax=0.3, cbar=False)
            ax.xaxis.set_ticks_position('top')
            ax.set_xlabel(cur)
            plt.xticks(rotation=45)
            k+=1
        plt.tight_layout()
        plt.suptitle(f'{self.vol_col} Corr ({cur})')
        plt.savefig(f'{self.vol_col} Corr ({cur}).png')
        pass

    def dist_vol(self):
        ''' distribution of volatility each year '''

        fig = plt.figure(figsize=(10, 10), dpi=120, constrained_layout=True)
        k = 1
        for cur, g in self.ratio.groupby(['currency_code']):
            ax = fig.add_subplot(2, 2, k)
            d1 = g.groupby(['year'])[self.vol_col].apply(list).to_dict()
            plt.boxplot(d1.values(), whis=2)
            ax.set_xticklabels(d1.keys(), rotation=90)
            ax.set_ylim((0, 1))
            ax.set_xlabel(cur)
            ax.axhline(y=0, color='r', linestyle='-')
            k+=1
        plt.suptitle(f'{self.vol_col} Dist')
        plt.savefig(f'{self.vol_col} Dist.png')

        for w, g1 in self.prem.groupby(['weeks_to_expire']):
            fig = plt.figure(figsize=(10, 10), dpi=120, constrained_layout=True)
            k = 1
            for cur, g in g1.groupby(['currency_code']):
                ax = fig.add_subplot(2, 2, k)
                d1 = g1.groupby(['year'])['value'].apply(list).to_dict()
                ax.boxplot(d1.values())
                ax.set_xticklabels(d1.keys(), rotation=90)
                # ax.set_ylim((-0.3, 0.3))
                ax.set_xlabel(cur)
                ax.axhline(y=0, color='r', linestyle='-')
                k+=1
            plt.suptitle(f'{self.vol_col} Premium Dist ({w})')
            plt.savefig(f'{self.vol_col} Premium Dist ({w}).png')
        pass

if __name__ == '__main__':
    vol_analysis(currency=['HKD', 'CNY', 'USD', 'EUR'], vol_col='vol_0_30')
    vol_analysis(currency=['HKD', 'CNY', 'USD', 'EUR'], vol_col='skew')

