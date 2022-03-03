import pandas as pd
import numpy as np
import datetime as dt
import os
from general.utils import to_excel
from general.sql_process import read_query
import ast
from dateutil.relativedelta import relativedelta

config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', 'use_average', 'down_mkt_pct']


class read_v2_top_excel:

    def __init__(self):
        df_top10 = pd.read_excel("v2_w4_d7_official.xlsx", f"top10")
        df_top10['weeks'] = 4
        df_top10_2 = pd.read_excel("v2_w13_d7_20220301195636_debug.xlsx", f"top10")
        df_top10_2['weeks'] = 13
        df_top10 = df_top10.append(df_top10_2)

        df_top10 = df_top10.loc[df_top10['currency_code'] != 'EUR']

        evaluate_topn_returns(df_top10)
        evaluate_config_accuracy(df_top10)

    @staticmethod
    def evaluate_topn_returns(df_top10):
        # c = df_top10.groupby(['weeks', 'currency_code', 'trading_day', 'y_type']).count()

        for n in [10, 20, 40, 80]:
            df_n_avg = df_top10.groupby(['weeks', 'currency_code']).apply(lambda x:
                                                                          x.nlargest(n, ['return'], keep='all')[
                                                                              'return'].mean()).unstack()
            print(df_n_avg)

    @staticmethod
    def evaluate_config_accuracy(df_top10):
        for col in config_col:
            df_config = df_top10.groupby(['weeks', 'y_type', col, 'currency_code'])['return'].mean().unstack()
            print(df_config)


class read_v2_actual_eval:
    name_sql = 'w13_d7_20220301195636_debug'
    weeks_to_expire = 13

    # name_sql = 'w4_d7_official'
    # weeks_to_expire = 4

    n_period_list = None
    n_config_list = None
    m_asc_list = None
    is_config_test = None

    def __init__(self):

        self.df_top10 = pd.read_excel(f"v2_{self.name_sql}.xlsx", f"top10")
        configs = self.find_config()

    def find_config(self):
        try:
            df = pd.read_csv(f'cached_factor_result_rank_backtest_eval_{self.name_sql}.csv')
        except:
            df = read_query(f"SELECT * FROM factor_result_rank_backtest_eval WHERE name_sql='{self.name_sql}'")
            df.to_csv(f'cached_factor_result_rank_backtest_eval_{self.name_sql}.csv')

        df['trading_day'] = pd.to_datetime(df['trading_day']) + pd.tseries.offsets.DateOffset(
            weeks=self.weeks_to_expire)
        df['net_ret'] = df['max_ret'] - df['min_ret']

        df_config = pd.DataFrame(df['config'].apply(ast.literal_eval).to_list())
        df = pd.concat([df, df_config], axis=1)

        # 1. test on configurations
        # self.set_test_config()
        # self.best_config(df)

        # 2. test return for top picks
        self.set_test_ret()
        self.best_config(df)

    def set_test_config(self):
        """ set configurations if we test many n_period """
        self.n_period_list = [12, 24, 36, 48]
        self.n_config_list = [10, 20, 40, 80]
        self.m_asc_list = [('mse', True), ('max_ret', False), ('net_ret', False)]
        self.is_config_test = True

    def set_return_config(self):
        """ set configurations if we test many n_period """
        self.n_period_list = [12]
        self.n_config_list = [10, 80]
        self.m_asc_list = [('net_ret', False)]
        self.is_config_test = False

    def best_config_test(self, df):
        """ test resutls on different configuration """
        date_list = list(reversed(sorted(df['trading_day'].unique())))

        ret_data = daily_trans_return()
        ret_data.holding(weeks_to_expire=self.weeks_to_expire)

        best_config_info = {}
        for (y_type, group), g in df.groupby(['y_type', 'group']):
            for n_period in self.n_period_list:
                for i in range(len(date_list)):
                    ret_series = ret_data.period_average(date_list[i])      # pd.Series: ticker -> ret

                    if i + n_period + 1 > len(date_list):
                        continue
                    backtest_date_list = date_list[i + 1:i + n_period + 1]
                    g_current = g.loc[df['trading_day'] == date_list[i]]
                    g_backtest = g.loc[df['trading_day'].isin(backtest_date_list)]

                    # assert len(g_backtest['actual'].unique()) == n_period
                    for (m, asc) in self.m_asc_list:
                        g_backtest_avg = g_backtest.groupby(config_col)[m].mean()
                        g_backtest_avg = g_backtest_avg.sort_values(ascending=asc)

                        for n_config in self.n_config_list:
                            config_best = g_backtest_avg.head(n_config).reset_index()

                            top_best = self.df_top10.loc[(self.df_top10['y_type'] == y_type) &
                                                         (self.df_top10['currency_code'] == group) &
                                                         (self.df_top10['trading_day'] == date_list[i])]
                            top_best = top_best.merge(config_best, on=config_col)

                            print('---> finish', (y_type, group, date_list[i], m, n_config, n_period))
                            if self.is_config_test:
                                best_config_info[(y_type, group, date_list[i], m, n_config, n_period)] = top_best[
                                    ['positive_pct', 'return']].mean()
                            # else:
                            #     tickers = top_best['tickers'] =

        if self.is_config_test:
            read_v2_actual_eval.save_test_config_excel(best_config_info)
        else:
            read_v2_actual_eval.save_test_return_excel(best_config_info)

    @staticmethod
    def save_test_config_excel(best_config_info):
        best_config_info = pd.DataFrame(best_config_info).transpose().reset_index().dropna(subset=['return'])
        best_config_info.columns = ['y_type', 'group', 'trading_day', 'm',
                                    'n_configs', 'n_period', 'positive_pct', 'return']
        print(best_config_info)

        xlsx_df = {}
        xlsx_df['all'] = best_config_info
        for i in ['positive_pct', 'return']:
            xlsx_df[f'{i}_agg'] = best_config_info.groupby(['n_configs', 'n_period', 'm', 'group'])[
                i].mean().unstack().reset_index()
        to_excel(xlsx_df, f'v2_{self.name_sql}_agg')
        return True

    @staticmethod
    def save_test_return_excel(best_config_info):
        return

class daily_trans_return:
    """ calculate stock holding return is hold all stocks for given periods """

    df_sum = None
    df = None
    weeks_to_expire = None

    def __init__(self):
        try:
            df = pd.read_csv('cache_tri.csv')
        except:
            df = read_query(f"SELECT ticker, trading_day, total_return_index as tri "
                            f"FROM data_tri WHERE trading_day > '2021-01-01' ORDER BY ticker, trading_day")
            df.to_csv('cache_tri.csv', index=False)

        df = daily_trans_return.__fill_business_day(df)
        df['tri'] = df.groupby(['ticker'])['tri'].pct_change().fillna(0)
        df['tri'] = np.log(df['tri'] + 1)
        self.df = df

    def holding(self, weeks_to_expire):
        self.df_sum = self.df.copy()
        self.df_sum['tri'] = self.df_sum.groupby('ticker')['tri'].rolling(weeks_to_expire*5,
                                                                          min_periods=weeks_to_expire*5).sum().values
        self.df_sum['tri'] = np.exp(self.df_sum['tri']) - 1
        self.weeks_to_expire = weeks_to_expire

    def period_average(self, trading_day):
        df_ret = self.df_sum.copy()
        df_ret = df_ret.loc[(df_ret['trading_day'] > trading_day) &
                            (df_ret['trading_day'] <= trading_day + relativedelta(weeks=self.weeks_to_expire))]
        df_ret = df_ret.groupby('ticker')['tri'].mean()
        return df_ret

    @staticmethod
    def __fill_business_day(result, date_col="trading_day"):
        """ Fill all the weekends between first / last day and fill NaN """

        # Construct indexes for all day between first/last day * all ticker used
        df = result[["ticker", date_col]].copy()
        df.trading_day = pd.to_datetime(df[date_col])
        result.trading_day = pd.to_datetime(result[date_col])
        df = df.sort_values(by=[date_col], ascending=True)

        # last_day = Sunday
        last_day = df.iloc[-1, 1] + pd.offsets.Week(weekday=6)
        daily = pd.bdate_range(df.iloc[0, 1], last_day, freq='B')
        indexes = pd.MultiIndex.from_product([df['ticker'].unique(), daily], names=['ticker', date_col])

        # Insert weekend/before first trading date to df
        df = df.set_index(['ticker', date_col]).reindex(indexes).reset_index()
        df = df.sort_values(by=['ticker', date_col], ascending=True)
        result = df.merge(result, how="left", on=["ticker", date_col])

        return result


if __name__ == '__main__':
    # read_v2_top_excel()
    # read_v2_actual_eval()
    data = daily_trans_return()
    data.holding(weeks_to_expire=4)
    data.period_average(dt.datetime(2021, 10, 31, 0, 0, 0))
