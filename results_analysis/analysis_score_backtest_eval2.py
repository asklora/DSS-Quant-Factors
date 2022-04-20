import pandas as pd
import numpy as np
import datetime as dt
import os
from general.utils import to_excel
from general.sql_process import read_query
import ast
from dateutil.relativedelta import relativedelta
import global_vars

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
        # c = df_top10.groupby(['weeks', 'currency_code', 'trading_day', 'pillar']).count()

        for n in [10, 20, 40, 80]:
            df_n_avg = df_top10.groupby(['weeks', 'currency_code']).apply(lambda x:
                                                                          x.nlargest(n, ['return'], keep='all')[
                                                                              'return'].mean()).unstack()
            print(df_n_avg)

    @staticmethod
    def evaluate_config_accuracy(df_top10):
        for col in config_col:
            df_config = df_top10.groupby(['weeks', 'pillar', col, 'currency_code'])['return'].mean().unstack()
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
        for (pillar, group), g in df.groupby(['pillar', 'group']):
            for n_period in self.n_period_list:
                for i in range(len(date_list)):
                    ret_series = ret_data.period_average(date_list[i])  # pd.Series: ticker -> ret

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

                            top_best = self.df_top10.loc[(self.df_top10['pillar'] == pillar) &
                                                         (self.df_top10['currency_code'] == group) &
                                                         (self.df_top10['trading_day'] == date_list[i])]
                            top_best = top_best.merge(config_best, on=config_col)

                            print('---> finish', (pillar, group, date_list[i], m, n_config, n_period))
                            if self.is_config_test:
                                best_config_info[(pillar, group, date_list[i], m, n_config, n_period)] = top_best[
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
        best_config_info.columns = ['pillar', 'group', 'trading_day', 'm',
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
    up_to_end_date = False  # if up_to_end_date -> use end_in_1w = False
    end_in_1w = True  # end date = start_date + weeks_to_expire + 7d (else + weeks_to_expire)

    def __init__(self):
        try:
            df = pd.read_csv('cache_tri.csv')
        except:
            df = read_query(f"SELECT ticker, trading_day, total_return_index as tri "
                            f"FROM data_tri WHERE trading_day > '2018-01-01' ORDER BY ticker, trading_day")
            df.to_csv('cache_tri.csv', index=False)

        df = daily_trans_return.__fill_business_day(df).ffill()


        df['tri'] = df.groupby(['ticker'])['tri'].pct_change().fillna(0)
        df['tri'] = np.log(df['tri'] + 1)

        self.df = df

    def holding(self, weeks_to_expire):
        self.weeks_to_expire = weeks_to_expire
        self.df_sum = self.df.copy()
        if self.up_to_end_date:
            pass
        else:
            self.df_sum['tri'] = self.df_sum.groupby('ticker')['tri'].rolling(weeks_to_expire * 5).sum().values
            self.df_sum['tri'] = self.df_sum['tri'].shift(-weeks_to_expire * 5) / (weeks_to_expire / 4)
            self.df_sum['tri'] = np.exp(self.df_sum['tri']) - 1

    def period_average(self, trading_day):
        df_ret = self.df_sum.copy()

        if self.end_in_1w:
            end_date = trading_day + relativedelta(days=7)
            df_ret = df_ret.loc[(df_ret['trading_day'] >= trading_day) & (df_ret['trading_day'] < end_date)]

            # end_date = trading_day - relativedelta(days=7)
            # df_ret = df_ret.loc[(df_ret['trading_day'] <= trading_day) & (df_ret['trading_day'] > end_date)]
        else:
            end_date = trading_day + relativedelta(weeks=self.weeks_to_expire)
            df_ret = df_ret.loc[(df_ret['trading_day'] >= trading_day) & (df_ret['trading_day'] < end_date)]

        if self.up_to_end_date:
            df_ret['tri'] = df_ret.groupby('ticker')['tri'].apply(lambda x: np.cumsum(x[::-1])[::-1])
        else:
            pass

        df_ret = df_ret.groupby('ticker')['tri'].mean()

        # x = df_ret.to_dict()
        # x1 = x['0883.HK']
        # x2 = x['UDR']

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


class top2_table_tickers_return:
    n_top_config = 1
    weeks_to_expire = 4

    def __init__(self, df=None, name_sql="w4_d-7_20220310130330_debug", xlsx_name=None):

        data = daily_trans_return()

        if type(df) == type(None):
            tbl_name = global_vars.backtest_top_table
            df_all = read_query(f"SELECT * FROM {tbl_name} WHERE trading_day > '2020-01-01' and name_sql='{name_sql}'"
                                f"and xlsx_name={xlsx_name}")

        df_all['trading_day'] = pd.to_datetime(df_all['trading_day'])

        df_new = []
        for (n_top_config, weeks_to_expire, n_top_ticker), df in df_all.groupby(['n_top_config', 'weeks_to_expire', 'n_top_ticker']):
            data.holding(weeks_to_expire=weeks_to_expire)
            for trading_day, g in df.groupby('trading_day'):
                ret_map = data.period_average(trading_day).to_dict()
                g['new_return'] = g['tickers'].apply(lambda x: np.mean([ret_map[e] for e in x.split(', ')]))
                df_new.append(g)
                print(df_new)

        df_new = pd.concat(df_new, axis=0)
        df_new_agg = df_new.groupby(['currency_code', 'weeks_to_expire', 'n_top_config',
                                     'n_backtest_period', 'n_top_ticker'])['new_return'].agg(
            ['min', 'max', 'mean', 'std', 'count']).reset_index()
        df_new_agg['sharpe'] = df_new_agg['mean'] / df_new_agg['std']

        num_col = ['min', 'max', 'mean', 'std', 'count', 'sharpe']
        to_excel({
            "all": df_new,
            "agg": df_new_agg,
            "n_backtest_period": df_new_agg.groupby(['weeks_to_expire', 'n_backtest_period'])[
                num_col].mean().reset_index(),
            "n_backtest_period_mean":
                df_new_agg.groupby(['weeks_to_expire', 'currency_code', 'n_backtest_period'
                                    ])['mean'].mean().unstack().reset_index(),
            "n_top_config": df_new_agg.groupby(['weeks_to_expire', 'n_top_config'])[num_col].mean().reset_index(),
            "n_top_config_mean":
                df_new_agg.groupby(['weeks_to_expire', 'currency_code', 'n_top_config'
                                    ])['mean'].mean().unstack().reset_index(),
            "n_top_ticker": df_new_agg.groupby(['weeks_to_expire', 'n_top_ticker'])[num_col].mean().reset_index(),
            "n_top_ticker_mean":
                df_new_agg.groupby(['weeks_to_expire', 'currency_code', 'n_top_ticker'
                                    ])['mean'].mean().unstack().reset_index(),
        },
            f'bk_score_eval_{xlsx_name if xlsx_name else input("input xlsx name:")}')

        # x = df_new_agg.groupby(['weeks_to_expire', 'currency_code', 'n_backtest_period']).mean().unstack()
        # print(df_new_agg.groupby(['weeks_to_expire']).mean())


# def top2_table_agg():
#     df = read_query("SELECT * FROM factor_result_rank_backtest_top2")
#     df['trading_day'] = pd.to_datetime(df['trading_day'])
#     df['year'] = df['trading_day'].dt.year
#
#     df_4 = df.loc[df['weeks_to_expire'] == 4]
#     print(df_4.describe())
#
#     # df_year = df_4.groupby(['currency_code', 'year'])['return'].mean().unstack().reset_index()
#     # df_year.to_csv('top2_df_year.csv')
#     # print(df_year)
#
#     df_since8 = df.loc[df['trading_day'] > dt.datetime(2021, 8, 1, 0, 0, 0)]
#
#     exit(200)


# def get_index_return_annual():
#     df = read_query("SELECT * FROM data_tri WHERE ticker like '.%%' AND EXTRACT(MONTH FROM trading_day)<2"
#                     "AND EXTRACT(year FROM trading_day)>=2016")
#
#     df['trading_day'] = pd.to_datetime(df['trading_day'])
#     df['year'] = df['trading_day'].dt.year
#     df = df.sort_values(by='trading_day').groupby(['ticker', 'year'])[['total_return_index']].mean()
#     df_return = df.groupby(['ticker'])['total_return_index'].pct_change().unstack()
#     df_return.to_csv('index_return_annual.csv')
#     print(df_return)
#
#     exit(200)


# class top2_table_bot_return:
#     n_top_config = 1
#     weeks_to_expire = 4
#
#     def get_bot_table(self):
#         df = pd.read_excel("best_bot_best_stock.xlsx", "Best Bot Best Stock 1m")
#         return df
#
#     # def get_bot_table_db(self):
#
#
#     def __init__(self, df=None):
#
#         data = self.get_bot_table()[['ticker', 'spot_date', 'bot_type', 'bot_return', 'expiry_return']]
#         # data[['bot_return', 'expiry_return']] = pd.to_numeric(data[['bot_return', 'expiry_return']], errors='coerce')
#
#         if type(df) == type(None):
#
#             tbl_name = global_vars.backtest_top_table
#             name_sql = name_sql
#             df = read_query(f"SELECT * FROM {tbl_name} "
#                             f"WHERE n_top_config={n_top_config} and trading_day > '2021-09-01' "
#                             f"and weeks_to_expire={weeks_to_expire}"
#                             f"and currency_code <> 'EUR'")
#
#         df_new = []
#         for n_top_config in [1, 3, 5, 10, 20, 40, 80]:
#             for weeks_to_expire in [4, 13]:
#                 df = read_query(f"SELECT * FROM factor_result_rank_backtest_top2 "
#                                 f"WHERE n_top_config={n_top_config} and trading_day > '2021-09-01' "
#                                 f"and weeks_to_expire={weeks_to_expire}"
#                                 f"and currency_code <> 'EUR'")
#                 df['trading_day'] = pd.to_datetime(df['trading_day'])
#                 for trading_day, g in df.groupby('trading_day'):
#                     # for bot_type in ['CLASSIC', 'UNO', 'UCDC']:
#                     data_return = data.loc[(data['spot_date'] > trading_day)
#                                            # & (data['spot_date'] <= (trading_day + relativedelta(weeks=weeks_to_expire)))
#                                            & (data['spot_date'] <= (trading_day + relativedelta(weeks=1)))
#                         # & (data['bot_type'] == bot_type)
#                     ]
#                     if len(data_return) == 0:
#                         continue
#                     ret_mapb = data_return.groupby('ticker')['bot_return'].mean().to_dict()
#                     ret_mape = data_return.groupby('ticker')['expiry_return'].mean().to_dict()
#                     g['bot_return'] = g['tickers'].apply(lambda x: np.mean([ret_mapb[e] if e in ret_mapb.keys()
#                                                                             else np.nan for e in x.split(', ')]))
#                     g['expiry_return'] = g['tickers'].apply(lambda x: np.mean([ret_mape[e] if e in ret_mape.keys()
#                                                                                else np.nan for e in x.split(', ')]))
#                     # g['bot_type'] = bot_type
#                     df_new.append(g)
#                     print(df_new)
#
#         df_new = pd.concat(df_new, axis=0)
#         df_new_agg = df_new.groupby(['currency_code', 'weeks_to_expire', 'n_top_config', 'n_backtest_period'])[
#             'bot_return'].agg(['min', 'max', 'mean', 'std', 'count']).reset_index()
#         df_new_agg['sharpe'] = df_new_agg['mean'] / df_new_agg['std']
#
#         num_col = ['min', 'max', 'mean', 'std', 'count', 'sharpe']
#         to_excel({
#             "all": df_new,
#             "agg": df_new_agg,
#             "n_backtest_period": df_new_agg.groupby(['weeks_to_expire', 'n_backtest_period'])[
#                 num_col].mean().reset_index(),
#             "n_backtest_period_mean":
#                 df_new_agg.groupby(['weeks_to_expire', 'currency_code', 'n_backtest_period'
#                                     ])['mean'].mean().unstack().reset_index(),
#             "n_top_config": df_new_agg.groupby(['weeks_to_expire', 'n_top_config'])[num_col].mean().reset_index(),
#             "n_top_config_mean":
#                 df_new_agg.groupby(['weeks_to_expire', 'currency_code', 'n_top_config'
#                                     ])['mean'].mean().unstack().reset_index(),
#         },
#             'top2_new_return_df_agg_-7d')
#
#         # x = df_new_agg.groupby(['weeks_to_expire', 'currency_code', 'n_backtest_period']).mean().unstack()
#         # print(df_new_agg.groupby(['weeks_to_expire']).mean())


if __name__ == '__main__':
    # get_index_return_annual()

    # 1. using original y return (e.g. stock_return_y_w4_d7)
    # top2_table_agg()

    # 2. using newly calculated return
    top2_table_tickers_return()

    # 3. using bot table returns
    # top2_table_bot_return()

    # read_v2_top_excel()
    # read_v2_actual_eval()
    # data = daily_trans_return()
    # data.holding(weeks_to_expire=4)
    # data.period_average(dt.datetime(2021, 8, 25, 0, 0, 0))
