import pandas as pd
import numpy as np
import datetime as dt
import os
from general.utils import to_excel
from general.sql_process import read_query
from functools import partial
import ast
from dateutil.relativedelta import relativedelta
import global_vars

config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', 'use_average', 'down_mkt_pct']


class eval3_factor_selection:

    n_top_config = 1
    weeks_to_expire = 4

    def __init__(self, df=None, name_sql="w4_d-7_20220310130330_debug"):

        # if type(df) == type(None):
        #     tbl_name = global_vars.production_factor_rank_backtest_eval_table
        #     df = read_query(f"SELECT * FROM {tbl_name} WHERE name_sql='{name_sql}'")
        #     df.to_pickle('cache.pkl')

        df = pd.read_pickle('cache.pkl')
        df['testing_period'] = pd.to_datetime(df['testing_period'])

        total_period = len(df['testing_period'].unique())

        col = ['group', 'group_code', 'testing_period', 'y_type']
        df = df.set_index(col)

        # count
        max_factor_count = pd.DataFrame(df['max_factor_count'].to_list(), index=df.index).fillna(0).stack().reset_index()
        max_factor_count = max_factor_count.rename(columns={'level_4': 'factor_name', 0: 'count_max'})
        max_factor_count = max_factor_count.groupby(col+['factor_name'])[['count_max']].mean()

        min_factor_count = pd.DataFrame(df['min_factor_count'].to_list(), index=df.index).fillna(0).stack().reset_index()
        min_factor_count = min_factor_count.rename(columns={'level_4': 'factor_name', 0: 'count_min'})
        min_factor_count = min_factor_count.groupby(col+['factor_name'])[['count_min']].mean()

        factor_count = max_factor_count.merge(min_factor_count, left_index=True, right_index=True).reset_index()

        factor_count['year'] = factor_count['testing_period'].dt.year
        factor_count = factor_count.groupby(['group', 'group_code', 'factor_name', 'y_type', 'year'])[['count_max', 'count_min']].mean().reset_index()
        factor_count = factor_count.replace(0, np.nan).dropna(subset=['count_max', 'count_min'], how='all')
        factor_count = factor_count.sort_values(by=['group', 'group_code', 'y_type', 'factor_name', 'year'])

        factor_count_h5 = factor_count.sort_values(by=['count_max'], ascending=False).groupby(['group', 'group_code', 'year']).head(5).sort_values(by=['group', 'group_code', 'year', 'count_max'], ascending=False)
        # factor_count_h10 = factor_count.sort_values(by=['count_max'], ascending=False).groupby(['group', 'group_code', 'year']).head(10).sort_values(by=['group', 'group_code', 'year', 'count_max'], ascending=False)

        # max factor
        max_factor_pred = pd.DataFrame(df['max_factor_pred'].to_list(), index=df.index).stack().reset_index()
        max_factor_pred = max_factor_pred.rename(columns={'level_4': 'factor_name', 0: 'pred'})
        max_factor_pred = max_factor_pred.groupby(col+['factor_name'])[['pred']].mean()

        max_factor_actual = pd.DataFrame(df['max_factor_actual'].to_list(), index=df.index).stack().reset_index()
        max_factor_actual = max_factor_actual.rename(columns={'level_4': 'factor_name', 0: 'actual'})
        max_factor_actual = max_factor_actual.groupby(col+['factor_name'])[['actual']].mean()

        max_factor = max_factor_pred.merge(max_factor_actual, left_index=True, right_index=True).reset_index()
        max_factor_mean = max_factor.groupby(['factor_name'])[['pred', 'actual']].mean()
        max_factor_count = max_factor.groupby(['factor_name'])['pred'].count() / total_period
        max_factor_mean = pd.concat([max_factor_mean, max_factor_count], axis=1).reset_index()

        # min factor
        min_factor_pred = pd.DataFrame(df['min_factor_pred'].to_list(), index=df.index).stack().reset_index()
        min_factor_pred = min_factor_pred.rename(columns={'level_4': 'factor_name', 0: 'pred'})
        min_factor_pred = min_factor_pred.groupby(col+['factor_name'])[['pred']].mean()

        min_factor_actual = pd.DataFrame(df['min_factor_actual'].to_list(), index=df.index).stack().reset_index()
        min_factor_actual = min_factor_actual.rename(columns={'level_4': 'factor_name', 0: 'actual'})
        min_factor_actual = min_factor_actual.groupby(col+['factor_name'])[['actual']].mean()

        min_factor = min_factor_pred.merge(min_factor_actual, left_index=True, right_index=True).reset_index()
        min_factor_mean = min_factor.groupby(['factor_name'])[['pred', 'actual']].mean()
        min_factor_count = min_factor.groupby(['factor_name'])['pred'].count() / total_period
        min_factor_mean = pd.concat([min_factor_mean, min_factor_count], axis=1).reset_index()

        actual = df.groupby('testing_period')['actual'].first()
        actual_s = df.groupby('testing_period')['actual_s'].first()

        print(df)

def actual_good_prem():
    premium = read_query(f"SELECT p.*, f.pillar FROM factor_processed_premium p "
                         f"INNER JOIN (SELECT name, pillar FROM factor_formula_ratios_prod) f ON p.field=f.name "
                         f"WHERE trading_day > '2016-01-01' and weeks_to_expire=4 and average_days=-7 ")
    print(premium)

    premium['trading_day'] = pd.to_datetime(premium['trading_day']) - pd.tseries.offsets.DateOffset(weeks=4)
    premium['year'] = premium['trading_day'].dt.year
    prem_std = premium.groupby(by=["group", "year", "field"])['value'].std().reset_index().sort_values(by=["group", "field", "year"])

    premium_l = premium.copy()
    premium_l['field'] += ' (L)'
    premium_l['value'] *= -1
    premium['field'] += ' (S)'
    premium = premium.append(premium_l)

    from calculation_rank import weight_qcut
    from functools import partial

    q = 1/3
    premium['weight'] = premium.groupby(by=["group", "trading_day"])['value'].transform(
        partial(weight_qcut, q_=[0., q, 1. - q, 1.]))
    premium['weight'] = np.where(premium['weight'] == 2, 1, 0)

    premium = premium.groupby(["group", "field", "pillar"])[['weight']].mean().reset_index()
    # premium_h5_count['value'] /= len(premium['trading_day'].unique())

    # premium_h5 = premium.sort_values('weight', ascending=False).groupby(["group", "trading_day"]).head(5).sort_values(
    #     by=['group', 'trading_day', 'value'], ascending=False)
    pass


def eval_sortino_ratio(name_sql='w4_d-7_20220324031027_debug'):
    """ calculate sortino / sharpe ratio for each of the factors
        -> result: if we use sortino ratio max_ret > net_ret
    """

    try:
        df = pd.read_pickle(f'cache_eval_{name_sql}.pkl')
    except Exception as e:
        tbl_name = global_vars.production_factor_rank_backtest_eval_table
        df = read_query(f"SELECT * FROM {tbl_name}")
        df.to_pickle(f'cache_eval_{name_sql}.pkl')

    df['testing_period'] = pd.to_datetime(df['testing_period'])
    col = df.select_dtypes(float).columns.to_list()
    # df_raw = df.groupby(['group', 'group_code', 'y_type', 'testing_period', 'q'])[['max_ret', 'net_ret', 'actual_s',
    #                                                                                'actual']].mean().unstack()
    # df_raw.to_csv(f'eval_raw_{name_sql}.csv')
    # exit(1)

    df = df.loc[df['y_type'] != "combine"]
    df = df.loc[((df['group'] != 'EUR') & (df['group'] == df['group_code'])) |
                ((df['group'] == 'EUR') & (df['group_code'] == 'USD'))]
    df['group'] = df['group'] + df['group_code']
    df = df.groupby(['group', 'testing_period'])[col].mean().reset_index()

    df['net_ret'] = df['max_ret'] - df['min_ret']
    df['net_ret_ab'] = df['net_ret'] - df['actual']
    df['net_ret_ab2_d'] = np.square(np.clip(df['net_ret_ab'], np.inf, 0))
    df['net_ret_ab2'] = np.square(df['net_ret_ab'])

    df['max_ret_ab'] = df['max_ret'] - df['actual']
    df['max_ret_ab2_d'] = np.square(np.clip(df['max_ret_ab'], np.inf, 0))
    df['max_ret_ab2'] = np.square(df['max_ret_ab'])

    xls = {}

    for d in ['2016-01-01', '2020-01-01', '2021-08-01']:
        df_p = df.loc[df['testing_period'] > dt.datetime.strptime(d, '%Y-%m-%d')]

        df_agg = df_p.groupby(['group', 'q'])[
            ['net_ret', 'net_ret_ab2', 'max_ret', 'max_ret_ab2', 'net_ret_ab2_d', 'max_ret_ab2_d']].mean()
        df_agg['net_ret_sortino'] = df_agg['net_ret'].div(np.sqrt(df_agg['net_ret_ab2_d']))
        df_agg['max_ret_sortino'] = df_agg['max_ret'].div(np.sqrt(df_agg['max_ret_ab2_d']))
        df_agg['net_ret_sharpe'] = df_agg['net_ret'].div(np.sqrt(df_agg['net_ret_ab2']))
        df_agg['max_ret_sharpe'] = df_agg['max_ret'].div(np.sqrt(df_agg['max_ret_ab2']))

        groupby_col = ['group', 'q']
        df_std = df_p.groupby(groupby_col)[
            ['net_ret', 'net_ret_ab', 'max_ret', 'max_ret_ab', 'actual']].std()
        df_std.columns = ['std_' + x for x in df_std]

        df_std = df_p.groupby(groupby_col)[['net_ret', 'max_ret']].std()
        df_std.columns = ['std_' + x for x in df_std]

        # calculate min period return
        df_min = df_p.groupby(groupby_col)[['net_ret', 'max_ret']].min()
        df_min.columns = ['min_' + x for x in df_min]

        df_quantile = df_p.groupby(groupby_col)[['net_ret', 'max_ret']].quantile(q=0.1)
        df_quantile.columns = ['q10_' + x for x in df_quantile]

        df_quantile2 = df_p.groupby(groupby_col)[['net_ret', 'max_ret']].quantile(q=0.05)
        df_quantile2.columns = ['q5_' + x for x in df_quantile2]

        xls[d] = pd.concat([df_agg, df_std, df_min, df_quantile2, df_quantile], axis=1).reset_index().drop(
            columns=['net_ret_ab2', 'max_ret_ab2', 'net_ret_ab2_d', 'max_ret_ab2_d'])

    to_excel(xls, f'sortino_ratio_{name_sql}_new')
    print(df)


def eval_sortino_ratio_top(name_sql='w4_d-7_20220312222718_debug'):
    """ calculate sortino / sharpe ratio for each of the factors
        -> result: if we use sortino ratio max_ret > net_ret
    """
    tbl_name = global_vars.production_factor_rank_backtest_top_table
    df = read_query(f"SELECT * FROM {tbl_name} WHERE name_sql='{name_sql}'")

    df[['return', 'bm_return']] /= 100
    # df.to_pickle('cache2.pkl')
    # df = pd.read_pickle('cache2.pkl')

    df['diff'] = df['return'] - df['bm_return']
    df['diff2_d'] = np.square(np.clip(df['diff'], np.inf, 0))
    df['diff2'] = np.square(df['diff'])
    df['return2_d'] = np.square(np.clip(df['return'], np.inf, 0))
    df['return2'] = np.square(df['return'])

    xls = {}
    groupby_col = ['currency_code', 'eval_metric', 'eval_q', 'n_top_config', 'n_top_ticker', 'n_backtest_period']
    for d in ['2016-01-01', '2020-01-01', '2021-08-01']:
        df_p = df.loc[df['trading_day'] > dt.datetime.strptime(d, '%Y-%m-%d').date()]

        # calculate avg return
        df_agg = df_p.groupby(groupby_col)[
            ['return', 'bm_return', 'diff', 'diff2_d', 'diff2', 'return2_d', 'return2']].mean()

        # calculate sortino / sharpe ratio
        df_agg['sortino_mkt'] = df_agg['diff'].div(np.sqrt(df_agg['diff2_d']))
        df_agg['sortino_0'] = df_agg['return'].div(np.sqrt(df_agg['return2_d']))
        df_agg['sharpe_mkt'] = df_agg['diff'].div(np.sqrt(df_agg['diff2']))
        df_agg['sharpe_0'] = df_agg['return'].div(np.sqrt(df_agg['return2']))

        # calculate std
        df_std = df_p.groupby(groupby_col)[['return', 'bm_return', 'diff']].std()
        df_std.columns = ['std_' + x for x in df_std]

        # calculate min period return
        df_min = df_p.groupby(groupby_col)[['return', 'bm_return', 'diff']].min()
        df_min.columns = ['min_' + x for x in df_min]

        df_quantile = df_p.groupby(groupby_col)[['return', 'bm_return', 'diff']].quantile(q=0.1)
        df_quantile.columns = ['q10_' + x for x in df_quantile]

        df_quantile2 = df_p.groupby(groupby_col)[['return', 'bm_return', 'diff']].quantile(q=0.05)
        df_quantile2.columns = ['q5_' + x for x in df_quantile2]

        xls[d] = pd.concat([df_agg, df_std, df_min, df_quantile, df_quantile2], axis=1).reset_index().drop(
            columns=['diff2_d', 'diff2', 'return2_d', 'return2'])

    to_excel(xls, f'sortino_ratio_{name_sql}_top')
    print(df)

if __name__ == '__main__':
    # actual_good_prem()
    # eval3_factor_selection()
    eval_sortino_ratio()
    # eval_sortino_ratio_top()
