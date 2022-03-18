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


def eval_sortino_ratio(name_sql='w4_d-7_20220312222718_debug'):
    """ calculate sortino / sharpe ratio for each of the factors
        -> result: if we use sortino ratio max_ret > net_ret
    """
    # tbl_name = global_vars.production_factor_rank_backtest_eval_table
    # df = read_query(f"SELECT * FROM {tbl_name} WHERE name_sql='{name_sql}'")
    # df.to_pickle('cache1.pkl')

    df = pd.read_pickle('cache1.pkl')
    df['testing_period'] = pd.to_datetime(df['testing_period'])

    df['net_ret'] = df['max_ret'] - df['min_ret']
    df['net_ret_ab'] = df['net_ret'] - df['actual_s']
    df['net_ret_ab2_d'] = np.square(np.clip(df['net_ret_ab'], np.inf, 0))
    df['net_ret_ab2'] = np.square(df['net_ret_ab'])

    df['max_ret_ab'] = df['max_ret'] - df['actual_s']
    df['max_ret_ab2_d'] = np.square(np.clip(df['max_ret_ab'], np.inf, 0))
    df['max_ret_ab2'] = np.square(df['max_ret_ab'])

    df = df.loc[df['testing_period'] > dt.datetime(2020, 1, 1)]
    df_std = df.groupby(['group', 'group_code', 'y_type'])[
        ['net_ret', 'net_ret_ab', 'max_ret', 'max_ret_ab', 'actual_s', 'actual']].std()

    df_agg = df.groupby(['group', 'group_code', 'y_type', 'q'])[
        ['net_ret', 'net_ret_ab2', 'max_ret', 'max_ret_ab2', 'net_ret_ab2_d', 'max_ret_ab2_d']].mean()
    df_agg['net_ret_sortino'] = df_agg['net_ret'].div(np.sqrt(df_agg['net_ret_ab2_d']))
    df_agg['max_ret_sortino'] = df_agg['max_ret'].div(np.sqrt(df_agg['max_ret_ab2_d']))
    df_agg['net_ret_sharpe'] = df_agg['net_ret'].div(np.sqrt(df_agg['net_ret_ab2']))
    df_agg['max_ret_sharpe'] = df_agg['max_ret'].div(np.sqrt(df_agg['max_ret_ab2']))
    print(df)

if __name__ == '__main__':
    # actual_good_prem()
    # eval3_factor_selection()
    eval_sortino_ratio()
