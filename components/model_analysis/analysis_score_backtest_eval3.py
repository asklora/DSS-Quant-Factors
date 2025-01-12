import pandas as pd
import numpy as np
import datetime as dt
from general.utils import to_excel
from general.sql.sql_process import read_query
import global_vars

config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', '_factor_reverse', 'down_mkt_pct']


class eval3_factor_selection:

    n_top_config = 1
    weeks_to_expire = 4

    def __init__(self, df=None, name_sql="w4_d-7_20220310130330_debug"):

        # if type(df) == type(None):
        #     tbl_name = global_vars.backtest_eval_table
        #     df = read_query(f"SELECT * FROM {tbl_name} WHERE name_sql='{name_sql}'")
        #     df.to_pickle('cache.pkl')

        df = pd.read_pickle('../cache.pkl')
        df['testing_period'] = pd.to_datetime(df['testing_period'])

        total_period = len(df['testing_period'].unique())

        col = ['group', 'group_code', 'testing_period', 'pillar']
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
        factor_count = factor_count.groupby(['group', 'group_code', 'factor_name', 'pillar', 'year'])[['count_max', 'count_min']].mean().reset_index()
        factor_count = factor_count.replace(0, np.nan).dropna(subset=['count_max', 'count_min'], how='all')
        factor_count = factor_count.sort_values(by=['group', 'group_code', 'pillar', 'factor_name', 'year'])

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


def eval_sortino_ratio(name_sql=None,
                       groupby_col=['_currency_code', 'name_sql', '_eval_removed_subpillar', '_eval_q']):
    """ calculate sortino / sharpe ratio for each of the factors
        -> result: if we use sortino ratio max_ret > net_ret
    """

    tbl_name = global_vars.backtest_eval_table
    if type(name_sql) == type(None):
        query = f"SELECT * FROM {tbl_name}"
        pkl_name = f'cache_{tbl_name}.pkl'
        xls_name = f'sortino_ratio_{tbl_name}'
    else:
        query = f"SELECT * FROM {tbl_name} WHERE name_sql='{name_sql}'"
        pkl_name = f'cache_eval_{name_sql}.pkl'
        xls_name = f'sortino_ratio_{name_sql}'

    try:
        df = pd.read_pickle(pkl_name)
    except Exception as e:
        df = read_query(query)
        df.to_pickle(pkl_name)

    df['_eval_q'] = df['_eval_q'].astype(str)
    df['_testing_period'] = pd.to_datetime(df['_testing_period'])
    # df['_name_sql'] = df['_name_sql'].replace({'w4_d-7_20220321173435_debug': "pre-defined",
    #                                            'w4_d-7_20220324031027_debug': "cluster"})
    #
    # df.loc[(df['_name_sql'] == "cluster") &
    #        (df['_train_currency'] == "CNY") &
    #        (df['_train_currency_code'] == "CNY") &
    #        (df['_q'] == '0.33')].to_csv('tests.xlsx')

    col = df.select_dtypes(float).columns.to_list()
    # df_raw = df.groupby(['group', 'group_code', 'pillar', 'testing_period', 'q'])[['max_ret', 'net_ret', 'actual_s',
    #                                                                                'actual']].mean().unstack()
    # df_raw.to_csv(f'eval_raw_{name_sql}.csv')
    # exit(1)

    # df = df.loc[df['_pillar'] != "combine"]
    # df = df.loc[((df['_train_currency'] != 'EUR') & (df['_train_currency'] == df['_train_currency_code'])) |
    #             ((df['_train_currency'] == 'EUR') & (df['_train_currency_code'] == 'USD'))]
    # df['_train_currency'] = df['_train_currency'] + df['_train_currency_code']
    df = df.groupby(['_testing_period'] + groupby_col)[col].mean().reset_index()

    df['net_ret'] = df['max_ret'] - df['min_ret']
    df['net_ret_ab'] = df['net_ret'] - df['actual']
    # df['net_ret_ab'] = df['net_ret'] - 0

    df['net_ret_ab2_d'] = np.square(np.clip(df['net_ret_ab'], -np.inf, 0))
    df['net_ret_ab2'] = np.square(df['net_ret_ab'])

    df['max_ret_ab'] = df['max_ret'] - df['actual']
    # df['max_ret_ab'] = df['max_ret'] - 0

    df['max_ret_ab2_d'] = np.square(np.clip(df['max_ret_ab'], -np.inf, 0))
    df['max_ret_ab2'] = np.square(df['max_ret_ab'])

    xls = {}
    for d in ['2016-01-01', '2020-01-01', '2021-08-01']:    # '2016-01-01', '2020-01-01',
        df_p = df.loc[df['_testing_period'] > dt.datetime.strptime(d, '%Y-%m-%d')]

        df_agg = df_p.groupby(groupby_col)[
            ['net_ret', 'net_ret_ab2', 'max_ret', 'max_ret_ab2', 'net_ret_ab2_d', 'max_ret_ab2_d']].mean()
        df_agg['net_ret_sortino'] = df_agg['net_ret'].div(np.sqrt(df_agg['net_ret_ab2_d']))
        df_agg['max_ret_sortino'] = df_agg['max_ret'].div(np.sqrt(df_agg['max_ret_ab2_d']))
        df_agg['net_ret_sharpe'] = df_agg['net_ret'].div(np.sqrt(df_agg['net_ret_ab2']))
        df_agg['max_ret_sharpe'] = df_agg['max_ret'].div(np.sqrt(df_agg['max_ret_ab2']))

        df_std = df_p.groupby(groupby_col)[
            ['net_ret', 'net_ret_ab', 'max_ret', 'max_ret_ab', 'actual']].std()
        df_std.columns = ['std_' + x for x in df_std]

        df_std = df_p.groupby(groupby_col)[['net_ret', 'max_ret']].std()
        df_std.columns = ['std_' + x for x in df_std]

        # calculate min period ret
        df_min = df_p.groupby(groupby_col)[['net_ret', 'max_ret']].min()
        df_min.columns = ['min_' + x for x in df_min]

        df_quantile = df_p.groupby(groupby_col)[['net_ret', 'max_ret']].quantile(q=0.1)
        df_quantile.columns = ['q10_' + x for x in df_quantile]

        df_quantile2 = df_p.groupby(groupby_col)[['net_ret', 'max_ret']].quantile(q=0.05)
        df_quantile2.columns = ['q5_' + x for x in df_quantile2]

        xls[d] = pd.concat([df_agg, df_std, df_min, df_quantile2, df_quantile], axis=1).reset_index().drop(
            columns=['net_ret_ab2', 'max_ret_ab2', 'net_ret_ab2_d', 'max_ret_ab2_d'])

    to_excel(xls, xls_name)
    print(df)


def eval_sortino_ratio_top(name_sql='w8_d-7_20220419172420'):
    """ calculate sortino / sharpe ratio for each of the factors
        -> result: if we use sortino ratio max_ret > net_ret
    """
    tbl_name = global_vars.backtest_top_table + '_debug'
    df = read_query(f"SELECT * FROM {tbl_name}")

    # filter
    # df = df.loc[df['trading_day'] < dt.date(2022, 4, 3)]
    # df = df.loc[df['currency_code'].isin(['HKD', 'USD'])]
    # df = df.loc[df['top_n'].isin([-10, -50, 50, 10])]
    df = df.loc[df['_eval_top_backtest_period'] == 12]
    df = df.loc[df["name_sql"] == name_sql]

    w = int(name_sql.split('_')[0][1:])
    df_index = read_query(f"SELECT trading_day, ticker as currency_code, value as index_ret "
                          f"FROM factor_processed_ratio "
                          f"WHERE ticker in ('.SPX', '.HSI', '.SXXGR', '.CSI300') AND field='stock_return_y_w{w}_d-7'",
                          db_url=global_vars.db_url_alibaba_prod)
    df_index["currency_code"] = df_index["currency_code"].replace({".SPX": "USD",
                                                                   ".HSI": "HKD",
                                                                   ".SXXGR": "EUR",
                                                                   ".CSI300": "CNY"})
    # df['trading_day'] = pd.to_datetime(df['trading_day'])
    # df_index['trading_day'] = pd.to_datetime(df_index['trading_day'])

    df = df.merge(df_index, on=["currency_code", "trading_day"], how="left")
    df[['ret', 'bm_ret']] /= 100
    # df.to_pickle('cache2.pkl')
    # df = pd.read_pickle('cache2.pkl')

    # df['ret2_d'] = np.square(np.clip(df['ret'], np.inf, 0))
    # df['bm_ret2_d'] = np.square(np.clip(df['bm_ret'], np.inf, 0))
    # df['index_ret2_d'] = np.square(np.clip(df['index_ret'], np.inf, 0))

    df['ret2_d'] = np.square(df['ret'])
    df['bm_ret2_d'] = np.square(df['bm_ret'])
    df['index_ret2_d'] = np.square(df['index_ret'])

    xls = {"raw": df}
    groupby_col = ['currency_code', '_eval_top_metric', 'top_n']
    for d in ['2019-03-15', '2017-03-15']:
        df_p = df.loc[df['trading_day'] > dt.datetime.strptime(d, '%Y-%m-%d').date()]
        d = f"{df_p['trading_day'].min()} ({len(df_p['trading_day'].unique())})"

        # calculate avg ret
        df_agg = df_p.groupby(groupby_col)[
            ['ret', 'bm_ret', 'index_ret', 'ret2_d', 'bm_ret2_d', 'index_ret2_d']].mean()

        # calculate sortino / sharpe ratio
        # df_agg['sortino'] = df_agg['ret'].div(np.sqrt(df_agg['ret2_d']))
        # df_agg['sortino_bm'] = df_agg['bm_ret'].div(np.sqrt(df_agg['bm_ret2_d']))
        # df_agg['sortino_index'] = df_agg['index_ret'].div(np.sqrt(df_agg['index_ret2_d']))

        df_agg['sharpe'] = df_agg['ret'].div(np.sqrt(df_agg['ret2_d']))
        df_agg['sharpe_bm'] = df_agg['bm_ret'].div(np.sqrt(df_agg['bm_ret2_d']))
        df_agg['sharpe_index'] = df_agg['index_ret'].div(np.sqrt(df_agg['index_ret2_d']))

        # # calculate std
        # df_std = df_p.groupby(groupby_col)[['ret', 'bm_ret', 'diff']].std()
        # df_std.columns = ['std_' + x for x in df_std]

        # calculate min period ret
        df_min = df_p.groupby(groupby_col)[['ret', 'bm_ret', 'index_ret']].min()
        df_min.columns = ['min_' + x for x in df_min]

        # df_quantile = df_p.groupby(groupby_col)[['ret', 'bm_ret', 'diff']].quantile(q=0.1)
        # df_quantile.columns = ['q10_' + x for x in df_quantile]
        #
        # df_quantile2 = df_p.groupby(groupby_col)[['ret', 'bm_ret', 'diff']].quantile(q=0.05)
        # df_quantile2.columns = ['q5_' + x for x in df_quantile2]

        final_df = pd.concat([df_agg, df_min], axis=1).reset_index().drop(
            columns=['ret2_d', 'bm_ret2_d', 'index_ret2_d'])

        final_df['sort'] = final_df['top_n'].map({-10: 0, -50: 1, 50: 2, 10: 3})
        final_df = final_df.sort_values(by=["currency_code", "sort"]).drop(columns=["sort"])

        xls[d] = final_df

    to_excel(xls, f'sortino_ratio_top_12_sharpe_{name_sql}')
    print(df)

if __name__ == '__main__':
    # actual_good_prem()
    # eval3_factor_selection()
    # eval_sortino_ratio(name_sql='w8_d-7_20220419172420')
    # eval_sortino_ratio(name_sql='w26_d-7_20220420143927')

    eval_sortino_ratio_top(name_sql='w8_20220428152336_debug')
    eval_sortino_ratio_top(name_sql='w26_20220429094107_debug')
