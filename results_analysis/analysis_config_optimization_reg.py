# simplfied configruation optimization model
# find "Rule" (e.g. when indx_ret > 5%) or combination (40% max + 60% net)

import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import gc

import global_vars
from general.send_slack import to_slack
from general.sql_process import read_query, read_table, trucncate_table_in_database, upsert_data_to_database
from general.utils import to_excel
from preprocess.load_data import download_clean_macros, download_index_return

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from lightgbm import cv, Dataset, train
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    LinearRegression,
    Lasso,
    ElasticNet,
    Ridge,
)
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, balanced_accuracy_score
from itertools import product, combinations, permutations
from functools import partial
from collections import Counter, ChainMap
import multiprocessing as mp
import gc

# def models(self):
#     self.model['rig_clf'] = RidgeClassifier(random_state=0).fit(X, y_cut)
#     self.model['lin_reg'] = LinearRegression().fit(X, y)
#     self.model['lasso'] = Lasso(random_state=0).fit(X, y)
#     self.model['ridge'] = Ridge(random_state=0).fit(X, y)
#     self.model['en'] = ElasticNet(random_state=0).fit(X, y)


def log_clf(X, y, y_cut):
    # for p in ['l2']:    # 'l1', 'l2', 'elasticnet'
    #     for t in range(8):
    model = LogisticRegression(penalty='l2', tol=.1, random_state=0, max_iter=1000, verbose=3).fit(X, y_cut)
    score = model.score(X, y_cut)
    print(f"score: ", score)
    # exit(10)
    print("coef: ", model.coef_)
    print("intercept:: ", model.intercept_)
    pred_prob = model.predict_proba(X_train)
    return pred_prob


def rdg_clf(X, y, y_cut):
    model = RidgeClassifier(random_state=0).fit(X, y_cut)
    score = model.score(X, y_cut)
    print("score: ", score)
    print("coef: ", model.coef_)
    print("intercept:: ", model.intercept_)
    pred = model.predict(X_train)
    return pred


def lin_reg(X, y, y_cut):

    new_row = int(len(X) // 4)    # 4 config for max_ret / is_usd

    X_config = np.reshape(X[:, -2:], (new_row, 4*2), order='F')
    y = np.reshape(y.values, (new_row, 4), order='F')
    X = X[:new_row, :-2]

    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    score = model.score(X, y)
    print("score: ", score)
    print("coef: ", model.coef_)
    print("intercept:: ", model.intercept_)

    y_cut_row = pd.DataFrame(y).apply(pd.qcut, q=3, labels=False, axis=1).values
    pred_cut_row = pd.DataFrame(pred).apply(pd.qcut, q=3, labels=False, axis=1).values
    score = accuracy_score(y_cut_row.flatten(), pred_cut_row.flatten())

    return y, pred


class load_date:

    def __init__(self, df, g, pillar):
        """ create DataFrame for x, y for all testing periods """

        # x += macros
        macros = self.download_macros(g)

        # x = all configurations
        # config_col_x = [x.strip('_') for x in df.filter(regex="^__|_q|testing_period").columns.to_list() if x != '__tree_type']
        df.columns = [x.strip('_') for x in df]
        # df = df.loc[df['pillar'] == pillar].copy()

        df['net_ret'] = df['max_ret'] - df['min_ret']
        df_pivot = df.groupby(['testing_period', 'group_code'])[['max_ret', 'net_ret']].mean().unstack()
        df_x = macros.merge(df_pivot, left_index=True, right_index=True, how='right')
        df_x.columns = ['-'.join(x) if len(x) == 2 else x for x in df_x]
        self.df_x = df_x

        pass

        # if g != 'USD':
        #     df_x['is_usd'] = (df_x['group_code'] == "USD").values

        # # add whether using [max_ret] column to x
        # df_x['max_ret'] = True
        # df_x_copy = df_x.copy()
        # df_x_copy['max_ret'] = False
        # self.df_x = pd.concat([df_x, df_x_copy], axis=0).reset_index(drop=True)

        # y = max_ret + net_ret
        # df_y_max = df['max_ret'].copy()
        # df_y_net = df['max_ret'] - df['min_ret']
        # self.df_y = pd.concat([df_y_max, df_y_net], axis=0).reset_index(drop=True)

    def download_macros(self, g):
        """ download macro data as input """
        # TODO: change to read from DB not cache
        # df_macros = download_clean_macros().set_index('trading_day')
        # df_macros.to_pickle('df_macros.pkl')
        df_macros = pd.read_pickle('df_macros.pkl')

        # df_index = download_index_return().set_index('trading_day')
        # df_index.to_pickle('df_index.pkl')
        df_index = pd.read_pickle('df_index.pkl')

        df = df_macros.merge(df_index, left_index=True, right_index=True)

        idx_map = {"CNY": ".CSI300", "HKD": ".HSI", "USD": ".SPX", "EUR": ".SXXGR"}
        index_col = ['stock_return_r12_7', 'stock_return_r1_0', 'stock_return_r6_2']

        df[f"cum_{idx_map[g]}_stock_return_r6_2"] = (1 + df[f"{idx_map[g]}_stock_return_r6_2"]) * \
                                                    (1 + df[f"{idx_map[g]}_stock_return_r1_0"]) - 1
        df[f"cum_{idx_map['USD']}_stock_return_r6_2"] = (1 + df[f"{idx_map['USD']}_stock_return_r6_2"]) * \
                                                        (1 + df[f"{idx_map['USD']}_stock_return_r1_0"]) - 1

        df[f"cum_{idx_map[g]}_stock_return_r12_7"] = (1 + df[f"{idx_map[g]}_stock_return_r12_7"]) * \
                                                     (1 + df[f"cum_{idx_map[g]}_stock_return_r6_2"]) - 1
        df[f"cum_{idx_map['USD']}_stock_return_r12_7"] = (1 + df[f"{idx_map['USD']}_stock_return_r12_7"]) * \
                                                         (1 + df[f"cum_{idx_map['USD']}_stock_return_r6_2"]) - 1

        g_cols = []
        for i in index_col:
            g_cols.append(f"{idx_map[g]}_{i}")
            if i != 'stock_return_r1_0':
                g_cols.append(f"cum_{idx_map[g]}_{i}")
            if g != "USD":
                g_cols.append(f"{idx_map['USD']}_{i}")
                if i != 'stock_return_r1_0':
                    g_cols.append(f"cum_{idx_map['USD']}_{i}")
                df[f'diff_{i}'] = df[f"{idx_map[g]}_{i}"] - df[f"{idx_map['USD']}_{i}"]
                g_cols.append(f'diff_{i}')

        vix_map = {"CNY": "VHSIVOL", "HKD": "VHSIVOL", "USD": "CBOEVIX", "EUR": "VSTOXXI"}
        g_cols.append(vix_map[g])

        gdp_map = {"CNY": ["CHGDP...C"], "HKD": ["CHGDP...C", "HKGDP...C"], "USD": ["USGDP...D"], "EUR": ["EMGDP...D"]}
        g_cols.extend(gdp_map[g])

        int_map = {"CNY": [],
                   "HKD": ["HKGBILL3"],
                   "USD": ["USGBILL3", "USINTER3"],
                   "EUR": ["EMIBOR3.", "EMINTER3"]}
        g_cols.extend(int_map[g])

        lead_map = {"CNY": ["CHCYLEADQ"], "HKD": ["CHCYLEADQ"], "USD": ["USCYLEADQ"], "EUR": []}
        g_cols.extend(lead_map[g])

        return df[g_cols]

    def get_train(self, qcut_q=3):
        """ split train / test sets """

        # [x]: split train / test
        X_train = self.df_x.drop(columns=['group_code', 'testing_period'])
        self.feature_names = X_train.columns.to_list()
        X_train = scale(X_train.values)

        # [y]: split train / test + qcut
        y_train = self.df_y
        y_train_cut, self.cut_bins = pd.qcut(y_train, q=qcut_q, retbins=True, labels=False)

        return X_train, y_train, y_train_cut


def trial_accuracy(*args):
    """ grid search """
    (x1, cutoff1, std1), (x2, cutoff2, std2), (x3, cutoff3, std3), ddf = args
    ddf1 = ddf.copy()

    print(x1, x2, x3)
    score_list = {}
    for i in ['>', '<', '~']:
        if i == '>':
            ddf[f'use_usd'] = ddf[x1] > cutoff1
            ddf1[f'use_net'] = ddf1[x1] > cutoff1
            ddf[f'use_net'] = np.where(ddf[f'use_usd'], ddf[x2] > cutoff2, ddf[x3] > cutoff3)
            ddf1[f'use_usd'] = np.where(ddf1[f'use_net'], ddf1[x2] > cutoff2, ddf1[x3] > cutoff3)
        elif i == '<':
            ddf[f'use_usd'] = ddf[x1] < cutoff1
            ddf1[f'use_net'] = ddf1[x1] < cutoff1
            ddf[f'use_net'] = np.where(ddf[f'use_usd'], ddf[x2] < cutoff2, ddf[x3] < cutoff3)
            ddf1[f'use_usd'] = np.where(ddf1[f'use_net'], ddf1[x2] < cutoff2, ddf1[x3] < cutoff3)
        else:
            ddf[f'use_usd'] = (ddf[x1] > (cutoff1 - std1)) & (ddf[x1] < (cutoff1 + std1))
            ddf1[f'use_net'] = (ddf[x1] > (cutoff1 - std1)) & (ddf[x1] < (cutoff1 + std1))
            ddf[f'use_net'] = np.where(ddf[f'use_usd'], (ddf[x2] > (cutoff2 - std2)) & (ddf[x2] < (cutoff2 + std2)),
                                       (ddf[x3] > (cutoff3 - std3)) & (ddf[x3] < (cutoff3 + std3)))
            ddf1[f'use_usd'] = np.where(ddf1[f'use_net'], (ddf[x2] > (cutoff2 - std2)) & (ddf[x2] < (cutoff2 + std2)),
                                        (ddf[x3] > (cutoff3 - std3)) & (ddf[x3] < (cutoff3 + std3)))

        ddf['Selection'] = ddf['use_net'] * 2 + ddf['use_usd']
        ddf1['Selection'] = ddf1['use_net'] * 2 + ddf1['use_usd']

        score = accuracy_score(ddf['Best'], ddf['Selection'])
        score1 = accuracy_score(ddf1['Best'], ddf1['Selection'])

        score_list[(f'usd{i}', x1, cutoff1, x2, cutoff2, x3, cutoff3)] = score
        score_list[(f'net{i}', x1, cutoff1, x2, cutoff2, x3, cutoff3)] = score1

    gc.collect()

    return score_list


def trial_accuracy_usd(*args):
    """ grid search """
    (x1, cutoff1, std1), ddf = args

    score_list = {}
    for i in ['>', '<', '~']:
        if i == '>':
            ddf[f'use_net'] = ddf[x1] > cutoff1
        elif i == '<':
            ddf[f'use_net'] = ddf[x1] < cutoff1
        else:
            ddf[f'use_net'] = (ddf[x1] > (cutoff1 - std1)) & (ddf[x1] < (cutoff1 + std1))

        score = accuracy_score(ddf['net_better'], ddf['use_net'])
        score_list[(f'net{i}', x1, cutoff1)] = score

    gc.collect()

    return score_list


if __name__ == "__main__":

    # --------------------------------- Parser --------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='multiclass')
    parser.add_argument('--name_sql', default=None)
    parser.add_argument('--qcut_q', type=int, default=3)
    args = parser.parse_args()

    # --------------------------------- Load Data -----------------------------------------------

    tbl_name = global_vars.production_factor_rank_backtest_eval_table
    if type(args.name_sql) == type(None):
        query = f"SELECT * FROM {tbl_name}"
        pkl_name = f'cache_{tbl_name}.pkl'
    else:
        query = f"SELECT * FROM {tbl_name} WHERE name_sql='{args.name_sql}'"
        pkl_name = f'cache_eval_{args.name_sql}.pkl'

    try:
        df = pd.read_pickle(pkl_name)
    except Exception as e:
        df = read_query(query)
        df.to_pickle(pkl_name)
    print(df)

    df = df.sort_values(by=['_testing_period'])
    # df = df.dropna(how='any')
    df['_testing_period'] = pd.to_datetime(df['_testing_period']).dt.normalize()
    config_col = df.filter(regex="^_").columns.to_list()

    # defined configurations
    # 1. HKD / CNY use clustered pillar
    df_cluster = df.loc[(df['_name_sql'].isin(['w4_d-7_20220324031027_debug', 'w4_d-7_20220329120327_debug'])) &
                        (df['_is_removed_subpillar']) &
                        (df['_q'].astype(float) == 0.33),
                        config_col + ['max_ret', 'min_ret']].fillna(0)
    df_cluster['_pillar'] = 'cluster'

    # 2. USD / EUR use clustered pillar
    df_pillar = df.loc[(~df['_name_sql'].isin(['w4_d-7_20220324031027_debug', 'w4_d-7_20220329120327_debug'])) &
                       (df['_q'].astype(float) == 0.33),
                       config_col + ['max_ret', 'min_ret']]
    df = df_cluster.append(df_pillar)
    del df_cluster, df_pillar

    df['_name_sql'] = df['_name_sql'].replace({'w4_d-7_20220324031027_debug': "adj_mse",
                                               'w4_d-7_20220329120327_debug': "mse",
                                               'w4_d-7_20220321173435_debug': "adj_mse",
                                               'w4_d-7_20220312222718_debug': "mse"})

    df = df.groupby(['_name_sql', '_group', '_group_code', '_testing_period', '_pillar']).mean().reset_index()
    df.to_pickle('mean-'+pkl_name)
    # df = pd.read_pickle('mean-'+pkl_name)

    # df_index = pd.read_pickle('df_index.pkl')
    # idx_map = {"CNY": ".CSI300_stock_return_r1_0", "HKD": ".HSI_stock_return_r1_0",
    #            "USD": ".SPX_stock_return_r1_0", "EUR": ".SXXGR_stock_return_r1_0"}
    # df_index = df_index.filter(regex='_stock_return_r1_0$').stack().reset_index()
    # df_index['level_1'] = df_index['level_1'].replace({v: k for k, v in idx_map.items()})
    # df_index['trading_day'] = pd.to_datetime(df_index['trading_day']) - pd.tseries.offsets.DateOffset(weeks=4)
    # df_index = df_index.set_index(['trading_day', 'level_1'])[0].rename('actual')

    df['_group_code'] = np.where(df['_group_code'] == df['_group'], 'own', 'usd')
    df_pivot = df.groupby(['_name_sql', '_testing_period', '_pillar', '_group', '_group_code'])[['max_ret', 'min_ret']].mean().unstack()
    df_pivot.columns = ['max_own', 'max_usd', 'min_own', 'min_usd']

    df_pivot['net_own'] = df_pivot['max_own'] - df_pivot['min_own']
    df_pivot['avg_own'] = df_pivot[['net_own', 'max_own']].mean(axis=1)
    df_pivot['net_usd'] = df_pivot['max_usd'] - df_pivot['min_usd']
    df_pivot['avg_usd'] = df_pivot[['net_usd', 'max_usd']].mean(axis=1)
    df_pivot['avg_max'] = df_pivot[['max_own', 'max_usd']].mean(axis=1)
    df_pivot['avg_net'] = df_pivot[['net_own', 'net_usd']].mean(axis=1)
    df_pivot.loc[df_pivot['net_usd'].isnull(), ['max_usd', 'net_usd', 'avg_usd', 'avg_max', 'avg_net']] = np.nan

    # for (x1, x2) in permutations(df_pivot, 2):
    #     for q in np.arange(0.2, 1.1, 0.2):
    #         df_pivot[f"{x1}_{x2}_{round(q, 1)}"] = df_pivot[x1] - q * df_pivot[x2]
    # to_excel(df_pivot.reset_index(), 'config_opt_combination_raw')

    # df_pivot['actual'] = df_pivot[['max_own', 'max_usd', 'min_own', 'min_usd']].mean(axis=1)
    # df_pivot = df_pivot.merge(df_index, left_on=['_testing_period', '_group'], right_index=True, how='left')
    # df_pivot['actual1'] = df_pivot.groupby(['_group', '_pillar'])['actual'].transform('mean')

    ret_cols =['max_own', 'max_usd', 'net_own', 'net_usd', 'avg_own', 'avg_usd', 'avg_max', 'avg_net']
    df_pivot = df_pivot.reset_index()
    df_pivot['best'] = df_pivot[ret_cols].idxmax(axis=1)
    df_pivot_count = df_pivot.groupby(['_name_sql', '_group', '_pillar'])['best'].apply(lambda x: dict(Counter(x)))

    def sortino(g, actual_col):
        """ sortino ratio with actual = (current period average) or (all time average) """
        if actual_col != 0:
            ab = np.subtract(g, g[[actual_col]])
        else:
            ab = g
        ab_down = np.clip(ab, -np.inf, 0)
        ab2_down = np.square(ab_down)
        ab2_down_std = np.sqrt(np.mean(ab2_down, axis=0))
        avg = np.mean(ab, axis=0)
        sortino = avg.div(ab2_down_std)
        return sortino

    def sharpe(g, actual_col):
        """ sharpe ratio with actual = (current period average) or (all time average) """
        if actual_col != 0:
            ab = np.subtract(g, g[[actual_col]])
        else:
            ab = g
        ab2 = np.square(ab)
        ab2_std = np.sqrt(np.mean(ab2, axis=0))
        avg = np.mean(ab, axis=0)
        sharpe = avg.div(ab2_std)
        return sharpe

    df_pivot_avg = df_pivot.groupby(['_name_sql', '_group', '_pillar'])[ret_cols].mean().stack().rename("mean")
    df_pivot_std = df_pivot.groupby(['_name_sql', '_group', '_pillar'])[ret_cols].std().stack().rename("std")
    df_pivot_min = df_pivot.groupby(['_name_sql', '_group', '_pillar'])[ret_cols].min().stack().rename("min")
    df_pivot_quantile = df_pivot.groupby(['_name_sql', '_group', '_pillar'])[ret_cols].quantile(q=0.05).stack().rename("q5")
    df_pivot_quantile1 = df_pivot.groupby(['_name_sql', '_group', '_pillar'])[ret_cols].quantile(q=0.1).stack().rename("q10")
    df_pivot_sort0 = df_pivot.groupby(['_name_sql', '_group', '_pillar']).apply(lambda x: sortino(x[ret_cols], actual_col=0)).stack().rename("sortino_0")
    # df_pivot_sort = df_pivot.groupby(['_name_sql', '_group', '_pillar']).apply(lambda x: sortino(x[ret_cols], actual_col='actual')).stack().rename("sortino")
    # df_pivot_sort1 = df_pivot.groupby(['_name_sql', '_group', '_pillar']).apply(lambda x: sortino(x[ret_cols], actual_col='actual1')).stack().rename("sortino_ts")
    df_pivot_sharpe0 = df_pivot.groupby(['_name_sql', '_group', '_pillar']).apply(lambda x: sharpe(x[ret_cols], actual_col=0)).stack().rename("sharpe_0")
    # df_pivot_sharpe = df_pivot.groupby(['_name_sql', '_group', '_pillar']).apply(lambda x: sharpe(x[ret_cols], actual_col='actual')).stack().rename("sharpe")
    # df_pivot_sharpe1 = df_pivot.groupby(['_name_sql', '_group', '_pillar']).apply(lambda x: sharpe(x[ret_cols], actual_col='actual1')).stack().rename("sharpe_ts")

    df_pivot_eval = pd.concat([df_pivot_avg, df_pivot_std, df_pivot_min, df_pivot_quantile, df_pivot_quantile1,
                               df_pivot_sort0, df_pivot_sharpe0], axis=1).reset_index()
    df_pivot_eval = df_pivot_eval.loc[df_pivot_eval['level_3'] != 'actual1']
    xlsx_dict = {'raw': df_pivot_eval.sort_values(by=['_group', '_pillar', 'sortino_0'], ascending=False).groupby(['_group', '_pillar']).head(20)}
    for i in ['mean', 'sortino_0', 'min']:
        xlsx_dict[i] = df_pivot_eval.groupby(['_group', '_pillar', 'level_3'])[i].mean().unstack().reset_index()
    xlsx_dict['count'] = df_pivot_count.unstack().reset_index()
    to_excel(xlsx_dict, 'config_opt_combination_top20')
    exit(200)


    # df_select1 = pd.read_excel('score_df_all3.xlsx', sheet_name='raw')
    # df_select1 = df_select1.loc[df_select1['Select by Level 0'] == "Y"]
    # print(df_select1)
    #
    # df_select2 = pd.read_excel('score_df_all3.xlsx', sheet_name='USD (>0.6)')
    # df_select2 = df_select2.loc[df_select2['Select by Level 0'] == "Y"]
    # print(df_select2)

    sql_result = vars(args).copy()  # data write to DB TABLE lightgbm_results
    # sql_result['name_sql2'] = input("config optimization model name_sql2: ")

    sql_result['class_weight'] = {i: 1 for i in range(args.qcut_q)}
    sql_result['class_weight'][0] = 2   # higher importance on really low iterations

    testing_period = np.sort(df['_testing_period'].unique())[-12:]
    print(df.dtypes)

    # df = df.loc[df['_group'] == "CNY"]  # TODO: debug CNY only

    final_df_dict = {}
    final_df_corr = {}
    score_df_list = []
    for (group, pillar), g in df.groupby(['_group', '_pillar']):
        # if group not in ['CNY']:
        #     continue

        print(group, pillar)
        sql_result['currency_code'] = group
        sql_result['pillar'] = pillar

        data = load_date(g, group, pillar)
        ddf = data.df_x.copy()

        # =================== grid search combination =======================

        if group == 'USD':
            y_cols = ddf.columns.to_list()[-2:]
            x_cols = ddf.columns.to_list()[:-2]
            y_cols_replace = dict(zip(y_cols, range(2)))
            ddf = ddf.rename(columns=y_cols_replace)
            ddf['net_better'] = ddf[1] > ddf[0]
            df_select = df_select2.loc[df_select2['pillar'] == pillar]
        else:
            y_cols = ddf.columns.to_list()[-4:]
            x_cols = ddf.columns.to_list()[:-4]
            y_cols_replace = dict(zip(y_cols, range(4)))
            ddf = ddf.rename(columns=y_cols_replace)
            ddf['Best'] = ddf.iloc[:, -4:].idxmax(axis=1)
            ddf['net_better'] = ddf[[2, 3]].sum(axis=1) > ddf[[0, 1]].sum(axis=1)
            ddf['usd_better'] = ddf[[1, 3]].sum(axis=1) > ddf[[0, 2]].sum(axis=1)
            df_select = df_select1.loc[(df_select1['pillar'] == pillar) & (df_select1['group'] == group)]

        # for y in y_cols:
        #     ddf[f'use_{y}'] = ddf[y] - ddf[[x for x in y_cols if x != y]].mean(axis=1)
        # ddf['use_usd'] = (ddf['max_ret-USD'] + ddf['n et_ret-USD'] - ddf[f'max_ret-{group}'] - ddf[f'net_ret-{group}']) / 2
        # ddf['use_max'] = (ddf['max_ret-USD'] - ddf['net_ret-USD'] + ddf[f'max_ret-{group}'] - ddf[f'net_ret-{group}']) / 2
        # ddf['average'] = (ddf['max_ret-USD'] + ddf['net_ret-USD'] + ddf[f'max_ret-{group}'] + ddf[f'net_ret-{group}']) / 4
        # final_df_corr[f"{group}_{pillar}"] = ddf.corr().iloc[:-11, -7:].reset_index()

        # =================== grid search =======================
    #     options = []
    #     for x in x_cols:
    #         std = np.round(np.std(ddf[x]), 2)
    #         for i in range(1, 5):
    #             cutoff = np.round(np.quantile(ddf[x], q=.2*i), 2)
    #             options.append([x, cutoff, std])
    #
    #     if group == 'USD':
    #         options_comb = product(options, [ddf])
    #         options_comb = [tuple(e) for e in options_comb]
    #         with mp.Pool(processes=12) as pool:
    #             results = pool.starmap(trial_accuracy_usd, options_comb)
    #         score_list = {k: v for element in results for k, v in element.items()}
    #         score_df = pd.DataFrame(score_list, index=['accuracy']).transpose()
    #         score_df = score_df[score_df['accuracy'] > 0.6]
    #         score_df = score_df.sort_values(by=['accuracy'], ascending=False).reset_index()
    #         score_df.columns = ['first', 'level_0', 'level_0_trh', 'accuracy']
    #     else:
    #         options_comb = product(options, options, options, [ddf])
    #         options_comb = [tuple(e) for e in options_comb]
    #         with mp.Pool(processes=12) as pool:
    #             results = pool.starmap(trial_accuracy, options_comb)
    #         score_list = {k: v for element in results for k, v in element.items()}
    #         score_df = pd.DataFrame(score_list, index=['accuracy']).transpose()
    #         score_df = score_df[score_df['accuracy'] > 0.4]
    #         score_df = score_df.sort_values(by=['accuracy'], ascending=False).reset_index()
    #         score_df.columns = ['first', 'level_0', 'level_0_trh', 'level_1+', 'level_1+_trh', 'level_1-', 'level_1-_trh', 'accuracy']
    #     score_df['pillar'] = pillar
    #     score_df['group'] = group
    #     print(score_df['accuracy'].max())
    #     score_df_list.append(score_df)
    #     gc.collect()
    #
    # score_df_all = pd.concat(score_df_list, axis=0)
    # score_df_all.to_csv(f'score_df_all6_CNY.csv', index=False)

        # =================== test assumption ====================

    #     ddf['use_net'] = ddf['diff_stock_return_r1_0'] > -0.04
    #     score_net = accuracy_score(ddf['net_better'], ddf['use_net'])
    #     print('net score: ', score_net)
    #
    #     ddf['use_usd'] = np.where(ddf['use_net'],
    #                               ddf['diff_stock_return_r6_2'] > 0.03,
    #                               ddf['.SPX_stock_return_r6_2'] > 0.06)
    #     score_usd = accuracy_score(ddf['usd_better'], ddf['use_usd'])
    #     print('usd score: ', score_usd)
    #
    #     ddf['Selection'] = ddf['use_net'] * 2 + ddf['use_usd']
    #     ddf['correct'] = ddf['Best'] == ddf['Selection']
    #     score = accuracy_score(ddf['Best'], ddf['Selection'])
    #     print('final score: ', score)
    #
    #     ddf['Selection Return'] = ddf.apply(lambda x: x[x['Selection']], axis=1)
    #     des = ddf.describe().transpose()
    #     exit(1)
    #
    #     final_df_dict[f"{group}_{pillar}"] = ddf.reset_index()
    #
    # # to_excel(final_df_dict, f'config_lin_reg_pivot3')
    # to_excel(final_df_corr, f'config_lin_reg_corr4')

    # final_df = pd.concat([data.df_x, true_df, pred_df], axis=1).sort_values(by=['testing_period'])
    # to_excel({"raw": final_df,
    #           "pivot": final_df.pivot(columns=["testing_period"], index=["max_ret", "is_usd"]).reset_index()},
    #          f'{group}_config_lin_reg')

    # input_df = data.df_x.copy()
    # input_df["max_ret"] = input_df["max_ret"].replace({True: "max", False: "net"})
    # input_df["is_usd"] = input_df["is_usd"].replace({True: "USD", False: "own"})
    #
    # X_train, y_train, y_train_cut = data.get_train(qcut_q=args.qcut_q)
    # true_df = pd.DataFrame(y_train, columns=["Return"])

    # # [CLF1] Logistic Regression
    # pred_prob = log_clf(X_train, y_train, y_train_cut)
    # pred_df = pd.DataFrame(pred_prob, columns=list(range(args.qcut_q)))
    # final_df = pd.concat([input_df, true_df, pred_df], axis=1).sort_values(by=['testing_period'])
    # final_df_agg = pd.pivot_table(final_df,
    #                               index=["testing_period", 'm_stock_return_r12_7', 'm_stock_return_r1_0',
    #                                      'm_stock_return_r6_2', 'm_vix'],
    #                               columns=["max_ret", "is_usd"],
    #                               values=['Return', 2],
    #                               aggfunc='mean').reset_index()
    # final_df_agg.columns = ['-'.join([str(e) for e in x if e != '']) for x in final_df_agg]
    # to_excel({"raw": final_df,
    #           "pivot": final_df_agg},
    #          f'{group}_config_log_clf')

    # # [CLF2] Ridge Classificatoin
    # pred_prob = rdg_clf(X_train, y_train, y_train_cut)
    # pred_df = pd.DataFrame(pred_prob, columns=['Pred'])
    # final_df = pd.concat([input_df, true_df, pred_df], axis=1).sort_values(by=['testing_period'])
    # final_df_agg = pd.pivot_table(final_df,
    #                               index=["testing_period", 'm_stock_return_r12_7', 'm_stock_return_r1_0',
    #                                      'm_stock_return_r6_2', 'm_vix'],
    #                               columns=["max_ret", "is_usd"],
    #                               values=['Return', 'Pred'],
    #                               aggfunc='mean').reset_index()
    # final_df_agg.columns = ['-'.join([str(e) for e in x if e != '']) for x in final_df_agg]
    # to_excel({"raw": final_df,
    #           "pivot": final_df_agg},
    #          f'{group}_config_rdg_clf')

    # Linear Regression

    # y, pred = lin_reg(X_train, y_train, y_train_cut)
    # true_df = pd.DataFrame(y, columns=["y-usd-max", "y-own-max", "y-usd-net", "y-own-net"]).reset_index(drop=True)
    # pred_df = pd.DataFrame(pred, columns=["y-usd-max", "y-own-max", "y-usd-net", "y-own-net"])
    # final_df = pd.concat([data.df_x.iloc[:new_row, :-2], true_df], axis=1).sort_values(by=['testing_period'])
    # final_df_dict[f"{group}_{pillar}"] = data.df_x.reset_index()


