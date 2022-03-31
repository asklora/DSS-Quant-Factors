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
from itertools import product
from functools import partial
from collections import Counter


# def models(self):
#     self.model['rig_clf'] = RidgeClassifier(random_state=0).fit(X, y_cut)
#     self.model['lin_reg'] = LinearRegression().fit(X, y)
#     self.model['lasso'] = Lasso(random_state=0).fit(X, y)
#     self.model['ridge'] = Ridge(random_state=0).fit(X, y)
#     self.model['en'] = ElasticNet(random_state=0).fit(X, y)


def log_clf(X, y, y_cut):
    model = LogisticRegression(random_state=0).fit(X, y_cut)
    score = model.score(X, y_cut)
    print("score: ", score)
    print("coef: ", model.coef_)
    print("intercept:: ", score.intercept_)
    pred_prob = model.predict_proba(X_train)
    return pred_prob


def lin_reg(X, y, y_cut):
    model = LogisticRegression(random_state=0).fit(X, y)
    score = model.score(X, y)
    print("score: ", score)
    print("coef: ", model.coef_)
    print("intercept:: ", score.intercept_)
    pred_prob = model.predict(X_train)
    return pred_prob


class load_date:

    def __init__(self, df, g):
        """ create DataFrame for x, y for all testing periods """

        # x += macros
        macros = self.download_macros(g)

        # x = all configurations
        # config_col_x = [x.strip('_') for x in df.filter(regex="^__|_q|testing_period").columns.to_list() if x != '__tree_type']
        df.columns = [x.strip('_') for x in df]
        df = df.groupby(['group', 'group_code', 'testing_period']).mean().reset_index()
        df_x = df[['group_code', 'testing_period']].merge(macros, left_on='testing_period', right_index=True, how='left')
        if g != 'USD':
            df_x['is_usd'] = (df_x['group_code'] == "USD").values

        # add whether using [max_ret] column to x
        df_x['max_ret'] = True
        df_x_copy = df_x.copy()
        df_x_copy['max_ret'] = False
        self.df_x = pd.concat([df_x, df_x_copy], axis=0).reset_index(drop=True)

        # y = max_ret + net_ret
        df_y_max = df['max_ret'].copy()
        df_y_net = df['max_ret'] - df['min_ret']
        self.df_y = pd.concat([df_y_max, df_y_net], axis=0).reset_index(drop=True)

    def download_macros(self, g):
        """ download macro data as input """
        # TODO: change to read from DB not cache
        df_macros = download_clean_macros().set_index('trading_day')
        df_macros.to_pickle('df_macros.pkl')
        # df_macros = pd.read_pickle('df_macros.pkl')

        # df_index = download_index_return().set_index('trading_day')
        # df_index.to_pickle('df_index.pkl')
        df_index = pd.read_pickle('df_index.pkl')

        df = df_macros.merge(df_index, left_index=True, right_index=True)

        idx_map = {"CNY": ".CSI300", "HKD": ".HSI", "USD": ".SPX", "EUR": ".SXXGR"}
        index_col = ['stock_return_r12_7', 'stock_return_r1_0', 'stock_return_r6_2']
        for i in index_col:
            df[f'm_{i}'] = df[f"{idx_map[g]}_{i}"] - df[f"{idx_map['USD']}_{i}"]

        vix_map = {"CNY": "VHSIVOL", "HKD": "VHSIVOL", "USD": "CBOEVIX", "EUR": "VSTOXXI"}
        df[f'm_vix'] = df[vix_map[g]]

        return df.filter(regex="^m_")

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


if __name__ == "__main__":

    # --------------------------------- Parser --------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='multiclass')
    parser.add_argument('--name_sql', default=None)
    parser.add_argument('--qcut_q', type=int, default=3)
    parser.add_argument('--nfold', type=int, default=5)
    parser.add_argument('--process', type=int, default=10)
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
    df = df.dropna(how='any')
    df['_testing_period'] = pd.to_datetime(df['_testing_period']).dt.normalize()
    config_col = df.filter(regex="^_").columns.to_list()

    # defined configurations
    # 1. HKD / CNY use clustered pillar
    df_na = df.loc[(df['_group'].isin(['HKD', 'CNY'])) & (df['_name_sql'] == 'w4_d-7_20220324031027_debug')]
    df_na = df_na.groupby([x for x in config_col if x != '_y_type'])[['max_ret', 'min_ret']].mean().reset_index()
    df_na['_y_type'] = 'cluster'

    # 2. USD / EUR use clustered pillar
    df_ws = df.loc[((df['_group'] == 'EUR') & (df['_name_sql'] == 'w4_d-7_20220321173435_debug')) |
                   ((df['_group'] == 'USD') & (df['_name_sql'] == 'w4_d-7_20220312222718_debug')),
                   config_col + ['max_ret', 'min_ret']]
    df = df_na.append(df_ws)
    del df_na, df_ws

    sql_result = vars(args).copy()  # data write to DB TABLE lightgbm_results
    # sql_result['name_sql2'] = input("config optimization model name_sql2: ")

    sql_result['class_weight'] = {i: 1 for i in range(args.qcut_q)}
    sql_result['class_weight'][0] = 2   # higher importance on really low iterations

    testing_period = np.sort(df['_testing_period'].unique())[-12:]
    print(df.dtypes)

    df = df.loc[df['_group'] == "CNY"]  # TODO: debug CNY only

    for (group, y_type), g in df.groupby(['_group', '_y_type']):
        print(group, y_type)
        sql_result['currency_code'] = group
        sql_result['y_type'] = y_type

        data = load_date(g, group)
        X_train, y_train, y_train_cut = data.get_train(qcut_q=args.qcut_q)
        true_df = pd.DataFrame(y_train, columns=["Return"])

        # Logistic Regression
        pred_prob = log_clf(X_train, y_train, y_train_cut)
        pred_df = pd.DataFrame(pred_prob, columns=list(range(args.qcut_q)))
        final_df = pd.concat([data.df_x, true_df, pred_df], axis=1).sort_values(by=['testing_period'])
        to_excel({"raw": final_df,
                  "pivot": pd.pivot(final_df, columns=["testing_period"], index=["max_ret", "is_usd"]).reset_index()},
                 f'{group}_config_log_clf')

        # Linear Regression
        pred_prob = lin_reg(X_train, y_train, y_train_cut)
        pred_df = pd.DataFrame(pred_prob, columns=["Prediction"])
        final_df = pd.concat([data.df_x, true_df, pred_df], axis=1).sort_values(by=['testing_period'])
        to_excel({"raw": final_df,
                  "pivot": final_df.pivot(columns=["testing_period"], index=["max_ret", "is_usd"]).reset_index()},
                 f'{group}_config_log_clf')
