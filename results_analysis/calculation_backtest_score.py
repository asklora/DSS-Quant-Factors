from typing import Any

import logger
from scipy.stats import skew
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, MinMaxScaler, power_transform
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

import global_vars
from general.sql_process import read_query, read_table, upsert_data_to_database
from general.send_slack import to_slack
from general.utils import to_excel
from collections import Counter, OrderedDict


def get_fundamental_scores(start_date='2016-01-10', sample_interval=1):
    """ get fundamental scores from ratio table """

    # Download: DataFrame for [fundamentals_score]
    trading_day_list = pd.date_range(dt.datetime.strptime(start_date, '%Y-%m-%d'), dt.datetime.now(), freq=f'{sample_interval}w')
    trading_day_list = [x.strftime('%Y-%m-%d') for x in trading_day_list]
    try:
        fundamentals_score = pd.read_csv(f'cached_fundamental_score_{start_date}.csv')
        fundamentals_score["trading_day"] = pd.to_datetime(fundamentals_score["trading_day"])
    except:
        print("=== Get [Factor Processed Ratio] history ===")
        conditions = ["r.ticker not like '.%%'"]
        if start_date:
            conditions.append(f"trading_day in {tuple(trading_day_list)}")
        ratio_query = f"SELECT r.*, currency_code FROM {global_vars.processed_ratio_table} r " \
                      f"INNER JOIN (SELECT ticker, currency_code FROM universe) u ON r.ticker=u.ticker " \
                      f"WHERE {' AND '.join(conditions)}".replace(",)", ")")
        fundamentals_score = read_query(ratio_query, global_vars.db_url_alibaba_prod)
        fundamentals_score = fundamentals_score.pivot(index=["ticker", "trading_day", "currency_code"],
                                                      columns=["field"], values="value").reset_index()
        fundamentals_score.to_csv(f'cached_fundamental_score_{start_date}.csv', index=False)

    # Download: DataFrame for [factor_formula]
    factor_formula = read_table(global_vars.factors_formula_table, global_vars.db_url_alibaba_prod)
    factor_formula = factor_formula.set_index(['name'])
    calculate_column = list(factor_formula.loc[factor_formula['scaler'].notnull()].index)
    calculate_column = sorted(set(calculate_column) & set(fundamentals_score.columns))

    # filter [fundamentals_score] for calculation scores
    label_col = ['ticker', 'trading_day', 'currency_code'] + fundamentals_score.filter(
        regex='^stock_return_y_').columns.to_list()
    fundamentals = fundamentals_score[label_col + calculate_column]
    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan).copy()
    fundamentals = fundamentals.dropna(subset=['trading_day'], how='any')

    # add industry_name (4-digit) to ticker
    query = "SELECT ticker, name_4 FROM universe u INNER JOIN icb_code_explanation i ON u.industry_code=i.code_8"
    df = read_query(query, global_vars.db_url_alibaba_prod)
    industry_name = df.set_index(["ticker"])["name_4"].to_dict()
    fundamentals["industry_name"] = fundamentals["ticker"].map(industry_name)

    return fundamentals, factor_formula


def scale_fundamental_scores(fundamentals):
    """ calculate score for single currency / pillar """

    fundamentals = fundamentals.set_index(['ticker', 'industry_name'])
    return_col = fundamentals.filter(regex='^stock_return_y_').columns.to_list()

    # transform original score -> 0 - 10 scale
    pipe = Pipeline(steps=[('trim', TrimOutlierTransformer(skew_trh=5, std_trh=2)),
                           ('robust', RobustScaler()),
                           ('minmax', MinMaxScaler(feature_range=(0, 10)))])
    adj_fundamentals = 10 - pipe.fit_transform(fundamentals)    # we use 10 - minmax(0, 10) because factor model is S-L

    # fillna for scores with average for that pillar
    adj_fundamentals = pd.DataFrame(adj_fundamentals, index=fundamentals.index, columns=fundamentals.columns)
    adj_fundamentals = adj_fundamentals.fillna(adj_fundamentals.mean(axis=0))
    adj_fundamentals[return_col] = fundamentals[return_col]     # return cols has no adjustment

    return adj_fundamentals


class TrimOutlierTransformer(BaseEstimator, TransformerMixin):
    """ trim outliers """

    def __init__(self, skew_trh=5, std_trh=2):
        self.skew_trh = skew_trh
        self.std_trh = std_trh

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # des = pd.DataFrame(X).describe()

        s = skew(X, nan_policy='omit')
        X = np.where((s < -self.skew_trh) | (s > self.skew_trh), power_transform(X), X)
        # s1 = skew(X, nan_policy='omit')

        # des1 = pd.DataFrame(X).describe()
        X = np.apply_along_axis(
            lambda x: np.clip(x, np.nanmean(X, axis=0) - self.std_trh * np.nanstd(X, axis=0),
                                 np.nanmean(X, axis=0) + self.std_trh * np.nanstd(X, axis=0)), axis=1, arr=X)
        # des2 = pd.DataFrame(X).describe()
        return X