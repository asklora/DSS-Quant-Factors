from typing import Any

import logging
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
from results_analysis.calculation_rank import rank_pred
from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION, INTEGER, TIMESTAMP
from collections import Counter, OrderedDict

universe_currency_code = ['HKD', 'CNY', 'USD', 'EUR']
# universe_currency_code = ['CNY']


# table dtypes for top backtest
top_dtypes = {
    "n_top": INTEGER,
    "currency_code": TEXT,
    "trading_day": DATE,
    "mode": TEXT,
    "mode_count": INTEGER,
    "positive_pct": DOUBLE_PRECISION,
    "return": DOUBLE_PRECISION,
    "bm_positive_pct": DOUBLE_PRECISION,
    "bm_return": DOUBLE_PRECISION,
    "tickers": TEXT,
    "updated": TIMESTAMP,
}


def get_fundamental_scores(start_date='2016-01-10', sample_interval=4):
    """ get fundamental scores from ratio table """

    # Download: DataFrame for [fundamentals_score]
    trading_day_list = pd.date_range(dt.datetime.strptime(start_date, '%Y-%m-%d'), dt.datetime.now(), freq='4w')
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
    factor_formula = read_table(global_vars.formula_factors_table_prod, global_vars.db_url_alibaba_prod)
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

    # TODO: decide what to do with ESG (added in production)
    # TODO: decide how to deal with extra_col
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


def get_minmax_factors(name_sql,
                       group="HKD",
                       group_code="HKD",
                       weeks_to_expire=4,
                       eval_q=0.33,
                       is_removed_subpillar=True,
                       y_type='cluster',
                       eval_metric='net_ret',
                       n_backtest_period=None,
                       n_config_pct=None):
    """ return DataFrame with (trading_day, list of good / bad factor on trading_day)"""

    # Download: DataFrame for [factor_rank]
    conditions = [f"is_valid",
                  f"_group='{group}'",
                  f"_group_code='{group_code}'",
                  f"_name_sql='{name_sql}'",
                  f"_weeks_to_expire={weeks_to_expire}",
                  f"_q={eval_q}",
                  f"_is_removed_subpillar={is_removed_subpillar}"]
    if y_type != "cluster":
        conditions.append(f"_y_type='{y_type}'")
    factor_eval_query = f"SELECT * FROM {global_vars.production_factor_rank_backtest_eval_table} " \
                        f"WHERE {' AND '.join(conditions)} ORDER BY _testing_period"
    factor_eval = read_query(factor_eval_query)
    factor_eval['_y_type'] = y_type

    # testing_period = start date of test sets (i.e. data cutoff should be 1 period later / last return = 2p later)
    factor_eval["trading_day"] = pd.to_datetime(factor_eval["_testing_period"]) + pd.tseries.offsets.DateOffset(weeks=4)
    factor_eval['net_ret'] = factor_eval['max_ret'] - factor_eval['min_ret']
    factor_eval['avg_ret'] = (factor_eval['max_ret'] + factor_eval['net_ret']) / 2

    # get config cols
    config_col = factor_eval.filter(regex='^__').columns.to_list()
    n_config = factor_eval[config_col].drop_duplicates().shape[0]

    # in case of cluster pillar combine different cluster selection
    factor_eval_agg = factor_eval.groupby(config_col + ['trading_day']).agg(
        {'max_factor': 'sum', 'min_factor': 'sum', 'max_ret': 'mean', 'net_ret': 'mean', 'avg_ret': 'mean'}).reset_index()

    # calculate different config rolling average return
    factor_eval_agg['rolling_ret'] = factor_eval_agg.groupby(config_col)[eval_metric].rolling(n_backtest_period, closed='left').mean().values

    # filter for configuration with rolling_ret (historical) > quantile
    factor_eval_agg['trh'] = factor_eval_agg.groupby('trading_day')['rolling_ret'].transform(
        lambda x: np.quantile(x, 1 - n_config_pct)).values
    factor_eval_agg_select = factor_eval_agg.loc[factor_eval_agg['rolling_ret'] >= factor_eval_agg['trh']]

    # count occurrence of factors each testing_period & keep factor with (e.g. > .5 occurrence in selected configs)
    if eval_metric == "max_ret":
        select_col = ['max_factor']
    else:
        select_col = ['max_factor', 'min_factor']   # i.e. net return
    period_agg = factor_eval_agg_select.groupby('trading_day')[select_col].sum()
    period_agg_filter = pd.DataFrame(index=period_agg.index, columns=select_col + [x + "_trh" for x in select_col])
    period_agg_counter = period_agg[select_col].applymap(lambda x: dict(Counter(x)))

    for col in select_col:
        min_occur_pct = 1
        while any(period_agg_filter[col + "_trh"].isnull()) and min_occur_pct > 0:
            min_occur = n_config * n_config_pct * min_occur_pct
            temp = period_agg[col].apply(lambda x: [k for k, v in dict(Counter(x)).items() if v >= min_occur])
            temp = pd.DataFrame(temp[temp.apply(lambda x: len(x) >= 2)])
            temp[col + "_trh"] = round(min_occur_pct, 2)
            period_agg_filter = period_agg_filter.fillna(temp)
            min_occur_pct -= 0.1
    period_agg_count = period_agg_filter[select_col].applymap(lambda x: len(x))

    # # TODO: rewrite add_factor_penalty -> [factor_rank] past period factor prediction
    # if self.add_factor_penalty:
    #     factor_rank['z'] = factor_rank['actual_z'] / factor_rank['pred_z']
    #     factor_rank['z_1'] = factor_rank.sort_values(['testing_period']).groupby(['group', 'factor_name'])['z'].shift(
    #         1).fillna(1)
    #     factor_rank['pred_z'] = factor_rank['pred_z'] * np.clip(factor_rank['z_1'], 0, 2)
    #     factor_rank['factor_weight_adj'] = factor_rank.groupby(by=['group', 'testing_period'])['pred_z'].transform(
    #         lambda x: pd.qcut(x, q=3, labels=False, duplicates='drop')).astype(int)
    #     factor_rank['factor_weight_org'] = factor_rank['factor_weight'].copy()
    #     conditions = [
    #         (factor_rank['factor_weight'] == factor_rank['factor_weight'].max())
    #         & (factor_rank['factor_weight_adj'] == factor_rank['factor_weight_adj'].max()),
    #         (factor_rank['factor_weight'] == 0) & (factor_rank['factor_weight_adj'] == 0)
    #     ]
    #     factor_rank['factor_weight'] = np.select(conditions, [2, 0], 1)
    #
    #     factor_rank.to_csv('factor_rank_adj.csv', index=False)

    return period_agg_filter


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


class backtest_score_history:
    """  calculate score with DROID v2 method & evaluate, write to table [factor_result_rank_backtest_top]

    Parameters
    ----------
    factor_rank (pd.DataFrame):
        factor_rank calculate from calculation_rank.py
    """

    weeks_to_expire = None
    add_factor_penalty = False
    xlsx_name = 'test'
    base = 5

    def __init__(self):

        # Download: DataFrame for [fundamentals_score] / [factor_formula]
        # fundamentals, factor_formula = get_fundamental_scores(start_date='2016-01-10', sample_interval=4)
        #
        # # fundamentals = fundamentals.loc[fundamentals['currency_code'] == 'HKD']     # TODO: remove after debug
        #
        # adj_fundamentals = fundamentals.groupby(['currency_code', 'trading_day']).apply(scale_fundamental_scores)
        # adj_fundamentals = adj_fundamentals.reset_index()
        #
        # print(fundamentals["trading_day"].min(), fundamentals["trading_day"].max())
        #
        # adj_fundamentals.to_pickle('cache_adj_fundamentals.pkl')
        adj_fundamentals = pd.read_pickle('cache_adj_fundamentals.pkl')

        # Download: DataFrame for list of selected factors
        models_key = ["name_sql", "group", "group_code", "weeks_to_expire", "eval_q", "is_removed_subpillar", "y_type",
                      "eval_metric", "n_backtest_period", "n_config_pct"]
        models_value = [
            ('w4_d-7_20220324031027_debug', "HKD", "HKD", 4, 0.33, True, "cluster", "net_ret", 36, 0.2),
            ('w4_d-7_20220324031027_debug', "CNY", "CNY", 4, 0.33, True, "cluster", "max_ret", 36, 0.2),
            ('w4_d-7_20220312222718_debug', "EUR", "USD", 4, 0.33, True, "momentum", "avg_ret", 36, 0.2),
            ('w4_d-7_20220312222718_debug', "EUR", "USD", 4, 0.33, True, "quality", "max_ret", 36, 0.2),
            ('w4_d-7_20220312222718_debug', "EUR", "USD", 4, 0.33, True, "value", "max_ret", 36, 0.2),
            ('w4_d-7_20220312222718_debug', "USD", "USD", 4, 0.33, True, "momentum", "avg_ret", 36, 0.2),
            ('w4_d-7_20220312222718_debug', "USD", "USD", 4, 0.33, True, "quality", "avg_ret", 36, 0.2),
            ('w4_d-7_20220312222718_debug', "USD", "USD", 4, 0.33, True, "value", "avg_ret", 36, 0.2)
        ]
        kwargs_df = pd.DataFrame(models_value, columns=models_key)
        kwargs_df["updated"] = dt.datetime.now()
        upsert_data_to_database(kwargs_df, global_vars.production_factor_rank_backtest_top_table + '_kwargs', how='append')

        score_df_list = []
        for values in models_value:
            kwargs = dict(zip(models_key, values))
            minmax_factors = get_minmax_factors(**kwargs)

            for trading_day, f in minmax_factors.to_dict(orient='index').items():
                g = adj_fundamentals.loc[(adj_fundamentals['trading_day'] == trading_day) &
                                         (adj_fundamentals['currency_code'] == kwargs["group"])].copy()
                g['return'] = g[f'stock_return_y_w{kwargs["weeks_to_expire"]}_d-7']
                if kwargs['eval_metric'] == "max_ret":
                    g[kwargs['y_type'] + '_score'] = self.base + g[f['max_factor']].mean(axis=1)
                else:
                    g[kwargs['y_type'] + '_score'] = self.base + g[f['max_factor']].mean(axis=1) - g[f['min_factor']].mean(axis=1)
                score_df_list.append(g)

        score_df = pd.concat(score_df_list, axis=0)
        score_df_comb = score_df.groupby(['currency_code', 'trading_day', 'ticker', 'industry_name']).mean().reset_index()
        score_df_comb['ai_score'] = score_df_comb.filter(regex='_score$').mean(axis=1)

        # Evaluate: calculate return for top 10 score / mode industry
        eval_best_all = {}  # calculate score
        n_top_ticker_list = [10, 20, 30, 50, -10, -50]
        for i in n_top_ticker_list:
            for (currency, trading_day), g_score in score_df_comb.groupby(['currency_code', 'trading_day']):
                try:
                    eval_best_all[(i, currency, trading_day)] = self.eval_best(g_score, best_n=i).copy()
                except Exception as e:
                    to_slack("clair").message_to_slack(
                        f" === ERROR in eval backtest ===: [best{i}, {currency}, {trading_day}] has no ai_score: {e}")

        print("=== Update top 10/20 to DB ===")
        df = self.write_topn_to_db(eval_best_all)

    def write_topn_to_db(self, eval_dict=None):
        """ for each backtest eval : write top 10 ticker to DB """

        # concat different trading_day top n selection ticker results
        df = pd.DataFrame(eval_dict).transpose().reset_index()
        df = df.rename(columns={"level_0": "top_n", "level_1": "currency_code", "level_2": "trading_day"})

        # keep this because ERROR: (psycopg2.ProgrammingError) can't adapt type 'numpy.int64'
        df.to_csv('test.csv', index=False)
        df = pd.read_csv('test.csv')
        df['trading_day'] = pd.to_datetime(df['trading_day'])

        # change data presentation for DB reading
        df['return'] = (pd.to_numeric(df['return']) * 100).round(2).astype(float)
        df['positive_pct'] = np.where(df['return'].isnull(), None,
                                      (df['positive_pct'] * 100).astype(float).round(0).astype(int))
        df['bm_return'] = (pd.to_numeric(df['bm_return']) * 100).round(2).astype(float)
        df['bm_positive_pct'] = np.where(df['bm_return'].isnull(), None,
                                        (df['bm_positive_pct'] * 100).astype(float).round(0).astype(int))
        df["trading_day"] = df["trading_day"].dt.date
        df["updated"] = dt.datetime.now()
        # df["add_factor_penalty"] = self.add_factor_penalty

        # write to DB
        upsert_data_to_database(df, global_vars.production_factor_rank_backtest_top_table + '_36',
                                primary_key=["currency_code", "trading_day", "top_n"],
                                how='append', db_url=global_vars.db_url_alibaba_prod,
                                dtype=top_dtypes)
        return df

    def eval_best(self, g_all, best_n=10, best_col='ai_score'):
        """ evaluate score history with top 10 score return & industry """

        top_ret = {}
        if best_n > 0:
            g = g_all.set_index('ticker').nlargest(best_n, columns=[best_col], keep='all')
        else:
            g = g_all.set_index('ticker').nsmallest(-best_n, columns=[best_col], keep='all')

        top_ret["ticker_count"] = g.shape[0]
        top_ret["return"] = g['return'].mean()
        top_ret["mode"] = g[f"industry_name"].mode()[0]
        top_ret["mode_count"] = np.sum(g[f"industry_name"] == top_ret["mode"])
        top_ret["positive_pct"] = np.sum(g['return'] > 0) / len(g)
        top_ret["bm_return"] = g_all['return'].mean()
        top_ret["bm_positive_pct"] = np.sum(g_all['return'] > 0) / len(g_all)
        top_ret["tickers"] = ', '.join(list(g.index))

        return top_ret


if __name__ == "__main__":
    backtest_score_history()
