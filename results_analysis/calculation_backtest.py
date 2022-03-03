import logging
from scipy.stats import skew
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import robust_scale, minmax_scale
import matplotlib.pyplot as plt

import global_vars
from general.sql_process import read_query, read_table, upsert_data_to_database
from general.send_slack import to_slack
from general.utils import to_excel
from results_analysis.calculation_rank import rank_pred

universe_currency_code = ['HKD', 'CNY', 'USD', 'EUR']
score_col = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra", "ai_score"]

from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION, INTEGER

# table dtype for top backtest
top_dtypes = {
    "currency_code": TEXT,
    "trading_day": DATE,
    "weeks_to_expire": INTEGER,
    "mode": TEXT,
    "mode_count": INTEGER,
    "positive_pct": DOUBLE_PRECISION,
    "return": DOUBLE_PRECISION,
    "tickers": TEXT,
    "name_sql": TEXT,
    "start_date": DATE,
    "n_period": INTEGER,
    "n_top": INTEGER
}

class score_scale:

    def __init__(self, fundamentals, calculate_column, universe_currency_code, factor_formula, weeks_to_expire, factor_rank):
        fundamentals = fundamentals.copy()

        # Scale factor scores
        fundamentals, calculate_column_score = score_scale.__scale1_trim_outlier(fundamentals, calculate_column)
        fundamentals = score_scale.__scale2_reverse_neg_factor(factor_rank, fundamentals)
        fundamentals, calculate_column_robust_score = score_scale.__scale3_robust_scaler(fundamentals, calculate_column)
        fundamentals, minmax_column = score_scale.__scale4_minmax_scaler(fundamentals, calculate_column)

        # Calculate AI Scores
        fundamentals = score_scale.__calc_ai_score_pillar(factor_rank, fundamentals)
        fundamentals = score_scale.__calc_ai_score_extra(factor_rank, fundamentals)
        # fundamentals = score_scale.__calc_esg(fundamentals)
        fundamentals = score_scale.__calc_ai_score(fundamentals)
        # fundamentals = score_scale.__calc_ai_score2(fundamentals)

        self.fundamentals = fundamentals
        self.calculate_column = calculate_column


    @staticmethod
    def __scale1_trim_outlier(fundamentals, calculate_column):
        ''' Scale 1: log transformation for high skewness & trim outlier to +/- 2 std -> [.*_score] columns '''
        def transform_trim_outlier(x):
            s = skew(x)
            if (s < -5) or (s > 5):
                x = np.log(x + 1 - np.min(x))
            m = np.median(x)
            # clip_x = np.clip(x, np.percentile(x, 0.01), np.percentile(x, 0.99))
            std = np.nanstd(x)
            return np.clip(x, m - 2 * std, m + 2 * std)

        calculate_column_score = []
        for column in calculate_column:
            column_score = column + "_score"
            fundamentals[column_score] = fundamentals.dropna(subset=[column]).groupby("currency_code")[
                column].transform(
                transform_trim_outlier)
            calculate_column_score.append(column_score)
        # print(calculate_column_score)

        return fundamentals, calculate_column_score

    @staticmethod
    def __scale2_reverse_neg_factor(factor_rank, fundamentals):
        ''' reverse ".*_score" columns in each currency for not long_large '''
        for group, g in factor_rank.groupby(['group']):
            neg_factor = [x + '_score' for x in g.loc[(g['long_large'] == False), 'factor_name'].to_list()]
            print(group, neg_factor)
            fundamentals.loc[(fundamentals['currency_code'] == group), neg_factor] *= -1

        return fundamentals

    @staticmethod
    def __scale3_robust_scaler(fundamentals, calculate_column):
        ''' apply robust scaler on [.*_score] -> [.*_robust_score] columns '''
        calculate_column_robust_score = []
        for column in calculate_column:
            try:
                column_score = column + "_score"
                column_robust_score = column + "_robust_score"
                fundamentals[column_robust_score] = fundamentals.dropna(subset=[column_score]).groupby("currency_code")[
                    column_score].transform(lambda x: robust_scale(x))
                calculate_column_robust_score.append(column_robust_score)
            except Exception as e:
                print(e)
        # print(calculate_column_robust_score)
        return fundamentals, calculate_column_robust_score

    @staticmethod
    def __scale4_minmax_scaler(fundamentals, calculate_column):
        ''' apply maxmin scaler on Currency / Industry on [.*_robust_score] -> [.*_minmax_currency_code] columns '''

        minmax_column = []
        for column in calculate_column:
            column_robust_score = column + "_robust_score"
            column_minmax_currency_code = column + "_minmax_currency_code"
            df_currency_code = fundamentals[["currency_code", column_robust_score]]
            df_currency_code = df_currency_code.rename(columns={column_robust_score: "score"})
            fundamentals[column_minmax_currency_code] = df_currency_code.dropna(
                subset=["currency_code", "score"]).groupby(
                'currency_code').score.transform(
                lambda x: minmax_scale(x.astype(float)) if x.notnull().sum() else np.full_like(x, np.nan))
            fundamentals[column_minmax_currency_code] = np.where(fundamentals[column_minmax_currency_code].isnull(),
                                                                 fundamentals[column_minmax_currency_code].mean() * 0.9,
                                                                 fundamentals[column_minmax_currency_code]) * 10
            minmax_column.append(column_minmax_currency_code)

            if column in ["environment", "social", "governance"]:  # for ESG scores also do industry partition
                column_minmax_industry = column + "_minmax_industry"
                df_industry = fundamentals[["industry_code", column_robust_score]]
                df_industry = df_industry.rename(columns={column_robust_score: "score"})
                fundamentals[column_minmax_industry] = df_industry.dropna(subset=["industry_code", "score"]).groupby(
                    "industry_code").score.transform(
                    lambda x: minmax_scale(x.astype(float)) if x.notnull().sum() else np.full_like(x, np.nan))
                fundamentals[column_minmax_industry] = np.where(fundamentals[column_minmax_industry].isnull(),
                                                                fundamentals[column_minmax_industry].mean() * 0.9,
                                                                fundamentals[column_minmax_industry]) * 10
                minmax_column.append(column_minmax_industry)

        # print(minmax_column)
        return fundamentals, minmax_column

    @staticmethod
    def __calc_ai_score_pillar(factor_rank, fundamentals):
        ''' calculate ai_score by each currency_code (i.e. group) for each of [Quality, Value, Momentum] '''

        # add column for 3 pillar score
        fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan

        for (group, pillar_name), g in factor_rank.groupby(["group", "pillar"]):
            print(f"Calculate Fundamentals [{pillar_name}] in group [{group}]")
            sub_g = g.loc[(g["factor_weight"] == 2) | (g["factor_weight"].isnull())]  # use all rank=2 (best class)
            score_col = [f"{x}_{y}_currency_code" for x, y in sub_g.loc[sub_g["scaler"].notnull(), ["factor_name", "scaler"]].to_numpy()]
            score_col += [x for x in sub_g.loc[sub_g["scaler"].isnull(), "factor_name"]]
            fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_{pillar_name}"] = \
                fundamentals.loc[fundamentals["currency_code"] == group, score_col].mean(axis=1).values

            sub_g_neg = g.loc[(g["factor_weight"] == 0)]  # use all rank=0 (worst class)
            score_col_neg = [f"{x}_{y}_currency_code" for x, y in sub_g_neg.loc[sub_g_neg["scaler"].notnull(), ["factor_name", "scaler"]].to_numpy()]
            score_col_neg += [x for x in sub_g_neg.loc[sub_g_neg["scaler"].isnull(), "factor_name"]]
            fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_{pillar_name}"] -= \
                fundamentals.loc[fundamentals["currency_code"] == group, score_col_neg].mean(axis=1).values

        return fundamentals

    @staticmethod
    def __calc_ai_score_extra(factor_rank, fundamentals):
        ''' calculate ai_score by each currency_code (i.e. group) for [Extra] '''

        for group, g in factor_rank.groupby("group"):
            try:
                print(f"Calculate Fundamentals [extra] in group [{group}]")
                sub_g = g.loc[(g["factor_weight"] == 2) | (g["factor_weight"].isnull())]  # use all rank=2 (best class)
                sub_g = sub_g.loc[(g["pred_z"] >= 1) | (
                    g["pred_z"].isnull())]  # use all rank=2 (best class) and predicted factor premiums with z-value >= 1

                if len(sub_g.dropna(subset=["pred_z"])) > 0:  # if no factor rank=2, don"t add any factor into extra pillar
                    score_col = [f"{x}_{y}_currency_code" for x, y in
                                 sub_g.loc[sub_g["scaler"].notnull(), ["factor_name", "scaler"]].to_numpy()]
                    fundamentals.loc[fundamentals["currency_code"] == group, "fundamentals_extra"] = fundamentals[
                        score_col].mean(axis=1)
                else:
                    fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_extra"] = \
                        fundamentals.loc[fundamentals["currency_code"] == group].filter(
                            regex="^fundamentals_").mean().mean()
            except Exception as e:
                print(e)
        return fundamentals

    @staticmethod
    def __calc_esg(fundamentals):
        ''' Calculate ESG Value '''
        print('Calculate ESG Value')
        esg_cols = ["environment_minmax_currency_code", "environment_minmax_industry", "social_minmax_currency_code",
                    "social_minmax_industry", "governance_minmax_currency_code", "governance_minmax_industry"]
        fundamentals["esg"] = fundamentals[esg_cols].mean(1)
        return fundamentals

    @staticmethod
    def __calc_ai_score(fundamentals):
        ''' Calculate AI Score '''
        print('Calculate AI Score')
        ai_score_cols = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra"]
        fundamentals["ai_score"] = fundamentals[ai_score_cols].mean(1)
        return fundamentals

    @staticmethod
    def __calc_ai_score2(fundamentals):
        ''' Calculate AI Score 2 '''
        print('Calculate AI Score 2')
        ai_score_cols2 = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "esg"]
        fundamentals["ai_score2"] = fundamentals[ai_score_cols2].mean(1)
        return fundamentals

    def score_update_scale(self):
        ''' scale for each score '''

        global score_col
        fundamentals, calculate_column = self.fundamentals, self.calculate_column
        print(fundamentals[score_col].describe())
        fundamentals[score_col] = fundamentals[score_col].round(1)
        return fundamentals

def get_fundamentals_score(start_date):
    ''' get fundamental scores from ratio table '''

    try:
        fundamentals_score = pd.read_csv(f'cached_fundamental_score_{start_date}.csv')
        fundamentals_score["trading_day"] = pd.to_datetime(fundamentals_score["trading_day"])
    except:
        print("=== Get [Factor Processed Ratio] history ===")
        conditions = ["r.ticker not like '.%%'"]
        if start_date:
            conditions.append(f"trading_day>='{start_date}'")
        ratio_query = f"SELECT r.*, currency_code FROM {global_vars.processed_ratio_table} r " \
                      f"INNER JOIN (SELECT ticker, currency_code FROM universe) u ON r.ticker=u.ticker " \
                      f"WHERE {' AND '.join(conditions)}"
        fundamentals_score = read_query(ratio_query, global_vars.db_url_alibaba_prod)
        fundamentals_score = fundamentals_score.pivot(index=["ticker", "trading_day", "currency_code"],
                                                      columns=["field"], values="value").reset_index()
        fundamentals_score.to_csv(f'cached_fundamental_score_{start_date}.csv', index=False)

    fundamentals_score = fundamentals_score.loc[fundamentals_score['currency_code'].isin(universe_currency_code)]
    return fundamentals_score

def update_factor_rank(factor_rank, factor_formula):
    ''' update factor_rank for 1) currency using USD; 2) factor not in model '''

    # for currency not predicted by Factor Model -> Use factor of USD
    for i in set(universe_currency_code) - set(factor_rank['group'].unique()):
        replace_rank = factor_rank.loc[factor_rank['group'] == 'USD'].copy()
        replace_rank['group'] = i
        factor_rank = factor_rank.append(replace_rank, ignore_index=True)

    factor_rank = factor_rank.merge(factor_formula, left_on=['factor_name'], right_index=True, how='outer')
    factor_rank['long_large'] = factor_rank['long_large'].fillna(True)
    factor_rank = factor_rank.dropna(subset=['pillar'])

    # for non calculating currency_code -> we add same for each one
    append_df = factor_rank.loc[factor_rank['keep']]
    for i in set(universe_currency_code):
        append_df['group'] = i
        factor_rank = factor_rank.append(append_df, ignore_index=True)
    return factor_rank

def backtest_score_history(factor_rank, name_sql):
    '''  calculate score with DROID v2 method & evaluate, write to table [factor_result_rank_backtest_top]

    Parameters
    ----------
    factor_rank (pd.DataFrame):
        factor_rank calculate from calculation_rank.py
    '''

    # DataFrame for [factor_rank]
    factor_rank["trading_day"] = pd.to_datetime(factor_rank["trading_day"])
    factor_rank['trading_day'] = factor_rank['trading_day'].dt.tz_localize(None)
    start_date = factor_rank['trading_day'].min() - relativedelta(weeks=27)     # back by 27 weeks since our max pred = 26w
    print(start_date)

    # DataFrame for [fundamentals_score]
    fundamentals_score = get_fundamentals_score(start_date=start_date)
    print(fundamentals_score["trading_day"].min(), fundamentals_score["trading_day"].max())
    print(sorted(list(factor_rank["trading_day"].unique())))

    # DataFrame for [factor_formula]
    factor_formula = read_table(global_vars.formula_factors_table_prod, global_vars.db_url_alibaba_prod)
    factor_formula = factor_formula.set_index(['name'])
    calculate_column = list(factor_formula.loc[factor_formula['scaler'].notnull()].index)
    calculate_column = sorted(set(calculate_column) & set(fundamentals_score.columns))

    factor_rank = update_factor_rank(factor_rank, factor_formula)

    label_col = ['ticker', 'trading_day', 'currency_code'] + fundamentals_score.filter(regex='^stock_return_y_').columns.to_list()
    fundamentals = fundamentals_score[label_col+calculate_column]
    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan).copy()
    # print(fundamentals)

    fundamentals = fundamentals.dropna(subset=['trading_day'], how='any')

    # add column for 3 pillar score
    fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan

    # add industry_name to ticker
    industry_name = get_industry_name()
    fundamentals["industry_name"] = fundamentals["ticker"].map(industry_name)

    # calculate score
    eval_best_all10 = {}
    eval_best_all20 = {}

    for (trading_day, weeks_to_expire), factor_rank_period in factor_rank.groupby(by=['trading_day', 'weeks_to_expire']):
        weeks_to_expire = int(weeks_to_expire)
        score_date = trading_day + relativedelta(weeks=weeks_to_expire)
        g = fundamentals.loc[fundamentals['trading_day']==score_date].copy()

        # Scale original fundamental score
        print(trading_day, score_date)
        g_score = score_scale(g.copy(1), calculate_column, universe_currency_code,
                              factor_rank_period, weeks_to_expire, factor_rank_period).score_update_scale()

        # Evaluate 2: calculate return for top 10 score / mode industry
        eval_best_all10[(score_date, weeks_to_expire)] = eval_best(g_score, weeks_to_expire, best_n=10).copy()
        eval_best_all20[(score_date, weeks_to_expire)] = eval_best(g_score, weeks_to_expire, best_n=20).copy()

    print("=== Update top 10/20 to DB ===")
    write_topn_to_db(eval_best_all10, 10, name_sql)
    write_topn_to_db(eval_best_all20, 20, name_sql)

    return True

def write_topn_to_db(eval_dict=None, n=None, name_sql=None):
    ''' for each backtest eval : write top 10 ticker to DB '''

    concat different trading_day top n selection ticker results
    df = pd.DataFrame(eval_dict).stack(level=[-2, -1]).unstack(level=1)
    df = df.reset_index().rename(columns={"level_0": "currency_code",
                                          "level_1": "trading_day",
                                          "level_2": "weeks_to_expire",})
    df.to_csv('test.csv', index=False)

    df = pd.read_csv('test.csv')
    df['trading_day'] = pd.to_datetime(df['trading_day'])

    # change data presentation for DB reading
    df['return'] = (pd.to_numeric(df['return']) * 100).round(2).astype(float)
    df['positive_pct'] = np.where(df['return'].isnull(), None, (df['positive_pct']*100).astype(float).round(0).astype(int))
    df["weeks_to_expire"] = df["weeks_to_expire"].astype(int)
    df['name_sql'] = name_sql
    df['start_date'] = df['trading_day'].min()
    df['n_period'] = len(df['trading_day'].unique())
    df['n_top'] = n
    df["trading_day"] = df["trading_day"].dt.date

    # write to DB
    upsert_data_to_database(df, global_vars.production_factor_rank_backtest_top_table,
                            primary_key=["name_sql", "start_date", "n_period", "n_top", "currency_code", "trading_day"],
                            how='update', db_url=global_vars.db_url_alibaba_prod,
                            dtype=top_dtypes)


def eval_best(fundamentals, weeks_to_expire, best_n=10, best_col='ai_score'):
    ''' evaluate score history with top 10 score return & industry '''

    top_ret = {}
    for name, g in fundamentals.groupby(["currency_code"]):
        g = g.set_index('ticker').nlargest(best_n, columns=[best_col], keep='all')
        top_ret[(name, "return")] = g[f'stock_return_y_w{weeks_to_expire}_d7'].mean()
        top_ret[(name, "mode")] = g[f"industry_name"].mode()[0]
        top_ret[(name, "mode_count")] = np.sum(g[f"industry_name"]==top_ret[(name, "mode")])
        top_ret[(name, "positive_pct")] = np.sum(g[f'stock_return_y_w{weeks_to_expire}_d7']>0)/len(g)
        top_ret[(name, "tickers")] = ', '.join(list(g.index))

    return top_ret

def get_industry_name():
    ''' get ticker -> industry name (4-digit) '''
    query = "SELECT ticker, name_4 FROM universe u INNER JOIN icb_code_explanation i ON u.industry_code=i.code_8"
    df= read_query(query, global_vars.db_url_alibaba_prod)
    return df.set_index(["ticker"])["name_4"].to_dict()

if __name__ == "__main__":
    pass