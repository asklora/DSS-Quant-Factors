import logging

from scipy.stats import skew
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import robust_scale, minmax_scale

import global_vars
from general.sql_process import read_query, read_table
from general.send_slack import to_slack
from general.utils import to_excel
from results_analysis.calculation_rank import rank_pred

universe_currency_code = ['USD', 'EUR', 'HKD']
score_col = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra", "ai_score"]

class score_scale:

    def __init__(self, fundamentals, calculate_column, universe_currency_code, factor_formula, weeks_to_expire, factor_rank):

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
        print(calculate_column_score)

        return fundamentals, calculate_column_score

    @staticmethod
    def __scale2_reverse_neg_factor(factor_rank, fundamentals):
        ''' reverse ".*_score" columns in each currency for not long_large '''

        for group, g in factor_rank.groupby(['group']):
            neg_factor = [x + '_score' for x in g.loc[(g['long_large'] == False), 'factor_name'].to_list()]
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
        print(calculate_column_robust_score)
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

        print(minmax_column)
        return fundamentals, minmax_column

    @staticmethod
    def __calc_ai_score_pillar(factor_rank, fundamentals):
        ''' calculate ai_score by each currency_code (i.e. group) for each of [Quality, Value, Momentum] '''

        # add column for 3 pillar score
        fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan
        # fundamentals[['dlp_1m','wts_rating']] = fundamentals[['dlp_1m','wts_rating']]/10    # adjust dlp score to 0 ~ 1 (originally 0 ~ 10)

        for (group, pillar_name), g in factor_rank.groupby(["group", "pillar"]):
            print(f"Calculate Fundamentals [{pillar_name}] in group [{group}]")
            sub_g = g.loc[(g["factor_weight"] == 2) | (g["factor_weight"].isnull())]  # use all rank=2 (best class)
            if len(sub_g.dropna(
                    subset=["pred_z"])) == 0:  # if no factor rank=2, use the highest ranking one & DLPA/ai_value scores
                sub_g = g.loc[g.nlargest(1, columns=["pred_z"]).index.union(g.loc[g["factor_weight"].isnull()].index)]

            score_col = [f"{x}_{y}_currency_code" for x, y in
                         sub_g.loc[sub_g["scaler"].notnull(), ["factor_name", "scaler"]].to_numpy()]
            score_col += [x for x in sub_g.loc[sub_g["scaler"].isnull(), "factor_name"]]
            fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_{pillar_name}"] = fundamentals[
                score_col].mean(axis=1)
        return fundamentals

    @staticmethod
    def __calc_ai_score_extra(factor_rank, fundamentals):
        ''' calculate ai_score by each currency_code (i.e. group) for [Extra] '''

        for group, g in factor_rank.groupby("group"):
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

# def get_tri_ret(start_date):
#     ''' calculate weekly / monthly return based on TRI '''
#
#     # calculate weekly return from TRI
#     tri = read_query(f"SELECT ticker, trading_day, total_return_index FROM data_tri "
#                      f"WHERE trading_day >= '{start_date}' "
#                      f"AND ticker='AAPL.O'"
#                      , global_vars.db_url_alibaba_prod)
#     tri['trading_day'] = pd.to_datetime(tri['trading_day'])
#     tri = tri.pivot(index=["trading_day"], columns=["ticker"], values="total_return_index")
#     tri = tri.resample('D').sum().replace(0, np.nan)
#     tri = tri.rolling(7, min_periods=1).mean().ffill()
#     triw = (tri.shift(-7)/tri - 1).stack().dropna(how="all").reset_index().rename(columns={0: "ret_week"})
#     trim = (tri.shift(-28)/tri - 1).stack().dropna(how="all").reset_index().rename(columns={0: "ret_month"})
#
#     return triw.merge(trim, on=["ticker", "trading_day"], how="outer")

def test_score_history(currency_code='USD', start_date='2020-10-01', name_sql=None):
    '''  calculate score with DROID v2 method & evaluate

    Parameters
    ----------
    currency_code (Str):
        score in which currency will be calculated
    start_date :
        start date to calculate AI Score
    name_sql :
        If None, AI Score calculated based on current [production_factor_rank_backtest_table] Table;
        Else, calculate rank using best model in name_sql = 'name_sql' in factor_model.
    '''

    if name_sql:
        print(f"=== Calculate [Factor Rank] for name_sql:[{name_sql}] ===")
        factor_rank = rank_pred(1/3, name_sql=name_sql).write_backtest_rank_(upsert_how=False)
    else:
        print("=== Get [Factor Rank] history from Backtest Table ===")
        conditions = [f"True"]
        if currency_code:
            conditions.append(f"\"group\"='{currency_code}'")
        if start_date:
            conditions.append(f"trading_day>='{start_date}'")
        factor_rank = read_query(f"SELECT * FROM {global_vars.production_factor_rank_backtest_table} "    # TODO: change table name
                                 f"WHERE {' AND '.join(conditions)}", global_vars.db_url_alibaba_prod)
        print(factor_rank.dtypes)

    print("=== Get [Factor Processed Ratio] history ===")
    conditions = ["r.ticker not like '.%%'"]
    if currency_code:
        conditions.append(f"currency_code='{currency_code}'")
    if start_date:
        conditions.append(f"trading_day>='{start_date}'")
    ratio_query = f"SELECT r.*, currency_code FROM {global_vars.processed_ratio_table} r " \
                  f"INNER JOIN (SELECT ticker, currency_code FROM universe) u ON r.ticker=u.ticker " \
                  f"WHERE {' AND '.join(conditions)}"
    fundamentals_score = read_query(ratio_query, global_vars.db_url_alibaba_prod)
    fundamentals_score = fundamentals_score.pivot(index=["ticker", "trading_day", "currency_code"], columns=["field"],
                                                  values="value").reset_index()

    fundamentals_score.to_csv('cached_fundamental_score.csv', index=False)
    factor_rank.to_csv('cached_factor_rank.csv', index=False)
    # fundamentals_score = pd.read_csv('cached_fundamental_score.csv')
    # factor_rank = pd.read_csv('cached_factor_rank.csv')

    fundamentals_score["trading_day"] = pd.to_datetime(fundamentals_score["trading_day"])
    factor_rank["trading_day"] = pd.to_datetime(factor_rank["trading_day"])
    factor_rank['trading_day'] = factor_rank['trading_day'].dt.tz_localize(None).apply(lambda x: x+relativedelta(hours=8))

    factor_rank['trading_day'] = factor_rank['trading_day'].apply(lambda x: x-relativedelta(hours=8))

    # Test return calculation
    # fundamentals_score_ret = fundamentals_score[["ticker","trading_day","stock_return_y_1week", "stock_return_y_4week"]]
    # tri_ret = get_tri_ret(start_date)
    # fundamentals_score_ret = fundamentals_score_ret.merge(tri_ret, on=["ticker", "trading_day"], how="left")
    # fundamentals_score_ret["diff_1w"] = fundamentals_score_ret["stock_return_y_1week"] - fundamentals_score_ret["ret_week"]*4
    # fundamentals_score_ret["diff_1m"] = fundamentals_score_ret["stock_return_y_4week"] - fundamentals_score_ret["ret_month"]

    factor_formula = read_table(global_vars.formula_factors_table_prod, global_vars.db_url_alibaba_prod)
    factor_rank = factor_rank.sort_values(by='last_update').drop_duplicates(subset=['trading_day','factor_name','group'], keep='last')
    fundamentals_score['trading_day'] = pd.to_datetime(fundamentals_score['trading_day'])

    # for currency not predicted by Factor Model -> Use factor of USD
    for i in set(universe_currency_code) - set(factor_rank['group'].unique()):
        replace_rank = factor_rank.loc[factor_rank['group'] == 'USD'].copy()
        replace_rank['group'] = i
        factor_rank = factor_rank.append(replace_rank, ignore_index=True)

    factor_rank = factor_rank.merge(factor_formula.set_index('name'), left_on=['factor_name'], right_index=True, how='outer')
    factor_rank['long_large'] = factor_rank['long_large'].fillna(True)
    factor_rank = factor_rank.dropna(subset=['pillar'])

    # for non calculating currency_code -> we add same for each one
    append_df = factor_rank.loc[factor_rank['keep']]
    for i in set(universe_currency_code):
        append_df['group'] = i
        factor_rank = factor_rank.append(append_df, ignore_index=True)

    factor_formula = factor_formula.set_index(['name'])
    calculate_column = list(factor_formula.loc[factor_formula['scaler'].notnull()].index)
    calculate_column = sorted(set(calculate_column) & set(fundamentals_score.columns))

    label_col = ['ticker', 'trading_day', 'currency_code'] + fundamentals_score.filter(regex='^stock_return_y_').columns.to_list()
    fundamentals = fundamentals_score[label_col+calculate_column]
    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan).copy()
    print(fundamentals)

    fundamentals = fundamentals.dropna(subset=['trading_day'], how='any')

    # add column for 3 pillar score
    fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan

    # add industry_name to ticker
    industry_name = get_industry_name()
    fundamentals["industry_name"] = fundamentals["ticker"].map(industry_name)

    # calculate score
    fundamentals_all = []
    eval_qcut_all = {}
    eval_best_all = {}
    if name_sql:
        options = [(int(name_sql.split('_')[0][1:]), int(name_sql.split('_')[1][1:]))]
    else:
        options = [(1,1), (4,7), (26,7), (26,28)]
    for name, g in fundamentals.groupby(['trading_day']):
        for weeks_to_expire, average_days in options:
            print(name)
            factor_rank_period = factor_rank.loc[(factor_rank['trading_day']==(name-relativedelta(weeks=weeks_to_expire)))&
                                                 (factor_rank['weeks_to_expire']==weeks_to_expire)]
            if len(factor_rank_period)==0:
                continue

            # Scale original fundamental score
            fundamentals = score_scale(g, calculate_column, universe_currency_code,
                                          factor_rank_period, weeks_to_expire, factor_rank_period).score_update_scale()
            fundamentals_all.append(fundamentals)

            # Evaluate 1: calculate return on 10-qcut portfolios
            eval_qcut_all[(name, f"{weeks_to_expire}_{average_days}")] = eval_qcut(fundamentals, score_col, weeks_to_expire, average_days)

            # Evaluate 2: calculate return for top 10 score / mode industry
            eval_best_all[(name, f"{weeks_to_expire}_{average_days}")] = eval_best(fundamentals, weeks_to_expire, average_days)

    print("=== Reorganize mean return dictionary to DataFrames ===")
    # 1. DataFrame for 10 group qcut - Mean Returns
    eval_qcut_df = pd.DataFrame(eval_qcut_all).stack(level=[-2, -1])
    eval_qcut_df = pd.DataFrame(eval_qcut_df.to_list(), index=eval_qcut_df.index)
    eval_qcut_df = eval_qcut_df.reset_index().rename(columns={"level_0": "currency_code",
                                                              "level_1": "score",
                                                              "level_2": "trading_day",
                                                              "level_3": "weeks_to_expire",})

    # 2. DataFrame for 10 group qcut - Mean Returns
    eval_qcut_df_avg = eval_qcut_df.groupby(["currency_code", "score", "weeks_to_expire"]).mean().reset_index()

    # 3. DataFrame for Top 10 picks - Mean Returns / mode indsutry(count) / positive pct
    eval_best_df = pd.DataFrame(eval_best_all).stack(level=[-2, -1]).unstack(level=1)
    eval_best_df = eval_best_df.reset_index().rename(columns={"level_0": "currency_code",
                                                              "level_1": "trading_day",
                                                              "level_2": "weeks_to_expire",})

    to_excel({"10 Qcut (Avg Ret)": eval_qcut_df,
              "10 Qcut (Avg Ret-Agg)": eval_qcut_df_avg,
              "Top 10 Picks": eval_best_df}, file_name=(name_sql if name_sql else ""))

    # fundamentals = pd.concat(fundamentals_all, axis=0)

    # Evaluate
    # eval_qcut_col_specific(["USD"], best_10_tickers_all, mean_ret_detail_all)   # period gain/loss/factors
    # save_description_history(score_history)                                     # score distribution
    return True

def save_description_history(df):
    ''' write statistics for description '''
    df = df.groupby('currency_code')['ai_score'].agg(['min','mean', 'median', 'max', 'std','count'])
    to_slack("clair").df_to_slack("AI Score distribution (Backtest)", df)

def eval_qcut(fundamentals, score_col, weeks_to_expire, average_days):
    ''' evaluate score history with score 10-qcut mean ret (over entire history) '''

    mean_ret = {}
    fundamentals = fundamentals.reset_index(drop=True)
    for name, df in fundamentals.groupby(["currency_code"]):
        for col in score_col:
            df['qcut'] = pd.qcut(df[col].dropna(), q=10, labels=False, duplicates='drop')
            mean_ret[(name, col)] = df.dropna(subset=[col]).groupby(['qcut'])[
                f'stock_return_y_w{weeks_to_expire}_d{average_days}'].mean().to_dict()

    return mean_ret

def eval_best(fundamentals, weeks_to_expire, average_days):
    ''' evaluate score history with top 10 score return & industry '''

    top_ret = {}
    for name, g in fundamentals.groupby(["currency_code"]):
        g = g.set_index('ticker').nlargest(10, columns=['ai_score'], keep='all')
        top_ret[(name, "return")] = g[f'stock_return_y_w{weeks_to_expire}_d{average_days}'].mean()
        top_ret[(name, "mode")] = g[f"industry_name"].mode()[0]
        top_ret[(name, "mode count")] = np.sum(g[f"industry_name"]==top_ret[(name, "mode")])
        top_ret[(name, "positive_pct")] = np.sum(g[f'stock_return_y_w{weeks_to_expire}_d{average_days}']>0)/len(g)

    return top_ret

def get_industry_name():
    ''' get ticker -> industry name (4-digit) '''
    query = "SELECT ticker, name_4 FROM universe u INNER JOIN icb_code_explanation i ON u.industry_code=i.code_8"
    df= read_query(query, global_vars.db_url_alibaba_prod)
    return df.set_index(["ticker"])["name_4"].to_dict()

if __name__ == "__main__":
    # can select name_sql based on
    name_sql = 'w8_d14_20220127195432_debug'
    test_score_history(name_sql=name_sql)