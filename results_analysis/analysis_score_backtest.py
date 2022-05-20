from global_vars import logger, LOGGER_LEVEL

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
count_neg = 0
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
            global count_neg
            count_neg += 1
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

def test_score_history(currency_code=None, start_date='2015-01-01', name_sql=None, top_config=3, use_usd=None, start_year=2016):
    '''  calculate score with DROID v2 method & evaluate, write to table [factor_result_rank_backtest_top]

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

    # if name_sql:
    try:
        print(f"=== Get [Factor Rank] from local ===")
        factor_rank = pd.read_csv(f'cached_factor_rank_{name_sql}.csv')
    except:
        print(f"=== Calculate [Factor Rank] for name_sql:[{name_sql}] ===")
        factor_rank = rank_pred(1/3, name_sql=name_sql, top_config=top_config).write_backtest_rank_(upsert_how=False)
    # else:
    #     print("=== Get [Factor Rank] history from Backtest Table ===")
    #     conditions = [f"True"]
    #     if currency_code:
    #         conditions.append(f"\"group\ in {tuple(currency_code)}")
    #     if start_date:
    #         conditions.append(f"trading_day>='{start_date}'")
    #     factor_rank = read_query(f"SELECT * FROM {global_vars.production_factor_rank_backtest_table} "
    #                              f"WHERE {' AND '.join(conditions)}", global_vars.db_url_alibaba_prod)
    #     print(factor_rank.dtypes)
    #     factor_rank.to_csv('cached_factor_rank.csv', index=False)

    factor_rank["trading_day"] = pd.to_datetime(factor_rank["trading_day"])
    factor_rank['trading_day'] = factor_rank['trading_day'].dt.tz_localize(None)

    try:
        fundamentals_score = pd.read_csv('cached_fundamental_score.csv')
    except:
        print("=== Get [Factor Processed Ratio] history ===")
        conditions = ["r.ticker not like '.%%'"]
        if currency_code:
            conditions.append(f"currency_code in {tuple(currency_code)}")
        if start_date:
            conditions.append(f"trading_day>='{start_date}'")
        ratio_query = f"SELECT r.*, currency_code FROM {global_vars.processed_ratio_table} r " \
                      f"INNER JOIN (SELECT ticker, currency_code FROM universe) u ON r.ticker=u.ticker " \
                      f"WHERE {' AND '.join(conditions)}"
        fundamentals_score = read_query(ratio_query, global_vars.db_url_alibaba_prod)
        fundamentals_score = fundamentals_score.pivot(index=["ticker", "trading_day", "currency_code"], columns=["field"],
                                                      values="value").reset_index()
        fundamentals_score.to_csv('cached_fundamental_score.csv', index=False)

    fundamentals_score["trading_day"] = pd.to_datetime(fundamentals_score["trading_day"])
    print(fundamentals_score['trading_day'].min())
    fundamentals_score = fundamentals_score.loc[fundamentals_score['currency_code'].isin(universe_currency_code)]
    fundamentals_score = fundamentals_score.loc[fundamentals_score['trading_day']>=dt.datetime(start_year,1,1)]

    if type(use_usd)==type(None):       # by default -> only EUR use USD rate
        factor_rank = factor_rank.loc[factor_rank['group']!='EUR']
    elif use_usd:
        factor_rank = factor_rank.loc[factor_rank['group']=='USD']

    print(fundamentals_score["trading_day"].min(), fundamentals_score["trading_day"].max())
    print(sorted(list(factor_rank["trading_day"].unique())))

    factor_formula = read_table(global_vars.factors_formula_table, global_vars.db_url_alibaba_prod)
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
    # print(fundamentals)

    fundamentals = fundamentals.dropna(subset=['trading_day'], how='any')

    # add column for 3 pillar score
    fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan

    # add industry_name to ticker
    industry_name = get_industry_name()
    fundamentals["industry_name"] = fundamentals["ticker"].map(industry_name)

    # calculate score
    eval_qcut_all = {}
    eval_best_all10 = {}
    eval_best_all20 = {}
    average_days = int(name_sql.split('_')[1][1:])

    for (trading_day, weeks_to_expire), factor_rank_period in factor_rank.groupby(by=['trading_day', 'weeks_to_expire']):
        weeks_to_expire = int(weeks_to_expire)
        score_date = trading_day + relativedelta(weeks=weeks_to_expire)
        g = fundamentals.loc[fundamentals['trading_day']==score_date].copy()

        # Scale original fundamental score
        print(trading_day, score_date)
        g_score = score_scale(g.copy(1), calculate_column, universe_currency_code,
                              factor_rank_period, weeks_to_expire, factor_rank_period).score_update_scale()

        # Evaluate 1: calculate return on 10-qcut portfolios
        eval_qcut_all[(score_date, f"{weeks_to_expire}_{average_days}")] = eval_qcut(g_score, score_col, weeks_to_expire, average_days).copy()

        # Evaluate 2: calculate return for top 10 score / mode industry
        eval_best_all10[(score_date, f"{weeks_to_expire}_{average_days}")] = eval_best(g_score, weeks_to_expire, average_days, best_n=10).copy()

        eval_best_all20[(score_date, f"{weeks_to_expire}_{average_days}")] = eval_best(g_score, weeks_to_expire, average_days, best_n=20).copy()

    print("=== Reorganize mean return dictionary to DataFrames ===")
    # 1. DataFrame for 10 group qcut - Mean Returns
    eval_qcut_df = pd.DataFrame(eval_qcut_all)
    eval_qcut_df = eval_qcut_df.stack(level=[-2, -1])
    eval_qcut_df = pd.DataFrame(eval_qcut_df.to_list(), index=eval_qcut_df.index)
    eval_qcut_df = eval_qcut_df.reset_index().rename(columns={"level_0": "currency_code",
                                                              "level_1": "score",
                                                              "level_2": "trading_day",
                                                              "level_3": "weeks_to_expire",})

    # 2. DataFrame for 10 group qcut - Mean Returns
    eval_qcut_df_avg = eval_qcut_df.groupby(["currency_code", "score", "weeks_to_expire"]).mean().reset_index()

    # 3.1. DataFrame for Top 10 picks - Mean Returns / mode indsutry(count) / positive pct
    eval_best_df10 = pd.DataFrame(eval_best_all10).stack(level=[-2, -1]).unstack(level=1)
    eval_best_df10 = eval_best_df10.reset_index().rename(columns={"level_0": "currency_code",
                                                              "level_1": "trading_day",
                                                              "level_2": "weeks_to_expire",})
    eval_best_df10['return'] = pd.to_numeric(eval_best_df10['return'])
    eval_best_df10['positive_pct'] = np.where(eval_best_df10['return'].isnull(), np.nan, eval_best_df10['positive_pct'])
    plot_topn_vs_mkt(eval_best_df10, weeks_to_expire, fig_name=f'{name_sql}_top10')
    eval_best_df10_agg = eval_best_df10.groupby(['currency_code'])[['positive_pct', 'return']].mean().reset_index()

    # 3.2. DataFrame for Top 20 picks - Mean Returns / mode indsutry(count) / positive pct
    eval_best_df20 = pd.DataFrame(eval_best_all20).stack(level=[-2, -1]).unstack(level=1)
    eval_best_df20 = eval_best_df20.reset_index().rename(columns={"level_0": "currency_code",
                                                              "level_1": "trading_day",
                                                              "level_2": "weeks_to_expire",})
    eval_best_df20['return'] = pd.to_numeric(eval_best_df20['return'])
    eval_best_df20['positive_pct'] = np.where(eval_best_df20['return'].isnull(), np.nan, eval_best_df20['positive_pct'])
    plot_topn_vs_mkt(eval_best_df20, weeks_to_expire, fig_name=f'{name_sql}_top20')
    eval_best_df20_agg = eval_best_df20.groupby(['currency_code'])[['positive_pct', 'return']].mean().reset_index()

    # 4. factor selection table
    factor_selection = read_query(f"SELECT * FROM {global_vars.backtest_eval_table} "
                                  f"WHERE name_sql = '{name_sql}'", global_vars.db_url_alibaba_prod)

    diff_config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', '_factor_reverse', 'down_mkt_pct']
    df = pd.concat([factor_selection, pd.DataFrame(factor_selection['config'].to_list())], axis=1)
    factor_selection_best = df.groupby(['group', 'pillar']).apply(lambda x: x.nsmallest(1, ['config_mean_mse'], keep='all'))
    factor_selection_best = factor_selection_best.drop(columns=diff_config_col)

    csv_name = 'top{}_{}'.format(top_config, start_year)
    if name_sql:
        csv_name += f"_{name_sql}"
    if use_usd:
        csv_name += f"_usd"

    to_excel({"10 Qcut (Avg Ret)": eval_qcut_df,
              "10 Qcut (Avg Ret-Agg)": eval_qcut_df_avg,
              "Top 10 Picks": eval_best_df10,
              "Top 10 Picks(agg)": eval_best_df10_agg,
              "Top 20 Picks": eval_best_df20,
              "Top 20 Picks(agg)": eval_best_df20_agg,
              "Factor Selection & Avg Premiums": factor_selection,
              "Best Factor Config": factor_selection_best}, file_name=csv_name)

    eval_best_df10 = pd.read_excel(csv_name+'.xlsx', "Top 10 Picks")
    eval_best_df20 = pd.read_excel(csv_name+'.xlsx', "Top 20 Picks")

    # update top 10 to DB
    def write_topn_to_db(df, n):
        ''' for each backtest eval : write top 10 ticker to DB '''

        tbl_name = global_vars.backtest_top_table
        df = df.convert_dtypes()
        df = df.rename(columns={"mode count": "mode_count"})
        df["return"] = (df["return"] * 100).round(2).astype(float)
        df["positive_pct"] = (df["positive_pct"] * 100).round(0)
        df["weeks_to_expire"] = pd.to_numeric(df["weeks_to_expire"].str.split('_', expand=True).values[:, 0])
        df['name_sql'] = name_sql
        df['start_date'] = df['trading_day'].min()
        df['n_period'] = len(df['trading_day'].unique())
        df['n_top'] = n
        df[["trading_day", "start_date"]] = df[["trading_day", "start_date"]].apply(lambda x: x.dt.date)
        upsert_data_to_database(df, tbl_name,
                                primary_key=["name_sql", "start_date", "n_period", "n_top", "currency_code", "trading_day"],
                                how='update', db_url=global_vars.db_url_alibaba_prod)

    # write_topn_to_db(eval_best_df10, 10)
    # write_topn_to_db(eval_best_df20, 20)

    return True

def test_score_history_v2(currency_code=None, start_date='2020-10-01', name_sql=None):
    '''  calculate score with DROID v2 method & evaluate (V2 only evaluate pillar score for all config)

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

    # print("=== Get [Factor Processed Ratio] history ===")
    # conditions = ["r.ticker not like '.%%'"]
    # if currency_code:
    #     conditions.append(f"currency_code in {tuple(currency_code)}")
    # if start_date:
    #     conditions.append(f"trading_day>='{start_date}'")
    # ratio_query = f"SELECT r.*, currency_code FROM {global_vars.processed_ratio_table} r " \
    #               f"INNER JOIN (SELECT ticker, currency_code FROM universe) u ON r.ticker=u.ticker " \
    #               f"WHERE {' AND '.join(conditions)}"
    # fundamentals_score = read_query(ratio_query, global_vars.db_url_alibaba_prod)
    # fundamentals_score = fundamentals_score.pivot(index=["ticker", "trading_day", "currency_code"], columns=["field"],
    #                                               values="value").reset_index()
    # fundamentals_score.to_csv('cached_fundamental_score.csv', index=False)
    # exit(200)

    fundamentals_score = pd.read_csv('cached_fundamental_score.csv')
    fundamentals_score["trading_day"] = pd.to_datetime(fundamentals_score["trading_day"])

    print(f"=== Calculate [Factor Rank] for name_sql:[{name_sql}] ===")
    # eval_table = read_query(f"SELECT * FROM factor_result_rank_backtest_eval WHERE name_sql='{name_sql}'")
    factor_rank_all = rank_pred(1/3, name_sql=name_sql, top_config=None).write_backtest_rank_(upsert_how=False)

    qcut_eval = []
    top10_eval = []
    error_iter = []

    for iter in factor_rank_all:

        try:
            factor_rank = iter["rank_df"]

            print(fundamentals_score["trading_day"].min(), fundamentals_score["trading_day"].max())
            print(sorted(list(factor_rank["trading_day"].unique())))
            factor_rank["trading_day"] = pd.to_datetime(factor_rank["trading_day"])
            factor_rank['trading_day'] = factor_rank['trading_day'].dt.tz_localize(None)

            # Test return calculation
            # fundamentals_score_ret = fundamentals_score[["ticker","trading_day","stock_return_y_1week", "stock_return_y_4week"]]
            # tri_ret = get_tri_ret(start_date)
            # fundamentals_score_ret = fundamentals_score_ret.merge(tri_ret, on=["ticker", "trading_day"], how="left")
            # fundamentals_score_ret["diff_1w"] = fundamentals_score_ret["stock_return_y_1week"] - fundamentals_score_ret["ret_week"]*4
            # fundamentals_score_ret["diff_1m"] = fundamentals_score_ret["stock_return_y_4week"] - fundamentals_score_ret["ret_month"]

            factor_formula = read_table(global_vars.factors_formula_table, global_vars.db_url_alibaba_prod)
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
            # print(fundamentals)

            fundamentals = fundamentals.dropna(subset=['trading_day'], how='any')

            # add column for 3 pillar score
            fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan

            # add industry_name to ticker
            industry_name = get_industry_name()
            fundamentals["industry_name"] = fundamentals["ticker"].map(industry_name)

            # calculate score
            eval_qcut_all = {}
            eval_best_all10 = {}

            average_days = int(name_sql.split('_')[1][1:])

            for (trading_day, weeks_to_expire), factor_rank_period in factor_rank.groupby(by=['trading_day', 'weeks_to_expire']):
                weeks_to_expire = int(weeks_to_expire)
                score_date = trading_day + relativedelta(weeks=weeks_to_expire)
                g = fundamentals.loc[fundamentals['trading_day']==score_date].copy()

                # Scale original fundamental score
                print(trading_day, score_date)
                g_score = score_scale(g.copy(1), calculate_column, universe_currency_code,
                                      factor_rank_period, weeks_to_expire, factor_rank_period).score_update_scale()

                current_score_col = 'fundamentals_'+iter["info"]["pillar"].to_list()[0]

                # Evaluate 1: calculate return on 10-qcut portfolios
                eval_qcut_all[(score_date, f"{weeks_to_expire}_{average_days}")] = eval_qcut(g_score, [current_score_col],
                                                                                             weeks_to_expire, average_days).copy()

                # Evaluate 2: calculate return for top 10 score / mode industry
                eval_best_all10[(score_date, f"{weeks_to_expire}_{average_days}")] = \
                    eval_best(g_score, weeks_to_expire, average_days, best_n=10, best_col=current_score_col).copy()

            print("=== Reorganize mean return dictionary to DataFrames ===")
            # 1. DataFrame for 10 group qcut - Mean Returns
            eval_qcut_df = pd.DataFrame(eval_qcut_all)
            eval_qcut_df = eval_qcut_df.stack(level=[-2, -1])
            eval_qcut_df = pd.DataFrame(eval_qcut_df.to_list(), index=eval_qcut_df.index)
            eval_qcut_df = eval_qcut_df.reset_index().rename(columns={"level_0": "currency_code",
                                                                      "level_1": "score",
                                                                      "level_2": "trading_day",
                                                                      "level_3": "weeks_to_expire",})

            def add_info_col(df):
                df = df.merge(iter["info"], left_on=["currency_code"], right_index=True)
                return df

            # 2. DataFrame for 10 group qcut - Mean Returns
            eval_qcut_df_avg = eval_qcut_df.groupby(["currency_code", "score", "weeks_to_expire"]).mean().reset_index()
            eval_qcut_df_avg_final = add_info_col(eval_qcut_df_avg).copy()
            qcut_eval.append(eval_qcut_df_avg_final)

            # 3.1. DataFrame for Top 10 picks - Mean Returns / mode indsutry(count) / positive pct
            eval_best_df10 = pd.DataFrame(eval_best_all10).stack(level=[-2, -1]).unstack(level=1)
            eval_best_df10 = eval_best_df10.reset_index().rename(columns={"level_0": "currency_code",
                                                                      "level_1": "trading_day",
                                                                      "level_2": "weeks_to_expire",})
            eval_best_df10['positive_pct'] = eval_best_df10['positive_pct'].replace(0, np.nan)
            eval_best_df10['return'] = pd.to_numeric(eval_best_df10['return'])
            eval_best_df10_final = add_info_col(eval_best_df10).copy()
            top10_eval.append(eval_best_df10_final)
        except Exception as e:
            print(e)
            error_iter.append(iter["info"])

    qcut_eval = pd.concat(qcut_eval, axis=0)
    top10_eval = pd.concat(top10_eval, axis=0)

    to_excel({"qcut": qcut_eval,
              "top10": top10_eval,
              }, file_name=('v2_'+name_sql if name_sql else ""))
    print(error_iter)
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
            try:
                df['qcut'] = pd.qcut(df[col].dropna(), q=10, labels=False, duplicates='drop')
                mean_ret[(name, col)] = df.dropna(subset=[col]).groupby(['qcut'])[
                    f'stock_return_y_w{weeks_to_expire}_d{average_days}'].mean().to_dict()
            except Exception as e:
                print(e)
    return mean_ret

def plot_topn_vs_mkt(top_n_df, weeks_to_expire, avg_days=7, fig_name='test'):
    ''' compare top n average return with market return

    Args:
        top_n_arr (pd.DataFrame): at least has colume (trading_day & return)
    '''
    group_index = {"USD": ".SPX", "HKD": ".HSI", "EUR": ".SXXGR", "CNY": ".CSI300"}

    date_list = list(top_n_df['trading_day'].dt.strftime('%Y-%m-%d').unique())

    fig = plt.figure(figsize=(10, 10), dpi=120, constrained_layout=True)

    k = 1
    for group, g in top_n_df.groupby(['currency_code']):
        ax = fig.add_subplot(2, 2, k)

        mkt = read_query(f"SELECT trading_day, value as mkt_return FROM factor_processed_ratio "
                         f"WHERE trading_day in {tuple(date_list)} "
                         f"AND field='stock_return_y_w{weeks_to_expire}_d{avg_days}' "
                         f"AND ticker='{group_index[group]}'".replace(",)", ")"))
        mkt['trading_day'] = pd.to_datetime(mkt['trading_day'])
        g = g.merge(mkt, on='trading_day').set_index('trading_day')[['return', 'mkt_return']]
        g[['return', 'mkt_return']] = g[['return', 'mkt_return']].add(1).cumprod(axis=0)
        ax.plot(g[['return', 'mkt_return']], label=['return', 'mkt_return'])
        ax.legend()
        ax.set_xlabel(group)
        for i in range(2):
            ax.annotate(g.iloc[-1, i].round(2), (g.index[-1], g.iloc[-1, i]), fontsize=10)
        k += 1

    plt.savefig(f'{fig_name}.png')
    plt.close()

def eval_best(fundamentals, weeks_to_expire, average_days, best_n=10, best_col='ai_score'):
    ''' evaluate score history with top 10 score return & industry '''

    top_ret = {}
    for name, g in fundamentals.groupby(["currency_code"]):
        g = g.set_index('ticker').nlargest(best_n, columns=[best_col], keep='all')
        top_ret[(name, "return")] = g[f'stock_return_y_w{weeks_to_expire}_d{average_days}'].mean()
        top_ret[(name, "mode")] = g[f"industry_name"].mode()[0]
        top_ret[(name, "mode count")] = np.sum(g[f"industry_name"]==top_ret[(name, "mode")])
        top_ret[(name, "positive_pct")] = np.sum(g[f'stock_return_y_w{weeks_to_expire}_d{average_days}']>0)/len(g)
        top_ret[(name, "tickers")] = ', '.join(list(g.index))

    return top_ret

def get_industry_name():
    ''' get ticker -> industry name (4-digit) '''
    query = "SELECT ticker, name_4 FROM universe u INNER JOIN icb_code_explanation i ON u.industry_code=i.code_8"
    df= read_query(query, global_vars.db_url_alibaba_prod)
    return df.set_index(["ticker"])["name_4"].to_dict()

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name_sql', type=str)
    # args = parser.parse_args()

    # can select name_sql based on
    #x'w4_d7_20220216100210_debug', 'w8_d7_20220215191634_debug', 'w26_d7_20220215152028_debug'
    # for name_sql in ['w4_d7_20220216100210_debug', 'w8_d7_20220215191634_debug', 'w26_d7_20220215152028_debug']:
    # for name_sql in [ 'w4_d7_official', 'w8_d7_official', 'w13_d7_official', 'w26_d7_official']:
    for name_sql in ['w4_d7_official', 'w13_d7_20220301195636_debug']:
        # for top_config in [10]:
        #     for start_year in [2016, 2018, 2020, 2021]:
        #         test_score_history(name_sql=name_sql, start_year=start_year, top_config=top_config)
        test_score_history_v2(name_sql=name_sql, start_date='2016-01-01', currency_code=universe_currency_code)

    # test_score_history(name_sql=args.name_sql)