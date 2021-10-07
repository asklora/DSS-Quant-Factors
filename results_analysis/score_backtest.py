from sqlalchemy import text
from scipy.stats import skew

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import robust_scale, minmax_scale, quantile_transform, MinMaxScaler
from preprocess.premium_calculation import trim_outlier
import global_vals
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from score_evaluate import score_eval
from utils import record_table_update_time

cur = ["'USD'","'HKD'"]

def score_update_scale(fundamentals, calculate_column, universe_currency_code, factor_rank):

    fundamentals['dummy'] = True
    groupby_col = ['currency_code']  # or 'dummy'

    def transform_trim_outlier(x):
        s = skew(x)
        if (s < -5) or (s > 5):
            x = np.log(x + 1 - np.min(x))
        m = np.median(x)
        # clip_x = np.clip(x, np.percentile(x, 0.01), np.percentile(x, 0.99))
        std = np.nanstd(x)
        return np.clip(x, m - 2 * std, m + 2 * std)

    # trim outlier to +/- 2 std
    calculate_column_score = []
    for column in calculate_column:
        column_score = column + "_score"
        fundamentals[column_score] = fundamentals.dropna(subset=[column]).groupby(groupby_col)[column].transform(
            transform_trim_outlier)
        calculate_column_score.append(column_score)
    print(calculate_column_score)

    # x1 = fundamentals.groupby("currency_code")[[x+'_score' for x in calculate_column]].skew()
    # y1 = fundamentals.groupby("currency_code")[[x+'_score' for x in calculate_column]].apply(pd.DataFrame.kurtosis)

    # apply robust scaler
    calculate_column_robust_score = []
    for column in calculate_column:
        try:
            column_score = column + "_score"
            column_robust_score = column + "_robust_score"
            fundamentals[column_robust_score] = fundamentals.dropna(subset=[column_score]).groupby(groupby_col)[
                column_score].transform(lambda x: robust_scale(x))
            calculate_column_robust_score.append(column_robust_score)
        except Exception as e:
            print(e)
    print(calculate_column_robust_score)

    # apply maxmin scaler on Currency / Industry
    minmax_column = ["uid", "ticker", "trading_day"]
    for column in calculate_column:
        column_robust_score = column + "_robust_score"
        column_minmax_currency_code = column + "_minmax_currency_code"
        df_currency_code = fundamentals[["currency_code", column_robust_score]]
        df_currency_code = df_currency_code.rename(columns={column_robust_score: "score"})
        fundamentals[column_minmax_currency_code] = df_currency_code.dropna(subset=["currency_code", "score"]).groupby(
            groupby_col).score.transform(
            lambda x: minmax_scale(x.astype(float)) if x.notnull().sum() else np.full_like(x, np.nan))
        fundamentals[column_minmax_currency_code] = np.where(fundamentals[column_minmax_currency_code].isnull(),
                                                             fundamentals[column_minmax_currency_code].mean() * 0.9,
                                                             fundamentals[column_minmax_currency_code]) * 10
        minmax_column.append(column_minmax_currency_code)

        if column in ["environment", "social", "goverment"]:  # for ESG scores also do industry partition
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

    # apply quantile transformation on before scaling scores
    tmp = fundamentals.melt(["ticker", "currency_code", "industry_code"], calculate_column)
    tmp["quantile_transformed"] = tmp.dropna(subset=["value"]).groupby(groupby_col + ["variable"])["value"].transform(
        lambda x: QuantileTransformer(n_quantiles=4).fit_transform(
            x.values.reshape(-1, 1)).flatten() if x.notnull().sum() else np.full_like(x, np.nan))
    tmp = tmp[["ticker", "variable", "quantile_transformed"]]
    tmp["variable"] = tmp["variable"] + "_quantile_currency_code"
    tmp = tmp.pivot(["ticker"], ["variable"]).droplevel(0, axis=1)
    fundamentals = fundamentals.merge(tmp, how="left", on="ticker")

    fundamentals["trading_day"] = find_nearest_specific_days(days=6)
    fundamentals = uid_maker(fundamentals, uid="uid", ticker="ticker", trading_day="trading_day")

    # add column for 3 pillar score
    fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan
    # fundamentals[['dlp_1m','wts_rating']] = fundamentals[['dlp_1m','wts_rating']]/10    # adjust dlp score to 0 ~ 1 (originally 0 ~ 10)

    # dataframe for details checking
    fundamentals_details = {}
    fundamentals_details_column_names = {}
    for i in universe_currency_code:
        if i:
            fundamentals_details[i] = {}
            fundamentals_details_column_names[i] = {}

    # calculate ai_score by each currency_code (i.e. group) for each of [Quality, Value, Momentum]
    for (group, pillar_name), g in factor_rank.groupby(["group", "pillar"]):
        print(f"Calculate Fundamentals [{pillar_name}] in group [{group}]")
        sub_g = g.loc[(g["factor_weight"] == 2) | (g["factor_weight"].isnull())]  # use all rank=2 (best class)
        if len(sub_g.dropna(
                subset=["pred_z"])) == 0:  # if no factor rank=2, use the highest ranking one & DLPA/ai_value scores
            sub_g = g.loc[g.nlargest(1, columns=["pred_z"]).index.union(g.loc[g["factor_weight"].isnull()].index)]

        fundamentals_details_column_names[group][pillar_name] = ','.join(sub_g['factor_name'])
        score_col = [f"{x}_{y}_currency_code" for x, y in
                     sub_g.loc[sub_g["scaler"].notnull(), ["factor_name", "scaler"]].to_numpy()]
        score_col += [x for x in sub_g.loc[sub_g["scaler"].isnull(), "factor_name"]]
        fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_{pillar_name}"] = fundamentals[
            score_col].mean(axis=1)

        # save used columns to pillars
        score_col_detail = ['ticker', f"fundamentals_{pillar_name}"] + sub_g.loc[
            sub_g["scaler"].notnull(), 'factor_name'].to_list() + score_col
        fundamentals_details[group][pillar_name] = fundamentals.loc[
            fundamentals['currency_code'] == group, score_col_detail].sort_values(by=[f"fundamentals_{pillar_name}"])

    # calculate ai_score by each currency_code (i.e. group) for [Extra]
    for group, g in factor_rank.groupby("group"):
        print(f"Calculate Fundamentals [extra] in group [{group}]")
        sub_g = g.loc[(g["factor_weight"] == 2) | (g["factor_weight"].isnull())]  # use all rank=2 (best class)
        sub_g = sub_g.loc[(g["pred_z"] >= 1) | (
            g["pred_z"].isnull())]  # use all rank=2 (best class) and predicted factor premiums with z-value >= 1

        if len(sub_g.dropna(subset=["pred_z"])) > 0:  # if no factor rank=2, don"t add any factor into extra pillar
            score_col = [f"{x}_{y}_currency_code" for x, y in
                         sub_g.loc[sub_g["scaler"].notnull(), ["factor_name", "scaler"]].to_numpy()]
            score_col_detail = sub_g.loc[sub_g["scaler"].notnull(), 'factor_name'].to_list() + score_col
            fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_extra"] = fundamentals[
                score_col].mean(axis=1)
            fundamentals_details_column_names[group]['extra'] = ','.join(sub_g['factor_name'])
        else:
            score_col_detail = []
            fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_extra"] = \
                fundamentals.loc[fundamentals["currency_code"] == group].filter(regex="^fundamentals_").mean().mean()
            fundamentals_details_column_names[group]['extra'] = ''

        # save used columns to pillars
        # fundamentals_details[group]['extra'] = fundamentals.loc[fundamentals['currency_code']==group,
        #        ['ticker', "fundamentals_extra"] + score_col_detail].sort_values(by=[ f"fundamentals_extra"])

    replace_table_datebase_ali(pd.DataFrame(fundamentals_details_column_names).transpose().reset_index(),
                               f"test_fundamental_score_current_names")

    # manual score check output to alibaba DB
    for group, v in fundamentals_details.items():
        pillar_df = []
        for pillar, df in v.items():
            pillar_df.append(df.set_index(['ticker']))
        pillar_df = pd.concat(pillar_df, axis=1)
        pillar_df.index = pillar_df.index.set_names(['index'])
        replace_table_datebase_ali(pillar_df.reset_index(), f"test_fundamental_score_details_{group}")

    fundamentals_factors_scores_col = fundamentals.filter(regex="^fundamentals_").columns

    print("Calculate ESG Value")
    esg_cols = ["environment_minmax_currency_code", "environment_minmax_industry", "social_minmax_currency_code",
                "social_minmax_industry", "goverment_minmax_currency_code", "goverment_minmax_industry"]
    fundamentals["esg"] = fundamentals[esg_cols].mean(1)

    print("Calculate AI Score")
    ai_score_cols = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra"]
    fundamentals["ai_score"] = fundamentals[ai_score_cols].mean(1)

    print("Calculate AI Score 2")
    ai_score_cols2 = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "esg"]
    fundamentals["ai_score2"] = fundamentals[ai_score_cols2].mean(1)

    print(fundamentals[["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra",
                        'esg']].describe())

    # scale ai_score with history min / max
    # print(fundamentals.groupby(['currency_code'])[["ai_score", "ai_score2"]].agg(['min','mean','median','max']).transpose()[['HKD','USD','CNY','EUR']])
    fundamentals[["ai_score_unscaled", "ai_score2_unscaled"]] = fundamentals[["ai_score", "ai_score2"]]
    score_history = get_ai_score_testing_history(backyear=1)
    for cur, g in fundamentals.groupby(['currency_code']):
        try:
            raise Exception('Scaling with current score')
            score_history_cur = score_history.loc[score_history['currency_code'] == cur]
            print(f'{cur} History Min/Max: ',
                  score_history_cur[["ai_score_unscaled", "ai_score2_unscaled"]].min().values,
                  score_history_cur[["ai_score_unscaled", "ai_score2_unscaled"]].max().values)
            print(f'{cur} Current Min/Max: ', g[["ai_score", "ai_score2"]].min().values,
                  g[["ai_score", "ai_score2"]].max().values)
            m1 = MinMaxScaler(feature_range=(0, 10)).fit(score_history_cur[["ai_score_unscaled", "ai_score2_unscaled"]])
            fundamentals.loc[g.index, ["ai_score", "ai_score2"]] = m1.transform(g[["ai_score", "ai_score2"]])
        except Exception as e:
            print(e)
            print(f'{cur} Current Min/Max: ', g[["ai_score", "ai_score2"]].min().values,
                  g[["ai_score", "ai_score2"]].max().values)
            fundamentals.loc[g.index, ["ai_score", "ai_score2"]] = MinMaxScaler(feature_range=(0, 10)).fit_transform(
                g[["ai_score", "ai_score2"]])

    return fundamentals, minmax_column, fundamentals_factors_scores_col

def score_history():
    ''' calculate score with DROID v2 method & evaluate '''

    with global_vals.engine.connect() as conn, global_vals.engine_ali.connect() as conn_ali:  # write stock_pred for the best hyperopt records to sql
        factor_formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}_prod', conn_ali)
        factor_rank = pd.read_sql(f'SELECT * FROM {global_vals.production_factor_rank_table}_history', conn_ali)
        universe = pd.read_sql(f"SELECT * FROM {global_vals.dl_value_universe_table} WHERE is_active AND currency_code in ({','.join(cur)})", conn)
        fundamentals_score = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}_monthly "
                                         f"WHERE (period_end>'2017-08-30') AND (ticker not like '.%%') ", conn_ali)
                                         # f"WHERE (period_end='2021-07-31') AND (ticker not like '.%%') ", conn_ali)
        # pred_mean = pd.read_sql(f"SELECT * FROM ai_value_lgbm_pred_final_eps", conn_ali)
    global_vals.engine_ali.dispose()

    # universe_ticker = set(universe['ticker'].to_list())
    # ratio_ticker = set(fundamentals_score['ticker'].to_list())
    # both_exist_ticker = set(universe_ticker) & ratio_ticker
    # print(universe_ticker - both_exist_ticker)
    # print(ratio_ticker - both_exist_ticker)

    fundamentals_score = fundamentals_score.loc[fundamentals_score['ticker'].isin(universe['ticker'].to_list())]
    print(len(set(fundamentals_score['ticker'].to_list())))

    # x = fundamentals_score.groupby('currency_code').apply(lambda x: x.isnull().sum()/len(x))
    # pred_mean = pd.pivot_table(pred_mean, index=['ticker'], columns=['y_type'], values='final_pred')
    # pred_mean.columns = ['ai_value_'+x for x in pred_mean.columns]
    # fundamentals_score = fundamentals_score.merge(pred_mean, on=['ticker','period_end'])

    # for currency not predicted by Factor Model -> Use factor of USD
    universe_currency_code = list(filter(None, universe['currency_code'].unique()))
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

    # change ratio to negative if original factor calculation using reverse premiums
    for (group_name, period_end), g in factor_rank.groupby(['group', 'period_end']):
        neg_factor = g.loc[(factor_rank['long_large'] == False),'factor_name'].to_list()
        fundamentals_score.loc[(fundamentals_score['period_end']==period_end)&(fundamentals_score['currency_code']==group_name), neg_factor] *= -1

    fundamentals = fundamentals_score[['ticker','period_end','currency_code','stock_return_y']+calculate_column]
    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan).copy()
    print(fundamentals)

    fundamentals = fundamentals.dropna(subset=["currency_code",'period_end'], how='any')

    # add column for 3 pillar score
    fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan
    fundamentals[['dlp_1m', 'wts_rating','earnings_pred_minmax_currency_code','revenue_pred_minmax_currency_code']] = np.nan  # ignore ai_value / DLPA
    # fundamentals['period_end'] = fundamentals['period_end'].dt.strftime('%Y-%m-%d')

    for name, g in fundamentals.groupby():
        # Scale original fundamental score
        g, minmax_column, fundamentals_factors_scores_col = score_update_scale(g, calculate_column, universe_currency_code, factor_rank)



    m1 = MinMaxScaler(feature_range=(0, 10)).fit(fundamentals[["ai_score"]])
    fundamentals[["ai_score_scaled"]] = m1.transform(fundamentals[["ai_score"]])
    print(fundamentals[["ai_score"]].describe())

    x = fundamentals.groupby(['currency_code']).agg(['min','mean','max','std'])

    score_col = ['ai_score', 'ai_score_scaled', 'fundamentals_value','fundamentals_quality','fundamentals_momentum','fundamentals_extra']
    label_col = ['ticker', 'period_end', 'currency_code', 'stock_return_y']

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
        fundamentals[label_col + score_col].to_sql(global_vals.production_score_history, **extra)
        record_table_update_time(global_vals.production_score_history, conn)
    global_vals.engine_ali.dispose()

if __name__ == "__main__":
    score_history()
    # score_eval()
