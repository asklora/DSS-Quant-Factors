from sqlalchemy import text
from scipy.stats import skew

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import robust_scale, minmax_scale, MinMaxScaler, QuantileTransformer
from preprocess.premium_calculation import trim_outlier
import global_vals
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from score_evaluate import score_eval
from utils import record_table_update_time

cur = ["'USD'"]

def score_update_scale(fundamentals, calculate_column, universe_currency_code, factor_rank):

    groupby_col = ['currency_code']  # or 'dummy'

    def transform_trim_outlier(x):
        s = skew(x)
        if (s < -5) or (s > 5):
            x = np.log(x + 1 - np.min(x))
        m = np.median(x)
        # clip_x = np.clip(x, np.percentile(x, 0.01), np.percentile(x, 0.99))
        std = np.nanstd(x)
        return np.clip(x, m - 2 * std, m + 2 * std)

    # 1. trim outlier to +/- 2 std
    calculate_column_score = []
    for column in calculate_column:
        column_score = column + "_score"
        fundamentals[column_score] = fundamentals.dropna(subset=[column]).groupby(groupby_col)[column].transform(
            transform_trim_outlier)
        calculate_column_score.append(column_score)
    print(calculate_column_score)

    # x1 = fundamentals.groupby("currency_code")[[x+'_score' for x in calculate_column]].skew()
    # y1 = fundamentals.groupby("currency_code")[[x+'_score' for x in calculate_column]].apply(pd.DataFrame.kurtosis)

    # 2. change ratio to negative if original factor calculation using reverse premiums
    for group_name, g in factor_rank.groupby(['group']):
        neg_factor = [x+'_score' for x in g.loc[(factor_rank['long_large'] == False), 'factor_name'].to_list()]
        fundamentals.loc[(fundamentals['currency_code'] == group_name), neg_factor] *= -1

    # 3. apply robust scaler
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

    # 4. apply maxmin scaler on Currency / Industry
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

    # add column for 3 pillar score
    fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan

    # dataframe for details checking
    fundamentals_details = {}
    fundamentals_details_column_names = {}
    for i in universe_currency_code:
        if i:
            fundamentals_details[i] = {}
            fundamentals_details_column_names[i] = {}

    mean_ret = {}
    best_score_ticker = {}
    mean_ret_detail_all = {}

    # calculate ai_score by each currency_code (i.e. group) for each of [Quality, Value, Momentum]
    for (group, pillar_name), g in factor_rank.groupby(["group", "pillar"]):
        print(f"Calculate Fundamentals [{pillar_name}] in group [{group}]")
        sub_g = g.loc[(g["factor_weight"] == 2) | (g["factor_weight"].isnull())]  # use all rank=2 (best class)
        if len(sub_g.dropna(subset=["pred_z"])) == 0:  # if no factor rank=2, use the highest ranking one & DLPA/ai_value scores
            sub_g = g.loc[g.nlargest(1, columns=["pred_z"]).index.union(g.loc[g["factor_weight"].isnull()].index)]

        score_col = [f"{x}_{y}_currency_code" for x, y in
                     sub_g.loc[sub_g["scaler"].notnull(), ["factor_name", "scaler"]].to_numpy()]
        score_col += [x for x in sub_g.loc[sub_g["scaler"].isnull(), "factor_name"]]
        fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_{pillar_name}"] = fundamentals[score_col].mean(axis=1)

        mean_ret[(group, pillar_name)] = score_ret_mean(fundamentals.loc[fundamentals["currency_code"] == group], score_col+[f"fundamentals_{pillar_name}"])

        mean_ret_detail = pd.DataFrame(mean_ret[(group, pillar_name)][f"fundamentals_{pillar_name}"],
                                       index=list(range(len(mean_ret[(group, pillar_name)][f"fundamentals_{pillar_name}"])))).transpose()
        f = sub_g[['factor_name', 'factor_weight', 'pred_z', 'long_large']].values
        f = [f'{a}({b}, {round(c, 2)}, {d})' for a, b, c, d in f]
        mean_ret_detail['score_col'] = ', '.join(f)
        mean_ret_detail_all[pillar_name] = mean_ret_detail


    # calculate ai_score by each currency_code (i.e. group) for [Extra]
    for group, g in factor_rank.groupby("group"):
        print(f"Calculate Fundamentals [extra] in group [{group}]")
        sub_g = g.loc[(g["factor_weight"] == 2) | (g["factor_weight"].isnull())]  # use all rank=2 (best class)
        sub_g = sub_g.loc[(g["pred_z"] >= 1) | (
            g["pred_z"].isnull())]  # use all rank=2 (best class) and predicted factor premiums with z-value >= 1

        if len(sub_g.dropna(subset=["pred_z"])) > 0:  # if no factor rank=2, don"t add any factor into extra pillar
            score_col = [f"{x}_{y}_currency_code" for x, y in
                         sub_g.loc[sub_g["scaler"].notnull(), ["factor_name", "scaler"]].to_numpy()]
            fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_extra"] = fundamentals[
                score_col].mean(axis=1)
        else:
            fundamentals.loc[fundamentals["currency_code"] == group, f"fundamentals_extra"] = \
                fundamentals.loc[fundamentals["currency_code"] == group].filter(regex="^fundamentals_").mean().mean()

        mean_ret[(group, 'extra')] = score_ret_mean(fundamentals.loc[fundamentals["currency_code"] == group], score_col+["fundamentals_extra"])

    print("Calculate AI Score")
    ai_score_cols = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra"]
    fundamentals["ai_score"] = fundamentals[ai_score_cols].mean(1)

    for group, g in fundamentals.groupby("currency_code"):
        mean_ret[(group, 'ai_score')] = score_ret_mean(g, ["ai_score"])
        best_score_ticker[group] = best_10_tickers(g, ai_score_cols+['ai_score'])

    print(fundamentals[["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra"]].describe())

    return fundamentals, mean_ret, best_score_ticker, mean_ret_detail_all

def score_history():
    ''' calculate score with DROID v2 method & evaluate '''

    with global_vals.engine.connect() as conn, global_vals.engine_ali.connect() as conn_ali:  # write stock_pred for the best hyperopt records to sql
        factor_formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}_prod', conn_ali)
        factor_rank = pd.read_sql(f'SELECT * FROM {global_vals.production_factor_rank_table}_history', conn_ali)
        universe = pd.read_sql(f"SELECT * FROM {global_vals.dl_value_universe_table} WHERE is_active AND currency_code in ({','.join(cur)})", conn)
        fundamentals_score = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}_monthly "
                                         f"WHERE (period_end>='2017-10-30') AND (ticker not like '.%%') ", conn_ali)
                                         # f"WHERE (period_end='2021-07-31') AND (ticker not like '.%%') ", conn_ali)
        # pred_mean = pd.read_sql(f"SELECT * FROM ai_value_lgbm_pred_final_eps", conn_ali)
    global_vals.engine_ali.dispose()

    fundamentals_score.to_csv('cached_fundamental_score.csv', index=False)
    factor_rank.to_csv('cached_factor_rank.csv', index=False)
    # fundamentals_score = pd.read_csv('cached_fundamental_score.csv')
    # factor_rank = pd.read_csv('cached_factor_rank.csv')

    fundamentals_score['period_end'] = pd.to_datetime(fundamentals_score['period_end'])
    fundamentals_score = fundamentals_score.loc[fundamentals_score['period_end']>dt.datetime(2020,2,1)]

    fundamentals_score = fundamentals_score.loc[fundamentals_score['ticker'].isin(universe['ticker'].to_list())]
    print(len(set(fundamentals_score['ticker'].to_list())))

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

    fundamentals = fundamentals_score[['ticker','period_end','currency_code','stock_return_y']+calculate_column]
    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan).copy()
    print(fundamentals)

    fundamentals = fundamentals.dropna(subset=["currency_code",'period_end'], how='any')

    # add column for 3 pillar score
    fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan
    fundamentals[['dlp_1m', 'wts_rating','earnings_pred_minmax_currency_code','revenue_pred_minmax_currency_code']] = np.nan  # ignore ai_value / DLPA
    # fundamentals['period_end'] = fundamentals['period_end'].dt.strftime('%Y-%m-%d')

    score_col = ['ai_score', 'ai_score_scaled', 'fundamentals_value','fundamentals_quality','fundamentals_momentum','fundamentals_extra']
    label_col = ['ticker', 'period_end', 'currency_code', 'stock_return_y']

    mean_ret_all = {}
    best_10_tickers_all = {}

    mean_ret_detail_all = {}
    mean_ret_detail_all['quality'] = []
    mean_ret_detail_all['value'] = []
    mean_ret_detail_all['momentum'] = []

    fundamentals_all = []

    for name, g in fundamentals.groupby(['period_end']):

        factor_rank_period = factor_rank.loc[factor_rank['period_end'].astype(str)==name.strftime('%Y-%m-%d')]

        # Scale original fundamental score
        fundamentals, mean_ret_all[name], best_10_tickers_all[name], mean_ret_detail = \
            score_update_scale(g, calculate_column, universe_currency_code, factor_rank_period)

        fundamentals_all.append(fundamentals)

        for p, df in mean_ret_detail.items():
            df['period_end'] = name
            mean_ret_detail_all[p].append(df)
        # results = fundamentals[label_col+score_col]
        # x = results.groupby(['currency_code']).agg(['min','mean','max','std'])
        # print(x)

    for c in [x[1:-1] for x in cur]:
        writer = pd.ExcelWriter(f"#{dt.datetime.today().strftime('%Y%m%d')}_backtest_{c}.xlsx")
        tic = {x:z for x, yz in best_10_tickers_all.items() for y, z in yz.items() if y==c}
        pd.DataFrame(tic).transpose().to_excel(writer, 'Top 10 Ticker')

        for p, df_list in mean_ret_detail_all.items():
            ddf = pd.concat(df_list, axis=0).reset_index(drop=True)
            ddf.to_excel(writer, f'{p} details')

        for p in ['momentum','quality', 'value', 'extra']:
            ret_p = {x:z for x, yz in mean_ret_all.items() for y, z in yz.items() if (y[0]==c) & (y[1]==p) if len(z)>0}
            ret_p = {x:y[f'fundamentals_{p}'] for x, y in ret_p.items()}
            ret_p = {x:np.pad(y, (10-len(y),0)) for x, y in ret_p.items()}
            ret_p_df = pd.DataFrame(ret_p, index=list(range(10))).transpose()
            ret_p_df.to_excel(writer, f'Qcut {p}')
        for p in ['ai_score']:
            ret_p1 = {x:z[p] for x, yz in mean_ret_all.items() for y, z in yz.items() if (y[0]==c) & (y[1]==p) if len(z)>0}
            ret_p1 = {x:np.pad(y, (10-len(y),0)) for x, y in ret_p1.items()}
            ret_p1_df = pd.DataFrame(ret_p1, index=list(range(10))).transpose()
            ret_p1_df.to_excel(writer, 'Qcut ai_score')
        writer.save()

    fundamentals = pd.concat(fundamentals_all, axis=0)
    m1 = MinMaxScaler(feature_range=(0, 10)).fit(fundamentals[["ai_score"]])
    fundamentals[["ai_score_scaled"]] = m1.transform(fundamentals[["ai_score"]])
    print(fundamentals[["ai_score"]].describe())

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
        fundamentals[label_col + score_col].to_sql(global_vals.production_score_history, **extra)
        record_table_update_time(global_vals.production_score_history, conn)
    global_vals.engine_ali.dispose()

def best_10_tickers(g, score_col):
    g = g.set_index('ticker').nlargest(10, columns=['ai_score'], keep='all')[['stock_return_y']+score_col]
    comb = pd.DataFrame(g.mean()).round(4).transpose()
    g = g.reset_index()[['ticker','stock_return_y']].values
    g = [f'{a}({round(b,2)})' for a, b in g]
    comb['ticker'] = ', '.join(g)
    return comb.transpose().to_dict()[0]

    # for name, g in fundamentals.groupby('currency_code'):
    #     df_dict[f'best10_{name}'] = record_tickers(g).iloc[:,:-2].reset_index().drop(columns=['level_2'])

# 2. Test 10-qcut return
def score_ret_mean(df, score_col):

    mean_ret = {}
    df = df.reset_index(drop=True)
    for col in score_col:
        df['qcut'] = pd.qcut(df[col].dropna(), q=10, labels=False, duplicates='drop')
        mean_ret[col.replace('_minmax_currency_code','')] = df.dropna(subset=[col]).groupby(['qcut'])['stock_return_y'].mean().values

    return mean_ret

if __name__ == "__main__":
    score_history()
    # score_eval()
