# This Code copies from DSS-Quant-Ingestion repo just for reference

import numpy as np
from scipy.stats import skew
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import robust_scale, minmax_scale, QuantileTransformer, PowerTransformer, MinMaxScaler
from general.data_process import remove_null, uid_maker, uid_maker_field
from general.slack import report_to_slack
from general.sql_output import (
    delete_data_on_database,
    upsert_data_to_database,
    update_ingestion_update_time,
)
from general.table_name import (
    get_universe_rating_detail_history_table_name,
    get_universe_rating_history_table_name,
    get_universe_rating_table_name,
    get_universe_rating_posneg_factor_table_name,
)
from general.sql_query import (
    get_active_universe,
    get_factor_calculation_formula,
    get_factor_rank,
    get_worldscope_summary_latest,
    get_ibes_monthly_latest,
    get_dsws_addition_latest,
    get_last_close_industry_code,
    get_currency_fx_rate_dict,
    get_ai_value_pred_final,
    get_specific_tri_avg,
    get_specific_volume_avg,
    get_dlpa_rating_latest,
    get_ingestion_name_source,
    get_currency_code_ibes_ws,
    get_factor_current_used,
    get_universe_rating_history_score,
    get_latest_universe_rating_detail_history,
    get_specific_tri,
    get_data_ohlcv,
    get_data_tri,
    get_universe_rating,
)
from general.date_process import (
    backdate_by_year,
    backdate_by_week,
    backdate_by_month,
    dateNow,
    datetimeNow
)
from es_logging.logger import log2es

def score_update_vol_rs(list_of_start_end, days_in_year=256):
    """ Calculate roger satchell volatility:
        daily = average over period from start to end: Log(High/Open)*Log(High/Close)+Log(Low/Open)*Log(Open/Close)
        annualized = sqrt(daily*256)
    """

    # download past prices since 1 months before the earliest volitility calculation month i.e. end
    tri = get_data_ohlcv(start_date=backdate_by_month(list_of_start_end[-1][-1]+1))
    tri["trading_day"] = pd.to_datetime(tri["trading_day"])
    tri = tri.sort_values(by=["ticker", "trading_day"], ascending=[True, False]).reset_index(drop=True)
    open_data, high_data, low_data, close_data = tri["open"].values, tri["high"].values, tri["low"].values, tri[
        "close"].values

    # Calculate daily volatility
    hc_ratio = np.divide(high_data, close_data)
    log_hc_ratio = np.log(hc_ratio.astype(float))
    ho_ratio = np.divide(high_data, open_data)
    log_ho_ratio = np.log(ho_ratio.astype(float))
    lo_ratio = np.divide(low_data, open_data)
    log_lo_ratio = np.log(lo_ratio.astype(float))
    lc_ratio = np.divide(low_data, close_data)
    log_lc_ratio = np.log(lc_ratio.astype(float))

    input1 = np.multiply(log_hc_ratio, log_ho_ratio)
    input2 = np.multiply(log_lo_ratio, log_lc_ratio)
    sum_ = np.add(input1, input2)

    # Calculate annualize volatility
    vol_col = []
    for l in list_of_start_end:
        start, end = l[0]*30, l[1]*30
        name_col = f"vol_{start}_{end}"
        vol_col.append(name_col)
        tri[name_col] = sum_
        tri[name_col] = tri.groupby("ticker")[name_col].rolling(end - start, min_periods=1).mean().reset_index(drop=1)
        tri[name_col] = tri[name_col].apply(lambda x: np.sqrt(x * days_in_year))
        tri[name_col] = tri[name_col].shift(start)

    # return tri on the most recent trading_day
    final_tri = tri[["ticker"]+vol_col].dropna(how="any").groupby(["ticker"]).last().reset_index()

    return final_tri

def score_update_skew(year=1):
    ''' calcuate skewnesss of stock return of past 1yr '''
    tri = get_data_tri(start_date=backdate_by_month(year*12))
    tri["skew"] = tri['total_return_index']/tri.groupby('ticker')['total_return_index'].shift(1)-1       # update tri to 1d before (i.e. all stock ret up to 1d before)
    tri["trading_day"] = pd.to_datetime(tri["trading_day"])
    tri = tri.groupby('ticker')['skew'].skew().reset_index()
    return tri

def score_update_stock_return(list_of_start_end_month, list_of_start_end_week):
    """ Calculate specific period stock return (months) """

    df = pd.DataFrame(get_active_universe()["ticker"])

    for l in list_of_start_end_month:         # stock return (month)
        name_col = f"stock_return_r{l[1]}_{l[0]}"
        tri_start = get_specific_tri_avg(backdate_by_month(l[0]), avg_days=7, tri_name=f"tri_{l[0]}m")
        tri_end = get_specific_tri_avg(backdate_by_month(l[1]), avg_days=7, tri_name=f"tri_{l[1]}m")
        tri = tri_start.merge(tri_end, how="left", on="ticker")
        tri[name_col] = tri[f"tri_{l[0]}m"] / tri[f"tri_{l[1]}m"]-1
        df = df.merge(tri[["ticker", name_col]], how="left", on="ticker")
        print(df)

    for l in list_of_start_end_week:         # stock return (week)
        name_col = f"stock_return_ww{l[1]}_{l[0]}"
        tri_start = get_specific_tri_avg(backdate_by_week(l[0]), avg_days=7, tri_name=f"tri_{l[0]}w")
        tri_end = get_specific_tri_avg(backdate_by_week(l[1]), avg_days=7, tri_name=f"tri_{l[1]}w")
        tri = tri_start.merge(tri_end, how="left", on="ticker")
        tri[name_col] = tri[f"tri_{l[0]}w"] / tri[f"tri_{l[1]}w"]-1
        df = df.merge(tri[["ticker", name_col]], how="left", on="ticker")
        print(df)

    return df

def score_update_fx_conversion(df, ingestion_source):
    """ Convert all columns to USD for factor calculation (DSS, WORLDSCOPE, IBES using different currency) """

    org_cols = df.columns.to_list()     # record original columns for columns to return

    curr_code = get_currency_code_ibes_ws()     # map ibes/ws currency for each ticker
    df = df.merge(curr_code, on='ticker', how='left')
    # df = df.dropna(subset=['currency_code_ibes', 'currency_code_ws', 'currency_code'], how='any')   # remove ETF / index / some B-share -> tickers will not be recommended

    # map fx rate for conversion for each ticker
    fx = get_currency_fx_rate_dict()
    df['fx_dss'] = df['currency_code'].map(fx)
    df['fx_ibes'] = df['currency_code_ibes'].map(fx)
    df['fx_ws'] = df['currency_code_ws'].map(fx)

    ingestion_source = ingestion_source.loc[ingestion_source['non_ratio']]     # no fx conversion for ratio items

    for name, g in ingestion_source.groupby(['source']):        # convert for ibes / ws
        cols = g['our_name'].to_list()
        df[cols] = df[cols].div(df[f'fx_{name}'], axis="index")

    df['close'] = df['close']/df['fx_dss']  # convert close price

    return df[org_cols]

def score_update_factor_ratios(df, formula, ingestion_source):
    """ Calculate all factor used referring to DB ratio table """

    print(df.columns)
    df = score_update_fx_conversion(df, ingestion_source)

    # Prepare for field requires add/minus
    add_minus_fields = formula[["field_num", "field_denom"]].dropna(how="any").to_numpy().flatten()
    add_minus_fields = [i for i in list(set(add_minus_fields)) if any(["-" in i, "+" in i, "*" in i])]
    add_minus_fields_1q = formula.loc[formula['name'].str[-3:]=='_1q', ["field_num"]].dropna(how="any").to_numpy().flatten()
    add_minus_fields_1q = [i for i in list(set(add_minus_fields_1q)) if any(["-" in i, "+" in i, "*" in i])]
    add_minus_fields_1y = formula.loc[formula['name'].str[-4:]=='_1yr', ["field_num"]].dropna(how="any").to_numpy().flatten()
    add_minus_fields_1y = [i for i in list(set(add_minus_fields_1y)) if any(["-" in i, "+" in i, "*" in i])]

    def field_calc(df, x):
        ''' transform fields need calculation before ratio calculation '''
        if x[0] in "*+-": raise Exception("Invalid formula")
        temp = df[x[0]].copy()
        n = 1
        while n < len(x):
            if x[n] == "+":
                temp += df[x[n + 1]].replace(np.nan, 0)
            elif x[n] == "-":
                temp -= df[x[n + 1]].replace(np.nan, 0)
            elif x[n] == "*":
                temp *= df[x[n + 1]]
            else:
                raise Exception(f"Unexpected operand/operator: {x[n]}")
            n += 2
        return temp

    for i in add_minus_fields:
        x = [op.strip() for op in i.split()]
        df[i] = field_calc(df, x)
    for i in add_minus_fields_1q:
        x = [op.strip()+'_1q'  if len(op)>1 else op.strip() for op in i.split()]
        df[i+'_1q'] = field_calc(df, x)
    for i in add_minus_fields_1y:
        x = [op.strip()+'_1y'  if len(op)>1 else op.strip() for op in i.split()]
        df[i+'_1y'] = field_calc(df, x)

    # a) Keep original values
    keep_original_mask = formula["field_denom"].isnull() & formula["field_num"].notnull()
    new_name = formula.loc[keep_original_mask, "name"].to_list()
    old_name = formula.loc[keep_original_mask, "field_num"].to_list()
    df[new_name] = df[old_name]

    # b) Time series ratios (Calculate 1m change first)
    print(f'      ------------------------> Calculate time-series ratio ')
    for r in formula.loc[formula['field_num']==formula['field_denom'], ['name','field_denom']].to_dict(orient='records'):  # minus calculation for ratios
        if r['name'][-2:] == 'yr':
            df[r["name"]] = df[r["field_denom"]] / df[r["field_denom"]+'_1y']-1
        elif r['name'][-1] == 'q':
            df[r["name"]] = df[r["field_denom"]] / df[r["field_denom"]+'_1q']-1

    # c) Divide ratios
    print(f"      ------------------------> Calculate dividing ratios ")
    for r in formula.loc[(formula["field_denom"].notnull())&
                         (formula["field_num"]!= formula["field_denom"])].to_dict(orient="records"):  # minus calculation for ratios
        df[r["name"]] = df[r["field_num"]] / df[r["field_denom"]]

    return df

def score_update_scale(fundamentals, calculate_column, universe_currency_code, factor_formula, weeks_to_expire):
    ''' scale factor original value -> (0,1) scores '''

    def transform_trim_outlier(x):
        s = skew(x)
        if (s < -5) or (s > 5):
            x = np.log(x + 1 - np.min(x))
        m = np.median(x)
        # clip_x = np.clip(x, np.percentile(x, 0.01), np.percentile(x, 0.99))
        std = np.nanstd(x)
        return np.clip(x, m - 2 * std, m + 2 * std)

    # Scale 1: log transformation for high skewness & trim outlier to +/- 2 std
    calculate_column_score = []
    for column in calculate_column:
        column_score = column + "_score"
        fundamentals[column_score] = fundamentals.dropna(subset=[column]).groupby("currency_code")[column].transform(
            transform_trim_outlier)
        calculate_column_score.append(column_score)
    print(calculate_column_score)

    # Scale 2: Reverse value for long_large = False (i.e. recommend short larger value)
    factor_rank = get_factor_rank(weeks_to_expire)
    factor_rank = factor_rank.merge(factor_formula, left_on=['factor_name'], right_index=True, how='outer')

    for i in set(factor_rank['group'].dropna().unique()):
        keep_rank = factor_rank.loc[factor_rank["group"].isnull()].copy()
        keep_rank["group"] = i
        factor_rank = factor_rank.append(keep_rank, ignore_index=True)
    factor_rank = factor_rank.dropna(subset=["group"])

    # for score not included in backtest (earnings_pred & wts_rating)
    factor_rank.loc[factor_rank['factor_name']=='earnings_pred', ['group', 'long_large','pred_z','factor_weight']] = \
        factor_rank.loc[factor_rank['factor_name']=='fwd_ey', ['group', 'long_large','pred_z','factor_weight']].values
    if weeks_to_expire==1:
        factor_rank.loc[factor_rank['factor_name'].isin(["wts_rating"]), 'long_large'] = True
        factor_rank.loc[factor_rank['factor_name'].isin(["wts_rating"]), 'pred_z'] = 2
        factor_rank.loc[factor_rank['factor_name'].isin(["wts_rating"]), 'factor_weight'] = 2
    factor_rank = factor_rank.dropna(subset=['pillar','pred_z'], how='any')

    # 2.1: use USD -> other currency
    replace_rank = factor_rank.loc[factor_rank['group'] == 'USD'].copy()
    for i in set(universe_currency_code) - set(factor_rank['group'].unique()):
        replace_rank['group'] = i
        factor_rank = factor_rank.append(replace_rank, ignore_index=True)

    # 2.2: reverse currency for not long_large
    for group, g in factor_rank.groupby(['group']):
        neg_factor = [x+'_score' for x in g.loc[(g['long_large'] == False), 'factor_name'].to_list()]
        fundamentals.loc[(fundamentals['currency_code'] == group), neg_factor] *= -1

    # Scale 3: apply robust scaler
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

    # Scale 4a: apply maxmin scaler on Currency / Industry
    minmax_column = []
    for column in calculate_column:
        column_robust_score = column + "_robust_score"
        column_minmax_currency_code = column + "_minmax_currency_code"
        df_currency_code = fundamentals[["currency_code", column_robust_score]]
        df_currency_code = df_currency_code.rename(columns={column_robust_score: "score"})
        fundamentals[column_minmax_currency_code] = df_currency_code.dropna(subset=["currency_code", "score"]).groupby(
            'currency_code').score.transform(lambda x: minmax_scale(x.astype(float)) if x.notnull().sum() else np.full_like(x, np.nan))
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

    # Scale 4b: apply quantile transformation on before scaling scores
    # tmp = fundamentals.melt(["ticker", "currency_code", "industry_code"], calculate_column)
    # tmp["quantile_transformed"] = tmp.dropna(subset=["value"]).groupby(groupby_col + ["variable"])["value"].transform(
    #     lambda x: QuantileTransformer(n_quantiles=4).fit_transform(
    #         x.values.reshape(-1, 1)).flatten() if x.notnull().sum() else np.full_like(x, np.nan))
    # tmp = tmp[["ticker", "variable", "quantile_transformed"]]
    # tmp["variable"] = tmp["variable"] + "_quantile_currency_code"
    # tmp = tmp.pivot(["ticker"], ["variable"]).droplevel(0, axis=1)
    # fundamentals = fundamentals.merge(tmp, how="left", on="ticker")

    # for currency not predicted by Factor Model -> Use factor of USD

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

    # Scale 5a: calculate ai_score by each currency_code (i.e. group) for each of [Quality, Value, Momentum]
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

    # Scale 5b: calculate ai_score by each currency_code (i.e. group) for [Extra]
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

    # upsert_data_to_database(pd.DataFrame(fundamentals_details_column_names).transpose().reset_index(),
    #                         f"test_fundamental_score_current_names_{weeks_to_expire}", how="replace")

    # # manual score check output to alibaba DB
    # for group, v in fundamentals_details.items():
    #     if group in ['HKD', 'USD']:
    #         pillar_df = []
    #         for pillar, df in v.items():
    #             pillar_df.append(df.set_index(['ticker']))
    #         pillar_df = pd.concat(pillar_df, axis=1)
    #         pillar_df.index = pillar_df.index.set_names(['index'])
    #         upsert_data_to_database(pillar_df.reset_index(),
    #                                 f"test_fundamental_score_details_{group}_{weeks_to_expire}", how="replace")

    print("Calculate ESG Value")
    esg_cols = ["environment_minmax_currency_code", "environment_minmax_industry", "social_minmax_currency_code",
                "social_minmax_industry", "governance_minmax_currency_code", "governance_minmax_industry"]
    fundamentals["esg"] = fundamentals[esg_cols].mean(1)

    print("Calculate AI Score")
    ai_score_cols = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra"]
    fundamentals["ai_score"] = fundamentals[ai_score_cols].mean(1)

    print("Calculate AI Score 2")
    ai_score_cols2 = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "esg"]
    fundamentals["ai_score2"] = fundamentals[ai_score_cols2].mean(1)

    print(fundamentals[["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra", 'esg']].describe())

    fundamentals_factors_scores_col = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra", "esg",
                                       "ai_score", "ai_score2", "wts_rating", "dlp_1m", "dlp_3m", "wts_rating2", "currency_code"]
    fundamentals[fundamentals_factors_scores_col] = fundamentals[fundamentals_factors_scores_col].round(1)

    return fundamentals.set_index('ticker')[fundamentals_factors_scores_col], fundamentals.set_index('ticker')[minmax_column]

def score_update_final_scale(fundamentals):
    ''' final scaling for ai_score '''

    # Scale 6: scale ai_score with history min / max
    # print(fundamentals.groupby(['currency_code'])[["ai_score", "ai_score2"]].agg(['min','mean','median','max']).transpose()[['HKD','USD','CNY','EUR']])
    fundamentals[["ai_score_unscaled", "ai_score2_unscaled"]] = fundamentals[["ai_score", "ai_score2"]]
    for cur, g in fundamentals.groupby(['currency_code']):
        try:
            score_history = get_universe_rating_history_score(backdate_by_year(1))
            score_history_cur = score_history.loc[score_history['currency_code'] == cur]
            # score_history_cur[["ai_score_unscaled", "ai_score2_unscaled"]] = score_history[["ai_score_unscaled", "ai_score2_unscaled"]]*1.1
            print(f'{cur} History Min/Max: ',
                  score_history_cur[["ai_score_unscaled", "ai_score2_unscaled"]].min().values,
                  score_history_cur[["ai_score_unscaled", "ai_score2_unscaled"]].max().values)
            print(f'{cur} Cur-bef Min/Max: ', g[["ai_score", "ai_score2"]].min().values,
                  g[["ai_score", "ai_score2"]].max().values)
            m1 = MinMaxScaler(feature_range=(0, 9)).fit(score_history_cur[["ai_score_unscaled", "ai_score2_unscaled"]])     # minmax -> (0, 9) to avoid overly large score
            fundamentals.loc[g.index, ["ai_score", "ai_score2"]] = m1.transform(g[["ai_score", "ai_score2"]])
            print(f'{cur} Cur-aft Min/Max: ', fundamentals.loc[g.index, ["ai_score", "ai_score2"]].min().values,
                  fundamentals.loc[g.index, ["ai_score", "ai_score2"]].max().values)
        except Exception as e:
            print(e)
            print(f'{cur} Current Min/Max: ', g[["ai_score", "ai_score2"]].min().values,
                  g[["ai_score", "ai_score2"]].max().values)
            fundamentals.loc[g.index, ["ai_score", "ai_score2"]] = MinMaxScaler(feature_range=(0, 10)).fit_transform(
                g[["ai_score", "ai_score2"]])
        fundamentals.loc[g.index, ["ai_score", "ai_score2"]] = MinMaxScaler(feature_range=(0, 10)).fit_transform(
            g[["ai_score", "ai_score2"]])

    fundamentals[['ai_score','ai_score2']] = fundamentals[['ai_score','ai_score2']].clip(0, 10)
    fundamentals[['ai_score','ai_score2',"esg"]] = fundamentals[['ai_score','ai_score2',"esg"]].round(1)

    return fundamentals.drop(columns=['currency_code'])

def score_update_return_adjustment(score):
    ''' adjust ai_score based on past week ai_score + past week return '''

    score = score.set_index(["ticker"])
    tri_0 = get_specific_tri(trading_day=dateNow()).set_index(["ticker"])[["tri"]]
    tri_1 = get_specific_tri(trading_day=backdate_by_week(1)).set_index(["ticker"])[["tri"]].rename(columns={"tri":"tri_1w"})
    df = pd.concat([score, tri_1, tri_0], axis=1)

    # calculate penalty = return * score
    df["ret"] = df["tri"]/df["tri_1w"] - 1
    df["penalty"] = df["ret"]*df["ai_score"]
    print(df["penalty"].describe())
    df["penalty"] = minmax_scale(df["penalty"], feature_range=(-1, .5))
    print(df["penalty"].describe())

    df["ai_score"] += df["penalty"]
    df["ai_score"] = np.clip(df["ai_score"], 0, 10)
    df["ai_score"] = df["ai_score"].round(2)
    return df.reset_index().drop(columns=["tri", "tri_1w", "ret"])

@update_ingestion_update_time(get_universe_rating_table_name())
@log2es("ai_score")
def update_universe_rating(ticker=None, currency_code=None):
    ''' Update ai_score -> universe_rating, universe_rating_history, universe_rating_detail_history '''

    # --------------------------------- Data Ingestion & Factor Calculation ---------------------------------------

    # Ingest 0: table formula for calculation
    factor_formula = get_factor_calculation_formula()       # formula for ratio calculation
    ingestion_source = get_ingestion_name_source()          # ingestion name & source

    # Ingest 1: DLPA & Universe
    print("{} : === Fundamentals Quality & Value Start Calculate ===".format(datetimeNow()))
    universe_rating = get_dlpa_rating_latest()

    # if DLPA results has problem will not using DLPA
    for col in ['dlp_1m', 'wts_rating']:
        if any(universe_rating[[col]].value_counts()/len(universe_rating) > .95):
            universe_rating[[col]] = np.nan

    # Ingest 2: fundamental score (Update: for mkt_cap/E/S/G only)
    print("=== Calculating Fundamentals Value & Fundamentals Quality ===")
    dsws_addition = get_dsws_addition_latest()
    dsws_addition = dsws_addition.filter(['ticker', 'mkt_cap','social','governance','environment'])
    print(dsws_addition)

    # Ingest 3: worldscope_summary & data_ibes_summary (Update: for mkt_cap/E/S/G only)
    print("=== Calculating Fundamentals Value & Fundamentals Quality ===")
    quarter_col = factor_formula.loc[factor_formula['name'].str[-3:]=='_1q', 'field_denom'].to_list()
    quarter_col = [i for x in quarter_col for i in x.split(' ')  if not any(['-' in i, '+' in i, '*' in i])]
    year_col = factor_formula.loc[factor_formula['name'].str[-4:]=='_1yr', 'field_denom'].to_list()
    year_col = [i for x in year_col for i in x.split(' ')  if not any(['-' in i, '+' in i, '*' in i])]
    ibes_col = ingestion_source.loc[ingestion_source['source']=='ibes','our_name'].to_list()
    ws_col = ingestion_source.loc[ingestion_source['source']=='ws','our_name'].to_list()
    ws_latest = get_worldscope_summary_latest(quarter_col=list(set(quarter_col) & set(ws_col)), year_col=list(set(year_col) & set(ws_col)))
    ws_latest = ws_latest.drop(columns=['mkt_cap'])     # use daily mkt_cap from fundamental_score
    print(ws_latest)
    ibes_latest = get_ibes_monthly_latest(quarter_col=list(set(quarter_col) & set(ibes_col)), year_col=list(set(year_col) & set(ibes_col)))
    print(ibes_latest)

    # Ingest 4.1: get last trading price for factor calculation
    close_price = get_last_close_industry_code(ticker=ticker, currency_code=currency_code)
    print(close_price)

    # Ingest 4.2: get volatility
    vol = score_update_vol_rs(list_of_start_end=[[0,1]])     # calculate RS volatility -> list_of_start_end in ascending sequence (start_month, end_month)
    print(vol)

    # Ingest 4.3: get skewness
    skew = score_update_skew(year=1)     # calculate RS volatility -> list_of_start_end in ascending sequence (start_month, end_month)
    print(skew)

    # Ingest 4.4: get different period stock return
    tri = score_update_stock_return(list_of_start_end_month=[[0,1],[2,6],[7,12]], list_of_start_end_week=[[0,1],[1,2],[2,4]])
    print(tri)

    #Ingest 4.5:  get last week average volume
    volume1 = get_specific_volume_avg(backdate_by_month(0), avg_days=7).set_index('ticker')
    volume2 = get_specific_volume_avg(backdate_by_month(0), avg_days=91).set_index('ticker')
    volume = (volume1/volume2).reset_index()
    print(volume)

    # Ingest 5: get earning_prediction from ai_value
    pred_mean = get_ai_value_pred_final()
    print(pred_mean)

    # merge scores used for calculation
    dsws_addition = close_price.merge(dsws_addition, how="left", on="ticker")
    dsws_addition = dsws_addition.merge(ws_latest, how="left", on="ticker")
    dsws_addition = dsws_addition.merge(ibes_latest, how="left", on="ticker")
    dsws_addition = dsws_addition.merge(vol, how="left", on="ticker")
    dsws_addition = dsws_addition.merge(skew, how="left", on="ticker")
    dsws_addition = dsws_addition.merge(pred_mean, how="left", on="ticker")
    dsws_addition = dsws_addition.merge(tri, how="left", on="ticker")
    dsws_addition = dsws_addition.merge(volume, how="left", on="ticker")

    # calculate ratios refering to table X
    dsws_addition = score_update_factor_ratios(dsws_addition, factor_formula, ingestion_source)

    # ------------------------------------ Factor Score Scaling ------------------------------------------

    factor_formula = factor_formula.set_index('name')
    calculate_column = list(factor_formula.loc[factor_formula["scaler"].notnull()].index)
    calculate_column = sorted(set(calculate_column))
    calculate_column += ["environment", "social", "governance"]

    fundamentals = dsws_addition[["ticker", "currency_code", "industry_code"] + calculate_column]
    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan).copy()

    # add DLPA scores
    fundamentals = fundamentals.merge(universe_rating, on="ticker", how="left")
    print(fundamentals)

    # Scale original fundamental score
    universe_currency_code = get_active_universe()['currency_code'].unique()
    fundamentals_factors_scores_col_diff = ["fundamentals_value", "fundamentals_quality", "fundamentals_momentum", "fundamentals_extra", "ai_score", "ai_score2"]
    fundamentals_1w, fundamentals_details_1w = score_update_scale(fundamentals, calculate_column, universe_currency_code, factor_formula, weeks_to_expire=4)
    fundamentals_1w = fundamentals_1w[fundamentals_factors_scores_col_diff]  # for only scores
    fundamentals_1m, fundamentals_details_1m = score_update_scale(fundamentals, calculate_column, universe_currency_code, factor_formula, weeks_to_expire=1)

    # details history table = average score of 1w + 1m
    universe_rating_detail_history = (fundamentals_details_1w + fundamentals_details_1m)/2

    fundamentals = fundamentals_1w.merge(fundamentals_1m, left_index=True, right_index=True, suffixes=('_weekly1','_monthly1'))
    for i in ['ai_score', 'ai_score2']:  # currently we use simple average of weekly score & monthly score
        fundamentals[i] = (fundamentals[i+'_weekly1'] + fundamentals[i+'_monthly1'])/2
    fundamentals = fundamentals.reset_index()
    fundamentals = score_update_final_scale(fundamentals)
    fundamentals = score_update_return_adjustment(fundamentals)

    print("=== Calculate Fundamentals Value & Fundamentals Quality DONE ===")
    if(len(fundamentals)) > 0 :
        print(fundamentals)

        # update universe_rating
        fundamentals["trading_day"] = dateNow()
        upsert_data_to_database(fundamentals[["ticker", "ai_score", "ai_score2", "trading_day"]], get_universe_rating_table_name(), "ticker", how="update", Text=True)
        delete_data_on_database(get_universe_rating_table_name(), f"ticker is not null", delete_ticker=True)

        # update universe_rating_history
        fundamentals = pd.melt(fundamentals, id_vars=["ticker", "trading_day"], var_name="field", value_name="value").dropna(how="any")
        fundamentals = uid_maker_field(fundamentals, uid="uid", ticker="ticker", trading_day="trading_day", field="field")
        upsert_data_to_database(fundamentals, get_universe_rating_history_table_name(), "uid", how="update", Text=True)

        # update universe_rating_detail_history
        universe_rating_detail_history["trading_day"] = dateNow()
        universe_rating_detail_history = pd.melt(universe_rating_detail_history.reset_index(), id_vars=["ticker", "trading_day"],
                                                 var_name="field", value_name="value").dropna(how="any")
        universe_rating_detail_history = uid_maker_field(universe_rating_detail_history, uid="uid", ticker="ticker", trading_day="trading_day", field="field")
        upsert_data_to_database(universe_rating_detail_history, get_universe_rating_detail_history_table_name(), "uid", how="update", Text=True)

        # update positive negative factor based on universe_rating
        update_positive_negative_factor()
        # update_top_universe_rating()

        # delete_data_on_database(get_universe_rating_history_table_name(), f"ticker is not null", delete_ticker=True)
        # delete_data_on_database(get_universe_rating_detail_history_table_name(), f"ticker is not null", delete_ticker=True)
        report_to_slack("{} : === Universe Fundamentals Quality & Value Updated ===".format(datetimeNow()))

# ================================== Update universe_positive/negative_factor =======================================

def factor_column_name_changes():
    ''' map factor name used in DB to name shown on APP '''

    name_df = get_factor_calculation_formula()
    name_df['index'] = np.where(name_df['scaler'].notnull(),
                                [f"{x}_{y}_currency_code" for x, y in name_df[["name","scaler"]].to_numpy()],
                                name_df["name"].values)
    name_dict = name_df.set_index('index')['app_name'].to_dict()
    esg_name = esg_factor_name_change()
    name_dict.update(esg_name)
    return name_dict

def esg_factor_name_change():
    return {"environment_minmax_currency_code": "Environment",
            "environment_minmax_industry": "Environment (ind.)",
            "social_minmax_currency_code": "Social",
            "social_minmax_industry": "Social (ind.)",
            "goverment_minmax_currency_code": "Goverment",
            "goverment_minmax_industry": "Goverment (ind.)"}

def update_positive_negative_factor():
    ''' update positive/negative factors -> table '''

    universe_rating_positive_negative = pd.DataFrame({"ticker":[], "positive_factor":[], "negative_factor":[]}, index=[])
    name_map = factor_column_name_changes()

    # factor currently used in ai_score calculation
    factor_use = get_factor_current_used().transpose().to_dict(orient='list')
    for cur, v in factor_use.items():   # work on different currency separately -> can use different factors
        lst = [x for x in ','.join(v).split(',') if len(x) > 0]
        lst = list(set(lst))
        lst += [x + '_minmax_currency_code' for x in lst]

        # select dataframe for given industry
        curr_details = get_latest_universe_rating_detail_history(currency_code=cur)
        curr_details = curr_details.pivot(index=["ticker"], columns=["field"], values="value")

        # unstack dataframe
        cols = list(set(lst) & set(curr_details.columns.to_list()))
        curr_details = curr_details[cols].unstack().reset_index()
        curr_details.columns = ["factor_name", "ticker", "score"]
        curr_details["factor_name"] = curr_details["factor_name"].map(name_map)

        # rules for positive / negative factors
        curr_details['score'] = pd.to_numeric(curr_details['score'])
        curr_details = curr_details.dropna(how='any')
        curr_des = curr_details.groupby(['factor_name'])['score'].agg(['mean','std'])
        curr_pos = (curr_des['mean'] + 0.4*curr_des['std']).to_dict()       # positive = factors > mean + 0.4std
        curr_neg = (curr_des['mean'] - 0.4*curr_des['std']).to_dict()       # negative = factors < mean - 0.4std

        for tick in curr_details['ticker'].unique():
            positive_factor = []
            negative_factor = []
            temp = curr_details.loc[curr_details["ticker"] == tick]

            positive = temp.loc[temp["score"] > temp["factor_name"].map(curr_pos)]
            positive = positive.sort_values(by=["score"], ascending=False).head(5)
            # if len(positive) == 0:
            #     positive = temp.nlargest(1,'score')     # if no factor > mean + 0.4*std -> use highest score one as pos

            negative = temp.loc[temp["score"] < temp["factor_name"].map(curr_neg)]
            negative = negative.sort_values(by=["score"]).head(5)
            # if len(negative) == 0:
            #     positive = temp.nsmallest(1,'score')

            for index, row in positive.iterrows():
                positive_factor.append(row["factor_name"])      # positive/negative only first 5 factor
            for index, row in negative.iterrows():
                negative_factor.append(row["factor_name"])
            positive_negative_result = pd.DataFrame({"ticker":[tick], "positive_factor":[positive_factor], "negative_factor":[negative_factor]}, index=[0])
            universe_rating_positive_negative = universe_rating_positive_negative.append(positive_negative_result)

    universe_rating_positive_negative["trading_day"] = dateNow()
    upsert_data_to_database(universe_rating_positive_negative, get_universe_rating_posneg_factor_table_name(), "ticker",
                            how="update", Text=True)

    # testing: What tickers has no postive/negative tickers?
    for cur in ['USD', 'HKD']:
        universe_rating = get_universe_rating(currency_code=cur)[["ticker", "ai_score"]]
        df_cur = universe_rating_positive_negative.merge(universe_rating, on=["ticker"], how="inner")
        for i in ['positive_factor', 'negative_factor']:
            df = df_cur.loc[df_cur[i].astype(str) == '[]']
            report_to_slack("*{} : === [{}] without {}: {}/{} ===*".format(dateNow(), cur, i, len(df), len(df_cur)))
            report_to_slack('```'+', '.join(["{0:<10}: {1:<5}".format(x,y) for x, y in df[['ticker', 'ai_score']].values])+'```')

def update_top_universe_rating():
    ''' calculate weekly return for weekly pick of top score '''

    start_date = backdate_by_week(16)

    # get top 10 pick for each currency each day
    df = get_universe_rating_history_score(start_date=start_date, column="ai_score_unscaled")
    df = df.rename(columns={"ai_score_unscaled": "ai_score"}).sort_values(by=["ai_score"]).dropna(how='any')
    df = df.groupby(["currency_code", "trading_day"]).tail(20)
    df["trading_day"] = pd.to_datetime(df["trading_day"])

    # only audit on HKD, USD, EUR & Monday's Pick
    df = df.loc[(df["trading_day"].dt.dayofweek==1) & (df["currency_code"].isin(["HKD","USD","EUR"]))]

    # calculate weekly return from TRI
    index_ticker = ['.SPX', '.HSI', '.SXXGR']
    tickers = df["ticker"].to_list() + index_ticker
    tri = get_data_tri(ticker=tickers, start_date=start_date)
    tri['trading_day'] = pd.to_datetime(tri['trading_day'])
    tri = tri.pivot(index=["trading_day"], columns=["ticker"], values="total_return_index")
    tri = tri.resample('D').pad()
    tri = (tri.shift(-7)/tri - 1).stack().dropna(how="all").reset_index().rename(columns={0: "return"})
    for i in range(4):
        tri[f"ret_w{i+1}"] = tri.groupby(["ticker"])["return"].shift(-i*7).values

    # index return columns
    tri_index = tri.loc[tri["ticker"].isin(index_ticker)]
    tri_index["ticker"] = tri_index["ticker"].replace({'.SPX': "USD", '.HSI': "HKD", '.SXXGR': "EUR"})
    ret_col = [f"ret_w{i+1}" for i in range(4)]
    tri_index["ret_cumulative_mkt"] = np.round((tri_index[ret_col]+1).fillna(1).prod(axis=1) - 1, 4)
    tri_index = tri_index.rename(columns={"ticker": "currency_code"})
    tri_index = tri_index[["trading_day", "currency_code", "ret_cumulative_mkt"]]

    # map return to top picks
    df = df.merge(tri, on=["ticker", "trading_day"], how="left")
    df = uid_maker(df)

    # calculate mean return for top 10 picks
    df_mean = df.groupby(["currency_code", "trading_day"]).mean().reset_index()
    df_mean = df_mean.merge(tri_index, on=["trading_day", "currency_code"], how="left")
    df_mean = uid_maker(df_mean, ticker="currency_code")

    # add mean return for each currency
    df = df.append(df_mean).sort_values(by=["trading_day"]).drop(columns=["return"])
    ret_col = [f"ret_w{i+1}" for i in range(4)]
    df["ret_cumulative"] = (df[ret_col]+1).fillna(1).prod(axis=1) - 1
    df[ret_col+["ret_cumulative"]] = np.round(df[ret_col+["ret_cumulative"]], 4)
    df = df.dropna(subset=ret_col, how="all")

    # count n positive / negative amount top picks
    def count_pos_neg(g):
        s1 = (g[["ret_w1", "ret_w2", "ret_cumulative"]] > 0).sum(axis=0)
        return s1
    df_count = df.loc[df["ticker"].notnull()]
    df_count = df_count.groupby(["currency_code", "trading_day"]).apply(count_pos_neg)
    df_count.columns = [x+'_positive' for x in df_count.columns.to_list()]
    df = df.merge(df_count, left_on=["currency_code", "trading_day"], right_index=True, how='left')
    df = df.drop(columns=["ret_w3", "ret_w4"])

    upsert_data_to_database(df, table="universe_rating_history_top", how="update")  # write to DB

if __name__ == '__main__':
    # update_universe_rating()
    update_top_universe_rating()