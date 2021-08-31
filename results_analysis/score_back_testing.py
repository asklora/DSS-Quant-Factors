from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import robust_scale, minmax_scale, quantile_transform
from preprocess.premium_calculation import trim_outlier
import global_vals
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

def score_history():
    ''' calculate score with DROID v2 method & evaluate '''

    with global_vals.engine.connect() as conn, global_vals.engine_ali.connect() as conn_ali:  # write stock_pred for the best hyperopt records to sql
        factor_formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}_prod', conn_ali)
        factor_rank = pd.read_sql(f'SELECT * FROM {global_vals.production_factor_rank_table}_history', conn_ali)
        universe = pd.read_sql(f'SELECT * FROM {global_vals.dl_value_universe_table}', conn)
        fundamentals_score = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}_weekavg "
                                         f"WHERE (period_end>'2017-08-30') AND (ticker not like '.%%') ", conn_ali)
        # pred_mean = pd.read_sql(f"SELECT * FROM ai_value_lgbm_pred_final_eps", conn_ali)
    global_vals.engine_ali.dispose()

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
    for (group_name, period_end), g in factor_rank.groupby(['group','period_end']):
        neg_factor = g.loc[(factor_rank['long_large'] == False),'factor_name'].to_list()
        fundamentals_score.loc[(fundamentals_score['period_end']==period_end)&(fundamentals_score['currency_code']==group_name), neg_factor] *= -1

    fundamentals = fundamentals_score[['ticker','period_end','currency_code','stock_return_y']+calculate_column]
    fundamentals = fundamentals.replace([np.inf, -np.inf], np.nan).copy()
    print(fundamentals)

    # trim outlier to +/- 2 std
    calculate_column_score = []
    for column in calculate_column:
        try:
            if fundamentals[column].notnull().sum():
                column_score = column + "_score"
                fundamentals[column_score] = fundamentals.groupby(["currency_code",'period_end'])[column].transform(
                    lambda x: np.clip(x, np.nanmean(x) - 2 * np.nanstd(x), np.nanmean(x) + 2 * np.nanstd(x)))
                calculate_column_score.append(column_score)
        except Exception as e:
            print(e)
            continue
    print(calculate_column_score)

    # apply robust scaler
    calculate_column_robust_score = []
    for column in calculate_column:
        try:
            column_score = column + "_score"
            column_robust_score = column + "_robust_score"
            fundamentals[column_robust_score] = fundamentals.groupby(["currency_code",'period_end'])[column_score].transform(
                lambda x: robust_scale(x))
            calculate_column_robust_score.append(column_robust_score)
        except Exception as e:
            print(e)

    # apply maxmin scaler on Currency / Industry
    for column in calculate_column:
        try:
            column_robust_score = column + "_robust_score"
            column_minmax_currency_code = column + "_minmax_currency_code"
            df_currency_code = fundamentals[["currency_code", column_robust_score, 'period_end']]
            df_currency_code = df_currency_code.rename(columns={column_robust_score: "score"})
            fundamentals[column_minmax_currency_code] = df_currency_code.groupby(["currency_code",'period_end']).score.transform(
                lambda x: minmax_scale(x.astype(float)) if x.notnull().sum() else np.full_like(x, np.nan))
            fundamentals[column_minmax_currency_code] = np.where(fundamentals[column_minmax_currency_code].isnull(), 0.4, fundamentals[column_minmax_currency_code])
        except Exception as e:
            print(e)

    # apply quantile transformation on before scaling scores
    try:
        tmp = fundamentals.melt(['ticker', 'currency_code'], calculate_column)
        tmp['quantile_transformed'] = tmp.groupby(['currency_code', 'variable','period_end'])['value'].transform(
            lambda x: quantile_transform(x.values.reshape(-1, 1), n_quantiles=4).flatten() if x.notnull().sum() else np.full_like(x, np.nan))
        tmp = tmp[['ticker', 'variable', 'quantile_transformed']]
        tmp['variable'] = tmp['variable'] + '_quantile_currency_code'
        tmp = tmp.pivot(['ticker'], ['variable']).droplevel(0, axis=1)
        fundamentals = fundamentals.merge(tmp, how='left', on='ticker')
    except Exception as e:
        print(e)

    # plot min/max distribution
    n = round(len(calculate_column)**0.5)+1
    for name, g in fundamentals.groupby('currency_code'):
        fig = plt.figure(figsize=(n*4, n*4), dpi=120, constrained_layout=True)
        k=1
        for col in calculate_column:
            ax = fig.add_subplot(n, n, k)
            try:
                ax.hist(g[col+"_minmax_currency_code"], bins=20)
            except:
                pass
            ax.set_xlabel(f'{col}')
            k+=1
        fig.savefig(f'minmax_{name}.png')
        plt.close(fig)

    # add column for 3 pillar score
    fundamentals[[f"fundamentals_{name}" for name in factor_rank['pillar'].unique()]] = np.nan
    fundamentals[['dlp_1m', 'wts_rating','earnings_pred_minmax_currency_code','revenue_pred_minmax_currency_code']] = np.nan  # ignore ai_value / DLPA

    # calculate ai_score by each currency_code (i.e. group) for each of 3 pillar
    for (group, pillar_name), g in factor_rank.groupby(['group', 'pillar']):
        print(f"Calculate Fundamentals [{pillar_name}] in group [{group}]")
        sub_g = g.loc[(g['factor_weight'] == 2) | (g['factor_weight'].isnull())]  # use all rank=2 (best class)
        if len(sub_g) == 0:  # if no factor rank=2, use the highest ranking one & DLPA/ai_value scores
            sub_g = g.loc[g.nlargest(1, columns=['pred_z']).index.union(g.loc[g['factor_weight'].isnull()].index)]

        score_col = [f'{x}_{y}_currency_code' for x, y in
                     sub_g.loc[sub_g['scaler'].notnull(), ['factor_name', 'scaler']].to_numpy()]
        score_col += [x for x in sub_g.loc[sub_g['scaler'].isnull(), 'factor_name']]
        fundamentals.loc[fundamentals['currency_code'] == group, f"fundamentals_{pillar_name}"] = fundamentals[
            score_col].mean(axis=1)

    # calculate ai_score by each currency_code (i.e. group) for "Extra" pillar
    for group, g in factor_rank.groupby('group'):
        print(f"Calculate Fundamentals [extra] in group [{group}]")
        sub_g = g.loc[(g['factor_weight'] == 2) & (g['pred_z'] >= 1)]  # use all rank=2 (best class) and predicted factor premiums with z-value >= 1

        if len(sub_g) > 0:  # if no factor rank=2, don't add any factor into extra pillar
            score_col = [f'{x}_{y}_currency_code' for x, y in sub_g.loc[sub_g['scaler'].notnull(), ['factor_name', 'scaler']].to_numpy()]
            fundamentals.loc[fundamentals['currency_code'] == group, f'fundamentals_extra'] = fundamentals[score_col].mean(axis=1)
        else:
            fundamentals.loc[fundamentals['currency_code'] == group, f'fundamentals_extra'] = 0.

    fundamentals_factors_scores_col = fundamentals.filter(regex='^fundamentals_').columns
    fundamentals[fundamentals_factors_scores_col] = (fundamentals[fundamentals_factors_scores_col] * 10).round(1)

    print("Calculate AI Score")
    fundamentals["ai_score"] = (fundamentals["fundamentals_value"] + fundamentals["fundamentals_quality"] + \
                                fundamentals["fundamentals_momentum"] + fundamentals["fundamentals_extra"]) / 4

    # plot score distribution
    fig = plt.figure(figsize=(12, 20), dpi=120, constrained_layout=True)
    k=1
    for col in list(fundamentals_factors_scores_col) + ['ai_score']:
        for cur in ['USD','HKD','EUR']:
            ax = fig.add_subplot(5, 3, k)
            df = fundamentals.loc[fundamentals['currency_code']==cur, col]
            ax.hist(df, bins=20)
            if k % 3 == 1:
                ax.set_ylabel(col, fontsize=20)
            if k > (5-1)*3:
                ax.set_xlabel(cur, fontsize=20)
            k+=1
    fig.savefig(f'score_dist.png')
    plt.close()

    score_col = ['ai_score','fundamentals_value','fundamentals_quality','fundamentals_momentum','fundamentals_extra']
    label_col = ['ticker', 'period_end', 'currency_code', 'stock_return_y']

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
        fundamentals[label_col + score_col].to_sql('data_fundamental_score_history_testing', **extra)
    global_vals.engine_ali.dispose()

    score_eval()

def score_eval(name=''):
    ''' evaluate score history with 1) descirbe, 2) score 10-qcut mean ret, 3) per period change '''
    score_col = ['ai_score','fundamentals_value','fundamentals_quality','fundamentals_momentum','fundamentals_extra']
    writer = pd.ExcelWriter(f'score_eval_{name}.xlsx')

    with global_vals.engine_ali.connect() as conn:
        fundamentals = pd.read_sql(f"SELECT ticker, period_end, currency_code, stock_return_y, {', '.join(score_col)} FROM data_fundamental_score_history_clair", conn)
    global_vals.engine_ali.dispose()

    # 1. Score describe
    for name, g in fundamentals.groupby(['currency_code']):
        df = g.describe().transpose()
        df['std'] = df.std()
        df.to_excel(writer, sheet_name=f'describe_{name}')

    # 2. Test 10-qcut return
    def score_ret_mean(score_col='ai_score', group_col='currency_code'):
        if group_col=='':
            fundamentals['currency_code'] = 'cross'
            group_col = 'currency_code'
        fundamentals['score_qcut'] = fundamentals.groupby([group_col])[score_col].transform(lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop'))
        mean_ret = pd.pivot_table(fundamentals, index=[group_col], columns=['score_qcut'], values='stock_return_y')
        mean_ret['count'] = fundamentals.groupby([group_col])[score_col].count()
        return mean_ret.transpose()

    for i in score_col:
        score_ret_mean(i).to_excel(writer, sheet_name=f'ret_{i}')
    score_ret_mean('ai_score','').to_excel(writer, sheet_name='ret_cross')

    # 3. Ticker Score Single-Period Change
    fundamentals = fundamentals.sort_values(by=['ticker','period_end'])
    fundamentals['ai_score_last'] = fundamentals.groupby(['ticker'])['ai_score'].shift(1)
    fundamentals['ai_score_change'] = fundamentals['ai_score'] - fundamentals['ai_score_last']
    df = fundamentals.groupby(['ticker'])['ai_score_change'].agg(['mean','std'])
    df.to_excel(writer, sheet_name='score_change')

    writer.save()

if __name__ == "__main__":
    score_history()
    # score_eval()
