from sqlalchemy import text
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
import numpy as np
import os
import re

import global_vals

model = 'lgbm'
r_name = 'newlastweekavg_pca'

iter_name = r_name

def download_stock_pred(plot_dist=True):
    ''' download training history and training prediction from DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.cv_number, S.y_type FROM {global_vals.result_pred_table}_{model}_reg P "
                     f"INNER JOIN {global_vals.result_score_table}_{model}_reg S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number','group'], keep='last')
    result_all = result_all.drop(['cv_number'], axis=1)
    result_all_comb = result_all.groupby(['group_code', 'testing_period', 'y_type', 'group'])[['pred','actual']].mean()
    # result_all_comb.columns = ['median','actual']
    # result_all_comb['mean'] = result_all.groupby(['group_code', 'testing_period', 'y_type', 'group'])['pred'].mean()

    fig = plt.figure(figsize=(16, 8), dpi=120, constrained_layout=True)  # create figure for test & train boxplot

    k=1
    if plot_dist:
        for name, g in result_all_comb.groupby(['group']):
            ax = fig.add_subplot(2, 3, k)
            ax.hist(g, bins=10)
            ax.set_ylabel(name, fontsize=20)
            if k == 1:
                plt.legend(['pred','actual'])
            k+=1
        plt.savefig(f'score/{model}_pred_reg_mean_{iter_name}.png')
        # plt.show()

    corr_dict = {}
    for name, g in result_all_comb.groupby(['group']):
        corr_dict[name] = g[['pred','actual']].corr().iloc[0,1]

    corr_group = pd.DataFrame(corr_dict, index=[0]).transpose()
    print(corr_group)

    return result_all_comb.reset_index(), corr_group

def combine_mode_group(df):
    ''' calculate accuracy score by each industry/currency group '''

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'group']):
        result_dict[name] = {}
        result_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        result_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        result_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict).transpose().reset_index()
    df.columns = ['group_code', 'y_type','group'] + df.columns.to_list()[3:]

    with global_vals.engine_ali.connect() as conn:
        icb_name = pd.read_sql(f"SELECT DISTINCT code_6 as group, name_6 as name FROM icb_code_explanation", conn)  # download training history
        icb_count = pd.read_sql(f"SELECT \"group\", avg(num_ticker) as num_ticker FROM icb_code_count GROUP BY \"group\"", conn)  # download training history
    global_vals.engine_ali.dispose()

    df = df.merge(icb_name, on=['group'], how='left')
    df = df.merge(icb_count, on=['group'], how='left')

    return df

def combine_mode_time(df):
    ''' calculate accuracy score by each industry/currency group '''

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'testing_period']):
        result_dict[name] = {}
        result_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        result_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        result_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict).transpose().reset_index()
    df.columns = ['group_code', 'y_type','testing_period'] + df.columns.to_list()[3:]

    return df

def calc_performance(df, accu_df, plot_performance_all=True, q=1/3):
    ''' calculate accuracy score by each industry/currency group '''

    # test on factor 'vol_30_90' first
    df = df.loc[df['y_type']=='vol_0_30']

    df['pred_low'] = df.groupby(['group_code', 'y_type', 'testing_period'])['pred'].transform(np.nanquantile, q=q)
    df['pred_high'] = df.groupby(['group_code', 'y_type', 'testing_period'])['pred'].transform(np.nanquantile, q=1-q)
    df['actual_low'] = df.groupby(['group_code', 'y_type', 'testing_period'])['actual'].transform(np.nanquantile, q=q)
    df['actual_high'] = df.groupby(['group_code', 'y_type', 'testing_period'])['actual'].transform(np.nanquantile, q=1-q)

    df[['pred_cut','actual_cut']] = 1
    df.loc[df['pred']<df['pred_low'], 'pred_cut'] = 0
    df.loc[df['pred']>df['pred_high'], 'pred_cut'] = 2
    df.loc[df['actual']<df['actual_low'], 'actual_cut'] = 0
    df.loc[df['actual']>df['actual_high'], 'actual_cut'] = 2

    # 1. calculate return per our prediction & actual class
    df_list = []
    for i in ['actual_cut','pred_cut']:
        r = pd.pivot_table(df, index=['group_code', 'y_type', 'testing_period'], columns=[i], values='actual', aggfunc='mean')
        r.columns = [f'{i}_{x}' for x in r.columns.to_list()]
        df_list.append(r)

    results = pd.concat(df_list, axis=1).reset_index().sort_values(['group_code', 'y_type', 'testing_period'])
    results['average'] = df.groupby(['group_code', 'y_type', 'testing_period'])['actual'].mean().sort_index().values

    # 2. add accuracy for each (group_code, y_type, testing_period)
    results = results.merge(accu_df, on=['group_code', 'y_type', 'testing_period'], how='left')

    # 4. plot performance with plt.plot again index
    cols = r.columns.to_list() + ['average']
    if plot_performance_all:
        # 4.2 - plot cumulative return for entire testing period
        k=1
        fig = plt.figure(figsize=(8, 8), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
        for part_name, g in results.groupby(['group_code']):
            ax = fig.add_subplot(2,1,k)

            # add index benchmark
            g[cols] = np.cumprod(g[cols] + 1, axis=1)
            ax.plot(g.set_index(['testing_period'])[cols])        # plot cumulative return for the year
            plt.ylim((0.8,2))
            ax.set_ylabel(part_name, fontsize=20)
            if k == 1:
                plt.legend(cols, loc='upper left', fontsize='large')
            k+=1
        # plt.show()
        plt.savefig(f'score/{model}_performance_{iter_name}.png')

    return results

def calc_pred_class():
    ''' Calculte the accuracy_score if combine CV by mean, median, mode '''

    df, corr_group = download_stock_pred()
    df = df.dropna(how='any')

    writer = pd.ExcelWriter(f'score/{model}_pred_reg_mean_{iter_name}.xlsx')

    result_time = combine_mode_time(df)
    result_performance = calc_performance(df, result_time)
    result_group = combine_mode_group(df)

    corr_group.to_excel(writer, sheet_name='corr_group')
    result_time.groupby(['group_code', 'y_type']).mean().to_excel(writer, sheet_name='average')
    result_group.to_excel(writer, sheet_name=f'group_all', index=False)
    pd.pivot_table(result_group, index=['group'], columns=['y_type'], values=['mae','mse','r2']).to_excel(writer,sheet_name='group_pivot')
    result_time.to_excel(writer, sheet_name=f'time_all', index=False)
    pd.pivot_table(result_time, index=['testing_period'], columns=['y_type'], values=['mae','mse','r2']).to_excel(writer,sheet_name='time_pivot')
    result_performance.to_excel(writer, sheet_name='performance', index=False)

    writer.save()

    # compare_all_similar_xls()

def compare_all_similar_xls():
    df_list = []
    for f in os.listdir('./score'):
        if ('_pred_reg_mean_' in f) and ('.xlsx' in f):
            df = pd.read_excel(open('score/' + f, 'rb'), sheet_name='average')
            df['name_sql'] = f
            df_list.append(df)

    all_df = pd.concat(df_list, axis=0)
    print(all_df)

if __name__ == "__main__":

    calc_pred_class()
    # compare_all_similar_xls()

