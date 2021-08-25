from sqlalchemy import text
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
import numpy as np
import os
import re
import seaborn as sns

import global_vals

model = 'rf_reg'
r_name = 'pca_top16_q3_mse_rerun_tv3'

def download_stock_pred(q, iter_name, save_xls=True, save_plot=True):
    ''' download training history and training prediction from DB '''

    if 'lgbm' in model:
        y_type = 'S.y_type'
    else:
        y_type = 'P.y_type'

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual_exact as actual, {y_type}, P.group as group_code, S.testing_period, S.cv_number, CONCAT(S.tree_type,'',S.use_pca) as alpha "
                     f"FROM {global_vals.result_pred_table}_{model} P "
                     f"INNER JOIN {global_vals.result_score_table}_{model} S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{iter_name}%' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'alpha', 'cv_number'], keep='last')

    if 'class' in model:
        result_all_final = pd.DataFrame(result_all.groupby(['group_code', 'testing_period', 'y_type', 'alpha']).apply(pd.DataFrame.mode)['pred'].dropna().reset_index(level=-1, drop=True))
        result_all_final['actual'] = result_all.groupby(['group_code', 'testing_period', 'y_type', 'alpha'])['actual'].mean()
        result_all = result_all_final.reset_index()
    else:
        result_all = result_all.groupby(['group_code', 'testing_period', 'y_type', 'alpha']).mean().reset_index()

    result_all_avg = result_all.groupby(['testing_period','group_code'])['actual'].mean().reset_index()

    # plt.plot(result_all.groupby(['pred'])['actual'].mean())
    # plt.show()
    # exit(1)

    corr_dict = {}
    for name, g in result_all.groupby([ 'alpha', 'y_type', 'group_code']):
        corr_dict[name] = g[['pred','actual']].corr().iloc[0, 1]
    corr_df = pd.DataFrame(corr_dict, index=[0]).stack(level=-1).reset_index(level=0, drop=True).transpose().reset_index()

    ret_dict = {}
    for name, g in result_all.groupby(['group_code', 'testing_period', 'alpha']):
        ret_dict[name] = {}
        if 'class' in model:
            max_g = g.loc[g['pred']==g['pred'].max()]
            min_g = g.loc[g['pred']==g['pred'].min()]
        elif q<1:
            max_g = g.loc[g['pred']>g['pred'].quantile(q=1-q)]
            min_g = g.loc[g['pred']<g['pred'].quantile(q=q)]
        else:
            max_g = g.loc[g['pred']>g['pred'].nlargest(q).iloc[-1]]
            min_g = g.loc[g['pred']<g['pred'].nsmallest(q).iloc[-1]]
        ret_dict[name]['max_factor'] = ','.join(list(max_g['y_type'].values))
        ret_dict[name]['min_factor'] = ','.join(list(min_g['y_type'].values))
        ret_dict[name]['max_ret'] = max_g['actual'].mean()
        ret_dict[name]['min_ret'] = min_g['actual'].mean()
        ret_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        ret_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        ret_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    result_all_comb = pd.DataFrame(ret_dict).transpose().reset_index()
    result_all_comb.columns = ['group_code', 'testing_period', 'alpha'] + result_all_comb.columns.to_list()[3:]
    result_all_comb[['max_ret','min_ret','mae','mse','r2']] = result_all_comb[['max_ret','min_ret','mae','mse','r2']].astype(float)

    result_all_comb = result_all_comb.merge(result_all_avg, on=['group_code', 'testing_period'])
    print(result_all_comb.groupby(['group_code', 'alpha']).mean())

    if save_xls:
        writer = pd.ExcelWriter(f'score/#{model}_pred_{iter_name}.xlsx')
        result_all_comb.groupby(['group_code','alpha']).mean().to_excel(writer, sheet_name='average')
        corr_df.to_excel(writer, sheet_name='corr')
        result_all_comb.to_excel(writer, sheet_name='group_time', index=False)
        pd.pivot_table(result_all, index=['alpha', 'group_code', 'testing_period'], columns=['y_type'], values=['pred','actual']).to_excel(writer, sheet_name='all')
        writer.save()

    if save_plot:
        num_alpha = len(result_all_comb['alpha'].unique())
        num_group = len(result_all_comb['group_code'].unique())
        fig = plt.figure(figsize=(num_group*8, num_alpha*4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
        k=1
        for name, g in result_all_comb.groupby(['group_code']):
            ax = fig.add_subplot(num_alpha, num_group, k)
            g[['max_ret','actual','min_ret']] = np.cumprod(g[['max_ret','actual','min_ret']] + 1, axis=0)
            plot_df = g.set_index(['testing_period'])[['max_ret','actual','min_ret']]
            ax.plot(plot_df)
            for i in range(3):
                ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
            if k % num_group ==1:
                ax.set_ylabel(name[0], fontsize=20)
            if k > num_group*num_alpha-num_group:
                ax.set_xlabel(name[1], fontsize=20)
            if k==1:
                plt.legend(['best','average','worse'])
            k+=1
        plt.savefig(f'score/#{model}_pred_{iter_name}.png')
        plt.close()

    return result_all_comb.groupby(['group_code','alpha']).mean()

# iter_name = 'pca_mse_allx'
def download_stock_pred_multi(iter_name, save_xls=True, plot_consol=True):
    ''' download training history and training prediction from DB '''

    if model == 'rf_reg':
        y_type = 'P.y_type'
    elif model == 'lgbm':
        y_type = 'S.y_type'

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, {y_type}, P.group as group_code, S.testing_period, S.cv_number, S.tree_type as alpha, S.name_sql, S.neg_factor "
                     f"FROM {global_vals.result_pred_table}_{model} P "
                     f"INNER JOIN {global_vals.result_score_table}_{model} S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{iter_name}%' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    df = result_all.drop_duplicates(subset=['name_sql', 'group_code', 'testing_period', 'y_type', 'alpha'], keep='last')
    # df['pred_rank'] = df.groupby(['group_code', 'testing_period', 'alpha', 'iter'])['pred'].rank().values

    # list out negative premiums
    df_neg = df[['group_code', 'testing_period', 'neg_factor']].drop_duplicates()
    all_neg_factor = list(set([i for x in df['neg_factor'].unique() for i in x.split(',')]))
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer().fit(all_neg_factor)
    df_neg[all_neg_factor] = df_neg['neg_factor'].apply(lambda x: lb.transform(x.split(',')).sum(axis=0)).values.tolist()
    df_neg = df_neg.drop(['neg_factor'], axis=1)

    result_all = df.groupby(['group_code', 'testing_period',  'y_type', 'alpha']).mean().reset_index()
    result_all_avg = result_all.groupby(['testing_period','group_code'])['actual'].mean().reset_index()

    corr_dict = {}
    for name, g in result_all.groupby([ 'alpha', 'y_type', 'group_code']):
        corr_dict[name] = g[['pred','actual']].corr().iloc[0, 1]
    corr_df = pd.DataFrame(corr_dict, index=[0]).stack(level=-1).reset_index(level=0, drop=True).transpose().reset_index()

    ret_dict = {}
    for name, g in result_all.groupby(['group_code', 'testing_period', 'alpha']):
        ret_dict[name] = {}
        max_g = g.loc[g['pred']>g['pred'].quantile(q=2/3)]
        min_g = g.loc[g['pred']<g['pred'].quantile(q=1/3)]
        ret_dict[name]['max_factor'] = ','.join(list(max_g['y_type'].values))
        ret_dict[name]['min_factor'] = ','.join(list(min_g['y_type'].values))
        ret_dict[name]['max_ret'] = max_g['actual'].mean()
        ret_dict[name]['min_ret'] = min_g['actual'].mean()
        ret_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        ret_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        ret_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    result_all_comb = pd.DataFrame(ret_dict).transpose().reset_index()
    result_all_comb.columns = ['group_code', 'testing_period', 'alpha'] + result_all_comb.columns.to_list()[3:]
    result_all_comb.iloc[:,5:] = result_all_comb.iloc[:,5:].astype(float)
    result_all_comb = result_all_comb.merge(result_all_avg, on=['group_code', 'testing_period'])
    print(result_all_comb.groupby(['group_code', 'alpha']).mean())

    if save_xls:
        writer = pd.ExcelWriter(f'score/#{model}_consol_pred_{iter_name}.xlsx')
        result_all_comb.groupby(['group_code','alpha']).mean().to_excel(writer, sheet_name='average')
        corr_df.to_excel(writer, sheet_name='corr')
        result_all_comb.to_excel(writer, sheet_name='group_time', index=False)
        pd.pivot_table(result_all, index=['alpha', 'group_code', 'testing_period'], columns=['y_type'], values=['pred','actual']).to_excel(writer, sheet_name='all')
        writer.save()

    if plot_consol:
        num_alpha = len(result_all_comb['alpha'].unique())
        num_group = len(result_all_comb['group_code'].unique())
        fig = plt.figure(figsize=(num_group*8, num_alpha*4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
        k=1
        for name, g in result_all_comb.groupby(['alpha','group_code']):
            ax = fig.add_subplot(num_alpha, num_group, k)
            g[['max_ret','actual','min_ret']] = np.cumprod(g[['max_ret','actual','min_ret']] + 1, axis=0)
            plot_df = g.set_index(['testing_period'])[['max_ret','actual','min_ret']]
            ax.plot(plot_df)
            for i in range(3):
                ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
            if k % num_group ==1:
                ax.set_ylabel(name[0], fontsize=20)
            if k > num_group*num_alpha-num_group:
                ax.set_xlabel(name[1], fontsize=20)
            if k==1:
                plt.legend(['best','average','worse'])
            k+=1
        plt.savefig(f'score/#{model}_consol_pred_{iter_name}.png')
    # plt.show()

    result_return = pd.pivot_table(result_all, index=['group_code', 'testing_period', 'alpha'], columns=['y_type'], values='pred')
    result_return = result_return.transpose().apply(pd.qcut, q=3, labels=[-1,0,1]).transpose().reset_index()
    result_return = result_return.merge(df_neg, on=['group_code', 'testing_period'], suffixes=['','_neg'])
    # x = result_return[all_neg_factor]
    # y = result_return[[x+'_neg' for x in all_neg_factor]]
    result_return[all_neg_factor] = -result_return[all_neg_factor].values*result_return[[x+'_neg' for x in all_neg_factor]].values

    return result_return.drop([x+'_neg' for x in all_neg_factor], axis=1)

def download_stock_pred_many_iters(iter_name, save_xls=True, plot_consol=True):
    ''' download training history and training prediction from DB '''

    if model == 'rf':
        y_type = 'P.y_type'
    elif model == 'lgbm':
        y_type = 'S.y_type'

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, {y_type}, P.group as group_code, S.testing_period, S.cv_number, S.name_sql as alpha, S.neg_factor "
                     f"FROM {global_vals.result_pred_table}_{model}_reg P "
                     f"INNER JOIN {global_vals.result_score_table}_{model}_reg S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{iter_name}%' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'alpha'], keep='last')
    result_all = result_all.groupby(['group_code', 'testing_period',  'y_type', 'alpha']).mean().reset_index()
    result_all_avg = result_all.groupby(['testing_period','group_code'])['actual'].mean().reset_index()

    corr_dict = {}
    for name, g in result_all.groupby([ 'alpha', 'y_type', 'group_code']):
        corr_dict[name] = g[['pred','actual']].corr().iloc[0, 1]
    corr_df = pd.DataFrame(corr_dict, index=[0]).stack(level=-1).reset_index(level=0, drop=True).transpose().reset_index()

    ret_dict = {}
    for name, g in result_all.groupby(['group_code', 'testing_period', 'alpha']):
        ret_dict[name] = {}
        max_g = g.loc[g['pred']>g['pred'].quantile(q=2/3)]
        min_g = g.loc[g['pred']<g['pred'].quantile(q=1/3)]
        ret_dict[name]['max_factor'] = ','.join(list(max_g['y_type'].values))
        ret_dict[name]['min_factor'] = ','.join(list(min_g['y_type'].values))
        ret_dict[name]['max_ret'] = max_g['actual'].mean()
        ret_dict[name]['min_ret'] = min_g['actual'].mean()
        ret_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        ret_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        ret_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    result_all_comb = pd.DataFrame(ret_dict).transpose().reset_index()
    result_all_comb.columns = ['group_code', 'testing_period', 'alpha'] + result_all_comb.columns.to_list()[3:]
    result_all_comb.iloc[:,5:] = result_all_comb.iloc[:,5:].astype(float)

    if save_xls:
        writer = pd.ExcelWriter(f'score/#{model}_consol_pred_{iter_name}.xlsx')
        result_all_comb.groupby(['group_code','alpha']).mean().to_excel(writer, sheet_name='average')
        corr_df.to_excel(writer, sheet_name='corr')
        result_all_comb.to_excel(writer, sheet_name='group_time', index=False)
        pd.pivot_table(result_all, index=['alpha', 'group_code', 'testing_period'], columns=['y_type'], values=['pred','actual']).to_excel(writer, sheet_name='all')
        writer.save()

    result_all_comb = result_all_comb.merge(result_all_avg, on=['group_code', 'testing_period'])
    print(result_all_comb.groupby(['group_code', 'alpha']).mean())

    if plot_consol:
        num_alpha = len(result_all_comb['alpha'].unique())
        num_group = len(result_all_comb['group_code'].unique())
        fig = plt.figure(figsize=(num_group*8, num_alpha*4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
        k=1
        for name, g in result_all_comb.groupby(['alpha','group_code']):
            ax = fig.add_subplot(num_alpha, num_group, k)
            g[['max_ret','actual','min_ret']] = np.cumprod(g[['max_ret','actual','min_ret']] + 1, axis=0)
            plot_df = g.set_index(['testing_period'])[['max_ret','actual','min_ret']]
            ax.plot(plot_df)
            plt.ylim((0.6, 1.9))
            for i in range(3):
                ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
            if k % num_group ==1:
                ax.set_ylabel(name[0], fontsize=20)
            if k > num_group*num_alpha-num_group:
                ax.set_xlabel(name[1], fontsize=20)
            if k==1:
                plt.legend(['best','average','worse'])
            k+=1
        plt.savefig(f'score/#{model}_consol_pred_{iter_name}.png')
    # plt.show()

# model = 'rf_reg'
def compare_all():
    with global_vals.engine_ali.connect() as conn:
        name_sql = pd.read_sql(f'SELECT DISTINCT name_sql from {global_vals.result_score_table}_{model}', conn)['name_sql'].to_list()
    global_vals.engine_ali.dispose()

    df_list = []
    for i in name_sql:
        print(i)
        df = download_stock_pred(4, i, False, False).reset_index()
        df['name_sql'] = i
        df_list.append(df)

    all = pd.concat(df_list, axis=0)
    all.to_csv('all_rf.csv', index=False)

def plot_corr_all_results():
    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, P.y_type, S.name_sql, P.group as group_code, S.testing_period, S.cv_number, CONCAT(S.tree_type,'',S.use_pca) as alpha FROM {global_vals.result_pred_table}_{model} P "
                     f"INNER JOIN {global_vals.result_score_table}_{model} S ON S.finish_timing = P.finish_timing "
                     f"WHERE P.actual IS NOT NULL and S.name_sql is not null ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    df = result_all[['pred','actual','name_sql']].dropna(how='any')
    print(df.describe())
    df = df.loc[df['name_sql'].isin(['pca_v2tryaddx_1'])]
    c = df.groupby('name_sql').corr()
    print(c)

    # Get Unique continents
    color_labels = df['name_sql'].unique()

    # Finally use the mapped values
    plt.scatter(df['pred'], df['actual'])
    plt.show()

    exit(1)

if __name__ == "__main__":
    # compare_all()
    download_stock_pred(1/3, iter_name=r_name)
    # download_stock_pred_multi(iter_name)
    # download_stock_pred_many_iters(r_name)
    # plot_corr_all_results()

