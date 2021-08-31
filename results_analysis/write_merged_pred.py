import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import argparse

from pandas.core.reshape.tile import qcut
import global_vals
from sqlalchemy import text
from sqlalchemy.dialects.postgresql.base import DATE, DOUBLE_PRECISION, TEXT, INTEGER, BOOLEAN, TIMESTAMP
from pandas.tseries.offsets import MonthEnd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix


stock_pred_dtypes = dict(
    period_end=DATE,
    factor_name=TEXT,
    group=TEXT,
    factor_weight=INTEGER,
    pred_z=DOUBLE_PRECISION,
    long_large=BOOLEAN,
    last_update=TIMESTAMP
)

def download_stock_pred(
        q,
        model,
        name_sql,
        rank_along='current',   # current / train_hist / test_hist
        keep_all_history=True,
        # return_summary=False,
        save_xls=False,
        save_plot=False):
    ''' organize production / last period prediction and write weight to DB '''

    # --------------------------------- Download Predictions ------------------------------------------
    if 'rf' in model:
        other_group_col = ['tree_type','use_pca']
    elif 'lasso' in model:
        other_group_col = ['use_pca','l1_ratio','alpha']

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, P.y_type as factor_name, P.group as \"group\", S.neg_factor, S.train_bins, "
                     f"S.testing_period as period_end, S.cv_number, {', '.join(['S.'+x for x in other_group_col])} "
                     f"FROM {global_vals.result_pred_table}_{model} P "
                     f"INNER JOIN {global_vals.result_score_table}_{model} S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{name_sql}%' ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn, chunksize=10000)
        result_all = pd.concat(result_all, axis=0, ignore_index=True)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group', 'period_end', 'factor_name', 'cv_number']+other_group_col, keep='last')
    result_all['period_end'] = pd.to_datetime(result_all['period_end'])

    neg_factor = result_all[['group','neg_factor']].drop_duplicates().set_index('group')['neg_factor'].to_dict()    # negative value columns
    result_all_avg = result_all.groupby(['group', 'period_end'])['actual'].mean()   # actual factor premiums

    # use average predictions from different validation sets
    result_all = result_all.groupby(['period_end','factor_name','group']+other_group_col)[['pred', 'actual']].mean()

    # --------------------------------- Add Rank & Evaluation Metrics ------------------------------------------

    if isinstance(q, int):    # if define top/bottom q factors as the best/worse
        q = q/len(result_all['factor_name'].unique())
    q_ = [0., q, 1.-q, 1.]

    # rank within current testing_period
    groupby_keys = ['group','period_end']
    result_all['factor_weight'] = result_all.groupby(level=groupby_keys)['pred'].transform(lambda x: pd.qcut(x, q=q_, labels=range(len(q_) - 1), duplicates='drop'))

    # calculate pred_z using mean & std of all predictions in entire testing history
    groupby_keys = ['group']
    result_all = result_all.join(result_all.groupby(level=groupby_keys)['pred'].agg(['mean', 'std']), on=groupby_keys, how='left')
    result_all['pred_z'] = (result_all['pred'] - result_all['mean']) / result_all['std']
    result_all = result_all.drop(['mean', 'std'], axis=1)

    def get_summary_stats_in_group(g):
        ''' Calculate basic evaluation metrics for factors '''

        ret_dict = {}

        max_g = g[g['factor_weight'] == 2]
        min_g = g[g['factor_weight'] == 0]

        ret_dict['max_factor'] = ','.join(list(max_g.index.get_level_values('factor_name').tolist()))
        ret_dict['min_factor'] = ','.join(list(min_g.index.get_level_values('factor_name').tolist()))
        ret_dict['max_ret'] = max_g['actual'].mean()
        ret_dict['min_ret'] = min_g['actual'].mean()
        ret_dict['mae'] = mean_absolute_error(g['pred'], g['actual'])
        ret_dict['mse'] = mean_squared_error(g['pred'], g['actual'])
        ret_dict['r2'] = r2_score(g['pred'], g['actual'])

        return pd.Series(ret_dict)

    result_all_comb = result_all.groupby(level=['group', 'period_end']+other_group_col).apply(get_summary_stats_in_group)
    result_all_comb = result_all_comb.loc[result_all_comb.index.get_level_values('period_end')<result_all_comb.index.get_level_values('period_end').max()]
    result_all_comb[['max_ret','min_ret','mae','mse','r2']] = result_all_comb[['max_ret','min_ret','mae','mse','r2']].astype(float)
    result_all_comb = result_all_comb.join(result_all_avg, on=['group', 'period_end']).reset_index()
    result_all_comb_mean = result_all_comb.groupby(['group'] + other_group_col).mean().reset_index()
    print(result_all_comb_mean)

    # --------------------------------- Save Local Evaluation ------------------------------------------

    if save_xls:    # save local for evaluation
        writer = pd.ExcelWriter(f'#{model}_pred_{name_sql}.xlsx')
        result_all_comb_mean.to_excel(writer, sheet_name='average', index=False)
        result_all_comb.to_excel(writer, sheet_name='group_time', index=False)
        pd.pivot_table(result_all, index=['group', 'period_end'], columns=['factor_name'], values=['pred','actual']).to_excel(writer, sheet_name='all')
        writer.save()

    if save_plot:    # save local for evaluation
        result_all_comb['other_group'] = result_all_comb[other_group_col].astype(str).agg('-'.join, axis=1)
        num_group = len(result_all_comb['group'].unique())
        num_other_group = len(result_all_comb['other_group'].unique())
        fig = plt.figure(figsize=(num_group*8, num_other_group*4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
        k=1
        for name, g in result_all_comb.groupby(['other_group', 'group']):
            ax = fig.add_subplot(num_other_group, num_group, k)
            g[['max_ret','actual','min_ret']] = (g[['max_ret','actual','min_ret']] + 1).cumprod(axis=0)
            plot_df = g[['max_ret','actual','min_ret']]
            ax.plot(plot_df)
            for i in range(3):
                ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
            if k % num_group == 1:
                ax.set_ylabel(name[0], fontsize=20)
            if k > (num_other_group-1)*num_group:
                ax.set_xlabel(name[1], fontsize=20)
            if k==1:
                plt.legend(['best','average','worse'])
            k+=1
        plt.ylabel('-'.join(other_group_col))
        plt.xlabel('group')
        plt.savefig(f'#{model}_pred_{name_sql}.png')
        plt.close()

    # ------------------------ Select Best Config (among other_group_col) ------------------------------

    result_all_comb_mean['net_ret'] = result_all_comb_mean['max_ret'] - result_all_comb_mean['min_ret']
    result_all_comb_mean_best = result_all_comb_mean.sort_values(['net_ret']).groupby(['group']).last()[other_group_col].reset_index()
    print(result_all_comb_mean_best)

    result_all = result_all.dropna(axis=0, subset=['factor_weight'])[['pred_z','factor_weight']].reset_index()
    result_all = result_all.merge(result_all_comb_mean_best, on=['group']+other_group_col, how='right')

    # --------------------------------- Save Prod Table to DB ------------------------------------------

    # count rank for debugging
    # factor_rank = result_all.set_index(['period_end','factor_name','group'])['factor_weight'].unstack()
    rank_count = result_all.groupby(['group','period_end'])['factor_weight'].apply(pd.value_counts)
    rank_count = rank_count.unstack().fillna(0)
    print(rank_count)

    # prepare table to write to DB
    if keep_all_history:        # keep only last testing i.e. for production
        period_list = result_all['period_end'].unique()
        tbl_suffix = '_history'
    else:
        period_list = [result_all['period_end'].max()]
        tbl_suffix = ''

    for period in period_list:
        print(period)
        result_col = ['group','period_end','factor_name','pred_z','factor_weight']
        df = result_all.loc[result_all['period_end']==period, result_col].copy().reset_index(drop=True)

        # basic info
        df['period_end'] = df['period_end'] + MonthEnd(1)
        df['factor_weight'] = df['factor_weight'].astype(int)
        df['long_large'] = False
        df['last_update'] = dt.datetime.now()

        for k, v in neg_factor.items():     # write neg_factor i.e. label factors
            df.loc[(df['group']==k)&(df['factor_name'].isin([x[2:] for x in v.split(',')])), 'long_large'] = True

        with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            if conn.dialect.has_table(global_vals.engine_ali, global_vals.production_factor_rank_table + tbl_suffix): # remove same period prediction if exists
                if keep_all_history:
                    latest_period_end = dt.datetime.strftime(df['period_end'][0], r'%Y-%m-%d')
                    delete_query_history = f"DELETE FROM {global_vals.production_factor_rank_table}{tbl_suffix} WHERE period_end='{latest_period_end}';"
                    conn.execute(delete_query_history)

            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000, 'dtype': stock_pred_dtypes}
            df.sort_values(['group','pred_z']).to_sql(global_vals.production_factor_rank_table+tbl_suffix, **extra)

            if (period==result_all['period_end'].max()):  # if keep_all_history also write to prod table
                extra['if_exists'] = 'replace'
                df.sort_values(['group', 'pred_z']).to_sql(global_vals.production_factor_rank_table, **extra)

        global_vals.engine_ali.dispose()

        # return factor_rank, rank_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', type=float, default=1/3)
    parser.add_argument('--model', type=str, default='rf_reg')
    parser.add_argument('--name_sql', type=str, default='default1_20210830')

    # parser.add_argument('--rank_along_testing_history', action='store_false', help='rank_along_testing_history = True')
    parser.add_argument('--keep_all_history', action='store_true', help='keep_last = True')
    parser.add_argument('--save_plot', action='store_true', help='save_plot = True')
    parser.add_argument('--save_xls', action='store_true', help='save_xls = True')
    # parser.add_argument('--return_summary', action='store_true', help='return_summary = True')

    args = parser.parse_args()

    if args.q.is_integer():
        q = int(args.q)
    elif args.q < .5:
        q = args.q
    else:
        raise Exception('q is either >= .5 or not a numeric')

    # Example
    # download_stock_pred(1/3, 'rf_reg', 'pca_trimold2', rank_along_testing_history=True, keep_last_period=True, save_plot=True, return_summary=True)

    download_stock_pred(
        q,
        args.model,
        args.name_sql,
        keep_all_history=args.keep_all_history,
        save_plot=args.save_plot,
        save_xls=args.save_xls,
        # return_summary=args.return_summary
    )

