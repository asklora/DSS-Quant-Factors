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
        rank_along_testing_history=True,
        keep_all_history=True,
        return_summary=False,
        save_xls=False,
        save_plot=False):
    ''' organize production / last period prediction and write weight to DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, P.y_type as factor_name, P.group as \"group\", S.neg_factor, S.testing_period as period_end, S.cv_number FROM {global_vals.result_pred_table}_rf_reg P "
                     f"INNER JOIN {global_vals.result_score_table}_rf_reg S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{name_sql}%' and tree_type ='rf' and use_pca = 0.6 ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group', 'period_end', 'factor_name', 'cv_number'], keep='last')
    result_all['period_end'] = pd.to_datetime(result_all['period_end'])

    neg_factor = result_all[['group','neg_factor']].drop_duplicates().set_index('group')['neg_factor'].to_dict()

    # use average predictions from different validation sets
    result_all = result_all.groupby(['period_end','factor_name','group'])[['pred', 'actual']].mean()
    result_all_avg = result_all.groupby(['group', 'period_end'])['actual'].mean()

    # if rank_along_testing_history:          # rank across all testing history
    groupby_keys = ['group']
    # else:                                   # rank for each period
    #     groupby_keys = ['group', 'period_end']

    if isinstance(q, int):    # if define top/bottom q factors as the best/worse
        q = q/len(result_all['factor_name'].unique())
    q_ = [0., q, 1.-q, 1.]

    # calculate pred_z using mean & std of all predictions in entire testing history
    result_all = result_all.join(result_all.groupby(level=groupby_keys)['pred'].agg(['mean', 'std']), on='group', how='left')
    result_all['pred_z'] = (result_all['pred'] - result_all['mean']) / result_all['std']
    result_all = result_all.drop(['mean', 'std'], axis=1)
    result_all['factor_weight'] = result_all.groupby(level=groupby_keys)['pred'].transform(lambda x: pd.qcut(x, q=q_, labels=range(len(q_)-1), duplicates='drop'))

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

    result_all_comb = result_all.groupby(level=['group', 'period_end']).apply(get_summary_stats_in_group)
    result_all_comb[['max_ret','min_ret','mae','mse','r2']] = result_all_comb[['max_ret','min_ret','mae','mse','r2']].astype(float)
    result_all_comb = result_all_comb.join(result_all_avg, on=['group', 'period_end'])

    if save_xls:    # save local for evaluation
        writer = pd.ExcelWriter(f'score/#{model}_pred_{name_sql}.xlsx')
        result_all_comb.groupby(['group']).mean().to_excel(writer, sheet_name='average')
        result_all_comb.to_excel(writer, sheet_name='group_time', index=False)
        pd.pivot_table(result_all, index=['group', 'period_end'], columns=['factor_name'], values=['pred','actual']).to_excel(writer, sheet_name='all')
        writer.save()

    if save_plot:    # save local for evaluation
        num_group = len(result_all_comb.index.get_level_values('group').unique())
        fig = plt.figure(figsize=(num_group*8, 4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
        k=1
        for name, g in result_all_comb.groupby(level=['group']):
            ax = fig.add_subplot(1, num_group, k)
            g[['max_ret','actual','min_ret']] = (g[['max_ret','actual','min_ret']] + 1).cumprod(axis=0)
            plot_df = g.droplevel(['group'])[['max_ret','actual','min_ret']]
            ax.plot(plot_df)
            for i in range(3):
                ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
            ax.set_xlabel(name, fontsize=20)
            if k==1:
                plt.legend(['best','average','worse'])
            k+=1
        plt.savefig(f'score/#{model}_pred_{name_sql}.png')
        plt.close()

    result_all = result_all.drop(['pred', 'actual'], axis=1)
    result_all = result_all.dropna(axis=0, subset=['factor_weight'])

    # count rank rank for debugging
    factor_rank = result_all['factor_weight'].unstack()
    rank_count = result_all.droplevel('factor_name', axis=0)['factor_weight'].reset_index().value_counts()
    rank_count = rank_count.unstack().fillna(0)

    result_all = result_all.reset_index()

    # prepare table to write to DB
    if keep_all_history:        # keep only last testing i.e. for production
        period_list = result_all['period_end'].unique()
        tbl_suffix = '_history'
    else:
        period_list = [result_all['period_end'].max()]
        tbl_suffix = ''

    for period in period_list:
        result_all = result_all.loc[result_all['period_end']==period].reset_index(drop=True).copy()

        # basic info
        result_all['period_end'] = result_all['period_end'] + MonthEnd(1)
        result_all['factor_weight'] = result_all['factor_weight'].astype(int)
        result_all['long_large'] = False
        result_all['last_update'] = dt.datetime.now()

        for k, v in neg_factor.items():     # write neg_factor i.e. label factors
            result_all.loc[(result_all['group']==k)&(result_all['factor_name'].isin([x[2:] for x in v.split(',')])), 'long_large'] = True

        with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000, 'dtype': stock_pred_dtypes}

            # remove same period prediction if exists
            if conn.dialect.has_table(global_vals.engine_ali, global_vals.production_factor_rank_table+tbl_suffix):
                latest_period_end = dt.datetime.strftime(result_all['period_end'][0], r'%Y-%m-%d')
                delete_query = f"DELETE FROM {global_vals.production_factor_rank_table}{tbl_suffix} WHERE period_end='{latest_period_end}';"
                conn.execute(delete_query)

            result_all.sort_values(['group','pred_z']).to_sql(global_vals.production_factor_rank_table+tbl_suffix, **extra)
        global_vals.engine_ali.dispose()

        return factor_rank, rank_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', type=float, default=1/3)
    parser.add_argument('--model', type=str, default='rf_reg')
    parser.add_argument('--name-sql', type=str, default='pca_top16_mse_rerun_tv3')

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
        # rank_along_testing_history=args.rank_along_testing_history,
        keep_all_history=args.keep_all_history,
        save_plot=args.save_plot,
        save_xls=args.save_xls,
        # return_summary=args.return_summary
    )

