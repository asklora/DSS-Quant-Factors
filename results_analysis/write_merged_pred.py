import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np

from pandas.core.reshape.tile import qcut
import global_vals
from sqlalchemy import text
from sqlalchemy.dialects.postgresql.base import DATE, TEXT, INTEGER, BOOLEAN, TIMESTAMP
from pandas.tseries.offsets import MonthEnd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix


model = 'rf_reg'
r_name = 'pca_trimold2'


stock_pred_dtypes = dict(
    period_end=DATE,
    factor_name=TEXT,
    group=TEXT,
    factor_weight=INTEGER,
    long_large=BOOLEAN,
    last_update=TIMESTAMP
)


def download_stock_pred(
        q,
        iter_name,
        rank_along_testing_history=True,
        keep_last_period=True,
        return_summary=False,
        save_xls=False,
        save_plot=False):
    ''' organize production / last period prediction and write weight to DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, P.y_type as factor_name, P.group as \"group\", S.neg_factor, S.testing_period as period_end, S.cv_number FROM {global_vals.result_pred_table}_rf_reg P "
                     f"INNER JOIN {global_vals.result_score_table}_rf_reg S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{iter_name}%' and tree_type ='rf' and use_pca = 0.2 ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group', 'period_end', 'factor_name', 'cv_number'], keep='last')
    result_all['period_end'] = pd.to_datetime(result_all['period_end'])

    neg_factor = result_all[['group','neg_factor']].drop_duplicates().set_index('group')['neg_factor'].to_dict()

    # use average predictions from different validation sets
    result_all = result_all.groupby(['period_end','factor_name','group'])[['pred', 'actual']].mean()
    result_all_avg = result_all.groupby(['group', 'period_end'])['actual'].mean()

    if rank_along_testing_history:
        groupby_keys = ['group']
    else:
        groupby_keys = ['group', 'period_end']

    if q < .5:
        q_ = [0., q, 1.-q, 1.]
    elif isinstance(q, int):
        # create equal-sized bins between 0 and 1 inclusively
        q_ = np.linspace(0., 1., q)
    else:
        raise Exception('q is either >= .5 or not a numeric')
    
    result_all['factor_weight'] = result_all.groupby(level=groupby_keys)['pred'].transform(lambda x: pd.qcut(x, q=q_, labels=range(len(q_)-1), duplicates='drop'))

    def get_summary_stats_in_group(g):
        ret_dict = {}

        max_g = g[g['factor_weight'] == len(q_)-2]
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

    if save_xls:
        writer = pd.ExcelWriter(f'score/#{model}_pred_{iter_name}.xlsx')
        result_all_comb.groupby(['group']).mean().to_excel(writer, sheet_name='average')
        result_all_comb.to_excel(writer, sheet_name='group_time', index=False)
        pd.pivot_table(result_all, index=['group', 'period_end'], columns=['factor_name'], values=['pred','actual']).to_excel(writer, sheet_name='all')
        writer.save()

    result_all_comb = result_all_comb.join(result_all_avg, on=['group', 'period_end'])

    if save_plot:
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
        plt.savefig(f'score/#{model}_pred_{iter_name}.png')
        plt.close()

    result_all = result_all.dropna(axis=0, subset=['factor_weight'])

    if return_summary:
        factor_rank = result_all['factor_weight'].unstack()
        rank_count = result_all.droplevel('factor_name', axis=0)['factor_weight'].reset_index().value_counts()
        rank_count = rank_count.unstack().fillna(0)

    result_all = result_all.reset_index()

    if keep_last_period:
        # keep only last testing i.e. for production
        result_all = result_all.loc[result_all['period_end']==result_all['period_end'].max()].reset_index(drop=True).copy()
    
    result_all['period_end'] = result_all['period_end'] + MonthEnd(1)
    result_all['factor_weight'] = result_all['factor_weight'].astype(int)
    result_all['long_large'] = False
    result_all['last_update'] = dt.datetime.now()

    for k, v in neg_factor.items():
        result_all.loc[(result_all['group']==k)&(result_all['factor_name'].isin([x[2:] for x in v.split(',')])), 'long_large'] = True

    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000, 'dtype': stock_pred_dtypes}

        if conn.dialect.has_table(global_vals.engine_ali, 'factor_result_pred_prod'):
            if keep_last_period:
                # remove same period prediction if exists
                latest_period_end = dt.datetime.strftime(result_all['period_end'][0], r'%Y-%m-%d')
                delete_query = f"DELETE FROM {global_vals.production_factor_rank_table} WHERE period_end='{latest_period_end}';"
            else:
                delete_query = f"DELETE FROM {global_vals.production_factor_rank_table} WHERE true;"
        
            conn.execute(delete_query)
        
        result_all.drop(['pred', 'actual'], axis=1).sort_values(['group','factor_weight']).to_sql(global_vals.production_factor_rank_table, **extra)

    if return_summary:
        return factor_rank, rank_count


if __name__ == "__main__":
    download_stock_pred(1/3, 'pca_trimold2', rank_along_testing_history=True, keep_last_period=True, save_plot=True, return_summary=True)

