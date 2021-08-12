from sqlalchemy import text
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
import numpy as np
import os
import re

import global_vals

r_name = 'lastweekavg_pca_new'
iter_name = r_name

def download_stock_pred():
    ''' download training history and training prediction from DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.alpha, S.y_type FROM {global_vals.result_pred_table}_lasso P "
                     f"INNER JOIN {global_vals.result_score_table}_lasso S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'alpha','group'], keep='last')

    result_all_avg = result_all.groupby(['testing_period','group_code','alpha'])['actual'].mean().reset_index()

    ret_dict = {}
    for name, g in result_all.groupby(['group_code', 'testing_period', 'alpha']):
        ret_dict[name] = {}
        ret_dict[name]['code'] = g.loc[g['pred']==g['pred'].max(), 'y_type'].values[0]
        ret_dict[name]['max_ret'] = g.loc[g['pred']==g['pred'].max(), 'actual'].values[0]
        ret_dict[name]['min_ret'] = g.loc[g['pred']==g['pred'].min(), 'actual'].values[0]
        ret_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        ret_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        ret_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    result_all_comb = pd.DataFrame(ret_dict).transpose().reset_index()
    result_all_comb.columns = ['group_code', 'testing_period', 'alpha'] + result_all_comb.columns.to_list()[3:]
    result_all_comb.iloc[:,4:] = result_all_comb.iloc[:,4:].astype(float)

    writer = pd.ExcelWriter(f'score/#lasso_pred_{iter_name}.xlsx')
    result_all_comb.groupby(['group_code','alpha']).mean().to_excel(writer, sheet_name='average')
    result_all_comb.to_excel(writer, sheet_name='all', index=False)
    writer.save()

    result_all_comb = result_all_comb.merge(result_all_avg, on=['group_code', 'testing_period', 'alpha'])
    print(result_all_comb.groupby(['alpha']).mean())

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
    plt.savefig(f'score/#lasso_pred_{iter_name}.png')
    # plt.show()

if __name__ == "__main__":

    download_stock_pred()


