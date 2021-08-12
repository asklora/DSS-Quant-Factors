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

def download_stock_pred(plot_dist=True):
    ''' download training history and training prediction from DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.alpha, S.y_type FROM {global_vals.result_pred_table}_lasso P "
                     f"INNER JOIN {global_vals.result_score_table}_lasso S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'alpha','group'], keep='last')

    ret_dict = {}
    for name, g in result_all.groupby(['group_code', 'testing_period', 'alpha']):
        ret_dict[name] = {}
        ret_dict[name]['ret'] = g.loc[g['pred']==g['pred'].max(), 'actual'].values[0]
        ret_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        ret_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        ret_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    result_all_comb = pd.DataFrame(ret_dict).transpose().reset_index()
    result_all_comb.columns = ['group_code', 'testing_period', 'alpha'] + result_all_comb.columns.to_list()[3:]

    writer = pd.ExcelWriter(f'score/#lasso_pred_{iter_name}.xlsx')
    result_all_comb.groupby(['alpha']).mean().to_excel(writer, sheet_name='average')
    result_all_comb.to_excel(writer, sheet_name='all', index=False)
    writer.save()

    print(result_all_comb.groupby(['alpha']).mean())

    num_alpha = len(result_all_comb['alpha'].unique())
    fig = plt.figure(figsize=(8, num_alpha*4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
    k=1
    for name, g in result_all_comb.groupby(['alpha']):
        ax = fig.add_subplot(num_alpha, 1, k)
        plot_df = pd.pivot_table(g, index=['testing_period'], columns=['group_code'], values='ret')
        cols = plot_df.columns.to_list()
        plot_df[cols] = np.cumprod(plot_df[cols] + 1, axis=0)
        ax.plot(plot_df)
        ax.set_ylabel(name, fontsize=20)
        if k==1:
            plt.legend(cols)
        k+=1
    plt.savefig(f'score/#lasso_pred_{iter_name}.png')
    # plt.show()

if __name__ == "__main__":

    download_stock_pred()


