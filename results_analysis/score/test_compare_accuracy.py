import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, \
    multilabel_confusion_matrix, roc_curve
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from pandas.tseries.offsets import MonthEnd

import global_vals

if __name__ == "__main__":

    df0 = pd.read_excel(open(f'lgbm_pred_accuracy_lastweekavg_rerun.xlsx', 'rb'), sheet_name='mode_time')
    df0 = df0.set_index(['group_code', 'period_end'])
    df0 = df0.loc[df0['y_type']=='vol_30_90'].drop(['y_type'], axis=1)
    df0.columns = ['a_rerun']

    csv_list = ['_timevalid_weight','_timevalid_unbalance','_timevalid_rerun', '_timevalid_rerun1']

    for csv_name in csv_list:
        print(csv_name)
        df = pd.read_excel(open(f'lgbm_pred_accuracy_lastweekavg{csv_name}.xlsx', 'rb'), sheet_name='mode_time')
        df = df.set_index(['group_code','period_end']).drop(['y_type'], axis=1)
        df.columns = ['a'+csv_name]
        df0 = df0.merge(df, left_index=True, right_index=True, how='outer')

    x = df0.groupby(['group_code']).mean()

    df0.to_csv('test_compare_accuracy.csv')
    fig = plt.figure(figsize=(12, 8), dpi=120, constrained_layout=True)
    col = ['a'] + ['a'+ x for x in csv_list]
    k=1
    for name, g in df0.reset_index().groupby(['group_code']):
        ax = fig.add_subplot(2, 1, k)
        g = g.set_index(['period_end'])
        ax.plot(g[col])
        ax.set_ylabel(name, fontsize=20)
        plt.ylim((0, 1))
        if k == 1:
            plt.legend(col, loc='upper left', fontsize='small')
        k += 1
    plt.show()