import pandas as pd
import numpy as np
import datetime as dt
import os
from general.utils import to_excel
from general.sql_process import read_query, read_table
from functools import partial
import ast
from dateutil.relativedelta import relativedelta
import global_vars


class main:

    def __init__(self):
        df = read_table(global_vars.factor_config_score_table)
        df['hpot_uid'] = df['uid'].str[20:-1]
        score_col = df.filter(regex='_train$|_valid$|_test$').columns.to_list()

        df_best = df.sort_values(by='logloss_valid').groupby(['hpot_uid']).first()
        df_best_agg = df_best.groupby(['currency_code'])[score_col].mean()

        df_best_agg.to_csv('save.csv')
        pass

        df_importance = read_table(global_vars.factor_config_importance_table)
        df_importance['split'] = df_importance.groupby(['uid'])['split'].apply(lambda x: x/x.max())
        df_best_importance = df_importance.merge(df_best[['uid', 'currency_code']], on='uid')

        df_best_importance = df_best_importance.groupby(['index', 'currency_code'])['split'].mean().unstack()
        df_best_importance.to_csv('save1.csv')


if __name__ == '__main__':
    main()
