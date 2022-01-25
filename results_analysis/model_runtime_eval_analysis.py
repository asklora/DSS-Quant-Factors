import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
import datetime as dt
import numpy as np
from global_vars import *
from general.sql_process import read_query

def download_model(name_sql='week1_20220125130042_debug', eval_metric='r2'):
    ''' evaluation runtime calculated metrics '''

    query = f"SELECT * FROM {result_score_table} WHERE name_sql='{name_sql}'"
    df = read_query(query, db_url_read)

    iter_unique_col = ['group_code', 'y_type', 'testing_period', 'cv_number']  # keep 'cv_number' in last one for averaging
    diff_config_col = ['tree_type', 'use_pca']

    # 2.1. remove duplicate samples from running twice when testing
    df = df.drop_duplicates(subset=iter_unique_col + diff_config_col, keep='last')

    # 2.2. find best in cv groups
    df_best = df.sort_values(by=[eval_metric+'_valid']).groupby(iter_unique_col[:-1] + diff_config_col).first()

    # 2.3. calculate average accuracy across testing_period
    df_best_avg = df_best.groupby(iter_unique_col[:-2] + diff_config_col).mean().filter(regex=f'^{eval_metric}_')

    return

if __name__ == "__main__":
    download_model()