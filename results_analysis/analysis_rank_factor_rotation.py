import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import argparse
import json
import ast
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial
from general.send_slack import to_slack
from general.utils import to_excel
import global_vars
from global_vars import *
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import DATE, DOUBLE_PRECISION, TEXT, INTEGER, BOOLEAN, TIMESTAMP, JSON
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from general.sql_process import (
    read_query,
    upsert_data_to_database,
    uid_maker,
    trucncate_table_in_database,
    delete_data_on_database,
)
from collections import Counter
from results_analysis.calculation_rank import weight_qcut
from xlsxwriter.utility import xl_rowcol_to_cell


def excel_cformat(writer, sheet_name):
    """ for func [factor_weight] excel format factor rotation """

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    cell_format1 = workbook.add_format()
    cell_format1.set_num_format('0%')

    worksheet = writer.sheets[sheet_name]
    last_cell = xl_rowcol_to_cell(worksheet.dim_rowmax, worksheet.dim_colmax)
    # percent_fmt = workbook.add_format({'num_format': '0.00%'})

    # Apply a conditional format to the cell range.
    if '_count' in sheet_name:
        worksheet.conditional_format(f'B2:{last_cell}', {'type': '2_color_scale',
                                                         'min_color': "white",
                                                         'max_color': 'green'})
        worksheet.set_column(f'B2:{last_cell}', None, cell_format1)
    else:
        worksheet.conditional_format(f'B2:{last_cell}', {'type': '3_color_scale',
                                                         'min_color': 'red',
                                                         'mid_color': "white",
                                                         'max_color': 'green'})
        worksheet.set_column(f'B2:{last_cell}', None, cell_format1)

    return writer

def factor_weight(use_max_ret, q = 1 / 3):
    """ analysis based on factor weight
        1. [mean]: if we select factors as True Positive (=1) / False Positive (=-1)
        2. [count]: how many times we select factor as good factors
    """

    pred = pd.read_pickle('pred_cache.pkl')
    pred['testing_period'] = pd.to_datetime(pred['testing_period'])

    # pred_weight
    try:
        pred['pred_weight'] = pred.groupby(by='uid_hpot')['pred'].transform(
            partial(weight_qcut, q_=[0., q, 1. - q, 1.]))
        pred['actual_weight'] = pred.groupby(by='uid_hpot')['actual'].transform(
            partial(weight_qcut, q_=[0., q, 1. - q, 1.]))
    except Exception as e:
        print(e)

    if use_max_ret:
        # label prediction (if use [max_ret])
        conditions = [
            ((pred['pred_weight'] == 2) & (pred['actual_weight'] == 2)),
            ((pred['pred_weight'] == 2) & (pred['actual_weight'] == 0)),
        ]
        pred['good_pred'] = np.select(conditions, [1, -1], 0)
        pred['select_pred'] = (pred['pred_weight'] == 2)
    else:
        # label prediction (if use [net_ret] = max_ret - min_ret)
        conditions = [
            ((pred['pred_weight'] == 2) & (pred['actual_weight'] == 2)) | (
                        (pred['pred_weight'] == 0) & (pred['actual_weight'] == 0)),
            ((pred['pred_weight'] == 2) & (pred['actual_weight'] == 0)) | (
                        (pred['pred_weight'] == 0) & (pred['actual_weight'] == 2)),
        ]
        pred['good_pred'] = np.select(conditions, [1, -1], 0)
        pred['select_pred'] = pred['pred_weight'] - 1

    # type date
    xls = {}
    valid_group = [('USD', 'USD'), ('USD', 'EUR'), ('CNY', 'CNY'), ('HKD', 'HKD')]
    for (group, group_code, y_type), g in pred.groupby(['group', 'group_code', 'y_type']):
        if (group_code, group) not in valid_group:
            continue
        g_mean = g.groupby(['testing_period', 'factor_name'])['good_pred'].mean().unstack().replace(0, np.nan)
        g_count = g.groupby(['testing_period', 'factor_name'])['select_pred'].mean().unstack().replace(0, np.nan)
        xls[f'{group}{group_code}_{y_type}_mean'] = g_mean.reset_index()
        xls[f'{group}{group_code}_{y_type}_count'] = g_count.reset_index()

    if use_max_ret:
        to_excel(xls, f'max_ret_factor_rotation_4-7', excel_cformat)
    else:
        to_excel(xls, f'net_ret_factor_rotation_4-7', excel_cformat)

def factor_pred_variance():
    """ analyse factor based on period variance """

    pred = pd.read_pickle('pred_cache.pkl')
    pred['testing_period'] = pd.to_datetime(pred['testing_period'])
    pred['factor_name'] = pred['factor_name'].str[:-4]

    from random_forest import adj_mse_score
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    def group_metrics(g):
        g_actual = g.groupby(['uid_hpot', 'factor_name'])['actual'].mean().unstack().fillna(0)
        g_pred = g.groupby(['uid_hpot', 'factor_name'])['pred'].mean().unstack()

        result = {}
        for k, func in {"mae": mean_absolute_error, "r2": r2_score, "mse": mean_squared_error, "adj_mse": adj_mse_score}.items():
            score = func(g_actual.values.T, g_pred.values.T, multioutput='raw_values')
            if type(score) == type({}):
                for score_k, score_v in score.items():
                    result[k] = score_v
            else:
                result[k + '_med'] = np.median(score)
                result[k + '_mean'] = np.mean(score)
        return result

    results = pred.groupby(['group', 'group_code', 'y_type', 'testing_period']).apply(group_metrics)
    results = pd.DataFrame(results.to_list(), index=results.index)
    return results

if __name__ == '__main__':
    factor_pred_variance()

