import pandas as pd
import numpy as np
from functools import partial
from general.utils import to_excel
from results_analysis.calculation_rank import weight_qcut
from xlsxwriter.utility import xl_rowcol_to_cell
from .configs import LOGGER_LEVELS
from utils import sys_logger
logger = sys_logger(__name__, LOGGER_LEVELS.ANALYSIS_RANK_FACTOR_ROTATION)

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
        # 
        pass

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
    valid_train_currency = [('USD', 'USD'), ('USD', 'EUR'), ('CNY', 'CNY'), ('HKD', 'HKD')]
    for (group, group_code, pillar), g in pred.groupby(['group', 'group_code', 'pillar']):
        if (group_code, group) not in valid_train_currency:
            continue
        g_mean = g.groupby(['testing_period', 'factor_name'])['good_pred'].mean().unstack().replace(0, np.nan)
        g_count = g.groupby(['testing_period', 'factor_name'])['select_pred'].mean().unstack().replace(0, np.nan)
        xls[f'{group}{group_code}_{pillar}_mean'] = g_mean.reset_index()
        xls[f'{group}{group_code}_{pillar}_count'] = g_count.reset_index()

    if use_max_ret:
        to_excel(xls, f'max_ret_factor_rotation_4-7', excel_cformat)
    else:
        to_excel(xls, f'net_ret_factor_rotation_4-7', excel_cformat)


def factor_pred_variance():
    """ analyse factor based on period variance (overall, i.e. no specification on single factor) """

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

    results = pred.groupby(['group', 'group_code', 'pillar', 'testing_period']).apply(group_metrics)
    results = pd.DataFrame(results.to_list(), index=results.index)
    return results


def factor_pred_variance_single(is_rank=True):
    """ analyse factor based on period variance (overall, i.e. no specification on single factor) """

    pred = pd.read_pickle('pred_cache.pkl')
    pred['testing_period'] = pd.to_datetime(pred['testing_period'])
    pred['factor_name'] = pred['factor_name'].str[:-4]

    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

    def group_metrics(g):
        g_actual = g.groupby(['uid_hpot', 'factor_name'])['actual'].mean().unstack().filnet_retlna(0)
        g_pred = g.groupby(['uid_hpot', 'factor_name'])['pred'].mean().unstack()

        if is_rank:
            g_actual = g_actual.values.argsort().argsort()
            g_pred = g_pred.values.argsort().argsort()

        result = {}
        for k, func in {"mae": mean_absolute_error, "r2": r2_score, "mse": mean_squared_error}.items():
            score = func(g_actual, g_pred, multioutput='raw_values')
            result[k + '_mean'] = np.mean(score)
        return result

    results = pred.groupby(['group', 'group_code', 'pillar', 'testing_period']).apply(group_metrics)
    results = pd.DataFrame(results.to_list(), index=results.index)
    return results


if __name__ == '__main__':
    factor_pred_variance()

