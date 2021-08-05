from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import datetime as dt
import numpy as np
from preprocess.load_data import load_data

import global_vals

r_name = 'biweekly_rerun'
iter_name = r_name

def download_stock_pred():
    ''' download training history and training prediction from DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_rf_class P "
                     f"INNER JOIN {global_vals.result_score_table}_rf_class S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number','group'], keep='last')
    result_all = result_all.drop(['cv_number'], axis=1)

    # save counting label to csv
    count_i = {}
    for name,g in result_all.groupby(['group_code', 'testing_period', 'y_type']):
        count_i[name] = g['pred'].value_counts().to_dict()
    pd.DataFrame(count_i).transpose().to_csv(f'score/rf_result_pred_count_{iter_name}.csv')

    return result_all

def combine_pred_class(df, agg_type):
    ''' download stock / ibes data and convert to qcut_median '''

    if agg_type == 'mean':  # use median/mean for cross listing & multiple cross-validation
        df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).mean().reset_index()
        df['pred'] = df['pred'].round()
    elif agg_type == 'median':
        df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).median().reset_index()
        df['pred'] = df['pred'].round()
    elif agg_type == 'mode':
        df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    else:
        raise ValueError("Invalid agg_type method. Expecting 'mean', 'median' or 'mode', got ", agg_type)

    result_dict = {}
    for name, g in df.dropna(how='any').groupby(['group_code', 'testing_period', 'y_type']):
        result_dict[name] = accuracy_score(g['pred'], g['actual'])

    return result_dict

def combine_mode_class(df):
    ''' calculate accuracy score when pred = 0 / 1 / 2 '''

    df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    df = df.dropna(how='any')

    result_dict = {}
    for name, g in df.groupby(['group_code', 'testing_period', 'y_type']):
        result_dict[name] = {}
        for name1, g1 in g.groupby(['pred']):
            result_dict[name][name1] = accuracy_score(g1['pred'], g1['actual'])

    df = pd.DataFrame(result_dict).transpose().reset_index()
    df.columns = ['group_code', 'testing_period', 'y_type'] + df.columns.to_list()[3:]
    return df

def combine_mode_group(df):
    ''' calculate accuracy score by each industry/currency group '''

    df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    df = df.dropna(how='any')

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'group']):
        result_dict[name] = accuracy_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict, index=['0']).transpose().reset_index()
    df.columns = ['group_code', 'y_type','group','accuracy']

    with global_vals.engine_ali.connect() as conn:
        icb_name = pd.read_sql(f"SELECT DISTINCT code_6 as group, name_6 as name FROM icb_code_explanation", conn)  # download training history
        icb_count = pd.read_sql(f"SELECT \"group\", avg(num_ticker) as num_ticker FROM icb_code_count GROUP BY \"group\"", conn)  # download training history
    global_vals.engine_ali.dispose()

    df = df.merge(icb_name, on=['group'])
    df = df.merge(icb_count, on=['group'])

    return df

def calc_pred_class():
    ''' Calculte the accuracy_score if combine CV by mean, median, mode '''

    df = download_stock_pred()
    df[['pred','actual']] = df[['pred','actual']].astype(int)
    result_group = combine_mode_group(df)
    result_class = combine_mode_class(df)

    results = {}
    for i in ['mean', 'median', 'mode']:
        results[i] = combine_pred_class(df, i)

    r = pd.DataFrame(results).reset_index()
    r.columns = ['group_code', 'testing_period', 'y_type'] + ['mean', 'median', 'mode']

    with pd.ExcelWriter(f'score/rf_result_pred_accuracy_{iter_name}.xlsx') as writer:
        r.to_excel(writer, sheet_name='original', index=False)
        r.groupby(['group_code', 'y_type']).mean().reset_index().to_excel(writer, sheet_name='cv_average', index=False)
        result_group.to_excel(writer, sheet_name='mode_group', index=False)
        result_class.to_excel(writer, sheet_name='mode_012', index=False)
        result_class.groupby(['group_code', 'y_type']).mean().reset_index().to_excel(writer, sheet_name='mode_012_avg', index=False)

if __name__ == "__main__":
    # df = pd.read_csv('y_conversion.csv')
    # ddf = df.loc[(df['currency_code']=='USD')]
    # ddf.to_csv('y_conversion_spx.csv')

    calc_pred_class()

    # print(df)
