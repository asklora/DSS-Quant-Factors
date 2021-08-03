from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
import numpy as np

import global_vals

# r_name = 'indoverfit'
# r_name = 'testing'
# r_name = 'laskweekavg'
r_name = 'lastweekavg'
# r_name = 'lastweekavg_newmacros'
r_name = 'biweekly'
r_name = 'biweekly_new'
r_name = 'biweekly_ma'
# r_name = 'test_stable9_re'


iter_name = r_name

def download_stock_pred(count_pred=True):
    ''' download training history and training prediction from DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_lgbm_class P "
                     f"INNER JOIN {global_vals.result_score_table}_lgbm_class S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history      
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number','group'], keep='last')
    result_all = result_all.drop(['cv_number'], axis=1)

    # save counting label to csv
    if count_pred:
        count_i = {}
        for name,g in result_all.groupby(['group_code', 'testing_period', 'y_type']):
            count_i[name] = g['pred'].value_counts().to_dict()
        pd.DataFrame(count_i).transpose().to_csv(f'score/result_pred_count_{iter_name}.csv')

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

    # df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    # df = df.dropna(how='any')

    result_dict = {}
    for name, g in df.groupby(['group_code', 'testing_period', 'y_type']):
        result_dict[name] = {}
        for name1, g1 in g.groupby(['pred']):
            result_dict[name][name1] = accuracy_score(g1['pred'], g1['actual'])

    df = pd.DataFrame(result_dict).transpose().reset_index()
    df.columns = ['group_code', 'testing_period', 'y_type'] + df.columns.to_list()[3:]
    return df

def calc_confusion(results):
    ''' calculate the confusion matrix for multi-class'''

    # calculate the mode of each CV
    # results = results.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)

    lst = []
    for name, df in results.groupby(['group_code', 'testing_period', 'y_type']):
        labels = list(set(df['actual'].dropna().unique()))
        x = multilabel_confusion_matrix(df['pred'], df['actual'], labels=labels)
        x = pd.DataFrame(x.reshape((2*len(labels),2)), columns=['Label-N','Label-P'], index=[f'{int(x)}{y}' for x in labels for y in['N','P']])
        # x = x.divide(x.sum(axis=1), axis=0)
        x = (x/len(df)).reset_index()
        x[['group_code', 'testing_period', 'y_type']] = name
        lst.append(x)

    confusion_df = pd.concat(lst).groupby(['y_type', 'group_code', 'index']).mean().reset_index()

    return confusion_df

def combine_mode_group(df):
    ''' calculate accuracy score by each industry/currency group '''
    #
    # df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    # df = df.dropna(how='any')

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'group']):
        result_dict[name] = accuracy_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict, index=['0']).transpose().reset_index()
    df.columns = ['group_code', 'y_type','group','accuracy']

    with global_vals.engine_ali.connect() as conn:
        icb_name = pd.read_sql(f"SELECT DISTINCT code_6 as group, name_6 as name FROM icb_code_explanation", conn)  # download training history
        icb_count = pd.read_sql(f"SELECT \"group\", avg(num_ticker) as num_ticker FROM icb_code_count GROUP BY \"group\"", conn)  # download training history
    global_vals.engine_ali.dispose()

    df = df.merge(icb_name, on=['group'], how='outer')
    df = df.merge(icb_count, on=['group'], how='outer')

    return df

def combine_mode_time(df):
    ''' calculate accuracy score by each industry/currency group '''

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'testing_period']):
        result_dict[name] = accuracy_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict, index=['0']).transpose().reset_index()
    df.columns = ['group_code', 'y_type','testing_period','accuracy']

    return df

def calc_pred_class():
    ''' Calculte the accuracy_score if combine CV by mean, median, mode '''

    df = download_stock_pred()
    df[['pred','actual']] = df[['pred','actual']].astype(int)

    df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    df = df.dropna(how='any')

    result_time = combine_mode_time(df)
    confusion_df = calc_confusion(df)
    result_group = combine_mode_group(df)
    result_class = combine_mode_class(df)

    # results = {}
    # for i in ['mean', 'median', 'mode']:
    #     results[i] = combine_pred_class(df, i)

    # r = pd.DataFrame(results).reset_index()
    # r.columns = ['group_code', 'testing_period', 'y_type'] + ['mean', 'median', 'mode']

    with pd.ExcelWriter(f'score/result_pred_accuracy_{iter_name}.xlsx') as writer:
        # r.to_excel(writer, sheet_name='original', index=False)
        # r.groupby(['group_code', 'y_type']).mean().reset_index().to_excel(writer, sheet_name='cv_average', index=False)
        result_time.groupby(['group_code', 'y_type']).mean().to_excel(writer, sheet_name='average')
        result_group.to_excel(writer, sheet_name='mode_group', index=False)
        result_time.to_excel(writer, sheet_name='mode_time', index=False)
        confusion_df.to_excel(writer, sheet_name='confusion', index=False)
        # result_class.to_excel(writer, sheet_name='mode_012', index=False)
        result_class.groupby(['group_code', 'y_type']).mean().reset_index().to_excel(writer, sheet_name='mode_012_avg', index=False)

if __name__ == "__main__":
    # df = pd.read_csv('y_conversion.csv')
    # ddf = df.loc[(df['currency_code']=='USD')]
    # ddf.to_csv('y_conversion_spx.csv')

    calc_pred_class()

    # print(df)
