import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, \
    multilabel_confusion_matrix, roc_curve
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates

import global_vals

# r_name = 'indoverfit'
# r_name = 'testing'
# r_name = 'laskweekavg'
r_name = 'lastweekavg'
# r_name = 'lastweekavg_newmacros'
r_name = 'biweekly'
r_name = 'biweekly_new'
r_name = 'biweekly_ma'
r_name = 'biweekly_crossar'

# r_name = 'test_stable9_re'

iter_name = r_name

def download_stock_pred(count_pred=True):
    ''' download training history and training prediction from DB '''

    try:
        result_all = pd.read_csv('result_all.csv')
    except Exception as e:
        print(e)
        with global_vals.engine_ali.connect() as conn:
            query = text(f"SELECT P.*, S.group_code, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_lgbm_class P "
                         f"INNER JOIN {global_vals.result_score_table}_lgbm_class S ON S.finish_timing = P.finish_timing "
                         f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
            result_all = pd.read_sql(query, conn)       # download training history
        global_vals.engine_ali.dispose()
        print(result_all.shape)

        # remove duplicate samples from running twice when testing
        result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number','group'], keep='last')
        # result_all = result_all.drop(['cv_number'], axis=1)

        # save counting label to csv
        if count_pred:
            count_i = pd.pivot_table(result_all, index=['cv_number','group'], columns=['group_code', 'testing_period', 'y_type'], values=['pred'])
            count_i = count_i.apply(pd.value_counts)
            count_i.transpose().to_csv(f'score/result_pred_count_{iter_name}.csv')

        # convert pred/actual class to int & combine 5-CV with mode
        result_all[['pred','actual']] = result_all[['pred','actual']].astype(int)
        result_all = result_all.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
        result_all = result_all.dropna(subset=['actual'])
        result_all.to_csv('result_all.csv', index=False)

    # label period_end as the time where we assumed to train the model (try to incorporate all information before period_end)
    result_all['period_end'] = pd.to_datetime(result_all['testing_period']).apply(lambda x: x + relativedelta(weeks=2))
    result_all = result_all.loc[result_all['period_end']<result_all['period_end'].max()] # remove last period no enough data to measure reliably

    # map the original premium to the prediction result
    result_all = add_org_premium(result_all)

    return result_all

def add_org_premium(df):
    ''' map the original premium to the prediction result '''

    factor = df['y_type'].unique()
    with global_vals.engine_ali.connect() as conn:
        actual_ret = pd.read_sql(f"SELECT \"group\", period_end, {','.join(factor)} "
                                 f"FROM {global_vals.factor_premium_table}_biweekly", conn)  # download training history
    global_vals.engine_ali.dispose()

    actual_ret = actual_ret.set_index(['group', 'period_end']).stack().reset_index()
    actual_ret.columns = ['group', 'period_end', 'y_type', 'premium']

    df = df.merge(actual_ret, on=['group', 'period_end', 'y_type'], how='left')
    return df

def test_mean_med_mode(df, agg_type):
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

    result_dict = {}
    for name, g in df.groupby(['group_code', 'period_end', 'y_type']):
        result_dict[name] = {}
        for name1, g1 in g.groupby(['pred']):
            result_dict[name][name1] = accuracy_score(g1['pred'], g1['actual'])

    df = pd.DataFrame(result_dict).transpose().reset_index()
    df.columns = ['group_code', 'testing_period', 'y_type'] + df.columns.to_list()[3:]

    return df

def combine_mode_group(df):
    ''' calculate accuracy score by each industry/currency group '''

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

def combine_mode_time(df, time_plot=True):
    ''' calculate accuracy score by each industry/currency group '''

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'period_end']):
        result_dict[name] = accuracy_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict, index=['0']).transpose().reset_index()
    df.columns = ['group_code', 'y_type','period_end','accuracy']

    if time_plot:
        df['period_end'] = pd.to_datetime(df['period_end'])
        df_plot = pd.pivot_table(df, index=['period_end'], columns=['y_type'], values=['accuracy'], aggfunc='mean')
        fig = plt.figure(figsize=(12, 4), dpi=120, constrained_layout=True)
        plt.plot(df_plot)
        plt.savefig(f'score/result_pred_accu_time_{iter_name}.png')

    return df

def calc_confusion(results):
    ''' calculate the confusion matrix for multi-class'''

    lst = []
    for name, df in results.groupby(['group_code', 'period_end', 'y_type']):
        labels = list(set(df['actual'].dropna().unique()))
        x = multilabel_confusion_matrix(df['pred'], df['actual'], labels=labels)
        x = pd.DataFrame(x.reshape((2*len(labels),2)), columns=['Label-N','Label-P'], index=[f'{int(x)}{y}' for x in labels for y in['N','P']])
        # x = x.divide(x.sum(axis=1), axis=0)
        x = (x/len(df)).reset_index()
        x[['group_code', 'testing_period', 'y_type']] = name
        lst.append(x)

    confusion_df = pd.concat(lst).groupby(['y_type', 'group_code', 'index']).mean().reset_index()

    return confusion_df

def calc_auc(results):
    ''' plot ROC curve and calculate AUC '''

    result_dict = {}
    for name, df in results.groupby(['group_code', 'period_end', 'y_type']):
        fpr, tpr, thresholds = roc_curve(df['actual'], df['pred'], pos_label=2)
        result_dict[name] = roc_auc_score(df['actual'], df['pred'])


def calc_performance(df, compare_with_index=True, plot_performance=True):
    ''' calculate accuracy score by each industry/currency group '''

    r = pd.pivot_table(df, index=['group_code', 'y_type', 'period_end'], columns=['pred'], values=['premium'], aggfunc='mean')
    r.columns = [x[1] for x in r.columns.to_list()]
    r['ret'] = r.iloc[:,2] - r.iloc[:,0]
    r = r.reset_index()

    r = r.loc[r['period_end']<r['period_end'].max()]
    r = r.loc[r['y_type']!='market_cap_usd']
    r['year'] = r['period_end'].dt.year
    col_list = list(r['y_type'].unique())

    fig = plt.figure(figsize=(28,8), dpi=120, constrained_layout=True)  # create figure for test & train boxplot

    if compare_with_index:
        with global_vals.engine_ali.connect() as conn:
            index_ret = pd.read_sql(f"SELECT ticker, period_end, stock_return_y FROM {global_vals.processed_ratio_table}_biweekly WHERE ticker like '.%%'",conn)
        global_vals.engine_ali.dispose()
        index_ret['period_end'] = pd.to_datetime(index_ret['period_end'])
        index_ret.columns = ['y_type','period_end','ret']
        r = pd.concat([r, index_ret], axis=0)

    if plot_performance:
        k=1
        for name, g in r.groupby(['group_code','year']):
            ax = fig.add_subplot(2,5,k)

            # add index benchmark
            g = pd.concat([g, index_ret.loc[index_ret['y_type'].isin(['.SPX','.CSI300'])]], axis=0)
            g['year'] = g['period_end'].dt.year     # fill in year

            g = pd.pivot_table(g, index=['period_end'], columns=['y_type'], values=['ret'], aggfunc='mean')
            g.columns = [x[1] for x in g.columns.to_list()]
            g = g.dropna(subset=col_list, how='all')
            g = g.fillna(0)

            g[g.columns.to_list()] = np.cumprod(g + 1, axis=0)

            ax.plot(g, label=g.columns.to_list())        # plot cumulative return for the year
            myFmt = mdates.DateFormatter('%m')
            ax.xaxis.set_major_formatter(myFmt)
            plt.ylim((0.8,1.8))
            plt.xlim([dt.date(int(name[1]), 1, 1), dt.date(int(name[1]), 12, 31)])

            if k in [1, 6]:
                ax.set_ylabel(name[0], fontsize=20)
            if k > 5:
                ax.set_xlabel(int(name[1]), fontsize=20)
            if k == 1:
                plt.legend(loc='upper left', fontsize='x-large')
            k+=1

        plt.savefig(f'score/performance_{iter_name}.png')

    return r

def calc_pred_class():
    ''' Calculte the accuracy_score if combine CV by mean, median, mode '''

    df = download_stock_pred()

    # calc_auc(df)
    result_performance = calc_performance(df)
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
        result_performance.to_excel(writer, sheet_name='performance', index=False)
        result_class.groupby(['group_code', 'y_type']).mean().reset_index().to_excel(writer, sheet_name='mode_012_avg', index=False)

if __name__ == "__main__":
    # df = pd.read_csv('y_conversion.csv')
    # ddf = df.loc[(df['currency_code']=='USD')]
    # ddf.to_csv('y_conversion_spx.csv')

    calc_pred_class()

    # print(df)
