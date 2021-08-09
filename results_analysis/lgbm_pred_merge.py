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

restart = True
model = 'lgbm'
period = 'weekavg' # biweekly / weekavg
r_name = 'lastweekavg_tv_maxret'

iter_name = r_name

def download_stock_pred(count_pred=True):
    ''' download training history and training prediction from DB '''

    try:
        if restart:
            raise Exception('------------> Restart')
        result_all = pd.read_csv('cache_result_all.csv')
    except Exception as e:
        print(e)
        with global_vals.engine_ali.connect() as conn:
            query = text(f"SELECT P.group, P.pred, P.actual, S.group_code, S.y_type, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_{model}_class P "
                         f"INNER JOIN {global_vals.result_score_table}_{model}_class S ON S.finish_timing = P.finish_timing "
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
            count_i = count_i.apply(pd.value_counts).transpose()
            count_i = count_i.apply(lambda x: x/count_i.sum(1).values).reset_index()
            with pd.ExcelWriter(f'score/{model}_pred_count_{iter_name}.xlsx') as writer:
                count_i.to_excel(writer, sheet_name='count', index=False)
                count_i['year'] = count_i['testing_period'].dt.year
                count_i.groupby(['year']).mean().to_excel(writer, sheet_name='year')
                count_i.groupby(['y_type']).mean().to_excel(writer, sheet_name='factor')
                pd.pivot_table(result_all, index=['group_code', 'y_type', 'testing_period','group'], columns=['cv_number']
                               , values=['pred','actual']).to_excel(writer, sheet_name='cv')
            print('----------------------> Finish writing counting to csv')

        # convert pred/actual class to int & combine 5-CV with mode
        result_all = result_all.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
        result_all = result_all.dropna(subset=['actual'])
        result_all.to_csv('cache_result_all.csv', index=False)

    result_all[['pred','actual']] = result_all[['pred','actual']].astype(int)

    # remove last period no enough data to measure reliably
    result_all = result_all.loc[result_all['testing_period']<result_all['testing_period'].max()]

    # label period_end as the time where we assumed to train the model (try to incorporate all information before period_end)
    if period == 'biweekly':
        result_all['period_end'] = pd.to_datetime(result_all['testing_period']).apply(lambda x: x + relativedelta(weeks=2))
    else:
        result_all['period_end'] = pd.to_datetime(result_all['testing_period']) + MonthEnd(1)

    # map the original premium to the prediction result
    result_all = add_org_premium(result_all)

    return result_all

def add_org_premium(df):
    ''' map the original premium to the prediction result '''

    factor = df['y_type'].unique()
    with global_vals.engine_ali.connect() as conn:
        actual_ret = pd.read_sql(f"SELECT \"group\", period_end, {','.join(factor)} "
                                 f"FROM {global_vals.factor_premium_table}_{period}", conn)  # download training history
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
        fig = plt.figure(figsize=(12, 8), dpi=120, constrained_layout=True)
        k=1
        for name, g in df.groupby(['group_code']):
            ax = fig.add_subplot(2,1,k)
            g['period_end'] = pd.to_datetime(g['period_end'])
            df_plot = pd.pivot_table(g, index=['period_end'], columns=['y_type'], values=['accuracy'], aggfunc='mean')
            ax.plot(df_plot, label=[x[1] for x in df_plot.columns.to_list()])
            ax.set_ylabel(name, fontsize=20)
            plt.ylim((0,1))
            if k==1:
                plt.legend(loc='upper left', fontsize='small')
            k+=1
        plt.savefig(f'score/{model}_pred_accu_time_{iter_name}.png')

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

def calc_performance(df, accu_df, plot_performance_yearly=False, plot_performance_all=True):
    ''' calculate accuracy score by each industry/currency group '''

    col_list = list(df['y_type'].unique())

    # test on factor 'vol_30_90' first
    df = df.loc[df['y_type']=='vol_30_90']

    # 1. calculate return per our prediction & actual class
    df_list = []
    for i in ['pred', 'actual']:
        r = pd.pivot_table(df, index=['group_code', 'y_type', 'period_end'], columns=[i], values=['premium'], aggfunc='mean')
        r.columns = [f'{i}_{x[1]}' for x in r.columns.to_list()]
        r[f'ret_{i}'] = r.iloc[:,2].fillna(0) - r.iloc[:,0].fillna(0)
        df_list.append(r)

    mean_premium = -df.groupby(['group_code', 'y_type', 'period_end'])['premium'].mean()
    mean_premium.name = 'ret_pred'
    mean_premium = mean_premium.reset_index()
    mean_premium['y_type'] = 'mean_premium'

    results = pd.concat(df_list, axis=1).reset_index().sort_values(['group_code', 'y_type', 'period_end'])
    results = pd.concat([results, mean_premium], axis=0)

    # 2. add accuracy for each (group_code, y_type, testing_period)
    results = results.merge(accu_df, on=['group_code', 'y_type', 'period_end'], how='left')

    # 3. read index_return from DB & add for plot
    with global_vals.engine_ali.connect() as conn:
        index_ret = pd.read_sql(f"SELECT ticker as y_type, period_end, stock_return_y as ret_pred "
                                f"FROM {global_vals.processed_ratio_table}_{period} "
                                f"WHERE ticker like '.%%' AND period_end >= '{df['period_end'].min().strftime('%Y-%m-%d')}'",conn)
    global_vals.engine_ali.dispose()
    index_ret['period_end'] = pd.to_datetime(index_ret['period_end'])
    index_ret_avg = index_ret.groupby(['period_end']).mean()['ret_pred'].reset_index()
    index_ret_avg['y_type'] = 'index_avg'
    index_ret = index_ret.loc[index_ret['y_type'].isin(['.SPX', '.CSI300','index_avg'])]

    results_plot = pd.concat([results, index_ret, index_ret_avg], axis=0)
    results_plot['year'] = results_plot['period_end'].dt.year

    # results_plot['ret_pred'].update(results_plot['ret_pred'].fillna(0))

    # 4. plot performance with plt.plot again index
    num_year = len(results_plot['year'].dropna().unique())
    currency = results_plot.loc[(results_plot['group_code'].isnull()) | (results_plot['group_code'] == 'currency')]
    industry = results_plot.loc[(results_plot['group_code'].isnull()) | (results_plot['group_code'] == 'industry')]
    if plot_performance_yearly:
        # 4.1 - plot cumulative return for every year
        k=1
        fig = plt.figure(figsize=(28, 8), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
        for part_name, part in {'currency':currency, 'industry':industry}.items():
            for name, g in part.groupby(['year']):
                ax = fig.add_subplot(2, num_year, k)
                g = pd.pivot_table(g, index=['period_end'], columns=['y_type'], values='ret_pred', aggfunc='mean')
                g[g.columns.to_list()] = np.cumprod(g + 1, axis=0)

                # define plot start point
                if period == 'biweekly':
                    start_index = g.index[0] - relativedelta(weeks=2)
                else:
                    start_index = g.index[0] - MonthEnd(1)
                g.loc[start_index, :] = 1

                # format plot
                ax.plot(g.sort_index(), label=g.columns.to_list())
                myFmt = mdates.DateFormatter('%m')
                ax.xaxis.set_major_formatter(myFmt)
                plt.ylim((0.8,1.8))
                plt.xlim([dt.date(int(name)-1, 12, 31), dt.date(int(name), 12, 31)])

                if k in [1, 1+num_year]:
                    ax.set_ylabel(part_name, fontsize=20)
                if k > num_year:
                    ax.set_xlabel(int(name), fontsize=20)
                if k == 1:
                    plt.legend(loc='upper left', fontsize='large')
                k+=1

        plt.savefig(f'score/{model}_performance_{iter_name}_yearly.png')

    if plot_performance_all:
        # 4.2 - plot cumulative return for entire testing period
        k=1
        fig = plt.figure(figsize=(8, 8), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
        for part_name, g in {'currency':currency, 'industry':industry}.items():
            ax = fig.add_subplot(2,1,k)

            # add index benchmark
            g = pd.pivot_table(g, index=['period_end'], columns=['y_type'], values='ret_pred', aggfunc='mean')
            g[g.columns.to_list()] = np.cumprod(g + 1, axis=0)
            ax.plot(g, label=g.columns.to_list())        # plot cumulative return for the year
            plt.ylim((0.8,1.8))
            ax.set_ylabel(part_name, fontsize=20)
            if k == 1:
                plt.legend(loc='upper left', fontsize='large')
            k+=1

        plt.savefig(f'score/{model}_performance_{iter_name}.png')

    return results

def calc_pred_class():
    ''' Calculte the accuracy_score if combine CV by mean, median, mode '''

    df = download_stock_pred()

    # calc_auc(df)
    result_time = combine_mode_time(df)
    result_performance = calc_performance(df, result_time)
    confusion_df = calc_confusion(df)
    result_group = combine_mode_group(df)
    result_class = combine_mode_class(df)

    with pd.ExcelWriter(f'score/{model}_pred_accuracy_{iter_name}.xlsx') as writer:
        result_time.groupby(['group_code', 'y_type']).mean().to_excel(writer, sheet_name='average')
        result_group.to_excel(writer, sheet_name='mode_group', index=False)
        pd.pivot_table(result_group, index=['group'], columns=['y_type'], values='accuracy').to_excel(writer, sheet_name='mode_group_pivot')

        result_time.to_excel(writer, sheet_name='mode_time', index=False)
        pd.pivot_table(result_time, index=['period_end'], columns=['y_type'], values='accuracy').to_excel(writer, sheet_name='mode_time_pivot')

        confusion_df.to_excel(writer, sheet_name='confusion', index=False)
        result_performance.to_excel(writer, sheet_name='performance', index=False)
        result_class.groupby(['group_code', 'y_type']).mean().reset_index().to_excel(writer, sheet_name='mode_012_avg', index=False)

if __name__ == "__main__":
    calc_pred_class()
