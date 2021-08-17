from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from preprocess.premium_calculation import trim_outlier
import global_vals

from results_analysis.lgbm_pred_merge_rotate import download_stock_pred_multi

def calc_score(testing_period=None):
    ''' select group with historically high prediction accuracy '''

    # download best factors
    f = download_stock_pred_multi('pca_mse_moretree', False, False)
    f = f.loc[(f['group_code']=='USD')&(f['alpha']=='extra')].set_index(['testing_period'])[['max_factor','min_factor']]

    # download membership table
    with global_vals.engine_ali.connect() as conn:
        mem = pd.read_sql(f"SELECT * FROM {global_vals.membership_table}_weekavg WHERE \"group\"='USD'", conn)  # download training history
    global_vals.engine_ali.dispose()


    # test selection process based on last testing_period
    if testing_period == None:
        testing_period = df['testing_period'].max()
        df_selection = df.loc[df['testing_period'] < testing_period]
        df_test = df.loc[df['testing_period'] >= testing_period]

    # # 1. Remove factor with majority proportion of prediction as 1
    # prc1 = {}
    # for name, g in df.groupby(['y_type']):
    #     prc1[name] = g['pred'].value_counts().to_dict()[1]/len(g)
    # x = pd.DataFrame(prc1, index=['index']).transpose().sort_values(by=['index'], ascending=False)
    # print(x)

    # calculate historic accuracy with all sample prior to prediction period
    group_acc = {}
    for name, g in df_selection.groupby(['group_code', 'group', 'y_type']):
        group_acc[name] = accuracy_score(g['pred'], g['actual'])
    group_acc_df = pd.DataFrame(group_acc, index=['0']).transpose().reset_index()
    group_acc_df.columns = ['group_code', 'group','y_type','accuracy']

    # 1. select factor used (accuracy high enough)
    avg = group_acc_df.groupby(['y_type']).mean().reset_index()
    factor = avg.loc[avg['accuracy']>0.5, 'y_type'].to_list()
    print(factor)

    # 2. select factor used (accuracy high enough)
    avg_g = group_acc_df.groupby(['group']).mean().sort_values(by=['accuracy'], ascending=False).reset_index()
    accu_group = avg_g.head(10)['group'].to_list()
    accu_group_ind = [x for x in accu_group if len(x)==6]
    accu_group_cur = [x for x in accu_group if len(x)==3]

    # calculate historic accuracy with all sample prior to prediction period
    # group_acc_test = {}
    # for name, g in df_test.groupby(['group_code', 'group', 'y_type']):
    #     group_acc_test[name] = accuracy_score(g['pred'], g['actual'])
    # group_acc_test_df = pd.DataFrame(group_acc, index=['0']).transpose().reset_index()
    # group_acc_test_df.columns = ['group_code', 'group','y_type','accuracy']
    #
    # group_acc_test_df = group_acc_test_df.merge(group_acc_df, on=['group_code', 'group', 'y_type'], suffixes=['','hist'])
    # group_acc_test_df = group_acc_test_df.sort_values(by=['accuracy'])
    #
    # print(group_acc_test_df)

    return df.set_index(['group','testing_period','y_type'])['pred'].unstack().reset_index(), factor, accu_group_ind, accu_group_cur

def download_ratios():
    ''' download ratios for all the stocks '''

    pred_df, factor_list, accu_group_ind, accu_group_cur = select_best_group()
    print(np.array([relativedelta(weeks=2)]*len(pred_df)))
    pred_df['period_end'] = pd.to_datetime(pred_df['testing_period'])+np.array([relativedelta(weeks=2)]*len(pred_df))
    pred_df = pred_df.drop(['testing_period'], axis=1)
    pred_df[factor_list] = (pred_df[factor_list]- 1)

    try:
        ratio = pd.read_csv('ratio.csv')
    except Exception as e:
        print(e)
        with global_vals.engine_ali.connect() as conn:
            ratio = pd.read_sql(f"SELECT ticker, period_end, icb_code, currency_code, stock_return_y, {','.join(factor_list)} "
                                f"FROM {global_vals.processed_ratio_table}_biweekly WHERE EXTRACT(YEAR FROM period_end) > 2019", conn)
        global_vals.engine_ali.dispose()
        ratio.to_csv('ratio.csv', index=False)

    ratio['icb_code'] = ratio['icb_code'].astype(str).str[:6]
    # ratio = ratio.loc[(ratio['icb_code'].isin(accu_group_ind))&(ratio['currency_code'].isin(accu_group_cur))]
    ratio[factor_list] = ratio[factor_list].replace([np.inf, -np.inf], np.nan)

    ratio_tf = []
    for name, g in ratio.groupby(['period_end','icb_code']):
        g_rs = RobustScaler(unit_variance=True).fit_transform(g[factor_list])
        # g[factor_list] = QuantileTransformer(n_quantiles=10).fit_transform(g[factor_list])
        ratio_tf.append(g_rs)
    ratio = ratio.sort_values(['period_end','icb_code'])
    ratio[[x+'_ind' for x in factor_list]] = np.concatenate(ratio_tf, axis=0)

    ratio_tf = []
    for name, g in ratio.groupby(['period_end','currency_code']):
        g_rs = RobustScaler(unit_variance=True).fit_transform(g[factor_list])
        # g[factor_list] = QuantileTransformer(n_quantiles=10).fit_transform(g[factor_list])
        ratio_tf.append(g_rs)
    ratio = ratio.sort_values(['period_end','currency_code'])
    ratio[[x+'_cur' for x in factor_list]] = np.concatenate(ratio_tf, axis=0)

    # ratio_tf = []
    # for name, g in ratio.groupby(['period_end']):
    #     g_rs = RobustScaler(unit_variance=True).fit_transform(g[factor_list])
    #     # g[factor_list] = QuantileTransformer(n_quantiles=10).fit_transform(g[factor_list])
    #     ratio_tf.append(g_rs)
    # x = np.concatenate(ratio_tf, axis=0)
    # ratio[factor_list] = np.concatenate(ratio_tf, axis=0)

    des = ratio.describe()

    ratio['period_end'] = pd.to_datetime(ratio['period_end'], format='%Y-%m-%d')
    ratio['icb_code'] = ratio['icb_code'].astype(str).str[:6]
    pred_df['period_end'] = pd.to_datetime(pred_df['period_end'], format='%Y-%m-%d')

    ratio = ratio.merge(pred_df, left_on=['icb_code','period_end'], right_on=['group','period_end'], how='left', suffixes=['','_pred_ind'])
    ratio = ratio.merge(pred_df, left_on=['currency_code','period_end'], right_on=['group','period_end'], how='left', suffixes=['','_pred_cur'])

    ratio['ind_score'] = np.nanmean(ratio[[x+'_ind' for x in factor_list]].values * ratio[[x+'_pred_ind' for x in factor_list]].values, axis=1)
    ratio['cur_score'] = np.nanmean(ratio[[x+'_cur' for x in factor_list]].values * ratio[[x+'_pred_cur' for x in factor_list]].values, axis=1)
    # ratio['ind_score'] = np.nanmean(ratio[factor_list].values * ratio[[x+'_pred_ind' for x in factor_list]].values, axis=1)
    # ratio['cur_score'] = np.nanmean(ratio[factor_list].values * ratio[[x+'_pred_cur' for x in factor_list]].values, axis=1)
    ratio['score'] = ratio[['ind_score','cur_score']].mean(axis=1)

    # ratio_tf['score_abs'] = ratio_tf['score'].abs()
    # ratio_tf.sort_values(['score_abs'], ascending=False).head(100).to_csv('high_score.csv')
    # exit(1)

    ratio = ratio.sort_values(['period_end','score'], ascending=False)
    score_df = ratio[['ticker','period_end','icb_code','currency_code','ind_score','cur_score','score','stock_return_y']]

    def show_dist_plot(df):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 6), dpi=120, constrained_layout=True)  # create figure for test & train boxplot

        ax = fig.add_subplot(1, 1, 1)
        test = df['score'].values.flatten()
        test = test[~np.isnan(test)]

        N, bins, patches = ax.hist(test, bins=1000, range=(-5, 5), weights=np.ones(len(test)) / len(test))

        # plt.show()
        plt.savefig(f'score_dist.png')

    show_dist_plot(score_df)
    print(score_df.describe())

    score_corr = {}
    cut10 = {}
    top01_group={}
    top01_group_score = {}
    top01_group_count = {}
    for name, g in score_df.groupby(['period_end']):
        g['score_qcut'] = pd.qcut(g['score'], q=10, labels=False, duplicates='drop')
        top01_group[name] = g.loc[g['score_qcut'].isin([0,1])].groupby(['icb_code','currency_code'])['stock_return_y'].mean()
        top01_group_score[name] = g.loc[g['score_qcut'].isin([0,1])].groupby(['icb_code','currency_code'])['score'].mean()
        top01_group_count[name] = g.loc[g['score_qcut'].isin([0,1])].groupby(['icb_code','currency_code'])['stock_return_y'].count()
        cut10[name] = g.groupby(['score_qcut'])['stock_return_y'].mean()
        score_corr[name] = {}
        for name1, g1 in g.groupby(['currency_code']):
            score_corr[name][name1] = g1['score'].corr(g1['stock_return_y'])

    score_corr_df = pd.DataFrame(score_corr).transpose()
    cut10_df = pd.DataFrame(cut10).transpose()

    top01 = pd.DataFrame(pd.DataFrame(top01_group).mean(axis=1))
    top01['count'] = pd.DataFrame(top01_group_count).mean(axis=1)
    top01['score'] = pd.DataFrame(top01_group_score).mean(axis=1)
    top01.columns = ['return','count','score']

    with pd.ExcelWriter(f'selection_test_{iter_name}_top1.xlsx') as writer:
        score_corr_df.to_excel(writer, sheet_name='corr')
        cut10_df.to_excel(writer, sheet_name='cut10')
        top01.reset_index().to_excel(writer, sheet_name='top01', index=False)

    return ratio_tf

if __name__ == "__main__":
    calc_score()

