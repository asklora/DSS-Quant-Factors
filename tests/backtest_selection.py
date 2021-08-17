from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from preprocess.premium_calculation import trim_outlier
import global_vals
from pandas.tseries.offsets import MonthEnd

from results_analysis.lgbm_pred_merge_rotate import download_stock_pred_multi

def calc_score():
    ''' select group with historically high prediction accuracy '''

    # download best factors
    f = download_stock_pred_multi('pca_mse_moretree', False, False)
    f = f.loc[(f['group_code']=='USD')&(f['alpha']=='extra')]
    f['testing_period'] = f['testing_period'] + MonthEnd(1)
    f = f.set_index(['testing_period'])
    factor_list = [x[2:] for x in f.columns.to_list()[2:]]

    # download membership table
    with global_vals.engine_ali.connect() as conn:
        ratio = pd.read_sql(f"SELECT * FROM {global_vals.processed_ratio_table}_weekavg WHERE currency_code='USD' AND period_end>'2017-08-31'", conn)  # download training history
    global_vals.engine_ali.dispose()

    ratio[factor_list] = ratio[factor_list].replace([np.inf, -np.inf], np.nan)

    ratio_tf = []
    for name, g in ratio.groupby(['period_end','icb_code']):
        try:
            g_rs = RobustScaler(unit_variance=True).fit_transform(g[factor_list])
        except:
            g_rs = StandardScaler().fit_transform(g[factor_list])
        # g[factor_list] = QuantileTransformer(n_quantiles=10).fit_transform(g[factor_list])
        ratio_tf.append(g_rs)
    # ratio[factor_list] = RobustScaler(unit_variance=True).fit_transform(ratio[factor_list])
    ratio[factor_list] = np.concatenate(ratio_tf, axis=0)
    ratio = ratio.merge(f, left_on=['period_end'], right_index=True)

    ratio['score'] = (ratio[factor_list].values*ratio[['y_'+x for x in factor_list]].values).sum(axis=1)
    ratio.sort_values(['period_end','score'],ascending=False)[['ticker','period_end','score','stock_return_y']].to_csv('score.csv', index=False)

    return ratio[['ticker','period_end','score','stock_return_y']]


def score_group():
    ''' download ratios for all the stocks '''

    score_df = calc_score()

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
        cut10[name] = g.groupby(['score_qcut'])['stock_return_y'].mean()

    cut10_df = pd.DataFrame(cut10).transpose()

    with pd.ExcelWriter(f'selection_test.xlsx') as writer:
        cut10_df.to_excel(writer, sheet_name='cut10')


if __name__ == "__main__":
    score_group()

