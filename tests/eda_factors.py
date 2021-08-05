import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, QuarterEnd
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.manifold import TSNE

with global_vals.engine_ali.connect() as conn:
    # factors = pd.read_sql(f'SELECT * FROM {global_vals.factor_premium_table}', conn)
    # factors_avg = pd.read_sql(f'SELECT * FROM {global_vals.factor_premium_table}_weekavg', conn)
    factors = factors_bi = pd.read_sql(f'SELECT * FROM {global_vals.factor_premium_table}_biweekly', conn)
    formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)
global_vals.engine_ali.dispose()

col_list = list(set(formula.loc[formula['x_col'],'name'].to_list()) - {'debt_issue_less_ps_to_rent'})
factor_list = formula.loc[formula['factors'], 'name'].to_list()

# factors.to_csv('eda/test_factor_premium.csv', index=False)

# factors = pd.read_csv('factor_premium.csv')

def correl_fama_website():
    ''' test correlation of our factor with monthly factor premiums posted on French website with Fama-French 5 factor model
    -----------------------------------------------------------------
    Results: (X)
    Bad correlation: possible reasons:
    1. not using value-weighted
    2. not using 30%/50% as indicated on website
    3. not the same universe (french website use everything in CRSP)
    '''

    website = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv')
    website['period_end'] = pd.to_datetime(website['period_end'], format="%Y%m") + MonthEnd(0)
    print(website.columns)

    website_factor = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    our_factor = ['vol_0_30', 'market_cap_usd', 'book_to_price', 'ebtda_1yr', 'assets_1yr']

    print(factors.dtypes)
    factor = factors.loc[factors['group']=='USD']
    # factor = factor.groupby(['period_end']).mean().reset_index(drop=False)
    factor['period_end'] = pd.to_datetime(factor['period_end'], format="%Y-%m-%d")

    df = pd.merge(website, factor, on=['period_end'])

    cr = df[website_factor+our_factor].corr()
    print(cr)
    cr.to_csv('eda_factor_corr_website.csv')

    # for i in np.arange(5):
    #     sub_df = df[['period_end', website_factor[i], our_factor[i]]].set_index('period_end')

def eda_missing():
    df = factors.filter(col_list).isnull().sum()/len(factors)
    df.sort_values(ascending=False).to_csv('eda/missing.csv')
    # plt.hist(df)
    # plt.savefig('eda/missing.png')

def eda_correl():
    ''' test correlation of our factor '''

    global formula, factors

    df = factors[col_list].transpose().reset_index()

    df = df.merge(formula[['pillar', 'name']], left_on=['index'], right_on=['name'])
    df = df.sort_values(['pillar','name']).set_index(['pillar', 'name']).drop(['index'], axis=1).transpose()
    df.columns = [x[0]+'-'+x[1] for x in df.columns.to_list()]

    cr = df.astype(float).corr()
    sns.set(style="whitegrid", font_scale=0.5)
    ax = sns.heatmap(cr, cmap='PiYG', vmin=-1, vmax=1, label='small')
    plt.tight_layout()
    plt.savefig('eda/test_eda_corr.png', dpi=300)

    cr_df = cr.stack(-1)
    cr_df = cr_df.reset_index()
    cr_df.columns=['f1','f2','corr']
    cr_df = cr_df.loc[cr_df['corr']!=1].drop_duplicates(subset=['corr'])
    cr_df['corr_abs'] = cr_df['corr'].abs()
    cr_df = cr_df.sort_values(by=['corr_abs'], ascending=False)
    cr_df.to_csv('eda/test_eda_corr.csv', index=False)

def eda_vif():
    ''' Calculate VIF -> no abnormal comes to our attention beyond correlations '''
    df = factors[col_list].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = add_constant(df)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)

    print(vif)
    vif.sort_values(ascending=False).to_csv('eda/test_eda_vif.csv')

def sharpe_ratio():
    ''' calculate the sharpe ratio '''

    writer = pd.ExcelWriter(f'eda/sharpe_ratio.xlsx')  # create excel records
    factors['period_end'] = pd.to_datetime(factors['period_end'])
    col_list = list(factors.select_dtypes(float).columns)

    factors['year'] = factors['period_end'].dt.year
    factors['month'] = factors['period_end'].dt.month

    def calc(f):
        ret = f.mean(axis=0)
        sd = f.std(axis=0)
        return ret/sd

    def calc_by_col(col):
        dic = {}
        dic['all'] = calc(factors)
        for name, g in factors.groupby(col):
            dic[name] = calc(g[col_list])

        df = pd.DataFrame(dic)
        df['all_abs'] = df['all'].abs()
        df = df.sort_values('all_abs', ascending=False)
        df.to_excel(writer, col)
        df.iloc[:,1:].transpose().describe().transpose().to_excel(writer, f'{col}_des')

    calc_by_col('group')
    calc_by_col('year')
    calc_by_col('month')
    writer.save()

def plot_trend():

    global factors
    mean = list(factors.mean().abs().sort_values(ascending=False).index)
    #
    # font = {'size': 5}
    plt.rcParams["font.size"] = 5
    plt.rcParams["figure.figsize"] = (20,2)
    plt.rcParams["figure.dpi"] = 200

    factors = factors.sort_values(by=['period_end','group'])
    factors['period_end'] = factors['period_end'].astype(str)

    # by market
    fig = plt.figure(figsize=(12, 20), dpi=300)  # create figure for test & train boxplot

    df = factors.loc[factors['group'].str[-1]!='0']
    for i in mean[:10]:
        df_new = pd.pivot_table(df, index=['period_end'], columns=['group'], values=[i])
        ax = df_new.plot.line(linewidth=1)
        plt.legend(ncol=1)
        plt.tight_layout()
        plt.show()
        exit(1)

    # factors = factors.filter(['period_end','group','fwd_ey',''])

    for name, g in factors.groupby(['group']):
        g = g.set_index('period_end')
        ax = g.plot.line(linewidth=1)
        plt.legend(ncol=1)
        plt.tight_layout()
        plt.show()
        # plt.savefig('eda/test_eda_trend.png')


def plot_autocorrel():
    ''' plot the level of autocorrelation '''

    global factors

    df = factors.copy(1)
    writer = pd.ExcelWriter(f'eda/auto_correlation.xlsx')  # create excel records

    # ax = sm.graphics.tsa.plot_acf(df['epsq_1q'].values.squeeze(), lags=40)
    # ax = pd.plotting.autocorrelation_plot(df['epsq_1q'].values)
    # exit(1)

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    n = 14
    col_list = list(df.select_dtypes(float).columns)

    def corr_results(df):
        ac = {}
        for col in col_list:
            ac[col] = {}
            ac[col]['N95'] = z95 / np.sqrt(df[col].notnull().sum())
            for t in np.arange(1, n, 1):
                ac[col][t] = df[col].autocorr(t)
        results = pd.DataFrame(ac).transpose()
        m = results < np.transpose([results['N95'].array] * n)
        results = results.mask(m, np.nan)
        return results

    r = corr_results(df.loc[df['group'].str[-1]!='0'].groupby(['period_end']).mean())
    r.to_excel(writer, 'curr_avg')

    r = corr_results(df.loc[df['group'].str[-1]=='0'].groupby(['period_end']).mean())
    r.to_excel(writer, 'ind_avg')

    for name, g in df.groupby(['group']):
        print(name)
        r = corr_results(g)
        r.to_excel(writer, name)

    writer.save()

    # plt.tight_layout()
    # plt.show()

def test_if_persistent():

    df = factors.copy(1)
    m = np.array([(df[col_list].mean()<0).values]*df.shape[0])
    df[col_list] = df[col_list].mask(m, -df[col_list])

    plt.figure(figsize=(16, 16))
    g = df.groupby(['period_end']).mean().reset_index(drop=False)

    # for name, g in df.groupby('group'):
        # ax = fig.add_subplot(n, n, k)

    g = g.fillna(0)
    date_list = g['period_end'].to_list()
    new_g = g[col_list].transpose().values
    new_g = np.cumprod(new_g + 1, axis=1)
    ddf = pd.DataFrame(new_g, index=col_list, columns=date_list).sort_values(date_list[-1], ascending=False)
    ddf.to_csv('eda/persistent.csv')
    ddf = ddf.iloc[:10,:].transpose()
    print(ddf)
    plt.plot(ddf, label=list(ddf.columns))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig('eda/persistent.png')

def test_tsne():

    global factors

    df = factors.copy(1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(-99.9)

    df = df.loc[df['group'].str[-1]!='0']
    df['group'] = df['group'].str[:2]

    df["y"] = pd.qcut(df['market_cap_usd'], q=10, labels=False)
    tsne_results = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(df.iloc[:,2:].values)

    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="group",
        # palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )

    plt.show()

def check_smb():

    # df = pd.read_csv('data_tri_final.csv', usecols=['ticker','trading_day','stock_return_y'])
    # prc = 0.0
    # print(df['stock_return_y'].quantile([prc, 1-prc]))
    # print(df.describe())
    # # plt.hist(df['stock_return_y'], bins=1000)
    # # plt.xlim(-0.5, 0.5)
    # # plt.show()
    # # df.to_csv('eda/stock_return_y.csv')
    # exit(1)

    # with global_vals.engine.connect() as conn:
    #     mem = pd.read_sql(f"SELECT ticker, period_end, market_cap_usd_cut FROM {global_vals.membership_table} WHERE \"group\" not like '%0'", conn)
    # global_vals.engine.dispose()
    #
    # df = pd.pivot_table(mem, columns=['period_end'], index=['ticker'], values=['market_cap_usd_cut'])
    # df.to_csv('eda/check_smb_mem.csv')
    #
    # exit(1)
    #
    # from preprocess.ratios_calculations import resample_to_monthly, FillAllDay
    #
    # tri = pd.read_csv('cache_tri.csv', low_memory=False, usecols=['ticker','trading_day','tri'])
    # tri = FillAllDay(tri)
    # tri.update(tri.groupby('ticker')['tri'].fillna(method='ffill'))
    # tri.update(tri.groupby('ticker')['tri'].fillna(method='bfill'))
    #
    # tri = resample_to_monthly(tri, date_col='trading_day')  # Resample to monthly stock tri
    #
    # print(tri.groupby(['ticker'])['tri'].first())
    # print(tri.groupby(['ticker'])['tri'].last())
    #
    # df = pd.DataFrame()
    # df['first'] = tri.groupby(['ticker'])['tri'].first().values
    # df['last'] = tri.groupby(['ticker'])['tri'].last().values
    # df['rate'] = df['last']/df['first']
    # df['ticker'] = tri.groupby(['ticker'])['tri'].last().index
    #
    # with global_vals.engine.connect() as conn:
    #     universe = pd.read_sql(f'SELECT ticker, currency_code FROM {global_vals.dl_value_universe_table}', conn)
    # global_vals.engine.dispose()
    #
    # df = df.merge(universe, on=['ticker'], how='left')
    #
    # # df = tri.groupby(['ticker'])['tri'].first()/tri.groupby(['ticker'])['tri'].last()
    # df.to_csv('eda/check_smb_from_tri.csv')
    #
    # exit(1)

    ratio = pd.read_csv('all_data.csv', usecols=['period_end','currency_code','ticker','market_cap_usd','stock_return_y'])
    ratio = ratio.sort_values(['stock_return_y']).dropna(how='any')

    premium = {}
    for name, g in ratio.groupby(['period_end', 'currency_code']):
        try:
            premium[name] = {}
            g[f'cut'] = pd.qcut(g['market_cap_usd'], q=[0, 0.2, 0.8, 1], retbins=False, labels=[0, 1, 2])
            premium[name]['small'] = g.loc[g['cut'] == 0, 'stock_return_y'].mean()
            premium[name]['small_comp'] = g.loc[g['cut'] == 0, 'ticker'].to_list()
            premium[name]['small_ret'] = g.loc[g['cut'] == 0, 'stock_return_y'].to_list()
            # premium[name]['small_me'] = g.loc[g['cut'] == 0, 'market_cap_usd'].to_list()

            premium[name]['mid'] = g.loc[g['cut'] == 1, 'stock_return_y'].mean()
            premium[name]['mid_comp'] = g.loc[g['cut'] == 1, 'ticker'].to_list()
            premium[name]['mid_ret'] = g.loc[g['cut'] == 1, 'stock_return_y'].to_list()
            # premium[name]['mid_me'] = g.loc[g['cut'] == 1, 'market_cap_usd'].to_list()

            premium[name]['large'] = g.loc[g['cut'] == 2, 'stock_return_y'].mean()
            premium[name]['large_comp'] = g.loc[g['cut'] == 2, 'ticker'].to_list()
            premium[name]['large_ret'] = g.loc[g['cut'] == 2, 'stock_return_y'].to_list()
            # premium[name]['large_me'] = g.loc[g['cut'] == 2, 'market_cap_usd'].to_list()

        except Exception as e:
            print(name, e)

    ddf = pd.DataFrame(premium)
    ddf.transpose().to_csv('eda/check_smb_premium.csv')
    exit(1)
    # df = pd.pivot_table(ratio.loc[ratio['currency_code']=='USD'], index=['period_end'], columns=['ticker'], values=['market_cap_usd'])


    ratio.loc[ratio['currency_code']=='USD'].dropna(how='any').to_csv('eda/check_smb_origin_usd.csv', index=False)
    # df = pd.pivot_table(ratio.loc[ratio['currency_code']=='CNY'], index=['period_end'], columns=['ticker'], values=['market_cap_usd'])
    ratio.loc[ratio['currency_code']=='CNY'].dropna(how='any').to_csv('eda/check_smb_origin_cny.csv', index=False)


    exit(0)

    df = factors.copy(1)
    # df = df.loc[factors['group'].str[-1]!='0']
    df = pd.pivot_table(df, index=['period_end'], columns=['group'], values=['vol_30_90'])

    arr = -df.values
    df_prod = np.cumprod(arr + 1, axis=0)
    ddf = pd.DataFrame(df_prod, index=df.index, columns=df.columns)
    ddf.to_csv('eda/check_vol_prod.csv')
    exit(0)

    plt.hist(df, label=list(df.columns))
    plt.legend(bbox_to_anchor=(0.7,1), loc='upper left', fontsize='xx-small')
    plt.show()

    # df.to_csv('eda/check_smb.csv')

def average_absolute_mean():
    df = factors[col_list].abs().mean().sort_values(ascending=False)
    df = pd.DataFrame({'abs':df, 'rank':list(range(len(df)))})
    df.to_csv('eda/average_absolute_mean.csv')
    print(df)

def test_cluster():

    from sklearn.cluster import KMeans, DBSCAN, OPTICS
    from sklearn import metrics
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn import decomposition

    aa = factors.copy(1)[col_list].values
    cluster_no = 5
    # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
    bb = np.nan_to_num(aa, -99.9)
    bb = StandardScaler().fit_transform(bb)

    kmeans = KMeans(cluster_no).fit(bb)
    centre = kmeans.cluster_centers_

    # pd.DataFrame(centre.transpose(), index=col_list).to_csv('eda/cluster_kmean.csv')

    # db = DBSCAN(eps=0.1, min_samples=4).fit(bb)

    db = OPTICS(min_samples=4).fit(bb)

    # centre = db.cluster_centers_
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    from collections import Counter
    print(Counter(labels))
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhoette Coefficient: %0.3f" % metrics.silhouette_score(bb, labels))
    exit(1)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(bb, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    # dbscan = make_pipeline(StandardScaler(), DBSCAN).fit(bb)
    # ppca = make_pipeline(StandardScaler(), decomposition.PCA).fit(bb)
    print(bb)

    # results = kmeans[-1].inertia_
    # print(results)

    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]

    kmeans = KMeans(n_clusters=5).fit(bb)
    # dbscan = DBSCAN(eps=0.3).fit(bb)

    clustering = kmeans.predict(bb)
    cluster_centers = np.zeros((cluster_no, bb.shape[-1]))

    # Finding cluster centers for later inference.
    for i in range(cluster_no):
        cluster_centers[i, :] = np.mean(bb[clustering == i], axis=0)

    clustering = np.reshape(clustering, (aa.shape[0], aa.shape[1]))

    return clustering

def dist_all():

    fig = plt.figure(figsize=(6, 18), dpi=120,  constrained_layout=True)  # create figure for test & train boxplot

    k=1
    name = ['original','avg','bi']
    for df in [factors_bi]:
        ax = fig.add_subplot(3,1,k)
        test = df[factor_list].values.flatten()
        test = test[~np.isnan(test)]
        max = test.max()
        min = test.min()

        prc = 0.01
        # test = np.abs(test)
        min = np.quantile(test, prc)
        max = np.quantile(test, 1-prc)

        print(pd.Series(test).describe())
        print(min, max)
        N, bins, patches = ax.hist(test, bins=1000, range=(-0.05,0.05), weights=np.ones(len(test)) / len(test))

        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        plt.ylim((0,0.01))
        ax.set_ylabel(name[k-1], fontsize=20)
        k+=1

    # plt.show()
    plt.savefig(f'eda/pdf_all_factor_cut.png')
    plt.close()

def dist_all_train_test():

    fig = plt.figure(figsize=(16, 16), dpi=120, constrained_layout=True)  # create figure for test & train boxplot

    first_testing = dt.date(2017,9,24)
    factor_list = formula.loc[formula['factors'],'name'].to_list()

    factors_bi['group_code'] = factors_bi['group'].str.len()
    factors_bi_train = factors_bi.loc[factors_bi['period_end']<first_testing]
    factors_bi_test = factors_bi.loc[factors_bi['period_end']>=first_testing]

    k=1
    labels = ['train-currency','train-industry','test-currency','test-industry']
    for df in [factors_bi_train, factors_bi_test]:
        for name, g in df.groupby(['group_code']):
            ax = fig.add_subplot(2,2,k)
            test = g[factor_list].values.flatten()
            test = test[~np.isnan(test)]

            low = np.quantile(test, 1/3)
            high = np.quantile(test, 2/3)

            # print(pd.Series(test).describe())
            print(low, high)
            N, bins, patches = ax.hist(test, bins=1000, range=(-0.1,0.1), weights=np.ones(len(test)) / len(test))

            for i in range(0, 1000):
                if (bins[i]>low) and (bins[i+1]<high):
                    patches[i].set_facecolor('g')
                else:
                    patches[i].set_facecolor('r')

            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
            plt.ylim((0,0.005))
            ax.set_xlabel(labels[k-1], fontsize=20)
            k+=1

    plt.savefig(f'dist_all_train_test.png')
    plt.close()

if __name__ == "__main__":
    # correl_fama_website()
    # eda_missing()
    # eda_correl()
    # eda_vif()
    # plot_autocorrel()
    # plot_trend()

    # sharpe_ratio()
    # test_if_persistent()
    # average_absolute_mean()

    # check_smb()

    ## Clustering
    # test_cluster()
    # test_tsne()

    # dist_all()
    dist_all_train_test()