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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


with global_vals.engine.connect() as conn:
    factors = pd.read_sql(f'SELECT * FROM {global_vals.factor_premium_table}', conn)
    formula = pd.read_sql(f'SELECT name, pillar FROM {global_vals.formula_factors_table}', conn)
global_vals.engine.dispose()

col_list = list(factors.select_dtypes(float).columns)

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

def eda_correl():
    ''' test correlation of our factor '''

    global formula, factors

    df = factors.iloc[:,2:].transpose().reset_index()

    df = df.merge(formula, left_on=['index'], right_on=['name'])
    df = df.sort_values(['pillar','name']).set_index(['pillar', 'name']).drop(['index'], axis=1).transpose()

    cr = df.corr()
    # cr_df = cr.stack()
    # cr_df.name='corr'
    # cr_df = cr_df.loc[cr_df!=1].drop_duplicates().reset_index()
    # cr_df['corr_abs'] = cr_df['corr'].abs()
    # cr_df = cr_df.sort_values(by=['corr_abs'], ascending=False)
    # cr_df.to_csv('eda/test_eda_corr.csv', index=False)

    sns.set(style="whitegrid", font_scale=0.5)
    ax = sns.heatmap(cr, cmap='PiYG', vmin=-1, vmax=1, label='small')
    plt.tight_layout()
    plt.savefig('eda/test_eda_corr.png', dpi=300)

def eda_vif():
    ''' Calculate VIF -> no abnormal comes to our attention beyond correlations '''
    df = factors.iloc[:,2:].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = add_constant(df)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)

    print(vif)
    vif.to_csv('eda/test_eda_vif.csv')

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

        exit(1)

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

def test_kmean(cluster_no=5):

    aa = factors.copy(1).iloc[:,2:].values

    # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
    bb = np.nan_to_num(aa, -99.9)

    clustering_model = KMeans(n_clusters=cluster_no).fit(bb)
    clustering = clustering_model.predict(bb)
    cluster_centers = np.zeros((cluster_no, bb.shape[-1]))

    # Finding cluster centers for later inference.
    for i in range(cluster_no):
        cluster_centers[i, :] = np.mean(bb[clustering == i], axis=0)

    clustering = np.reshape(clustering, (aa.shape[0], aa.shape[1]))
    return clustering

def test_if_persistent():

    df = factors.copy(1)
    col_list = list(df.select_dtypes(float).columns)
    m = np.array([(df.iloc[:,2:].mean()<0).values]*df.shape[0])
    df.iloc[:,2:] = df.iloc[:,2:].mask(m, -df.iloc[:,2:])

    plt.figure(figsize=(16, 16))
    g = df.groupby(['period_end']).mean().reset_index(drop=False)

    # for name, g in df.groupby('group'):
        # ax = fig.add_subplot(n, n, k)

    g = g.fillna(0)
    date_list = g['period_end'].to_list()
    new_g = g.iloc[:,1:].transpose().values
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
    # tri = pd.read_csv('data_tri.csv', low_memory=False, usecols=['ticker','trading_day','tri'])
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
    ratio = ratio.sort_values(['market_cap_usd']).dropna(how='any')

    premium = {}
    for name, g in ratio.groupby(['period_end', 'currency_code']):
        try:
            premium[name] = {}
            g[f'cut'] = pd.qcut(g['market_cap_usd'], q=[0, 0.2, 0.8, 1], retbins=False, labels=[0, 1, 2])
            premium[name]['small'] = g.loc[g['cut'] == 0, 'stock_return_y'].mean()
            premium[name]['small_comp'] = list(set(g.loc[g['cut'] == 0, 'ticker']))
            premium[name]['mid'] = g.loc[g['cut'] == 1, 'stock_return_y'].mean()
            premium[name]['mid_comp'] = list(set(g.loc[g['cut'] == 1, 'ticker']))
            premium[name]['large'] = g.loc[g['cut'] == 2, 'stock_return_y'].mean()
            premium[name]['large_comp'] = list(set(g.loc[g['cut'] == 2, 'ticker']))

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
    df = factors[col_list].abs().mean()
    df.to_csv('eda/average_absolute_mean.csv')
    print(df)


if __name__ == "__main__":
    # correl_fama_website()
    # eda_correl()
    # sharpe_ratio()
    # eda_vif()
    # plot_autocorrel()
    # plot_trend()

    # test_if_persistent()
    # check_smb()
    # average_absolute_mean()

    ## Clustering
    test_kmean()
    # test_tsne()