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

with global_vals.engine.connect() as conn:
    factors = pd.read_sql(f'SELECT * FROM {global_vals.factor_premium_table}', conn)
    formula = pd.read_sql(f'SELECT name, pillar FROM {global_vals.formula_factors_table}', conn)
global_vals.engine.dispose()

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

    def calc(f):
        ret = f.mean(axis=0)
        sd = f.std(axis=0)
        return ret/sd

    dic = {}
    dic['all'] = calc(factors)
    for name, g in factors.groupby('group'):
        dic[name] = calc(g)

    df = pd.DataFrame(dic)
    df.to_csv('eda/test_eda_sharpe_by_group.csv')
    df.iloc[:,1:].transpose().describe().transpose().to_csv('eda/test_eda_sharpe_by_group_des.csv')

    dic = {}
    dic['all'] = calc(factors)
    for name, g in factors.groupby('period_end'):
        dic[name] = calc(g)

    df = pd.DataFrame(dic)
    df.to_csv('eda/test_eda_sharpe_by_time.csv')
    df.iloc[:,1:].transpose().describe().transpose().to_csv('eda/test_eda_sharpe_by_time_des.csv')

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




if __name__ == "__main__":
    # correl_fama_website()
    # eda_correl()
    # sharpe_ratio()
    # eda_vif()

    plot_trend()