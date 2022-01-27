import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn import linear_model

from global_vars import *
from general.sql_process import read_table, read_query

import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vars
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
from sklearn.preprocessing import minmax_scale

def download_premium():
    ''' download premium and calculate factor importance '''

    query = f"SELECT * FROM {factor_premium_table}"
    df = read_query(query, db_url_read)

    for i in [1, 3, 5, 10]:
        start_date = (dt.datetime.today() - relativedelta(years=i)).date()
        df_period = df.loc[df["trading_day"] >= start_date]
        df_period_avg = df_period.groupby(["field", "group", "weeks_to_expire"])["value"].mean().unstack(level=[-2, -1])
        df_period_avg = df_period_avg.sort_values(by=[("USD", 1)], ascending=False)

        df_period["value"] = df_period["value"].abs()
        df_period_abs_avg = df_period.groupby(["field", "group", "weeks_to_expire"])["value"].mean().unstack(level=[-2, -1])
        df_period_abs_avg = df_period_abs_avg.sort_values(by=[("USD", 1)], ascending=False)
        print(df_period_avg)

def eda_missing(df, col_list):
    print('======= Missing ========')
    df = df.groupby(["group"])[col_list].apply(lambda x : x.isnull().sum()/len(x)).transpose()['USD']
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df[df>0].sort_values(ascending=False))

    return df

def eda_correl(df, col_list, formula):
    ''' test correlation of our factor '''

    print('======= Correlation ========')
    df = df[col_list]
    cr = df.astype(float).corr()

    cr_df = cr.stack(-1)
    cr_df = cr_df.reset_index()
    cr_df.columns=['f1','f2','corr']
    cr_df = cr_df.loc[cr_df['corr']!=1]
    cr_df['corr_abs'] = cr_df['corr'].abs()
    cr_df = cr_df.sort_values(by=['corr_abs'], ascending=False)
    cr_df = cr_df.drop_duplicates(subset=['f1'], keep='first').sort_values(by=['f1'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(cr_df.loc[cr_df['corr_abs']>0.2].set_index(['f1', 'f2'])['corr'])

    return cr_df.set_index(['f1'])['corr_abs']
    # # plot heatmap by pillar
    # df = df.merge(formula[['pillar', 'name']], left_on=['index'], right_on=['name'])
    # df = df.sort_values(['pillar','name']).set_index(['pillar', 'name']).drop(['index'], axis=1).transpose()
    # df.columns = [x[0]+'-'+x[1] for x in df.columns.to_list()]
    #
    # sns.set(style="whitegrid", font_scale=0.5)
    # ax = sns.heatmap(cr, cmap='PiYG', vmin=-1, vmax=1, label='small')
    # plt.tight_layout()
    # plt.show()

def eda_vif(df, col_list):
    ''' Calculate VIF -> no abnormal comes to our attention beyond correlations '''

    print('======= VIF ========')
    df = df[col_list].replace([np.inf, -np.inf], np.nan).fillna(0)
    X = add_constant(df)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(vif.sort_values(ascending=False).head(10))

def sharpe_ratio(factors, col_list):
    ''' calculate the sharpe ratio '''

    print('======= Sharpe Ratio ========')
    factors['trading_day'] = pd.to_datetime(factors['trading_day'])

    factors['year'] = factors['trading_day'].dt.year
    factors['month'] = factors['trading_day'].dt.month

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

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            df = df.iloc[:,1:].transpose().describe().transpose()
            print('----->', col)
            print(df)

    calc_by_col('group')
    calc_by_col('year')
    calc_by_col('month')

def plot_autocorrel(df, col_list):
    ''' plot the level of autocorrelation '''
    print('======= Autocorrelation ========')

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

    for name, g in df.groupby(['group']):
        print(name)
        r = corr_results(g)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print('---->', name)
            print(r)

def test_performance(df, col_list):

    print('======= Performance ========')

    curr_list = ['USD']  # 'TWD','JPY','SGD'
    df = df.loc[df['group'].isin(curr_list)]

    m = np.array([(df[col_list].mean()<0).values]*df.shape[0])
    df[col_list] = df[col_list].mask(m, -df[col_list])

    plt.figure(figsize=(16, 16))
    g = df.groupby(['trading_day']).mean().reset_index(drop=False)

    g = g.fillna(0)
    date_list = g['trading_day'].to_list()
    new_g = g[col_list].transpose().values
    new_g = np.cumprod(new_g/4 + 1, axis=1)
    ddf = pd.DataFrame(new_g, index=col_list, columns=date_list).sort_values(date_list[-1], ascending=False)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(ddf.iloc[:,-1].sort_values(ascending=False).head(10))

    ddf = ddf.iloc[:10,:].transpose()
    print(ddf)
    plt.plot(ddf, label=list(ddf.columns))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-large')
    plt.tight_layout()
    plt.show()

def average_absolute_mean(df, col_list):
    print('======= Mean Absolute Premium ========')

    df = df[col_list].abs().mean().sort_values(ascending=False)
    df = pd.DataFrame({'abs':df, 'rank':list(range(len(df)))})
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    return df['abs']

def test_premiums(name):

    with global_vars.engine_ali.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM {global_vars.factor_premium_table}_{name} WHERE not trim_outlier', conn)
        formula = pd.read_sql(f'SELECT * FROM {global_vars.formula_factors_table_prod}', conn)
    global_vars.engine_ali.dispose()

    df['trading_day'] = pd.to_datetime(df['trading_day'], format='%Y-%m-%d')  # convert to datetime
    df = df.pivot_table(index=['trading_day', 'group'], columns=['factor_name'], values=['premium']).droplevel(0, axis=1)
    df.columns.name = None
    df = df.reset_index()

    x_list = list(set(formula.loc[formula['x_col'], 'name'].to_list()) - {'debt_issue_less_ps_to_rent'})
    col_list = list(set(formula['name'].to_list()) & (set(df.columns.to_list())))
    factor_list = formula.loc[formula['factors'], 'name'].to_list()

    df_miss = eda_missing(df, col_list)
    df_corr = eda_correl(df, col_list, formula)
    eda_vif(df, col_list)
    plot_autocorrel(df, col_list)
    sharpe_ratio(df, col_list)
    test_performance(df, col_list)
    df_mean = average_absolute_mean(df, col_list)

    df = pd.concat([df_corr, df_miss, df_mean], axis=1)
    df.columns = ['corr','miss','mean']
    df['score'] = 1 - minmax_scale(df['corr']) + minmax_scale(df['mean'])

    with global_vars.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT * FROM {global_vars.formula_factors_table_prod}', conn)
        formula['rank'] = formula['name'].map(df['score'].to_dict())
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
        formula.to_sql(global_vars.formula_factors_table_prod, **extra)
    global_vars.engine_ali.dispose()

if __name__ == "__main__":
    test_premiums('weekly1')
    download_premium()