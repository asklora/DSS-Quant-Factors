from sqlalchemy import create_engine
import global_vars
import pandas as pd

from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope

import gc
import itertools
import multiprocessing as mp

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import robust_scale, QuantileTransformer, scale, minmax_scale, quantile_transform
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from sklearn import metrics
from s_dbw import S_Dbw
from descriptive_factor.fuzzy_metrics import *
from pyclustertend import hopkins
from dateutil.relativedelta import relativedelta
import datetime as dt

clustering_metrics1 = [
    metrics.calinski_harabasz_score,
    metrics.davies_bouldin_score,
    metrics.silhouette_score,
    S_Dbw,
]

select_cols = [
    'avg_market_cap_usd', 'avg_ca_turnover_re', 'avg_roe', 'avg_volume', 'change_earnings',  # hierarchical
    'avg_volume_1w3m', 'avg_debt_to_asset', 'change_volume', 'skew', 'icb_code', 'avg_div_yield', 'avg_gross_margin', 'change_tri_fillna', # fcm
    'avg_ca_turnover_re', 'icb_code', 'avg_fa_turnover_re', 'avg_market_cap_usd'    # gaussian
]

# select_cols = ['change_earnings', 'avg_roe', 'avg_inv_turnover', 'avg_market_cap_usd', 'change_dividend', 'avg_cash_ratio', 'avg_roic',
#                'avg_gross_margin', 'avg_debt_to_asset', 'icb_code', 'change_tri_fillna', 'skew', 'change_volume',
#                'avg_volume_1w3m', 'icb_code', 'avg_inv_turnover', 'change_dividend', 'avg_div_yield']

# select_cols = ['change_earnings', 'avg_roe', 'avg_inv_turnover', 'avg_market_cap_usd', 'avg_cash_ratio', 'avg_roic',
#                'avg_gross_margin', 'avg_debt_to_asset', 'icb_code', 'change_tri_fillna', 'skew', 'change_volume',
#                'avg_volume_1w3m', 'icb_code', 'avg_inv_turnover', 'avg_div_yield']

select_cols = list(set(select_cols))

high_corr_removed_cols = [
    # 'avg_cash_ratio',   # icb_code
    # 'avg_capex_to_dds', # icb_code
    # 'avg_volume_1w3m',  # change_tri_fillna, vol
    'avg_gross_margin',     # avg_fx_turnover_re
    'avg_ca_turnover_re',   # avg_fx_turnover_re
    'change_volume',        # avg_volume_1w3m
]

good_cols = list(set(select_cols) - set(high_corr_removed_cols))
good_mom_cols = ['skew','avg_volume_1w3m','change_tri_fillna']
print(len(good_cols), good_cols)

# --------------------------------- Prepare Datasets ------------------------------------------

def prep_factor_dateset(use_cached=True, list_of_interval=[7, 30, 91], currency='USD'):
    sample_df = {}
    list_of_interval_redo = []

    if use_cached:
        for i in list_of_interval:
            try:
                sample_df[i] = pd.read_csv(f'dcache_sample_{i}.csv', usecols=['ticker','trading_day']+good_cols)
            except:
                sample_df[i] = pd.read_csv(f'dcache_sample_{i}.csv')
                # list_of_interval_redo.append(i)
            sample_df[i]['trading_day'] = pd.to_datetime(sample_df[i]['trading_day'])
            sample_df[i] = sample_df[i].loc[sample_df[i]['trading_day'] > (dt.datetime.today() - relativedelta(years=7))]

    else:
        list_of_interval_redo = list_of_interval

    if len(list_of_interval_redo) > 0:
        print(f'-----------------------------> No local cache for {list_of_interval_redo} -> Process')
        df_dict = combine_tri_worldscope(use_cached=use_cached, save=True, currency=[currency]).get_results(
            list_of_interval_redo)
        sample_df.update(df_dict)

    return sample_df

# -------------------------------- Test Cluster -----------------------------------------------

def calc_corr_csv(df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    df = df.drop(columns=['trading_day','ticker'])

    X = add_constant(df.fillna(0))
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)

    c = df.corr().unstack().reset_index()
    c['corr_abs'] = c[0].abs()
    c = c.sort_values('corr_abs', ascending=True)

    return c

def test_missing(df):

    # writer = pd.ExcelWriter(f'missing_by_ticker.xlsx')

    # fill missing
    df['trading_day'] = pd.to_datetime(df['trading_day'])
    df = df.loc[df['trading_day'] > (dt.datetime.today() - relativedelta(years=5))]
    # df['avg_div_yield'] = df['avg_div_yield'].fillna(0)
    # df = df['avg_div_yield'].fillna(0)

    df_miss = df.groupby('ticker').apply(lambda x: x.notnull().sum())
    df_sum = pd.DataFrame(df_miss.sum(0))

    # df.to_excel(writer, sheet_name='by ticker')

    return df

def fill_all_day_interpolate(result, date_col="trading_day"):
    ''' Fill all the weekends between first / last day and fill NaN'''

    # Construct indexes for all day between first/last day * all ticker used
    df = result[["ticker", date_col]].copy()
    df.trading_day = pd.to_datetime(df[date_col])
    result.trading_day = pd.to_datetime(result[date_col])
    df = df.sort_values(by=[date_col], ascending=True)
    daily = pd.date_range(df.iloc[0, 1], df.iloc[-1, 1], freq='D')
    indexes = pd.MultiIndex.from_product([df['ticker'].unique(), daily], names=['ticker', date_col])

    # Insert weekend/before first trading date to df
    df = df.set_index(['ticker', date_col]).reindex(indexes).reset_index()
    df = df.sort_values(by=['ticker', date_col], ascending=True)
    result = df.merge(result, how="left", on=["ticker", date_col])

    num_col = result.select_dtypes(float).columns.to_list()
    result[num_col] = result.groupby(['ticker'])[num_col].apply(pd.DataFrame.interpolate, limit_direction='forward', limit_area='inside')
    result[num_col] = result.groupby(['ticker'])[num_col].ffill()

    return result

class test_cluster:

    def __init__(self, testing_interval=91):

        self.testing_interval = testing_interval
        # sample_df = prep_factor_dateset(list_of_interval=[testing_interval], use_cached=True)
        # self.df = sample_df[testing_interval]

        sample_df = prep_factor_dateset(list_of_interval=[1,7,91], use_cached=True)

        # add df for annually change
        df_1yr = sample_df[91].copy()
        # avg_cols = ['avg_debt_to_asset', 'avg_div_yield', 'avg_fa_turnover_re', 'avg_roe']
        avg_cols = []
        change_cols = ['change_earnings']
        # df_1yr[avg_cols] = df_1yr.groupby(['ticker'])[avg_cols].rolling(4).mean().reset_index(level=0, drop=True)
        df_1yr[change_cols] = df_1yr[change_cols] + 1
        df_1yr[change_cols] = df_1yr.groupby(['ticker'])[change_cols].rolling(4).apply(lambda x: np.prod(x)).reset_index(level=0, drop=True)
        df_1yr[change_cols] = df_1yr[change_cols] - 1
        df_1yr = df_1yr[['ticker','trading_day']+avg_cols+change_cols]

        # merge all period table
        df = sample_df[91].merge(fill_all_day_interpolate(sample_df[7])[['ticker','trading_day']+good_mom_cols],
                                 on=['ticker','trading_day'], how='left', suffixes=('_91','_7'))
        df = df.merge(fill_all_day_interpolate(sample_df[1]), on=['ticker','trading_day'], how='left', suffixes=('','_1'))
        df = df.merge(df_1yr, on=['ticker','trading_day'], how='left', suffixes=('','_365'))

        # df = df.merge(fill_all_day_interpolate(sample_df[7]), on=['ticker','trading_day'], how='left', suffixes=('','_7'))
        self.df = df.drop(columns=['avg_volume_1w3m_91'])

        # test_missing(self.df)
        # calc_corr_csv(self.df)

        self.cols = self.df.select_dtypes(float).columns.to_list()
        print(len(self.df['ticker'].unique()))

    def multithread_combination(self, name_sql=None, n_processes=12):

        print(len(good_cols), good_cols)
        self.score_col = 'cophenetic'
        period_list = list(range(1, round(365*5/self.testing_interval)))
        all_groups = itertools.product(period_list, [3, 4], [name_sql])
        all_groups = [tuple(e) for e in all_groups]
        with mp.Pool(processes=n_processes) as pool:
            pool.starmap(self.combination_test, all_groups)

    def combination_test(self, *args):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        period, n_cols, name_sql = args
        # print(args)

        df = self.df.groupby(['ticker']).nth(-period).reset_index().copy(1)
        df = df.replace([-np.inf, np.inf], [np.nan, np.nan])
        df = trim_outlier_std(df)

        all_results = []
        for cols in itertools.combinations(self.cols, n_cols):

            # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
            X = df[list(cols)].values
            X[X == 0] = np.nan
            X = np.nan_to_num(X, 0)

            m = test_method(X)
            m['factors'] = ', '.join(cols)
            print(cols, m)
            all_results.append(m)
            gc.collect()

        all_results = pd.DataFrame(all_results)
        all_results['period'] = period
        all_results['name_sql'] = name_sql
        best = all_results.nlargest(1, self.score_col, keep='first')

        print(args, '-----> Return: ', best['factors'].values[0])

        thread_engine_ali = create_engine(global_vars.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with thread_engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
            all_results[['period', 'factors', self.score_col, 'name_sql']].to_sql("des_factor_hierarchical", **extra)
        thread_engine_ali.dispose()

    def multithread_stepwise(self, name_sql=None, n_processes=12):

        print(len(self.cols), self.cols)
        self.score_col = 'cophenetic'
        period_list = list(range(1, round(365*5/self.testing_interval)))
        all_groups = itertools.product(period_list, self.cols, [name_sql], )
        all_groups = [tuple(e) for e in all_groups]
        with mp.Pool(processes=n_processes) as pool:
            pool.starmap(self.stepwise_test, all_groups)

    def stepwise_test(self, *args):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        period, init_col, name_sql = args
        print(period, init_col)

        df = self.df.groupby(['ticker']).nth(-period).reset_index().copy(1)
        df = df.replace([-np.inf, np.inf], [np.nan, np.nan])
        df = trim_outlier_std(df)

        init_cols = [init_col]
        init_score = 0

        next_factor = True
        while next_factor:
            all_results = []
            for col in self.cols:

                # testing on FCM
                if col not in init_cols:
                    cols = init_cols + [col]
                else:
                    continue

                # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
                X = df[cols].values
                X[X == 0] = np.nan
                X = np.nan_to_num(X, -1)

                m = test_method(X)
                m['factors'] = ', '.join(cols)
                all_results.append(m)
                gc.collect()

            all_results = pd.DataFrame(all_results)
            all_results['period'] = period
            all_results['name_sql'] = name_sql
            best = all_results.nlargest(1, self.score_col, keep='first')

            if best[self.score_col].values[0] > init_score:
                best_best = best.copy(1)
                init_cols = best['factors'].values[0].split(', ')
                init_score = best[self.score_col].values[0]
            else:
                print('-----> Return: ',init_col, period, init_score, init_cols)
                next_factor = False

                thread_engine_ali = create_engine(global_vars.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")
                with thread_engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
                    extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
                    best_best[['period', 'factors', self.score_col, 'name_sql']].to_sql("des_factor_hierarchical", **extra)
                thread_engine_ali.dispose()
        return

        # pd.concat(all_results_all, axis=0).to_csv(f'hierarchy_diffinit_{self.testing_interval}_{self.score_col}.csv')

def test_method(X):
    ''' test conditions on different conditions of cluster methods '''

    kwargs = {'distance_threshold': 0, 'linkage': 'complete', 'n_clusters':None}
    model = AgglomerativeClustering(**kwargs).fit(X)
    # y = model.labels_

    # calculate matrics
    m = {}
    # m[element]['n_clusters'] = model.n_clusters_
    m['cophenetic'] = get_cophenet_corr(model, X)
    m['hopkins'] = hopkin_static(X)
    m['avg_distance'] = np.mean(model.distances_/np.sqrt(X.shape[1]))

    return m

# ------------------------------------------ Trim Data -----------------------------------------------

def trim_outlier_std(df):
    ''' trim outlier on testing sets '''

    def trim_scaler(x):
        x = np.clip(x, np.nanpercentile(x, 1), np.nanpercentile(x, 99))
        # x = np.clip(x, np.nanmean(x) - 2 * np.nanstd(x), np.nanmean(x) + 2 * np.nanstd(x))
        x = robust_scale(x)
        return quantile_transform(np.reshape(x, (x.shape[0], 1)), output_distribution='normal')[:, 0]

    cols = df.select_dtypes(float).columns.to_list()
    for col in cols:
        if col != 'icb_code':
            x = trim_scaler(df[col])
        else:
            x = df[col].values
        x = scale(x.T)
        x = minmax_scale(x)
        df[col] = x
    return df

def trim_outlier_quantile(df):
    ''' trim outlier on testing sets based on quantile transformation '''

    cols = df.select_dtypes(float).columns.to_list()
    df[cols] = QuantileTransformer(n_quantiles=4).fit_transform(df[cols])
    return df

# -------------------------------------- Calculate Stat -----------------------------------------------

def hopkin_static(X, m=0.3):
    return hopkins(X, round(X.shape[0]*m))

def get_cophenet_corr(model, X):

    linkage_matrix = get_linkage_matrix(model)
    c, d = cophenet(linkage_matrix, pdist(X))
    return c

def get_linkage_matrix(model):

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    return linkage_matrix

if __name__ == "__main__":
    testing_interval = 91
    testing_name = 'all_comb_new_multiperiod1'
    
    data = test_cluster(testing_interval=testing_interval)
    # data.multithread_stepwise('{}:{}'.format(testing_name, testing_interval), n_processes=6)
    data.multithread_combination('{}:{}'.format(testing_name, testing_interval), n_processes=10)


