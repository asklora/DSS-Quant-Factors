from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, MeanShift
import matplotlib.animation as animation
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, robust_scale, minmax_scale, QuantileTransformer, \
    PowerTransformer, MinMaxScaler, scale
from sklearn import decomposition
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestCentroid
from scipy.cluster.hierarchy import single, cophenet
from scipy.spatial.distance import pdist
from fcmeans import FCM

import global_vars
from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import gc
import itertools
from collections import Counter
from scipy.cluster.hierarchy import dendrogram
import scipy.spatial

from sklearn import metrics
from s_dbw import S_Dbw
# from jqmcvi import base
from descriptive_factor.fuzzy_metrics import *
from dateutil.relativedelta import relativedelta

import matplotlib

print(matplotlib.__version__)

clustering_metrics_fuzzy = [
    # partition_coefficient,
    xie_beni_index,
    # fukuyama_sugeno_index,
    # cluster_center_var,
    # fuzzy_hypervolume,
    # beringer_hullermeier_index,
    # bouguessa_wang_sun_index,
]

# --------------------------------- Prepare Datasets ------------------------------------------

def prep_factor_dateset(use_cached=True, list_of_interval=[7, 30, 91], currency='USD'):
    sample_df = {}
    list_of_interval_redo = []

    if use_cached:
        for i in list_of_interval:
            try:
                sample_df[i] = pd.read_csv(f'dcache_sample_{i}.csv')
            except:
                list_of_interval_redo.append(i)
    else:
        list_of_interval_redo = list_of_interval

    if len(list_of_interval_redo) > 0:
        print(f'-----------------------------> No local cache for {list_of_interval_redo} -> Process')
        df_dict = combine_tri_worldscope(use_cached=use_cached, save=True, currency=[currency]).get_results(list_of_interval_redo)
        sample_df.update(df_dict)

    return sample_df

# -------------------------------- Test Cluster -----------------------------------------------

class test_cluster:

    def __init__(self, last=-1, testing_interval=91, use_cached=True):

        self.testing_interval = testing_interval

        sample_df = prep_factor_dateset(list_of_interval=[testing_interval], use_cached=use_cached)
        self.df = sample_df[testing_interval]
        self.df['trading_day'] = pd.to_datetime(self.df['trading_day'])

        print(len(self.df['ticker'].unique()))
        self.cols = self.df.select_dtypes(float).columns.to_list()

    def history(self, columns, t, iter_conditions_dict):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        all_results_all = []
        condition_values = list(iter_conditions_dict.values())
        condition_keys = list(iter_conditions_dict.keys())
        score_col = 'xie_beni_index'

        if not columns:
            columns = self.cols

        for col in columns:
            all_results = []
            for month in t:
                df = self.df.loc[self.df['trading_day']>dt.datetime.now()-relativedelta(months=month)]
                df.loc[:,[col]] = trim_outlier_std(df[[col]])
                # df = trim_outlier_quantile(df)

                df = df.set_index(['ticker', 'trading_day'])[col].unstack()
                df = df.replace([-np.inf, np.inf], [np.nan, np.nan])

                for element in itertools.product(*condition_values):
                    kwargs = dict(zip(condition_keys, list(element)))

                    X = df.values
                    X = np.nan_to_num(X, -1)

                    m = test_method(X, kwargs)
                    m['factors'] = col
                    m['lookback'] = month
                    # print(m)

                    all_results.append(m)
                    gc.collect()

            all_results = pd.DataFrame(all_results).nsmallest(1, ['xie_beni_index'], keep='first')
            all_results_all.append(all_results)

        x = pd.concat(all_results_all, axis=0)
        pd.concat(all_results_all, axis=0).to_csv(f'history_{self.testing_interval}.csv')

def test_method(X, kwargs):
    ''' test conditions on different conditions of cluster methods '''

    new_kwargs = kwargs.copy()
    if new_kwargs['n_clusters']<1:
        new_kwargs['n_clusters'] = int(round(X.shape[0]*new_kwargs['n_clusters']))
    model = FCM(**new_kwargs)
    model.fit(X)

    # y = model.predict(X)
    u = model.u
    v = model.centers

    # calculate matrics
    m = {}
    m.update(kwargs)
    # m[element]['count'] = Counter(y)
    for i in clustering_metrics_fuzzy:
        m[i.__name__] = round(i(X, u, v.T, kwargs['m']), 4)
    # print(m)

    return m

# -------------------------------- Plot Cluster -----------------------------------------------

def trim_outlier_std(df):
    ''' trim outlier on testing sets '''

    def trim_scaler(x):
        x = np.clip(x, np.nanpercentile(x, 1), np.nanpercentile(x, 99))
        # x = np.clip(x, np.nanmean(x) - 2 * np.nanstd(x), np.nanmean(x) + 2 * np.nanstd(x))
        x = robust_scale(x)
        return x

    cols = df.select_dtypes(float).columns.to_list()
    for col in cols:
        if col != 'industry_code':
            x = trim_scaler(df[col])
        else:
            x = df[col].values
        x = scale(x.T)
        df[col] = x
    return df

def trim_outlier_quantile(df):
    ''' trim outlier on testing sets based on quantile transformation '''

    cols = df.select_dtypes(float).columns.to_list()
    df[cols] = QuantileTransformer(n_quantiles=4).fit_transform(df[cols])
    return df

# -------------------------------- Plot Cluster -----------------------------------------------

def plot_scatter_hist(df, cols, cluster_method, suffixes):
    n = len(cols)
    fig = plt.figure(figsize=(n * 4, n * 4), dpi=60, constrained_layout=True)
    k = 1
    for v1 in cols:
        for v2 in cols:
            print(v1, v2)
            ax = fig.add_subplot(n, n, k)
            if v1 == v2:
                ax.hist(df[v1], bins=20)
                ax.set_xlabel(v1)
            else:
                plt_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                             'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
                colors = dict(zip(list(df['cluster'].unique()), plt_color[:df['cluster'].max() + 1]))
                ax.scatter(df[v1], df[v2], c=list(df['cluster'].map(colors).values), alpha=.5)
                ax.set_xlabel(v1)
                ax.set_ylabel(v2)
            k += 1

    fig.savefig(f'cluster_selected_{cluster_method}_{suffixes}.png')
    plt.close(fig)

def plot_pca_scatter_hist(df, cluster_method):
    X = np.nan_to_num(df.apply(pd.to_numeric, errors='coerce').values[:, 1:-1], -1)
    # X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=None).fit(X)  # calculation Cov matrix is embeded in PCA
    print(pca.explained_variance_ratio_)
    print(pca.transform(X))
    X_trans = pca.transform(X)
    v1, v2 = X_trans[:, 0], X_trans[:, 1]

    fig = plt.figure(figsize=(4, 4), dpi=60, constrained_layout=True)
    plt_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = dict(zip(list(df['cluster'].unique()), plt_color[:df['cluster'].max() + 1]))
    plt.scatter(v1, v2, c=list(df['cluster'].map(colors).values))

    fig.savefig(f'cluster_selected_pca_{cluster_method}.png')
    plt.close(fig)

if __name__ == "__main__":
    data = test_cluster(last=-1, testing_interval=30, use_cached=True)
    # data.history(['industry_code', 'change_tri_fillna', 'skew', 'avg_volume_1w3m',
    #               'avg_div_payout', 'avg_debt_to_asset', 'avg_fa_turnover'], [6, 12, 24, 36, 48, 60], {'n_clusters':[2], 'm':[2]})    # data = test_cluster(last=-1, testing_interval=91, use_cached=True)
    data.history(None, [1, 3, 6, 12, 24, 36, 48], {'n_clusters':[2], 'm':[2]})
    # data.stepwise_test({'n_clusters':[1], 'm':[1.25]})    # data = test_cluster(last=-1, testing_interval=91, use_cached=True)

