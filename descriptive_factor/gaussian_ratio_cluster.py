from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, MeanShift
import matplotlib.animation as animation
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, robust_scale, minmax_scale, QuantileTransformer, \
    PowerTransformer, MinMaxScaler
from sklearn import decomposition
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestCentroid
from scipy.cluster.hierarchy import single, cophenet
from scipy.spatial.distance import pdist
from fcmeans import FCM

import global_vals
from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from hierarchy_ratio_cluster import trim_outlier_std, trim_outlier_quantile
import gc
import itertools
from collections import Counter
from scipy.cluster.hierarchy import dendrogram
import scipy.spatial

from sklearn import metrics
from s_dbw import S_Dbw
# from jqmcvi import base
from descriptive_factor.fuzzy_metrics import *

import matplotlib

print(matplotlib.__version__)

clustering_metrics1 = [
    metrics.calinski_harabasz_score,
    metrics.davies_bouldin_score,
    metrics.silhouette_score,
    S_Dbw,
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

        print(len(self.df['ticker'].unique()))

        # self.cols = 'avg_market_cap_usd, avg_ebitda_to_ev, vol, change_tri_fillna, icb_code, change_volume, change_earnings, ' \
        #             'ret_momentum, avg_interest_to_earnings, change_ebtda, avg_roe, avg_ni_to_cfo, avg_div_payout, change_dividend, ' \
        #             'avg_inv_turnover, change_assets, avg_gross_margin, avg_debt_to_asset, skew, avg_fa_turnover'.split(', ')
        self.cols = self.df.select_dtypes(float).columns.to_list()
        # self.df = trim_outlier_std(self.df)
        # x = self.df.std()

        # x = x.describe()
        # self.df = trim_outlier_std(self.df)
        # self.df = trim_outlier_quantile(self.df)
        # x = self.df.describe()
        # print(x)

    def stepwise_test(self, iter_conditions_dict):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        all_results_all = []
        condition_values = list(iter_conditions_dict.values())
        condition_keys = list(iter_conditions_dict.keys())

        for i in range(1, round(365*5/self.testing_interval)):
            df = self.df.groupby(['ticker']).nth(-i).reset_index().copy(1)
            df = df.replace([-np.inf, np.inf], [np.nan, np.nan])
            df = trim_outlier_std(df)

            for element in itertools.product(*condition_values):
                kwargs = dict(zip(condition_keys, list(element)))

                init_cols = ['icb_code']
                score_col = 'S_Dbw'
                init_score = 10000
                cols_list = ['vol', 'change_tri_fillna', 'ret_momentum', 'avg_inv_turnover', 'avg_ni_to_cfo',
                             'change_dividend', 'avg_fa_turnover', 'change_earnings', 'change_assets', 'change_ebtda']
                cols_list = self.cols
                next_factor = True
                while next_factor:
                    all_results = []
                    for col in cols_list:

                        # testing on FCM
                        if col not in init_cols:
                            cols = init_cols + [col]
                        else:
                            continue

                        # cols = ['icb_code', 'vol', 'change_tri_fillna', 'avg_fa_turnover', 'change_dividend']

                        # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
                        X = df[cols].values
                        X = np.nan_to_num(X, -1)

                        try:
                            m = test_method(X, kwargs)
                            # print(i, kwargs, m[score_col], cols)
                        except Exception as e:
                            print('************** ERROR: ', i, element, cols, e)

                        m['factors'] = ', '.join(cols)
                        # print(f'cols: {cols}: {m[score_col]}')
                        all_results.append(m)
                        gc.collect()

                    all_results = pd.DataFrame(all_results)
                    all_results['period'] = i
                    best = all_results.nsmallest(1, score_col, keep='first')

                    if best[score_col].values[0] < init_score:
                        best_best = best.copy(1)
                        init_cols = best['factors'].values[0].split(', ')
                        init_score = best[score_col].values[0]
                    else:
                        all_results_all.append(best_best)
                        print('-----> Return: ', i, kwargs, init_score, init_cols)
                        next_factor = False

        pd.concat(all_results_all, axis=0).to_csv(f'new2_stepwise_{self.testing_interval}.csv')
        # print(best.transpose())

def test_method(X, kwargs):
    ''' test conditions on different conditions of cluster methods '''

    new_kwargs = kwargs.copy()
    new_kwargs['n_components'] = int(round(X.shape[0]*new_kwargs['n_components']))
    model = GaussianMixture(**new_kwargs)
    y = model.fit_predict(X)

    # calculate matrics
    m = {}
    m.update(kwargs)
    # m[element]['count'] = Counter(y)
    for i in clustering_metrics1:
        m[i.__name__] = i(X, y)
    # print(m)

    return m

# -------------------------------- Plot Cluster -----------------------------------------------

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
    data = test_cluster(last=-1, testing_interval=91, use_cached=True)
    data.stepwise_test({'n_components':[0.01, 0.02]})    # data = test_cluster(last=-1, testing_interval=91, use_cached=True)

    # data.stepwise_test({'n_clusters':[1], 'm':[1.25]})    # data = test_cluster(last=-1, testing_interval=91, use_cached=True)

