from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, MeanShift
from fcmeans import FCM
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

import global_vals
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

import matplotlib

print(matplotlib.__version__)

clustering_metrics1 = [
    metrics.calinski_harabasz_score,
    metrics.davies_bouldin_score,
    metrics.silhouette_score,
    S_Dbw,
]
# clustering_metrics2 = [base.dunn]
clustering_metrics_fuzzy = [
    # partition_coefficient,
    xie_beni_index,
    # fukuyama_sugeno_index,
    # cluster_center_var,
    fuzzy_hypervolume,
    # beringer_hullermeier_index,
    # bouguessa_wang_sun_index,
]

# --------------------------------- Prepare Datasets ------------------------------------------

def prep_factor_dateset(use_cached=True, list_of_interval=[7, 30, 91], currency='USD'):
    sample_df = {}
    corr_df = {}
    list_of_interval_redo = []

    if use_cached:
        for i in list_of_interval:
            try:
                sample_df[i] = pd.read_csv(f'dcache_sample_{i}.csv')
                corr_df[i] = pd.read_csv(f'sample_corr_{i}.csv')
            except:
                list_of_interval_redo.append(i)
    else:
        list_of_interval_redo = list_of_interval

    if len(list_of_interval_redo) > 0:
        print(f'-----------------------------> No local cache for {list_of_interval_redo} -> Process')
        df_dict = combine_tri_worldscope(use_cached=use_cached, save=True, currency=[currency]).get_results(
            list_of_interval_redo)
        sample_df.update(df_dict)

    return sample_df

# -------------------------------- Test Cluster -----------------------------------------------

class test_cluster:

    def __init__(self, last=-1, testing_interval=91):

        self.testing_interval = testing_interval

        sample_df = prep_factor_dateset(list_of_interval=[testing_interval], use_cached=True)
        self.df = sample_df[testing_interval]

        print(len(self.df['ticker'].unique()))

        if last:
            self.df = self.df.groupby(['ticker']).nth(last).reset_index()
            self.df = self.df.replace([-np.inf, np.inf], [np.nan, np.nan])

        self.cols = self.df.select_dtypes(float).columns.to_list()
        # self.df = trim_outlier_quantile(self.df)
        self.df = trim_outlier_std(self.df)
        print(self.df.describe())

    def stepwise_test(self, cluster_method, kwargs):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        print(self.df.shape)
        print(self.df.describe().transpose())
        cols = init_cols = ['icb_code']
        score_col = 'cophenetic'
        init_score = 0
        cols_list = ['vol', 'change_tri_fillna', 'ret_momentum', 'avg_inv_turnover', 'avg_ni_to_cfo', 'change_dividend',
                     'avg_fa_turnover', 'change_earnings', 'change_assets', 'change_ebtda']
        cols_list = self.cols

        all_results_all = []
        next_factor = True
        while next_factor:
        # while len(cols)<=20:

            all_results = []
            for col in cols_list:

                # testing on FCM
                if col not in init_cols:
                    cols = init_cols + [col]
                else:
                    continue

                # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
                X = self.df[cols].values
                X = np.nan_to_num(X, -1)
                # labels = self.cols

                labels = None
                m, model = test_method(X, labels=labels, cluster_method=cluster_method, iter_conditions_dict=kwargs, save_csv=False)
                get_cophenet_corr(model, X)

                m['factors'] = ', '.join(cols)
                # print(f'cols: {cols}: {m[score_col]}')
                all_results.append(m)
                gc.collect()

            all_results = pd.DataFrame(all_results)
            # best = all_results.nsmallest(1, score_col, keep='first')
            best = all_results.nlargest(1, score_col, keep='first')
            init_cols = best['factors'].values[0].split(', ')
            all_results_all.append(all_results.loc[all_results['factors'] == best['factors'].values[0]])

            if best[score_col].values[0] > init_score:
                init_score = best[score_col].values[0]
                print(init_score, init_cols)
                # plt.plot(model.distances_)
                # plt.show()
                # plot_dendrogram(model, labels)
            else:
                init_score = best[score_col].values[0]
                print(init_score, init_cols)
                next_factor = False

        pd.concat(all_results_all, axis=0).to_csv(f'new_stepwise__{self.testing_interval}.csv')
        # print(best.transpose())

def test_method(X, labels, cluster_method, iter_conditions_dict, save_csv=True):
    ''' test conditions on different conditions of cluster methods '''

    m = {}
    condition_values = list(iter_conditions_dict.values())
    condition_keys = list(iter_conditions_dict.keys())

    for element in itertools.product(*condition_values):
        kwargs = dict(zip(condition_keys, list(element)))

        # if fuzzy clustering
        if cluster_method.__name__ == 'FCM':
            model = cluster_method(**kwargs)
            model.fit(X)

            y = model.predict(X)
            u = model.u
            v = model.centers

            # centre_df = pd.DataFrame(v, columns=self.cols)

            # calculate matrics
            m[element] = {}
            m[element]['count'] = Counter(y)
            for i in self.clustering_metrics_fuzzy:
                m[element][i.__name__] = i(X, u, v.T, kwargs['m'])
            print(element, m[element])

        # if crisp clustering
        else:
            model = cluster_method(**kwargs).fit(X)
            try:
                y = model.predict(X)
            except:
                y = model.labels_

            # calculate matrics
            m[element] = {}
            # m[element]['count'] = Counter(y)
            m[element]['n_clusters'] = model.n_clusters_
            m[element]['cophenetic'] = get_cophenet_corr(model, X)
            # for i in clustering_metrics1:
            #     m[element][i.__name__] = i(X, y)
            # m[element]['min_distance'] = np.min(model.distances_/np.sqrt(X.shape[1]))
            m[element]['avg_distance'] = np.mean(model.distances_/np.sqrt(X.shape[1]))
            # m[element]['max_distance'] = np.max(model.distances_/np.sqrt(X.shape[1]))

    if save_csv:
        results = pd.DataFrame(m).transpose()
        results.index = results.index.set_names(condition_keys)
        results = results.reset_index()
        results.to_csv(f'cluster_{cluster_method.__name__}.csv', index=False)
        return results
    else:
        return m[element], model

# -------------------------------- Plot Cluster -----------------------------------------------

def trim_outlier_std(df):
    ''' trim outlier on testing sets '''

    def trim_scaler(x):
        x = np.clip(x, np.nanmean(x) - 2 * np.nanstd(x), np.nanmean(x) + 2 * np.nanstd(x))
        # x = np.clip(x, np.percentile(x, 0.10), np.percentile(x, 0.90))
        x = robust_scale(x)
        return x

    for col in df.select_dtypes(float).columns.to_list():
        if col != 'icb_code':
            x = trim_scaler(df[col])
        else:
            x = df[col].values
        x = StandardScaler().fit_transform(np.expand_dims(x, 1))[:, 0]
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

def plot_dendrogram(model, labels):
    ''' Create linkage matrix and then plot the dendrogram '''

    # create the counts of samples under each node

    linkage_matrix = get_linkage_matrix(model)

    # Plot the corresponding dendrogram
    # dendrogram(linkage_matrix,  truncate_mode=None, labels=labels, orientation='left')
    dendrogram(linkage_matrix,  truncate_mode='level', orientation='left')

    plt.tight_layout()
    plt.show()
    # exit(1)

if __name__ == "__main__":
    data = test_cluster(last=-1, testing_interval=30)
    data.stepwise_test(AgglomerativeClustering, {'distance_threshold': [0], 'linkage': ['average'],
                                                 'n_clusters':[None]})


