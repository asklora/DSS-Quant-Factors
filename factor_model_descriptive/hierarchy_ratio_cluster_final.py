from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, MeanShift, FeatureAgglomeration
from fcmeans import FCM
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
from dateutil.relativedelta import relativedelta

import global_vars
from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
from pyclustertend import hopkins

import pandas as pd
import matplotlib.pyplot as plt

import itertools
from scipy.cluster.hierarchy import dendrogram
from descriptive_factor.fuzzy_metrics import *

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
        self.cols = self.df.select_dtypes(float).columns.to_list()

        self.feature_cluster_plot()

        if last:
            self.df = self.df.groupby(['ticker']).nth(last).reset_index().copy(1)
        self.df = self.df.replace([-np.inf, np.inf], [np.nan, np.nan])
        self.df = trim_outlier_std(self.df)

        print(len(self.df['ticker'].unique()))

        self.indice = self.df['ticker'].to_list()

    def feature_cluster_plot(self):
        df = trim_outlier_std(self.df)
        X = df[self.cols].values
        X = np.nan_to_num(X, -1)
        agglo = FeatureAgglomeration(distance_threshold=0, n_clusters=None)
        model = agglo.fit(X)
        plot_dendrogram(model, labels=self.df.columns.to_list(), suffix=f'_features{self.testing_interval}')
        exit(1)

    def stepwise_test(self, cluster_method, kwargs):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        # cols = ['industry_code','avg_inv_turnover', 'change_dividend', 'avg_market_cap_usd', 'avg_roic']
        cols = ['industry_code','change_tri_fillna', 'vol']

        X = self.df[cols].values
        X = np.nan_to_num(X, -1)
        m, model = test_method(X, cluster_method=cluster_method, iter_conditions_dict=kwargs, save_csv=False, indice=self.indice)
        m['factors'] = ', '.join(cols)

def test_method(X, cluster_method, iter_conditions_dict, indice, save_csv=True):
    ''' test conditions on different conditions of cluster methods '''

    m = {}
    condition_values = list(iter_conditions_dict.values())
    condition_keys = list(iter_conditions_dict.keys())

    for element in itertools.product(*condition_values):
        kwargs = dict(zip(condition_keys, list(element)))
        model = cluster_method(**kwargs).fit(X)

        # calculate matrics
        m[element] = {}
        m[element]['n_clusters'] = model.n_clusters_
        m[element]['cophenetic'] = get_cophenet_corr(model, X)
        m[element]['hopkins'] = hopkin_static(X)
        m[element]['avg_distance'] = np.mean(model.distances_/np.sqrt(X.shape[1]))
        print(m[element])
        plot_dendrogram(model, indice)

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

def plot_dendrogram(model, labels, suffix=''):
    ''' Create linkage matrix and then plot the dendrogram '''

    # create the counts of samples under each node

    linkage_matrix = get_linkage_matrix(model)
    # fig = plt.figure(figsize=(40, 80), dpi=120)
    dendrogram(linkage_matrix,  color_threshold=0.7*max(linkage_matrix[:,2]), truncate_mode='level',
               orientation='left', labels=labels, leaf_font_size=5)
    plt.tight_layout()
    plt.savefig(f'hierarchical_dendrogram{suffix}.png')
    # exit(1)

if __name__ == "__main__":
    data = test_cluster(last=-1, testing_interval=30)
    data.stepwise_test(AgglomerativeClustering, {'distance_threshold': [0], 'linkage': ['average'], 'n_clusters':[None]})


