from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, MeanShift
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
from sqlalchemy import create_engine

import global_vals
from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
from pyclustertend import hopkins

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
import multiprocessing as mp

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
        df_dict = combine_tri_worldscope(use_cached=use_cached, save=True, currency=[currency]).get_results(
            list_of_interval_redo)
        sample_df.update(df_dict)

    return sample_df

# -------------------------------- Test Cluster -----------------------------------------------

class test_cluster:

    def __init__(self, testing_interval=91):

        self.testing_interval = testing_interval
        sample_df = prep_factor_dateset(list_of_interval=[testing_interval], use_cached=True)
        self.df = sample_df[testing_interval]
        self.cols = self.df.select_dtypes(float).columns.to_list()
        print(len(self.df['ticker'].unique()))

    def multithread_stepwise(self):

        self.score_col = 'cophenetic'
        period_list = list(range(1, round(365*5/self.testing_interval)))
        all_groups = itertools.product(period_list, self.cols)
        with mp.Pool(processes=12) as pool:
            pool.starmap(self.stepwise_test, all_groups)

    def stepwise_test(self, *args):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        period, init_col = args

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
            best = all_results.nlargest(1, self.score_col, keep='first')

            if best[self.score_col].values[0] > init_score:
                best_best = best.copy(1)
                init_cols = best['factors'].values[0].split(', ')
                init_score = best[self.score_col].values[0]
            else:
                print('-----> Return: ',init_col, period, init_score, init_cols)
                next_factor = False

                thread_engine_ali = create_engine(global_vals.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")
                with thread_engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
                    extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
                    best_best[['period', 'factors', self.score_col]].to_sql("des_factor_hierarchical", **extra)
                thread_engine_ali.dispose()

        # pd.concat(all_results_all, axis=0).to_csv(f'hierarchy_diffinit_{self.testing_interval}_{self.score_col}.csv')

def test_method(X):
    ''' test conditions on different conditions of cluster methods '''

    kwargs = {'distance_threshold': [0], 'linkage': ['average'], 'n_clusters':[None]}
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
        return x

    cols = df.select_dtypes(float).columns.to_list()
    for col in cols:
        if col != 'icb_code':
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
    data = test_cluster(testing_interval=91)
    data.multithread_stepwise()


