from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, MeanShift
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.preprocessing import robust_scale, minmax_scale, QuantileTransformer, PowerTransformer, MinMaxScaler
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from scipy.cluster.hierarchy import dendrogram

from sklearn import metrics
from s_dbw import S_Dbw
from jqmcvi import base
from descriptive_factor.ratio_cluster_metrics import *

def get_least_corr_important_factors():

    # df = combine_tri_worldscope(use_cached=True, save=True, currency=['USD']).get_results(list_of_interval=[365])[365]
    # df.dropna(subset=['change_tri_fillna']).to_csv('factor_30_sample.csv')

    # c = df.corr().unstack().reset_index()
    # c['corr_abs'] = c[0].abs()
    # c.sort_values('corr_abs', ascending=False).to_csv('factor_30_corr.csv')

    df = pd.read_csv('factor_30_sample.csv')
    df = df.replace([-np.inf, np.inf], [np.nan, np.nan])

    # # Usage PFA
    # pfa = PFA(n_features=8)
    # df_num = df.select_dtypes(float)
    # pfa.fit(df_num)
    # # To get the transformed matrix
    # x = pfa.features_
    # print(x)
    # # To get the column indices of the kept features
    # column_indices = [df_num.columns.to_list()[i] for i in pfa.indices_]
    # print(column_indices)
    # df = df[column_indices]

    return df

class PFA(object):
    ''' perform principal_feature_analysis on initial columns '''
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        X = np.nan_to_num(X, -99.9)
        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q).fit(X)  # calculation Cov matrix is embeded in PCA
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

class test_cluster:

    def __init__(self):
        self.cols = ['ticker', 'trading_day']
        self.cols += ['avg_market_cap_usd','vol','change_tri_fillna']
        # self.cols += ['avg_market_cap_usd','avg_volume','vol','change_assets','change_tri_fillna']
        self.cols += ['icb_code']

        self.df = get_least_corr_important_factors()[self.cols]
        self.clustering_metrics1 = [
            metrics.calinski_harabasz_score,
            metrics.davies_bouldin_score,
            metrics.silhouette_score,
            S_Dbw,
        ]
        self.clustering_metrics2 = [
            base.dunn
        ]

        self.trim_outlier()

    def trim_outlier(self):
        ''' trim outlier on testing sets '''

        def trim_scaler(x):
            x = np.clip(x, np.nanmean(x)-2*np.nanstd(x), np.nanmean(x)+2*np.nanstd(x))
            return robust_scale(x)

        for col in self.cols[2:-1]:
            self.df[col] = trim_scaler(self.df[col])

    def plot_scatter_hist(self):
        n = len(self.cols)-2
        fig = plt.figure(figsize=(n * 4, n * 4), dpi=60, constrained_layout=True)
        k=1
        for v1 in self.cols[2:]:
            for v2 in self.cols[2:]:
                print(v1, v2)
                ax = fig.add_subplot(n, n, k)
                if v1==v2:
                    ax.hist(self.df[v1], bins=20)
                    ax.set_xlabel(v1)
                else:
                    ax.scatter(self.df[v1], self.df[v2])
                    ax.set_xlabel(v1)
                    ax.set_ylabel(v2)
                k+=1
        plt.show()
        fig.savefig(f'cluster_selected_corr.png')
        plt.close(fig)

    def test_method(self, cluster_method, iter_conditions_dict):
        ''' test conditions on different conditions of cluster methods '''

        # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
        X = self.df.groupby(['ticker']).last()
        X = X.iloc[:,2:].values
        X = np.nan_to_num(X, -1)
        X = StandardScaler().fit_transform(X)

        m = {}
        condition_values = list(iter_conditions_dict.values())
        condition_keys = list(iter_conditions_dict.keys())

        for element in itertools.product(*condition_values):
            kwargs = dict(zip(condition_keys, list(element)))
            model = cluster_method(**kwargs).fit(X)
            try:
                y = model.predict(X)
            except:
                y = model.labels_

            # calculate matrics
            m[element] = {}
            m[element]['count'] = Counter(y)
            for i in self.clustering_metrics1:
                m[element][i.__name__] = i(X, y)
            for i in self.clustering_metrics2:
                m[element][i.__name__] = i([X[y==k] for k in set(y)])
            print(element, m[element])

        results = pd.DataFrame(m).transpose()
        results.to_csv(f'cluster_{cluster_method.__name__}.csv')

    def get_cluster(self, cluster_method, kwargs):
        ''' write cluster results to csv '''
        # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
        results = self.df.groupby(['ticker']).last()
        X = results.iloc[:,2:].values
        X = np.nan_to_num(X, -99.9)
        X = StandardScaler().fit_transform(X)

        model = cluster_method(**kwargs).fit(X)
        try:
            results['cluster'] = model.predict(X)
        except:
            results['cluster'] = model.labels_

        results.to_csv(f'result_cluster_{cluster_method.__name__}.csv')

        if cluster_method.__name__ == 'AgglomerativeClustering':
            plot_dendrogram(model, truncate_mode='level', p=3)
            plt.show()

def plot_dendrogram(model, **kwargs):
    ''' Create linkage matrix and then plot the dendrogram '''

    # create the counts of samples under each node
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

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

if __name__ == "__main__":
    data = test_cluster()

    # data.plot_scatter_hist()

    # data.test_method(KMeans, {'n_clusters': list(range(3,11))})
    # data.test_method(OPTICS, {'min_samples': list(range(1, 10, 1))})
    # data.test_method(DBSCAN, {'min_samples': [1], 'eps': list(np.arange(0.5, 2, 0.5))})
    # data.test_method(AgglomerativeClustering, {'n_clusters': list(range(3,11)), 'linkage':['ward', 'single']}) #  'complete', 'average',
    # data.test_method(AffinityPropagation, {'damping': list(np.arange(0.5, 1, 0.1))})
    # data.test_method(MeanShift, {'bandwidth': [None]})


    # data.get_cluster(DBSCAN, {'min_samples': 1, 'eps': 0.5})
    # data.get_cluster(AgglomerativeClustering, {'linkage': 'single', 'n_clusters': 6}) #  'complete', 'average',
    data.get_cluster(MeanShift, {'bandwidth': [None]})
