from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter

from sklearn import metrics
from s_dbw import S_Dbw
from jqmcvi import base
from descriptive_factor.ratio_cluster_metrics import *

def get_least_corr_important_factors():

    # df = combine_tri_worldscope(use_cached=True, save=True, currency=['USD']).get_results(list_of_interval=[30])[30]
    # df.dropna(subset=['change_tri_fillna']).to_csv('factor_30_sample.csv')

    df = pd.read_csv('factor_30_sample.csv')
    # c = df.corr().unstack().reset_index()
    # c['corr_abs'] = c[0].abs()
    # c.sort_values('corr_abs', ascending=False).to_csv('factor_30_corr.csv')
    return df

class test_cluster:

    def __init__(self):
        self.cols = ['ticker', 'trading_day', 'avg_market_cap_usd','avg_earnings_yield','avg_volume','vol','change_assets',
                'icb_code', 'change_tri_fillna']
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

    def plot_scatter_hist(self):
        n = len(self.cols)-2
        fig = plt.figure(figsize=(n * 4, n * 4), dpi=120, constrained_layout=True)
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
        fig.savefig(f'cluster_selected_corr.png')
        plt.close(fig)

    def test_method(self, cluster_method, iter_conditions_dict):

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

def root_mean_square_std_dev(y, c):

    centroids = np.zeros((len(y), len(X[0])), dtype=float)
    for k in range(len(y)):
        centroids[k] = c[y[k]]

if __name__ == "__main__":
    data = test_cluster()
    # data.plot_scatter_hist()
    # data.test_method(KMeans, {'n_clusters': list(range(3,11))})
    data.test_method(OPTICS, {'min_samples': list(range(1, 10, 1))})
    # data.test_method(DBSCAN, {'min_samples': list(range(10, 100, 10)), 'eps': list(range(0.1, 1, 0.2))})