from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN, OPTICS, AgglomerativeClustering, MeanShift
from fcmeans import FCM
import matplotlib.animation as animation
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, robust_scale, minmax_scale, QuantileTransformer, PowerTransformer, MinMaxScaler
from sklearn import decomposition
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestCentroid

import global_vals
from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
from collections import Counter
from scipy.cluster.hierarchy import dendrogram
import scipy.spatial

from sklearn import metrics
from s_dbw import S_Dbw
# from jqmcvi import base
from descriptive_factor.fuzzy_metrics import *

# --------------------------------- Prepare Datasets ------------------------------------------

def prep_factor_dateset(use_cached=True, list_of_interval=[7, 30, 91]):

    sample_df = {}
    corr_df = {}
    list_of_interval_redo = []

    def calc_corr_csv(df):
        c = df.corr().unstack().reset_index()
        c['corr_abs'] = c[0].abs()
        c = c.sort_values('corr_abs', ascending=True)
        c.to_csv(f'sample_corr_{i}.csv')
        return c

    if use_cached:
        for i in list_of_interval:
            try:
                sample_df[i] = pd.read_csv(f'dcache_sample_{i}.csv')
                corr_df[i] = pd.read_csv(f'sample_corr_{i}.csv')
            except:
                list_of_interval_redo.append(i)

    if len(list_of_interval_redo)>0:
        print(f'-----------------------------> No local cache for {list_of_interval_redo} -> Process')
        df_dict = combine_tri_worldscope(use_cached=True, save=True, currency=['USD']).get_results(list_of_interval_redo)
        for k, df in df_dict:
            corr_df[k] = calc_corr_csv(df)
        sample_df.update(df_dict)

    return sample_df, corr_df

# ----------------------------------------- Similar Stocks --------------------------------------------------

def get_most_close_ticker():

    df = prep_factor_dateset()

    # # Usage PFA
    # pfa = PFA(n_features=10)
    # df_num = df.select_dtypes(float)
    # pfa.fit(df_num)
    # cols = [df_num.columns.to_list()[i] for i in pfa.indices_]  # To get the column indices of the kept features
    # print(cols)

    cols = ['vol', 'skew', 'avg_inv_turnover', 'avg_capex_to_dda', 'change_tri_fillna', 'avg_earnings_yield',
            'avg_gross_margin', 'avg_market_cap_usd', 'avg_roe', 'currency_code','icb_code']

    df_last = df.groupby(['ticker'])[cols].last()
    X = df_last.values
    X = np.nan_to_num(X, -1)
    X = StandardScaler().fit_transform(X)

    nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(X)

    # KNN on sample ticker
    pt_idx = list(df_last.index).index('IBM')
    distances, indices = nbrs.kneighbors([X[pt_idx]])
    pt_close = df_last.iloc[indices[0], :]
    pt_close['distance'] = distances[0]
    pt_close = pt_close.transpose()

    # KNN on all tickers
    all_distances, all_indices = nbrs.kneighbors(X)
    le = LabelEncoder().fit(list(df_last.index))
    knn_tickers = pd.DataFrame(all_indices).apply(le.inverse_transform)

    print(pt_close)

# -------------------------------- Feature Selection -----------------------------------------------

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

        # kmeans = AgglomerativeClustering(n_clusters=self.n_features).fit(A_q)
        # clusters = kmeans.labels_
        # cluster_centers = NearestCentroid().fit(A_q, clusters).centroids_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

# -------------------------------- Test Cluster -----------------------------------------------

class test_cluster:

    def __init__(self, last=True):

        testing_interval = 91

        sample_df, corr_df = prep_factor_dateset(list_of_interval=[testing_interval])
        self.df = sample_df[testing_interval]
        self.c = corr_df[testing_interval]

        if last:
            self.df = self.df.groupby(['ticker']).last().reset_index()
            self.df = self.df.replace([-np.inf, np.inf], [np.nan, np.nan])

        self.label_cols = ['ticker', 'trading_day']
        self.cols = ['avg_market_cap_usd', 'avg_ebitda_to_ev', 'vol', 'change_tri_fillna', 'icb_code']
        self.cols = 'avg_market_cap_usd, avg_ebitda_to_ev, vol, change_tri_fillna, icb_code, change_volume, change_earnings, ' \
                    'ret_momentum, avg_interest_to_earnings, change_ebtda, avg_roe, avg_ni_to_cfo, avg_div_payout, change_dividend, ' \
                    'avg_inv_turnover, change_assets, avg_gross_margin, avg_debt_to_asset, skew, avg_fa_turnover'.split(', ')
        # self.cols = self.df.select_dtypes(float).columns.to_list()

        self.clustering_metrics1 = [
            metrics.calinski_harabasz_score,
            metrics.davies_bouldin_score,
            metrics.silhouette_score,
            S_Dbw,
        ]
        # self.clustering_metrics2 = [base.dunn]
        self.clustering_metrics_fuzzy = [
            partition_coefficient,
            xie_beni_index,
            fukuyama_sugeno_index,
            fuzzy_hypervolume,
            beringer_hullermeier_index,
            bouguessa_wang_sun_index,
        ]

        # x = x.describe()
        self.trim_outlier_std()
        # x = x.describe()

        # self.plot_scatter_hist(self.df, self.cols)

    def trim_outlier_std(self):
        ''' trim outlier on testing sets '''

        def trim_scaler(x):
            x = np.clip(x, np.nanmean(x)-2*np.nanstd(x), np.nanmean(x)+2*np.nanstd(x))
            # x = np.clip(x, np.percentile(x, 0.10), np.percentile(x, 0.90))
            x = robust_scale(x)
            return x

        for col in self.cols:
            if col != 'icb_code':
                x = trim_scaler(self.df[col])
                x = StandardScaler().fit_transform(np.expand_dims(x, 1))[:, 0]
                self.df[col] = x

    def stepwise_test_method(self, cluster_method, iter_conditions_dict):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        self.cols = ['icb_code']    # step wise start with icb_code

        all_results = []
        while len(self.cols) != len(self.c['level_0'].unique()):

            # add lst correlated factors
            lst = self.c.loc[(self.c['level_0'].isin(self.cols)) & (~self.c['level_1'].isin(self.cols))]
            lst = lst.groupby(['level_1'])['corr_abs'].max().sort_values()
            self.cols.append(lst.index[0])
            print(f'-----------------> add {lst.index[0]}')

            # testing on different cluster methods
            results = self.test_method(cluster_method, iter_conditions_dict, save_csv=False)
            results['factors'] = ', '.join(self.cols)
            all_results.append(results)

        all_results = pd.concat(all_results, axis=0)
        all_results.to_csv(f'stepwise_cluster_{cluster_method.__name__}.csv', index=False)

    def pillar_test_method(self, cluster_method, iter_conditions_dict):
        ''' test cluster using features manually classified to 4 pillar '''

        with global_vals.engine_ali.connect() as conn:
            uni = pd.read_sql(f'SELECT name, pillar FROM {global_vals.formula_factors_table}_descriptive', conn)
        global_vals.engine_ali.dispose()

        all_results = []
        for p, g in uni.groupby(['pillar']):

            # pillar columns
            p_cols = g['name'].to_list()
            cols = ['avg_'+x for x in p_cols] + ['change_'+x for x in p_cols] + p_cols
            self.cols = self.df.filter(cols).columns.to_list()
            print(self.cols)

            # testing on different cluster methods
            # results = self.test_method(cluster_method, iter_conditions_dict, save_csv=False)

            results = self.get_cluster(cluster_method, iter_conditions_dict, suffixes=p)

        #     results['pillar'] = p
        #     all_results.append(results)
        #
        # all_results = pd.concat(all_results, axis=0)
        # all_results.to_csv(f'pillar_cluster_{cluster_method.__name__}.csv', index=False)

    def test_method(self, cluster_method, iter_conditions_dict, save_csv=True):
        ''' test conditions on different conditions of cluster methods '''

        print('Testing on: ', self.cols)

        # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
        X = self.df[self.cols].values
        X = np.nan_to_num(X, -1)

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
                u = model.soft_predict(X)
                v = model.centers

                centre_df = pd.DataFrame(v, columns=self.cols)

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
                m[element]['count'] = Counter(y)
                for i in self.clustering_metrics1:
                    m[element][i.__name__] = i(X, y)
                # for i in self.clustering_metrics2:
                #     m[element][i.__name__] = i([X[y==k] for k in set(y)])
                print(element, m[element])

        results = pd.DataFrame(m).transpose()
        results.index = results.index.set_names(condition_keys)
        results = results.reset_index()

        if save_csv:
            results.to_csv(f'cluster_{cluster_method.__name__}.csv', index=False)

        return results

    def get_cluster(self, cluster_method, kwargs, suffixes='', save_csv=True):
        ''' write cluster results to csv '''

        # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
        results = self.df[self.cols]
        X = results.values
        X = np.nan_to_num(X, -1)
        X = StandardScaler().fit_transform(X)

        if cluster_method.__name__ == 'FCM':
            model = cluster_method(**kwargs)
            model.fit(X)
        else:
            model = cluster_method(**kwargs).fit(X)

        try:
            results['cluster'] = model.predict(X)
        except:
            results['cluster'] = model.labels_

        # x1 = model.predict(X)
        # x2 = model.soft_predict(X)

        if save_csv:
            results.to_csv(f'result_cluster_{cluster_method.__name__}.csv')

        plot_scatter_hist(results, self.cols, cluster_method.__name__, suffixes=suffixes)
        # plot_pca_scatter_hist(results, cluster_method.__name__)

        if cluster_method.__name__ == 'AgglomerativeClustering':
            plot_dendrogram(model, truncate_mode='level', p=3)
            plt.show()

# -------------------------------- Plot Cluster -----------------------------------------------

def plot_scatter_hist(df, cols, cluster_method, suffixes):

    n = len(cols)
    fig = plt.figure(figsize=(n * 4, n * 4), dpi=60, constrained_layout=True)
    k=1
    for v1 in cols:
        for v2 in cols:
            print(v1, v2)
            ax = fig.add_subplot(n, n, k)
            if v1==v2:
                ax.hist(df[v1], bins=20)
                ax.set_xlabel(v1)
            else:
                plt_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                             'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
                colors = dict(zip(list(df['cluster'].unique()), plt_color[:df['cluster'].max()+1]))
                ax.scatter(df[v1], df[v2], c=list(df['cluster'].map(colors).values), alpha=.5)
                ax.set_xlabel(v1)
                ax.set_ylabel(v2)
            k+=1

    fig.savefig(f'cluster_selected_{cluster_method}_{suffixes}.png')
    plt.close(fig)
    
def plot_pca_scatter_hist(df, cluster_method):

    X = np.nan_to_num(df.apply(pd.to_numeric, errors='coerce').values[:,1:-1], -1)
    # X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=None).fit(X)  # calculation Cov matrix is embeded in PCA
    print(pca.explained_variance_ratio_)
    print(pca.transform(X))
    X_trans = pca.transform(X)
    v1, v2 = X_trans[:,0], X_trans[:,1]

    fig = plt.figure(figsize=(4, 4), dpi=60, constrained_layout=True)
    plt_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = dict(zip(list(df['cluster'].unique()), plt_color[:df['cluster'].max()+1]))
    plt.scatter(v1, v2, c=list(df['cluster'].map(colors).values))

    fig.savefig(f'cluster_selected_pca_{cluster_method}.png')
    plt.close(fig)

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

    # get_most_close_ticker()

    # data.plot_scatter_hist()

    # data.test_method(KMeans, {'n_clusters': list(range(3,11))})
    # data.test_method(MeanShift, {'bandwidth': [None]})
    # data.test_method(AffinityPropagation, {'damping': list(np.arange(0.5, 1, 0.1))})
    # data.test_method(OPTICS, {'min_samples': list(range(1, 10, 1))})
    # data.test_method(DBSCAN, {'min_samples': [1], 'eps': list(np.arange(0.5, 2, 0.5))})
    # data.test_method(AgglomerativeClustering, {'n_clusters': list(range(3,11)), 'linkage':['ward']}) #  'complete', 'average',
    # data.test_method(GaussianMixture, {'n_components': list(range(3,11))})
    # data.test_method(FCM, {'n_clusters': list(range(5,10)), 'm':list(np.arange(1.25, 2.01, 0.25))})

    # data.stepwise_test_method(GaussianMixture, {'n_components': list(range(3, 11))})
    # data.stepwise_test_method(AgglomerativeClustering, {'n_clusters': list(range(3, 11)), 'linkage':['ward']})
    data.stepwise_test_method(FCM, {'n_clusters': list(range(5, 20, 2)), 'm':list(np.arange(1.25, 2.01, 0.25))})

    # data.pillar_test_method(AgglomerativeClustering, {'n_clusters': list(range(3, 11)), 'linkage':['ward']})
    # data.pillar_test_method(FCM, {'n_clusters': list(range(5,10)), 'm':list(np.arange(1.25, 2.01, 0.25))})
    # data.pillar_test_method(FCM, {'n_clusters': 9, 'm': 1.5})

    # data.get_cluster(DBSCAN, {'min_samples': 1, 'eps': 0.5})
    # data.get_cluster(AgglomerativeClustering, {'linkage': 'ward', 'n_clusters': 9, 'compute_distances': True}) #  'complete', 'average',
    # data.get_cluster(MeanShift, {'bandwidth': [None]})
    # data.get_cluster(GaussianMixture, {'n_components': 9})
    # data.get_cluster(FCM, {'n_clusters': 9, 'm': 1.25})


