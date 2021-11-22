import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import scale, minmax_scale, robust_scale, quantile_transform
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.stats import skew

from dateutil.relativedelta import relativedelta
import datetime as dt

from fcmeans import FCM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from descriptive_factor.fuzzy_metrics import xie_beni_index
from s_dbw import S_Dbw

import matplotlib.pyplot as plt


class read_item_df:
    def __init__(self, testing_interval=7, plot=False, currency=None):
        self.item_df = pd.read_csv(f'dcache_sample_{testing_interval}.csv')
        self.item_df = self.item_df.sort_values(by=['ticker', 'trading_day'])
        self.item_df_org = self.item_df.copy(1)

        self.item_df = self.item_df.loc[self.item_df['ticker'].str[0] != '.']

        if currency=='HKD':
            self.item_df = self.item_df.loc[self.item_df['ticker'].str[-3:]=='.HK']
        else:
            self.item_df = self.item_df.loc[self.item_df['ticker'].str[-3:]!='.HK']

        miss_df = self.item_df.isnull().sum()/len(self.item_df)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(miss_df[miss_df > 0].sort_values(ascending=False))

        self.item_df.iloc[:, 2:] = self.__trim_outlier_std(self.item_df.iloc[:, 2:].fillna(0), plot=plot).values
        self.peroid_list = sorted(list(self.item_df['trading_day'].unique()))
        self.peroid_list.reverse()
        self.orginal_cols = self.item_df.columns.to_list()[2:]

    def time_after(self, start, end):
        ''' item_df updated to the latest date '''
        self.item_df['trading_day'] = pd.to_datetime(self.item_df['trading_day'])
        self.item_df = self.item_df.loc[self.item_df['trading_day'] >= (dt.datetime.now()-relativedelta(years=start))]
        self.item_df = self.item_df.loc[self.item_df['trading_day'] <= (dt.datetime.now()-relativedelta(years=end))]

    def __trim_outlier_std(self, df, plot=False):
        ''' trim outlier on testing sets '''

        def trim_scaler(x):
            # s = skew(x)
            # if (s < -5) or (s > 5):
            #     x = np.log(x + 1 - np.min(x))
            x = x.values
            # m = np.median(x)
            # std = np.nanstd(x)
            # x = np.clip(x, m - 2 * std, m + 2 * std)
            return quantile_transform(np.reshape(x, (x.shape[0], 1)), output_distribution='normal', n_quantiles=1000)[:,0]

        cols = df.select_dtypes(float).columns.to_list()
        for col in cols:
            if plot:
                fig = plt.figure(figsize=(8, 4), dpi=60, constrained_layout=True)
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.hist(df[col], bins=20)
            if col != 'icb_code':
                x = trim_scaler(df[col])
            else:
                x = df[col].values
            x = scale(x.T)
            # x = minmax_scale(x)
            df[col] = x
            if plot:
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.hist(df[col], bins=20)
                plt.suptitle(col)
                plt.show()
                plt.close(fig)
        return df

    def select_comb_x(self, all_cols, n_cols=2):
        comb_list = itertools.combinations(all_cols, n_cols)
        return comb_list

    def org_x(self, cols=None):
        return self.item_df[cols].values

    def pca_x(self, cols, n=2):
        X = self.item_df[cols].values
        model = PCA(n_components=n).fit(X)  # calculation Cov matrix is embeded in PCA
        X = model.transform(X)
        components = pd.DataFrame(model.components_, columns=cols).transpose()
        explained_ratio = np.cumsum(model.explained_variance_ratio_)
        print(explained_ratio)
        return X, explained_ratio, components

    def svd_x(self, cols, n=2):
        X = self.item_df[cols].values
        model = TruncatedSVD(n_components=n).fit(X)  # calculation Cov matrix is embeded in PCA
        X = model.transform(X)
        explained_ratio = np.cumsum(model.explained_variance_ratio_)
        print(explained_ratio)
        return X, explained_ratio


def cluster_fcm(X, n_clusters=2, m=2):
    model = FCM(n_clusters=n_clusters, m=m)
    model.fit(X)
    u = model.u
    v = model.centers
    score = xie_beni_index(X, u, v.T, 2)
    y = model.predict(X)
    return score, y

def cluster_gaussian(X, n_clusters=2):
    model = GaussianMixture(n_components=n_clusters)
    y = model.fit_predict(X)
    score = S_Dbw(X, y)
    return score, y

def cluster_hierarchical(X, precomputed=False, distance_threshold=None, n_clusters=2):
    kwargs = {'distance_threshold': distance_threshold, 'linkage': 'complete', 'n_clusters': n_clusters}
    if precomputed:
        kwargs['affinity'] = 'precomputed'
        model = AgglomerativeClustering(**kwargs).fit(X)
    else:
        kwargs['affinity'] = 'euclidean'
        model = AgglomerativeClustering(**kwargs).fit(X)
    y = model.labels_
    score = S_Dbw(X, y)
    return score, y

def plot_scatter_2d(X, y, annotate=None):
    ''' plot variance in 2D space (variance on 2D side) '''
    plt.scatter(X[:,0], X[:,1], c=y, cmap="Set1", alpha=.5)
    if annotate:
        for i in range(len(annotate)):
            plt.annotate(annotate[i], (X[:,0], X[:,1]), fontsize=5)
    plt.tight_layout()
    plt.show()

def plot_scatter_nd(X, y, cols=None):
    n = X.shape[1]
    fig = plt.figure(figsize=(n * 4, n * 4), dpi=60, constrained_layout=True)
    k = 1
    for v1 in range(n):
        for v2 in range(n):
            # print(v1, v2)
            ax = fig.add_subplot(n, n, k)
            if v1 == v2:
                ax.hist(X[:,v1], bins=20)
                ax.set_xlabel(v1)
            if v1 < v2:
                ax.scatter(X[:,v1], X[:,v2], c=y, cmap="Set1", alpha=.5)
                if cols:
                    ax.set_xlabel(cols[v1])
                    ax.set_ylabel(cols[v2])
            k += 1

    plt.show()
    plt.close(fig)

def plot_dendrogram(model, **kwargs):
    ''' Create linkage matrix and then plot the dendrogram '''

    from scipy.cluster.hierarchy import dendrogram

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

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    kwargs_den = kwargs.copy()
    kwargs_den['orientation'] ='left'

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs_den)
    plt.tight_layout()
    plt.show()
    # plt.savefig('dendrogram_{}.png'.format(kwargs['suffix']))

def feature_hierarchical_plot(df):
    ''' perform agglomorative hierarchical cluster on descriptive factors at different interval'''

    agglo = FeatureAgglomeration(distance_threshold=0, n_clusters=None)
    model = agglo.fit(df.values)
    plot_dendrogram(model, labels=df.columns.to_list())

def selection_feature_hierarchical(df, i=0.5):
    ''' perform agglomorative hierarchical cluster on descriptive factors at different interval (Update: i=n_cluster)'''

    X = df.values
    agglo = FeatureAgglomeration(distance_threshold=0, n_clusters=None)
    model = agglo.fit(X)

    distance_threshold = np.max(model.distances_)*i
    agglo = FeatureAgglomeration(distance_threshold=distance_threshold, n_clusters=None)
    model = agglo.fit(X)
    print(f'---> {i} feature cluster for {model.n_clusters_} clusters')

    y = model.labels_
    lst = []
    for i in set(y):
        lst.append(list(np.array(df.columns.to_list())[y==i]))
    return lst

def distance_comp():
    from scipy.spatial import distance
    d_tri = distance.pdist(X, 'euclidean')
    d_tri = distance.squareform(d_tri)

def report_to_slack(message, channel='U026B04RB3J'):

    from slack_sdk import WebClient
    SLACK_API = "xoxb-305855338628-1139022048576-2KsNu5mJCbgRGh8z8S8NOdGI"

    try:
        client = WebClient(token=SLACK_API, timeout=30)
        client.chat_postMessage(
            channel=channel,
            text=message,
        )
    except Exception as e:
        print(e)

