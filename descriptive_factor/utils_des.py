import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, minmax_scale, robust_scale
from sklearn.decomposition import PCA, TruncatedSVD

from fcmeans import FCM
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from descriptive_factor.fuzzy_metrics import xie_beni_index
from s_dbw import S_Dbw

import matplotlib.pyplot as plt


class read_item_df:
    def __init__(self, testing_interval=7):
        self.item_df = pd.read_csv(f'dcache_sample_{testing_interval}.csv')
        self.item_df = self.item_df.sort_values(by=['ticker', 'trading_day'])
        self.item_df = self.item_df.loc[self.item_df['ticker'].str[0] != '.']
        self.item_df.iloc[:, 2:] = self.__trim_outlier_std(self.item_df.iloc[:, 2:].fillna(0)).values
        self.peroid_list = sorted(list(self.item_df['trading_day'].unique()))

    def time_spot(self, n):
        ''' item_df updated to the latest date '''
        self.item_df = self.item_df.loc[self.item_df['trading_day'] == self.peroid_list[n]]

    def time_after(self, n):
        ''' item_df updated to the latest date '''
        self.item_df = self.item_df.loc[self.item_df['trading_day'] >= self.peroid_list[n]]

    def __trim_outlier_std(self, df):
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
            x = minmax_scale(x)
            df[col] = x
        return df
    
    def org_x(self, cols):
        return self.item_df[cols].values

    def pca_x(self, cols, n=2):
        X = self.item_df[cols].values
        model = PCA(n_components=n).fit(X)  # calculation Cov matrix is embeded in PCA
        X = model.transform(X)
        print(np.cumsum(model.explained_variance_ratio_))
        return X

    def svd_x(self, cols, n=2):
        X = self.item_df[cols].values
        model = TruncatedSVD(n_components=n).fit(X)  # calculation Cov matrix is embeded in PCA
        X = model.transform(X)
        print(np.cumsum(model.explained_variance_ratio_))
        return X

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

def cluster_hierarchical(X, precomputed, distance_threshold=None, n_clusters=None):
    kwargs = {'distance_threshold': distance_threshold, 'linkage': 'complete', 'n_clusters': n_clusters}
    if precomputed:
        kwargs['affinity'] = 'precomputed'
        model = AgglomerativeClustering(**kwargs).fit(X)
    else:
        kwargs['affinity'] = 'affinity'
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

def distance_comp():
    from scipy.spatial import distance
    d_tri = distance.pdist(X, 'euclidean')
    d_tri = distance.squareform(d_tri)