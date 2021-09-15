from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
import pandas as pd
import numpy as np

def get_least_corr_important_factors():

    # df = combine_tri_worldscope(use_cached=True, save=True, currency=['USD']).get_results(list_of_interval=[30])[30]
    # df.dropna(subset=['change_tri_fillna']).to_csv('factor_30_sample.csv')

    df = pd.read_csv('factor_30_sample.csv')
    # c = df.corr()
    # c.unstack().to_csv('factor_30_corr.csv')
    return df

def test_cluster(method='kmean'):
    ''

    cols = ['ticker', 'trading_day', 'avg_market_cap_usd','avg_earnings_yield','avg_volume','vol','change_assets','icb_code','change_tri_fillna']
    df = get_least_corr_important_factors()[cols]

    print(df.corr().unstack().abs().sort_values(ascending=False))

    # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
    X = df.iloc[:,2:].values
    X = np.nan_to_num(X, -1)
    X = StandardScaler().fit_transform(X)

    if method == 'kmean':
        cluster_no = 5
        kmeans = KMeans(cluster_no).fit(X)
        y = kmeans.predict(X)
    elif method == 'optics':
        opt = OPTICS(min_samples=1).fit(X)
        y = opt.labels_
    elif method == 'dbscan':
        db = DBSCAN(min_samples=3, eps=0.9).fit(X)
        y = db.labels_

    df['cluster'] = y
    df_c = pd.pivot_table(df, index=['ticker'], columns=['trading_day'], values='cluster')
    df_c.to_csv('df_cluster.csv')
    print(df_c)

if __name__ == "__main__":
    test_cluster()