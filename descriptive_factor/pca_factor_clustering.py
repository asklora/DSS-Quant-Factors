import numpy as np
from sqlalchemy import create_engine
import global_vals
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from collections import Counter

from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
from hierarchy_ratio_cluster import trim_outlier_std

import gc
import itertools
import multiprocessing as mp

from fcmeans import FCM
from descriptive_factor.fuzzy_metrics import *
from sklearn.cluster import FeatureAgglomeration, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, minmax_scale
from sklearn import metrics
from s_dbw import S_Dbw
from descriptive_factor.fuzzy_metrics import xie_beni_index

from hierarchy_ratio_cluster import good_cols, fill_all_day_interpolate, relativedelta, test_missing, good_mom_cols, trim_outlier_std
from hierarchy_ratio_cluster import test_method as h_test_cluster
from fcm_ratio_cluster import test_method as f_test_cluster
from gaussian_ratio_cluster import test_method as g_test_cluster

# --------------------------------- Test subset composition ------------------------------------------

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

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    kwargs_den = kwargs.copy()
    kwargs_den.pop('suffix')
    kwargs_den['orientation'] ='left'

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs_den)
    plt.tight_layout()
    plt.savefig('dendrogram_{}.png'.format(kwargs['suffix']))

def feature_hierarchical_plot(testing_interval=7):
    ''' perform agglomorative hierarchical cluster on descriptive factors at different interval'''

    df = pd.read_csv(f'dcache_sample_{testing_interval}.csv')

    df = df.loc[df['ticker'].str[0]!='.']
    # df = df.loc[df['trading_day'] > (dt.datetime.today() - relativedelta(years=5))]

    df = df.iloc[:, 2:].fillna(0)
    X = trim_outlier_std(df).values

    print(df.dtypes)
    print(df.isnull().sum().sum())

    agglo = FeatureAgglomeration(distance_threshold=0, n_clusters=None)
    model = agglo.fit(X)
    plot_dendrogram(model, labels=df.columns.to_list(), suffix=f'_features{testing_interval}')

# --------------------------------- Test subset composition ------------------------------------------

def feature_subset_pca(testing_interval=7):
    ''' perfrom PCA on different feature subsets '''

    all_df_all = pd.read_csv(f'dcache_sample_{testing_interval}.csv')

    all_df_all = all_df_all.loc[all_df_all['ticker'].str[0]!='.']
    all_df_all['trading_day'] = pd.to_datetime(all_df_all['trading_day'])

    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT pillar, name FROM {global_vals.formula_factors_table}_descriptive', conn)
    global_vals.engine_ali.dispose()

    org_name = formula.loc[formula['pillar']=='momentum', 'name'].to_list()
    org_name = org_name + ['avg_' + x for x in org_name] + ['change_' + x for x in org_name]

    # org_name = ['change_tri_fillna', 'avg_volume_1w3m', 'ret_momentum', 'vol', 'icb_code']

    all_df_all = all_df_all.fillna(0).filter(['trading_day','ticker']+org_name)
    print(all_df_all.columns.to_list()[2:])

    for i in range(1, 10):
        df = all_df_all.loc[all_df_all['trading_day'] > (dt.datetime.today() - relativedelta(years=i))]
        X = trim_outlier_std(df.iloc[:,2:]).values
        pca = PCA(n_components=None).fit(X)  # calculation Cov matrix is embeded in PCA
        r = pca.explained_variance_ratio_
        # plt.plot(np.cumsum(r))
        # plt.xlabel(p)
        # plt.show()

        print(i, np.cumsum(r))

def feature_subset_cluster(testing_interval=7):
    ''' try cluster on features '''

    all_df_all = pd.read_csv(f'dcache_sample_{testing_interval}.csv')
    all_df_all = all_df_all.loc[all_df_all['ticker'].str[0]!='.']
    all_df_all['trading_day'] = pd.to_datetime(all_df_all['trading_day'])
    all_df_all = all_df_all.loc[all_df_all['trading_day'] == all_df_all['trading_day'].max()]

    d_icb = all_df_all['icb_code'].values
    # d_icb = np.array([np.cos(d_icb/85103035*2*np.pi), np.sin(d_icb/85103035*2*np.pi)]).T
    # d_icb = scipy.spatial.distance.pdist(d_icb, 'cosine')
    # d_icb = scipy.spatial.distance.squareform(d_icb)
    all_df_all[['icb_code1', 'icb_code2']] = np.array(
        [scale(np.cos(d_icb / 85103035 * 2 * np.pi)), scale(np.sin(d_icb / 85103035 * 2 * np.pi))]).T

    org_name = ['change_tri_fillna', 'avg_volume_1w3m', 'ret_momentum', 'vol']

    r = {}
    for col in all_df_all.columns.to_list():
        if col not in org_name + ['trading_day','ticker']:
            org_name += [col]
            org_name += ['icb_code1', 'icb_code2']
            df = all_df_all.fillna(0).filter(['trading_day','ticker'] + org_name)
            # print(org_name)

            # df = all_df_all.loc[all_df_all['trading_day'] > (dt.datetime.today() - relativedelta(years=1))]
            df.iloc[:, 2:-2] = trim_outlier_std(df.iloc[:, 2:-2])
            # print(df.describe())
            # X = df.iloc[:, 2:].values

            # d_tri = scipy.spatial.distance.pdist(X[:, [-1]], 'euclidean')
            # d_tri = scipy.spatial.distance.squareform(d_tri)

            X = df[org_name].values

            # kwargs = {'distance_threshold': 0, 'linkage': 'complete', 'n_clusters': None, 'affinity': 'euclidean'}
            # kwargs = {'distance_threshold': None, 'linkage': 'complete', 'n_clusters': 15, 'affinity': 'euclidean'}
            # model = AgglomerativeClustering(**kwargs).fit(X)
            # df['cluster'] = model.labels_
            # plot_dendrogram(model, labels=df['ticker'].to_list(), suffix=f'1w_{testing_interval}', leaf_font_size=2)

            for i in range(15, 16):
                model = FCM(n_clusters=i, m=1.5)
                model.fit(X)
                df['cluster'] = model.predict(X)
                u = model.u
                v = model.centers
                score = xie_beni_index(X, u, v.T, 2)
                r[col] = score
                print(org_name, i, score)

            org_name = org_name[:3]
            # plot_scatter_hist(df, org_name, 'h2')
    df = pd.DataFrame(r, index=[0]).transpose()
    print(df)

def calc_metrics(model, X):

    clustering_metrics = [
        metrics.calinski_harabasz_score,
        metrics.davies_bouldin_score,
        metrics.silhouette_score,
        S_Dbw,
    ]

    m = {}
    for i in clustering_metrics:
        m[i.__name__] = i(X, model.labels_)
    print(m)

def plot_scatter_hist(df, cols, suffixes):

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
                ax.scatter(df[v1], df[v2], c=df['cluster'], cmap="gist_rainbow", alpha=.5)
                ax.set_xlabel(v1)
                ax.set_ylabel(v2)
            k+=1

    plt.legend()
    plt.tight_layout()
    plt.show()
    # fig.savefig(f'cluster_selected_{suffixes}.png')
    # plt.close(fig)

# --------------------------------- Test on User Portfolio ------------------------------------------

def read_port_from_firebase():
    ''' read user portfolio details from firestore '''

    import firebase_admin
    from firebase_admin import credentials, firestore

    with global_vals.engine.connect() as conn:
        rating = pd.read_sql(f'SELECT ticker, ai_score FROM universe_rating', conn)
        rating = rating.set_index('ticker')['ai_score'].to_dict()
    global_vals.engine.dispose()

    # Get a database reference to our posts
    if not firebase_admin._apps:
        cred = credentials.Certificate(global_vals.firebase_url)
        default_app = firebase_admin.initialize_app(cred)

    db = firestore.client()
    doc_ref = db.collection(u"prod_portfolio").get()

    object_list = []
    for data in doc_ref:
        format_data = {}
        data = data.to_dict()
        format_data['index'] = data.get('profile').get('email')
        format_data['return'] = data.get('total_profit_pct')
        for i in data.get('active_portfolio'):
            format_data['ticker'] = i.get('ticker')
            format_data['spot_date'] = i.get('spot_date')
            format_data['pct_profit'] = i.get('pct_profit')
            format_data['bot'] = i.get('bot_details').get('bot_apps_name')
            object_list.append(format_data.copy())

    result = pd.DataFrame(object_list).sort_values(by=['return'], ascending=False)
    result['rating'] = result['ticker'].map(rating)

    result.to_csv('port_result.csv', index=False)
    return result

def analyse_port():
    ''' analyse ticker, pct_return of users '''

    result = pd.read_csv('port_result.csv')

    mean_index = result.groupby('index').mean()
    mean_bot = result.groupby('bot').mean()
    mean_date = result.groupby('spot_date').mean()
    count_ticker = result.groupby('ticker')['return'].count().to_frame().reset_index()

    with global_vals.engine.connect() as conn:
        rating = pd.read_sql(f'SELECT ticker, ai_score FROM universe_rating', conn)
        rating = rating.set_index('ticker')['ai_score'].to_dict()
    global_vals.engine.dispose()

    count_ticker['rating'] = count_ticker['ticker'].map(rating)
    print(result['bot'].unique())

# --------------------------------- Test on Fund Portfolio ------------------------------------------

def read_fund_port():
    ''' Get top 5 holdings for mid-large size fund '''

    with global_vals.engine.connect() as conn, global_vals.engine_ali.connect() as conn_ali:
        uni = pd.read_sql(f"SELECT * FROM universe WHERE currency_code='USD'", conn)
        port = pd.read_sql('SELECT * FROM data_factor_eikon_fund_holdings', conn_ali)
        size = pd.read_sql('SELECT * FROM data_factor_eikon_fund_size ORDER BY tna', conn_ali)
    global_vals.engine.dispose()

    # filter fund with size in middle-low range (10% - 50%)
    size = size.loc[(size['tna']>size['tna'].quantile(0.1))&(size['tna']<size['tna'].quantile(0.5))]
    port = port.loc[port['fund'].isin(size['ric'].to_list())]

    # filter ticker in our universe
    port['ticker'] = port['ticker'].str.replace('.OQ', '.O')
    df = port.merge(uni, on=['ticker'], how='inner')

    # first 5 holdings
    valid_fund = (df.groupby('fund')['ticker'].count()>5)
    valid_fund = valid_fund.loc[valid_fund]
    df = df.loc[df['fund'].isin(list(valid_fund.index))]
    df = df.groupby('fund')[['fund', 'ticker']].head(5)

    f = Counter(df['fund'].to_list())
    print(f)
    t = Counter(df['ticker'].to_list())
    print(t)

    return df

if __name__ == "__main__":

    # feature_cluster()
    # feature_subset_pca()
    # feature_subset_cluster()

    # read_port_from_firebase()
    # analyse_port()

    read_fund_port()