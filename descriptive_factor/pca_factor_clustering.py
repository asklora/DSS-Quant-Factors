from sqlalchemy import create_engine
import global_vals
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
from hierarchy_ratio_cluster import trim_outlier_std

import gc
import itertools
import multiprocessing as mp

from fcmeans import FCM
from descriptive_factor.fuzzy_metrics import *
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA

from hierarchy_ratio_cluster import good_cols, fill_all_day_interpolate, relativedelta, test_missing, good_mom_cols, trim_outlier_std

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

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=kwargs['labels'], orientation='left')
    plt.tight_layout()
    plt.savefig('dendrogram_{}.png'.format(kwargs['suffix']))

def feature_hierarchical_plot(testing_interval=7):

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

    all_df = pd.read_csv(f'dcache_sample_{testing_interval}.csv')

    all_df = all_df.loc[all_df['ticker'].str[0]!='.']
    # all_df = all_df.loc[df['trading_day'] > (dt.datetime.today() - relativedelta(years=5))]

    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT pillar, name FROM {global_vals.formula_factors_table}_descriptive', conn)
    global_vals.engine_ali.dispose()

    p = 'all'

    # fac_dict = {}
    # for p, g in formula.groupby('pillar'):
    #     org_name = g['name'].to_list()
    #     fac_dict[p] = org_name + ['avg_'+x for x in org_name] + ['change_'+x for x in org_name]

    df = all_df.iloc[:, 2:].fillna(0)
    # df = df.filter(fac_dict[p])
    print(p, df.columns.to_list())

    X = trim_outlier_std(df).values
    pca = PCA(n_components=None).fit(X)  # calculation Cov matrix is embeded in PCA
    r = pca.explained_variance_ratio_
    plt.plot(np.cumsum(r))
    plt.xlabel(p)
    plt.show()

    print(np.cumsum(r))

if __name__ == "__main__":

    # feature_cluster()
    feature_subset_pca()