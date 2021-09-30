from sqlalchemy import create_engine
import global_vals
import pandas as pd

from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
from hierarchy_ratio_cluster import trim_outlier_std

import gc
import itertools
import multiprocessing as mp

from sklearn import metrics
from s_dbw import S_Dbw
from sklearn.mixture import GaussianMixture
from descriptive_factor.fuzzy_metrics import *

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
        df_dict = combine_tri_worldscope(use_cached=use_cached, save=True, currency=[currency]).get_results(list_of_interval_redo)
        sample_df.update(df_dict)

    return sample_df

# -------------------------------- Test Cluster -----------------------------------------------

class test_cluster:

    def __init__(self, testing_interval=91, use_cached=True):

        self.testing_interval = testing_interval
        sample_df = prep_factor_dateset(list_of_interval=[testing_interval], use_cached=use_cached)
        self.df = sample_df[testing_interval]
        self.cols = self.df.select_dtypes(float).columns.to_list()
        print(len(self.df['ticker'].unique()))

    def multithread_stepwise(self, name_sql=None, fcm_args=None, n_processes=12):

        self.score_col = 'S_Dbw'
        period_list = list(range(1, round(365*5/self.testing_interval)))
        all_groups = itertools.product(period_list, self.cols, [name_sql], fcm_args['n_clusters'])
        all_groups = [tuple(e) for e in all_groups]
        with mp.Pool(processes=n_processes) as pool:
            pool.starmap(self.stepwise_test, all_groups)

    def stepwise_test(self, *args):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        period, init_col, name_sql, n_clusters = args
        print(period, init_col)

        df = self.df.groupby(['ticker']).nth(-period).reset_index().copy(1)
        df = df.replace([-np.inf, np.inf], [np.nan, np.nan])
        df = trim_outlier_std(df)

        init_cols = [init_col]
        init_score = 10000

        next_factor = True
        while next_factor:
            all_results = []
            for col in self.cols:

                if col not in init_cols:
                    cols = init_cols + [col]
                else:
                    continue

                # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
                X = df[cols].values
                X[X == 0] = np.nan
                X = np.nan_to_num(X, -1)

                m = test_method(X, n_clusters)
                m['factors'] = ', '.join(cols)
                all_results.append(m)
                gc.collect()

            all_results = pd.DataFrame(all_results)
            all_results['period'] = period
            all_results['name_sql'] = name_sql
            best = all_results.nsmallest(1, self.score_col, keep='first')

            if best[self.score_col].values[0] < init_score:
                best_best = best.copy(1)
                init_cols = best['factors'].values[0].split(', ')
                init_score = best[self.score_col].values[0]
            else:
                print('-----> Return: ',init_col, period, init_score, init_cols)
                next_factor = False

                thread_engine_ali = create_engine(global_vals.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")
                with thread_engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
                    extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
                    best_best[['period', 'n_clusters', 'factors', self.score_col, 'name_sql']].to_sql("des_factor_gaussian", **extra)
                thread_engine_ali.dispose()

def test_method(X, n_clusters):
    ''' test conditions on different conditions of cluster methods '''

    n_clusters = int(round(X.shape[0]*n_clusters))
    model = GaussianMixture(n_components=n_clusters)
    y = model.fit_predict(X)

    # calculate matrics
    m = {}
    m.update({'n_clusters':n_clusters})
    for i in clustering_metrics1:
        m[i.__name__] = i(X, y)

    return m


if __name__ == "__main__":

    testing_interval = 91
    testing_name = 'all_init'
    fcm_args = {'n_clusters':[0.01, 0.02]}

    data = test_cluster(testing_interval=testing_interval, use_cached=True)
    data.multithread_stepwise('{}:{}'.format(testing_name, testing_interval), fcm_args, n_processes=12)

