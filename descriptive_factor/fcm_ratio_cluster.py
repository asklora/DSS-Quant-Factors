from sqlalchemy import create_engine
import global_vals
import pandas as pd
import datetime as dt

from descriptive_factor.descriptive_ratios_calculations import combine_tri_worldscope
from hierarchy_ratio_cluster import trim_outlier_std

import gc
import itertools
import multiprocessing as mp

from fcmeans import FCM
from descriptive_factor.fuzzy_metrics import *

from hierarchy_ratio_cluster import good_cols, fill_all_day_interpolate, relativedelta, test_missing, prep_factor_dateset, good_mom_cols

clustering_metrics_fuzzy = [
    # partition_coefficient,
    xie_beni_index,
    # fukuyama_sugeno_index,
    # cluster_center_var,
    # fuzzy_hypervolume,
    # beringer_hullermeier_index,
    # bouguessa_wang_sun_index,
]

# -------------------------------- Test Cluster -----------------------------------------------

class test_cluster:

    def __init__(self, testing_interval=91, use_cached=True):

        self.testing_interval = testing_interval
        # sample_df = prep_factor_dateset(list_of_interval=[testing_interval], use_cached=True)
        # self.df = sample_df[testing_interval]


        sample_df = prep_factor_dateset(list_of_interval=[7, 91], use_cached=True)
        df = sample_df[91].merge(fill_all_day_interpolate(sample_df[7])[['ticker','trading_day']+good_mom_cols],
                                 on=['ticker','trading_day'], how='left', suffixes=('_91','_7'))
        # df = df.merge(fill_all_day_interpolate(sample_df[7]), on=['ticker','trading_day'], how='left', suffixes=('','_7'))
        self.df = df.drop(columns=['avg_volume_1w3m_91'])

        # test_missing(self.df)
        # calc_corr_csv(self.df)

        self.cols = self.df.select_dtypes(float).columns.to_list()
        print(len(self.df['ticker'].unique()))

    def multithread_combination(self, name_sql=None, fcm_args=None, n_processes=12):

        self.score_col = 'xie_beni_index'
        period_list = list(range(1, round(365*5/self.testing_interval)))
        all_groups = itertools.product(period_list, [5], [name_sql], fcm_args['n_clusters'], fcm_args['m'])
        all_groups = [tuple(e) for e in all_groups]
        with mp.Pool(processes=n_processes) as pool:
            pool.starmap(self.combination_test, all_groups)

    def combination_test(self, *args):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        period, n_cols, name_sql, n_clusters, m_arg = args
        # print(args)

        df = self.df.groupby(['ticker']).nth(-period).reset_index().copy(1)
        df = df.replace([-np.inf, np.inf], [np.nan, np.nan])
        df = trim_outlier_std(df)

        all_results = []
        for cols in itertools.combinations(self.cols, n_cols):

            # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
            X = df[list(cols)].values
            X[X == 0] = np.nan
            X = np.nan_to_num(X, 0)

            m = test_method(X, n_clusters, m_arg)
            m['factors'] = ', '.join(cols)
            print(cols, m)

            all_results.append(m)
            gc.collect()

        all_results = pd.DataFrame(all_results)
        all_results['period'] = period
        all_results['name_sql'] = name_sql
        best = all_results.nlargest(1, self.score_col, keep='first')

        print(args, '-----> Return: ', best['factors'].values[0])

        thread_engine_ali = create_engine(global_vals.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with thread_engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
            all_results[['period', 'n_clusters', 'm', 'factors', self.score_col, 'name_sql']].to_sql("des_factor_fcm", **extra)
        thread_engine_ali.dispose()

    def multithread_stepwise(self, name_sql=None, fcm_args=None, n_processes=12):

        self.score_col = 'xie_beni_index'
        period_list = list(range(1, round(365*5/self.testing_interval)))

        all_groups = itertools.product(period_list, self.cols, [name_sql], fcm_args['n_clusters'], fcm_args['m'])
        all_groups = [tuple(e) for e in all_groups]
        with mp.Pool(processes=n_processes) as pool:
            pool.starmap(self.stepwise_test, all_groups)

    def stepwise_test(self, *args):
        ''' test on different factors combinations -> based on initial factors groups -> add least correlated one first '''

        period, init_col, name_sql, n_clusters, m_arg = args
        print(args)

        df = self.df.groupby(['ticker']).nth(-period).reset_index().copy(1)
        df = df.replace([-np.inf, np.inf], [np.nan, np.nan])
        df = trim_outlier_std(df)

        init_cols = [init_col]
        init_score = 10000

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

                m = test_method(X, n_clusters, m_arg)
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
                    best_best[['period', 'n_clusters', 'm', 'factors', self.score_col, 'name_sql']].to_sql("des_factor_fcm", **extra)
                thread_engine_ali.dispose()

def test_method(X, n_clusters, m_arg):
    ''' test conditions on different conditions of cluster methods '''

    n_clusters = int(round(X.shape[0]*n_clusters))
    model = FCM(n_clusters=n_clusters, m=m_arg)
    model.fit(X)

    # y = model.predict(X)
    u = model.u
    v = model.centers

    # calculate matrics
    m = {}
    m.update({'n_clusters':n_clusters, 'm': m_arg})
    # m[element]['count'] = Counter(y)
    for i in clustering_metrics_fuzzy:
        m[i.__name__] = i(X, u, v.T, m_arg)

    return m


if __name__ == "__main__":

    testing_interval = 91
    testing_name = 'all_comb_new_multiperiod'
    fcm_args = {'n_clusters':[0.01, 0.02], 'm':[2]}

    data = test_cluster(testing_interval=testing_interval, use_cached=True)
    # data.multithread_stepwise('{}:{}'.format(testing_name, testing_interval), fcm_args, n_processes=12)
    data.multithread_combination('{}:{}'.format(testing_name, testing_interval), fcm_args, n_processes=12)

