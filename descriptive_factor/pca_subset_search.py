from sklearn.decomposition import PCA
import multiprocessing as mp
import itertools
import pandas as pd
from hierarchy_ratio_cluster import trim_outlier_std
import datetime as dt
from dateutil.relativedelta import relativedelta
import global_vals
from sqlalchemy import create_engine

# def feature_subset_cluster1(testing_interval=7):
#     ''' search 1: PCA '''

class test:

    def __init__(self):
        self.multithread_search()

    def load_data(self, testing_interval=91, year=2):
        df = pd.read_csv(f'dcache_sample_{testing_interval}.csv')
        df = df.loc[df['ticker'].str[0] != '.']
        df = df.loc[df['ticker'].str[-3:]=='.HK']
        df['trading_day'] = pd.to_datetime(df['trading_day'])
        df.iloc[:, 2:] = trim_outlier_std(df.iloc[:, 2:].fillna(0))
        df = df.loc[df['trading_day'] > (dt.datetime.today() - relativedelta(years=year))].copy()
        cols = df.columns.to_list()[2:]
        return df, cols

    def multithread_search(self, n_processes=12):

        for testing_interval in [7, 30, 91]:
            self.df, cols = self.load_data(testing_interval, year=2)
            subset = itertools.combinations(cols, 5)
            with mp.Pool(processes=n_processes) as pool:
                pool.starmap(self.combination_test, subset)

    def combination_test(self, *args):
        cols = args
        X = self.df[list(cols)]
        pca = PCA(n_components=None).fit(X)  # calculation Cov matrix is embeded in PCA
        r = pca.explained_variance_ratio_
        print(r[0], cols)

        thread_engine_ali = create_engine(global_vals.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with thread_engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
            pd.DataFrame({'cols':','.join(cols), 'score':r[0]}, index=[0]).to_sql("des_factor_pca", **extra)
        thread_engine_ali.dispose()


if __name__=="__main__":
    test()
