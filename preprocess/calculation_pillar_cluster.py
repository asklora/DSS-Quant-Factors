import numpy as np
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from collections import Counter
from sqlalchemy.dialects.postgresql import DATE, TEXT, INTEGER, JSON
import multiprocessing as mp
from global_vars import (
    logger,
    pillar_cluster_table,
    processed_ratio_table,
    factors_formula_table
)
from general.sql_process import read_query, upsert_data_to_database

from sklearn.preprocessing import scale
from sklearn.cluster import FeatureAgglomeration
from functools import partial

dtype_pillar = dict(
    pillar=TEXT,
    factor_name=JSON,
    testing_period=DATE,
    weeks_to_expire=INTEGER,
    group=TEXT,
    subpillar_trh=INTEGER,
    pillar_trh=INTEGER,
    lookback=INTEGER
)


class calc_pillar_cluster:

    def __init__(self, period_list, weeks_to_expire, currency_code='USD', subpillar_trh=5, pillar_trh=2, lookback=5,
                 save_to_db=True):
        """
        Parameters
        ----------
        testing_period (Datetime): testing_period
        weeks_to_expire (Int): weeks_to_expire
        group (Str): currency_code
        subpillar_trh (Int):
            nth smallest distances below which factors belong to same sub-pillar (i.e. factors won't be selected together)
        pillar_trh (Int):
            nth largest distances below which factors belong to same pillar
        lookback (Int):
            number of years for sample lookback periods prior to testing_period for clustering
        """

        # download past [5] year ratio for clustering
        end_date = np.max(period_list) + relativedelta(weeks=weeks_to_expire)
        start_date = np.min(period_list) - relativedelta(years=lookback)

        conditions = [f"ticker in (SELECT ticker FROM universe WHERE currency_code='{currency_code}')",
                      f"trading_day <= '{end_date}'",
                      f"trading_day > '{start_date}'"]
        query = f"SELECT * FROM {processed_ratio_table} WHERE {' AND '.join(conditions)}"
        df = read_query(query.replace(",)", ")"))
        # df.to_pickle('cache_factor_ratio1.pkl')
        # df = pd.read_pickle('cache_factor_ratio1.pkl')

        # get active factor list
        df_formula = read_query(f"SELECT name FROM {factors_formula_table} WHERE is_active")
        df_active_factor = df_formula['name'].to_list()

        # pivot ratio table & filter by active factor list
        df['trading_day'] = pd.to_datetime(df['trading_day'])
        df = df.set_index(['trading_day', 'ticker', 'field'])['value'].unstack()
        df = df.filter(df_active_factor)
        self.df = df.fillna(0)
        self.feature_names = np.array(df.columns.to_list())

        all_groups = [tuple([p]) for p in period_list]
        with mp.Pool(processes=1) as pool:
            results = pool.starmap(partial(self._calc_cluster, weeks_to_expire=weeks_to_expire, subpillar_trh=5,
                                           pillar_trh=2, lookback=5), all_groups)
        results = pd.concat(results, axis=0)

        if save_to_db:
            config = dict(weeks_to_expire=weeks_to_expire, currency_code=currency_code, subpillar_trh=subpillar_trh,
                          pillar_trh=pillar_trh, lookback=lookback)
            results = results.reset_index().rename(columns={"index": "pillar"})
            results["updated"] = dt.datetime.now()
            for k, v in config.items():
                results[k] = v
            primary_key = list(config.keys()) + ['pillar', "testing_period"]
            upsert_data_to_database(results, pillar_cluster_table, primary_key=primary_key,
                                    how="update", dtype=dtype_pillar)
            logger.info(f"=== Update Cluster Pillar to [{pillar_cluster_table}] ===")

    def _calc_cluster(self, *args, weeks_to_expire=4, subpillar_trh=5, pillar_trh=2, lookback=5):
        """ calculte pillar / subpillar """

        testing_period, = args
        end_date = testing_period + relativedelta(weeks=weeks_to_expire)
        start_date = end_date - relativedelta(years=lookback)
        df = self.df.loc[(self.df.index.get_level_values("trading_day").date <= end_date) &
                         (self.df.index.get_level_values("trading_day").date > start_date)]

        # general cluster for distance calculation
        try:
            X = scale(df)
            agglo = FeatureAgglomeration(n_clusters=None, distance_threshold=0, linkage='average')
            agglo.fit(X)

            # find [subpillar]
            subpillar_dist = agglo.distances_[subpillar_trh]
            subpillar_cluster = FeatureAgglomeration(n_clusters=None, distance_threshold=subpillar_dist, linkage='average')
            subpillar_cluster.fit(X)
            subpillar_label = subpillar_cluster.labels_
            logger.info(Counter(subpillar_label))
            subpillar = {f"subpillar_{k}": list(self.feature_names[subpillar_label == k]) for k, v in
                         dict(Counter(subpillar_label)).items()
                         if (v < len(self.feature_names) * subpillar_trh) and (v > 1)}

            # find [pillar]
            pillar_dist = agglo.distances_[-pillar_trh]
            pillar_cluster = FeatureAgglomeration(n_clusters=None, distance_threshold=pillar_dist, linkage='average')
            pillar_cluster.fit(X)
            pillar_label = pillar_cluster.labels_
            logger.info(Counter(pillar_label))
            pillar = {f"pillar_{k}": list(self.feature_names[pillar_label == k]) for k, v in dict(Counter(pillar_label)).items()}

            df_pillar = pd.DataFrame({"factor_list": {**pillar, **subpillar}})
            df_pillar["testing_period"] = testing_period
            return df_pillar
        except Exception as e:
            logger.info(e)
            return pd.DataFrame()

if __name__ == '__main__':
    subpillar_dict, pillar_dict = calc_pillar_cluster(dt.date(2016, 1, 1), 4)
