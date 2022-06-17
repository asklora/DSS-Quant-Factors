import numpy as np
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from collections import Counter
from sqlalchemy.dialects.postgresql import DATE, TEXT, INTEGER, JSON
import multiprocessing as mp

from typing import List

from utils import (
    sys_logger,
    models,
    read_query,
    upsert_data_to_database,
    err2slack,
    timestampNow
)
from sklearn.preprocessing import scale
from sklearn.cluster import FeatureAgglomeration
from functools import partial

logger = sys_logger(__name__, "DEBUG")

processed_ratio_table = models.PreprocessRatio.__table__.schema + '.' + models.PreprocessRatio.__table__.name
factors_formula_table = models.FormulaRatio.__table__.schema + '.' + models.FormulaRatio.__table__.name
pillar_cluster_table = models.FormulaPillarCluster.__table__.name


class clusterFeature:

    def __init__(self, df):
        """
        general cluster for distance calculation
        """
        self.X = scale(df)
        self.feature_names = np.array(df.columns.to_list())
        self.agglo = FeatureAgglomeration(n_clusters=None, distance_threshold=0, linkage='average')
        self.agglo.fit(self.X)

    def find_subpillar(self, subpillar_trh):
        subpillar_dist = self.agglo.distances_[subpillar_trh]
        subpillar_cluster = FeatureAgglomeration(n_clusters=None, distance_threshold=subpillar_dist, linkage='average')
        subpillar_cluster.fit(self.X)
        subpillar_label = subpillar_cluster.labels_
        logger.info(Counter(subpillar_label))
        subpillar = {f"subpillar_{k}": list(self.feature_names[subpillar_label == k]) for k, v in
                     dict(Counter(subpillar_label)).items()
                     if (v < len(self.feature_names) * subpillar_trh) and (v > 1)}

        return subpillar

    def find_pillar(self, pillar_trh):
        pillar_dist = self.agglo.distances_[-pillar_trh]
        pillar_cluster = FeatureAgglomeration(n_clusters=None, distance_threshold=pillar_dist, linkage='average')
        pillar_cluster.fit(self.X)
        pillar_label = pillar_cluster.labels_
        logger.info(Counter(pillar_label))
        pillar = {f"pillar_{k}": list(self.feature_names[pillar_label == k]) for k, v in
                  dict(Counter(pillar_label)).items()}

        return pillar


class calcPillarCluster:

    active_factor = read_query(f"SELECT name FROM {factors_formula_table} WHERE is_active")['name'].to_list()

    def __init__(self, period_list: List[dt.datetime], weeks_to_expire: int, currency_code: str = 'USD',
                 subpillar_trh: int = 5, pillar_trh: int = 2, lookback: int = 5, processes: int = 1):
        """
        Parameters
        ----------
        subpillar_trh (Int):
            nth smallest distances below which factors belong to same sub-pillar (i.e. factors won't be selected together)
        pillar_trh (Int):
            nth largest distances below which factors belong to same pillar
        lookback (Int):
            number of years for sample lookback periods prior to testing_period for clustering
        """

        self.period_list = period_list
        self.weeks_to_expire = weeks_to_expire
        self.currency_code = currency_code
        self.subpillar_trh = subpillar_trh
        self.pillar_trh = pillar_trh
        self.lookback = lookback
        self.processes = processes

    def _download_pivot_ratio(self):
        """ download past lookback = [5] year ratio for clustering """

        end_date = np.max(self.period_list) + relativedelta(weeks=self.weeks_to_expire)
        start_date = np.min(self.period_list) - relativedelta(years=self.lookback)

        conditions = [f"ticker in (SELECT ticker FROM universe WHERE currency_code='{self.currency_code}')",
                      f"trading_day <= '{end_date}'",
                      f"trading_day > '{start_date}'",
                      f"field in {tuple(self.active_factor)}"]
        query = f"SELECT * FROM {processed_ratio_table} WHERE {' AND '.join(conditions)}"
        df = read_query(query.replace(",)", ")"))

        df = df.pivot(index=["ticker", "trading_day"], columns=["field"], values='value')
        df = df.fillna(0)

        return df

    def write_all(self):
        """ write all cluster results to table  """
        with mp.Pool(processes=self.processes) as pool:
            df = self._download_pivot_ratio()
            all_train_currencys = [tuple([p]) for p in self.period_list]
            results = pool.starmap(partial(self._calc_cluster, ratio_df=df), all_train_currencys)
        results = pd.concat([x for x in results if type(x) != type(None)], axis=0)
        results = results.reset_index().rename(columns={"index": "pillar"})

        static_info = {"updated": timestampNow(),
                       "weeks_to_expire": self.weeks_to_expire,
                       "currency_code": self.currency_code,
                       "subpillar_trh": self.subpillar_trh,
                       "pillar_trh": self.pillar_trh,
                       "lookback": self.lookback}
        results = results.assign(**static_info)

        upsert_data_to_database(results, pillar_cluster_table, how="update")
        logger.info(f"=== Update Cluster Pillar to [{pillar_cluster_table}] ===")

        return results

    # @err2slack("clair")
    def _calc_cluster(self, *args, ratio_df=None) -> pd.DataFrame:
        """ calculte pillar / subpillar """

        testing_period, = args
        subset_end_date = testing_period + relativedelta(weeks=self.weeks_to_expire)
        subset_start_date = subset_end_date - relativedelta(years=self.lookback)
        df = ratio_df.loc[(ratio_df.index.get_level_values("trading_day") <= subset_end_date.date()) &
                          (ratio_df.index.get_level_values("trading_day") > subset_start_date.date())]

        cluster_cls = clusterFeature(df)
        subpillar = cluster_cls.find_pillar(self.subpillar_trh)
        pillar = cluster_cls.find_pillar(self.pillar_trh)

        df_pillar = pd.DataFrame({"factor_list": {**pillar, **subpillar}})
        df_pillar["testing_period"] = testing_period

        return df_pillar
