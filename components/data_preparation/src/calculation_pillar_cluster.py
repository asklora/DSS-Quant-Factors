import numpy as np
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from collections import Counter
import multiprocessing as mp
from sqlalchemy import select, and_
from typing import List, Union
from itertools import product

from utils import (
    sys_logger,
    models,
    read_query,
    read_query_list,
    upsert_data_to_database,
    err2slack,
    timestampNow,
    dateNow,
    str_to_date,
    recreate_engine,
    backdate_by_month,
    backdate_by_day,
)
from sklearn.preprocessing import scale
from sklearn.cluster import FeatureAgglomeration
from functools import partial

logger = sys_logger(__name__, "DEBUG")

universe_table = models.Universe.__table__.schema + '.' + models.Universe.__table__.name
processed_ratio_table = models.FactorPreprocessRatio.__table__.schema + '.' + models.FactorPreprocessRatio.__table__.name
factors_formula_table = models.FactorFormulaRatio.__table__.schema + '.' + models.FactorFormulaRatio.__table__.name
pillar_cluster_table = models.FactorFormulaPillarCluster.__tablename__


class clusterFeature:
    """
    general cluster for distance calculation
    """

    def __init__(self, df):
        self.X = scale(df)
        self.feature_names = np.array(df.columns.to_list())
        self.agglo = FeatureAgglomeration(n_clusters=None, distance_threshold=0, linkage='average')
        self.agglo.fit(self.X)

    def find_subpillar(self, subpillar_trh):
        """
        subpillar groups = factors with distance <  subpillar_trh
        """
        subpillar_dist = self.agglo.distances_[subpillar_trh]
        subpillar_cluster = FeatureAgglomeration(n_clusters=None, distance_threshold=subpillar_dist, linkage='average')
        subpillar_cluster.fit(self.X)
        subpillar_label = subpillar_cluster.labels_
        logger.debug(Counter(subpillar_label))
        subpillar = {f"subpillar_{k}": list(self.feature_names[subpillar_label == k]) for k, v in
                     dict(Counter(subpillar_label)).items()
                     if (v < len(self.feature_names) * subpillar_trh) and (v > 1)}

        return subpillar

    def find_pillar(self, pillar_trh):
        """
        pillar groups = factors with distance <  pillar_trh
        """

        pillar_dist = self.agglo.distances_[-pillar_trh]
        pillar_cluster = FeatureAgglomeration(n_clusters=None, distance_threshold=pillar_dist, linkage='average')
        pillar_cluster.fit(self.X)
        pillar_label = pillar_cluster.labels_
        logger.debug(Counter(pillar_label))
        pillar = {f"pillar_{k}": list(self.feature_names[pillar_label == k]) for k, v in
                  dict(Counter(pillar_label)).items()}

        return pillar


class calcPillarCluster:
    """
    calculate pillar / subpillar using hierarchical clustering on ratios over past (n = self.lookback) years
    - pillar: factors will be evaluated together (i.e. equivalent to value/momentum/quality) in multi-output RF tree
    - subpillar: factors will not be selected together if both deemed as "good" / "bad" factors
    """

    active_factor = read_query_list(f"SELECT name FROM {factors_formula_table} WHERE is_active")

    def __init__(self, weeks_to_expire: int, sample_interval: int,
                 currency_code_list: List[str] = ('USD',),
                 start_date: dt.datetime = None, end_date: dt.datetime = None,
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

        self.period_list = self._testing_period_list(start_date, end_date, sample_interval, weeks_to_expire)
        self.weeks_to_expire = weeks_to_expire
        self.currency_code_list = currency_code_list
        self.subpillar_trh = subpillar_trh
        self.pillar_trh = pillar_trh
        self.lookback = lookback
        self.processes = processes

    def _testing_period_list(self, start_date, end_date, sample_interval, weeks_to_expire):
        """
        Data processing: align period_list Timestamp to match premium table i.e. testing_period = (last_data_date - weeks_to_expire)
        """

        # end on last sunday so that we use last week TRI to calculate average TRI
        if type(end_date) == type(None):
            end_date = pd.to_datetime(pd.date_range(end=backdate_by_day(1), freq=f"W-MON", periods=1)[0])
        if type(start_date) == type(None):
            start_date = pd.to_datetime(backdate_by_month(3))

        periods = (end_date - start_date).days // (7*sample_interval) + 1
        date_list = pd.date_range(end=end_date, freq=f"{sample_interval}W-SUN", periods=periods)
        period_list = list([x - relativedelta(weeks=weeks_to_expire) for x in date_list])

        return period_list

    def write_all(self):
        """
        Data output: write all cluster results to table [factor_formula_pillar_cluster]
        """

        with mp.Pool(processes=self.processes, initializer=recreate_engine) as pool:
            df = self._download_pivot_ratio()
            all_groups = product(self.currency_code_list, self.period_list)
            all_groups = [tuple(e) for e in all_groups]
            results = pool.starmap(partial(self._calc_cluster, ratio_df=df), all_groups)

        results = pd.concat([x for x in results if type(x) != type(None)], axis=0)
        results = results.reset_index().rename(columns={"index": "pillar"})

        static_info = {"updated": timestampNow(),
                       "weeks_to_expire": self.weeks_to_expire,
                       "subpillar_trh": self.subpillar_trh,
                       "pillar_trh": self.pillar_trh,
                       "lookback": self.lookback}
        results = results.assign(**static_info)

        upsert_data_to_database(results, pillar_cluster_table, how="update")
        logger.info(f"=== Update Cluster Pillar to [{pillar_cluster_table}] ===")

        return results

    def _download_pivot_ratio(self):
        """
        download past lookback = [5] year ratio for clustering
        """

        end_date = np.max(self.period_list) + relativedelta(weeks=self.weeks_to_expire)
        start_date = np.min(self.period_list) - relativedelta(years=self.lookback)

        conditions = [f"currency_code in {tuple(self.currency_code_list)}",
                      f"trading_day <= '{end_date}'",
                      f"trading_day > '{start_date}'",
                      f"field in {tuple(self.active_factor)}"]
        query = f"SELECT r.*, u.currency_code FROM {processed_ratio_table} r " \
                f"INNER JOIN {universe_table} u on r.ticker::text = u.ticker::text " \
                f"WHERE {' AND '.join(conditions)}"
        df = read_query(query.replace(",)", ")"))

        df = df.pivot(index=["ticker", "trading_day", "currency_code"], columns=["field"], values='value')
        df = df.fillna(0)

        return df

    @err2slack("clair", )
    def _calc_cluster(self, *args, ratio_df=None) -> Union[None, pd.DataFrame]:
        """
        calculate pillar / subpillar using hierarchical clustering
        """

        currency_code, testing_period = args

        subset_end_date = testing_period + relativedelta(weeks=self.weeks_to_expire)
        subset_start_date = subset_end_date - relativedelta(years=self.lookback)
        df = ratio_df.loc[
            (ratio_df.index.get_level_values("currency_code") == currency_code) &
            (ratio_df.index.get_level_values("trading_day") <= subset_end_date.date()) &
            (ratio_df.index.get_level_values("trading_day") > subset_start_date.date())]

        if len(df) == 0:
            return None

        cluster_cls = clusterFeature(df)
        subpillar = cluster_cls.find_pillar(self.subpillar_trh)
        pillar = cluster_cls.find_pillar(self.pillar_trh)

        df_pillar = pd.DataFrame({"factor_list": {**pillar, **subpillar}})
        df_pillar["testing_period"] = testing_period
        df_pillar["currency_code"] = currency_code

        return df_pillar
