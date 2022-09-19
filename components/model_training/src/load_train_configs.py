import datetime as dt
import os

import pandas as pd
from typing import List
from itertools import product
from sqlalchemy import select, and_, func, text

from utils import (
    to_slack,
    report_to_slack,
    read_query,
    read_table,
    upsert_data_to_database,
    models,
    sys_logger,
    err2slack,
    backdate_by_week,
    dateNow
)
from .load_data import calcTestingPeriod

logger = sys_logger(__name__, "DEBUG")

pillar_cluster_table = models.FactorFormulaPillarCluster.__table__.schema + '.' + models.FactorFormulaPillarCluster.__table__.name
pillar_defined_table = models.FactorFormulaPillarDefine.__table__.schema + '.' + models.FactorFormulaPillarDefine.__table__.name


class loadTrainConfig(calcTestingPeriod):

    _auto_select_options = {
        "factor_pca": [0.4, 0],
        "factor_reverse": [0, 1, 2],  # No reverse, reverse by average, reverse by lasso
        "y_qcut": [0, 10],
        "valid_pct": [.2],
        "valid_method": [2010, 2012, 2014],
        "down_mkt_pct": [0.5, 0.7],
        "tree_type": ['rf'],
    }

    def __init__(self,
                 weeks_to_expire: int,
                 sample_interval: int = 4,
                 backtest_period: int = 1,
                 restart: str = None,
                 currency_code: str = None):
        super().__init__(weeks_to_expire, sample_interval, backtest_period, currency_code, restart)

        self.weeks_to_expire = weeks_to_expire
        self.restart = restart
        self.sample_interval = sample_interval
        self.backtest_period = backtest_period
        self.currency_code = currency_code

    def get_all_groups(self) -> List[tuple]:
        """
        Returns
        -------
        all_groups: List[(dict)}
            each dict defined the training configuration of:
            - testing_period
            - pillar
            - train_currency
            - pred_currency
            + other configuration defined in [FactorFormulaTrainConfig] ...
            + other grid search configuration (i.e. _auto_select_options) ...
        """
        all_configs_df = self.merge_groups_df()
        all_configs_df = self._replace_pillar_name_with_factor_list(all_configs_df)

        if self.restart:
            finish_configs_df = self._restart_finished_configs()
            diff_config_col = [x for x in finish_configs_df.columns.to_list()
                               if x != "count"]
            all_configs_df = all_configs_df.merge(finish_configs_df,
                                                  how="left",
                                                  on=diff_config_col)
            all_configs_df = all_configs_df.loc[all_configs_df['count'].isnull()
                                                ].drop(columns=["count"])

        report_to_slack(f"=== rest iterations (n={len(all_configs_df)}) ===")

        all_groups = [tuple([e]) for e in all_configs_df.to_dict("records")]
        return all_groups

    @property
    def _defined_configs(self):
        """
        for production: use defined configs
        """

        conditions = [
            models.FactorFormulaTrainConfig.weeks_to_expire == self.weeks_to_expire,
        ]
        if self.currency_code:
            conditions.append(models.FactorFormulaTrainConfig.pred_currency.like(f"%{self.currency_code}%"))
        if not os.getenv("DEBUG").lower == "true":
            conditions.append(models.FactorFormulaTrainConfig.id == 0)

        query = select(models.FactorFormulaTrainConfig).where(and_(*conditions))
        df = read_query(query)
        defined_configs = df.drop(columns=["finished", "id"]).to_dict("records")

        if len(defined_configs) <= 0:
            raise ValueError(f"No train config selected from [{models.FactorFormulaTrainConfig.__tablename__}] by {__name__}")

        return defined_configs

    def merge_groups_df(self):
        _auto_select_configs = [dict(zip(self._auto_select_options.keys(), e)) for e in
                                product(*self._auto_select_options.values())]

        all_configs = product([{**a, **l, **{"testing_period": p}}
                                for a in self._defined_configs
                                for l in _auto_select_configs
                                for p in self._testing_period_list[-self.backtest_period:]])

        all_configs_df = pd.DataFrame([tuple(e)[0] for e in all_configs])
        return all_configs_df

    def _replace_pillar_name_with_factor_list(self, configs_df):
        """
        Map pillar name to factor list by merging pillar tables & configs
        """

        cluster_pillar_df = self.__cluster_pillar_map(configs_df)
        defined_pillar_df = self.__defined_pillar_map(configs_df)
        configs_df = cluster_pillar_df.append(defined_pillar_df)
        return configs_df

    def __cluster_pillar_map(self, config_df) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame:
            Columns (add):
                pillar:         str, name of pillars (i.e. pillar_1/2/3...)
                factor_list:    list, list of factors belongs to this pillar
        """

        tbl = models.FactorFormulaPillarCluster
        query = select(tbl.currency_code.label("train_currency"), tbl.testing_period, tbl.pillar, tbl.factor_list)\
            .where(tbl.weeks_to_expire == self.weeks_to_expire)
        map_df = read_query(query)
        map_df["testing_period"] = pd.to_datetime(map_df["testing_period"])
        map_df = map_df.loc[map_df["pillar"].str.startswith("pillar")]

        config_df = config_df.loc[config_df["pillar"] == "cluster"].drop(columns=["pillar"])
        config_df = config_df.merge(map_df, on=["train_currency", "testing_period"], how="left")

        return config_df

    def __defined_pillar_map(self, df) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame:
            Columns (add):
                pillar:         str, value / quality / momentum
                factor_list:    list, list of factors belongs to this pillar
        """
        map_df = read_query(f"SELECT pillar, factor_list FROM {pillar_defined_table}")

        config_df = df.loc[df["pillar"] != "cluster"]
        config_df = config_df.merge(map_df, on=["pillar"], how="left")
        return config_df

    def _restart_finished_configs(self):
        diff_config_col = models.FactorResultScore.base_columns + \
                          models.FactorResultScore.config_define_columns + \
                          models.FactorResultScore.config_opt_columns

        query = select(*diff_config_col, func.count(models.FactorResultScore.uid).label("count"))\
            .where(models.FactorResultScore.uid == self.restart)\
            .group_by(*diff_config_col)
        df = read_query(query)

        return df