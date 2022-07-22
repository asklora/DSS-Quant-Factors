import pandas as pd
import numpy as np
import datetime as dt
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import Counter
from sqlalchemy import select, and_, Integer, func
from contextlib import closing
from typing import List

from utils import (
    read_query,
    models,
    sys_logger,
    err2slack,
    recreate_engine,
    upsert_data_to_database,
    timestampNow
)
from .load_eval_configs import load_eval_config

logger = sys_logger(__name__, "DEBUG")

pillar_cluster_table = models.FactorFormulaPillarCluster.__table__.schema + '.' + models.FactorFormulaPillarCluster.__table__.name


class cleanSubpillar:
    """
    Get subpillar map: factors will not be selected together if both deemed as "good" / "bad" factors
    """

    def __init__(self, start_date: dt.datetime, weeks_to_expire: int):
        self.start_date = start_date
        self.weeks_to_expire = weeks_to_expire

    def get_subpillar(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Columns:
                testing_period: dt.datetime
                currency_code:  str
                subpillar:      str
                factor_name:    str

         * Non-duplicated index with (testing_period, currency_code, factor_name)
        """

        df = self._download_subpillars()
        df = self._reformat_subpillar(df)
        return df

    def _download_subpillars(self):
        """
        download subpillar from pillar cluster table
        """

        conditions = [
            models.FactorFormulaPillarCluster.pillar.like("subpillar_%%"),
            models.FactorFormulaPillarCluster.testing_period >= self.start_date,
            models.FactorFormulaPillarCluster.weeks_to_expire == self.weeks_to_expire,
        ]
        query = select(models.FactorFormulaPillarCluster).where(and_(*conditions))
        subpillar = read_query(query)

        subpillar['testing_period'] = pd.to_datetime(subpillar['testing_period'])

        return subpillar

    def _reformat_subpillar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        separate each factor in factor_list: List[str] Columns into separate rows
        """

        arr_len = df['factor_list'].str.len().values
        arr_info = np.repeat(df[["testing_period", "currency_code", "pillar"]].values, arr_len, axis=0)
        arr_factor = np.array([e for x in df["factor_list"].to_list() for e in x])[:, np.newaxis]

        idx = pd.MultiIndex.from_arrays(arr_info.T, names=["testing_period", "currency_code", "subpillar"])
        df_new = pd.DataFrame(arr_factor, index=idx, columns=["factor_name"]).reset_index()

        df_new = df_new.sort_values(by=["testing_period", "currency_code", "subpillar"])
        df_new = df_new.drop_duplicates(subset=["testing_period", "currency_code", "factor_name"])

        return df_new


class cleanPrediction:
    """
    Data Processing: Combine prediction and score table
    """

    pred_pillar_list = None  # evaluate all pillars
    pred_start_testing_period = '2015-09-01'
    pred_start_uid = 20000000  # first training datetime

    if_combine_pillar = False  # combine cluster pillars

    save_cache = False  # for debugging

    def __init__(self, name_sql: str):
        self.name_sql = name_sql

    def get_prediction(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Columns:
                pred:       float64, predicted premium
                actual:     float64, actual premium
                + models.FactorResultScore.base_columns
                + ..config_define_columns
                + ..config_opt_columns
        """

        if self.save_cache:
            try:
                pred = self.__load_cache_prediction()
            except Exception as e:
                pred = self.__download_prediction_from_db()
        else:
            pred = self.__download_prediction_from_db()

        return self._reformat_pred(pred)

    def _reformat_pred(self, pred):
        pred['uid_hpot'] = pred['uid'].str[:20]  # first 20-digit uid = Hyperopt start time (i.e. same for 10 iterations)
        pred = self.__reverse_neg_factor(pred)

        if self.if_combine_pillar:
            pred['pillar'] = "combine"

        return pred

    def __reverse_neg_factor(self, pred):
        """
        get negative factors for all (testing_period, group)
        """

        pred['is_negative'] = pred.apply(lambda x: x['factor_name'] in x["neg_factor"], axis=1)
        pred.loc[pred['is_negative'], ["actual", "pred"]] *= -1

        return pred.drop(columns=["neg_factor"])

    # @err2slack("factor")
    def __download_prediction_from_db(self):
        """
        merge factor_stock & factor_model_stock
        """

        logger.info(f'=== Download prediction history on name_sql=[{self.name_sql}] ===')

        conditions = [
            models.FactorResultScore.name_sql == self.name_sql,
            models.FactorResultScore.testing_period >= self.pred_start_testing_period,
            func.left(models.FactorResultScore.uid, 8).cast(Integer) >= self.pred_start_uid,
        ]
        if self.pred_pillar_list:
            conditions.append(models.FactorResultScore.name_sql.in_(tuple(self.pred_pillar_list)))

        query = select(*models.FactorResultPrediction.__table__.columns,
                       *models.FactorResultScore.base_columns,
                       *models.FactorResultScore.config_define_columns,
                       *models.FactorResultScore.config_opt_columns,
                       models.FactorResultScore.neg_factor) \
            .join_from(models.FactorResultPrediction, models.FactorResultScore) \
            .where(and_(*conditions))
        pred = read_query(query).fillna(0)

        if len(pred) == 0:
            raise Exception(f"No prediction for ({self.name_sql}). Please check records exists in FactorResultScore.")

        if self.save_cache:
            pred.to_pickle(f'pred_{self.name_sql}.pkl')

        pred["testing_period"] = pd.to_datetime(pred["testing_period"])

        return pred

    def __load_cache_prediction(self):
        pred = pd.read_pickle(f"pred_{self.name_sql}.pkl")
        logger.debug(f'=== Load local prediction history on name_sql=[{self.name_sql}] ===')
        pred["testing_period"] = pd.to_datetime(pred["testing_period"])
        return pred


class groupSummaryStats:
    """
    Calculate good/bad factor and evaluation metrics into results (pd.Series) for each group;

    One group defined as all factors for
    - 1 pillar
    - 1 testing_period
    - 1 currency_code
    - 1 weeks_to_expire

    call get_stats() in groupby.apply function in evalFactor to calculate stats for all groups
    """

    def __init__(self, eval_q: float = 0.33, eval_removed_subpillar: bool = True, **kwargs):
        """
        Parameters
        ----------
        eval_q :
            default = 0.33, i.e. top 1/3 with highest predicted premium as good factors; bottom 1/3 as bad factors;
        eval_removed_subpillar:
            If True, factors within same subpillar will only be selected once.
        """
        self.eval_q = eval_q
        self.eval_removed_subpillar = eval_removed_subpillar

    def get_stats(self, g: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        g : pd.DataFrame
            Columns:
                pred:           float64, predicted premium for the group;
                actual:         float64, predicted premium for the group;
                subpillar:      str, name of subpillar;
                factor_name:    str, name of factors;

                + other columns(...) from the groupby function to label the group

        Returns
        ----------
        results: pd.DataFrame
            Columns:
                eval_q:                     float64, from inputs parameters
                eval_removed_subpillar:     bool, from inputs parameters
                max/min_factor:             list, list of factor name select as good/bad factors
                max/min_factor_pred:        dict, dict of {factor_name: prediction premium} for good/bad factors
                max/min_factor_actual:      dict, dict of {factor_name: actual premium} for good/bad factors
                max/min_ret:                float64, good/bad factors average actual premiums
                mae / mse / r2:             float64, metrics calculated with all factors pred and actual\
            Index = [0]
        """

        if len(g) > 1:
            max_g, min_g = self.__stats_if_multioutput(g)
        else:
            max_g, min_g = self.__stats_if_singleoutput(g)

        accu_dict = self.__group_accuracy(g)  # can only calculate accuracy when > 1 factors
        max_select_dict = self.__group_df_to_dict(prefix="max", g=max_g)
        min_select_dict = self.__group_df_to_dict(prefix="min", g=min_g)

        results = pd.Series({**accu_dict, **max_select_dict, **min_select_dict,
                                "eval_q"                : self.eval_q,
                                "eval_removed_subpillar": self.eval_removed_subpillar}).to_frame().T
        results.index.name = "group_idx"
        return results

    def __stats_if_multioutput(self, g: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Multi-factor pillar: good / bad factors = top / bottom eval_q
        """

        max_g = g.loc[g['pred'] > g['pred'].quantile(1 - self.eval_q)].sort_values(by=["pred"], ascending=False)
        min_g = g.loc[g['pred'] < g['pred'].quantile(self.eval_q)].sort_values(by=["pred"])
        if self.eval_removed_subpillar:
            max_g = max_g.drop_duplicates(subset=['subpillar'], keep="first")
            min_g = min_g.drop_duplicates(subset=['subpillar'], keep="first")

        return max_g, min_g

    def __stats_if_singleoutput(self, g: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        for cluster pillar in case only 1 factor in cluster
        """

        max_g = g.loc[g['pred'] > .005]  # b/c 75% quantile pred is usually 0.005+
        min_g = g.loc[g['pred'] < -.005]

        return max_g, min_g

    def __group_df_to_dict(self, prefix: str, g: pd.DataFrame):
        """
        Returns
        -------
            Dict for group average return, selected factor name, prediction / actual details
        """

        if len(g) > 0:
            result_dict = {
                f"{prefix}_factor"       : g['factor_name'].tolist(),
                f"{prefix}_factor_pred"  : g.groupby("factor_name")['pred'].mean().to_dict(),
            }
            if len(g.dropna(how="any")) > 0:
                result_dict[f"{prefix}_factor_actual"] = g.groupby(['factor_name'])['actual'].mean().to_dict()
                result_dict[f"{prefix}_ret"] = g['actual'].mean()
            return result_dict
        else:
            return {}

    def __group_accuracy(self, g):
        """
        Returns
        -------
            Dict for MSE / MAE / R2 on group
        """

        if len(g.dropna(how="any")) > 1:
            return {
                "mae": mean_absolute_error(g['pred'], g['actual']),
                "mse": mean_squared_error(g['pred'], g['actual']),
                "r2" : r2_score(g['pred'], g['actual'])
            }
        else:
            return {}


class evalFactor:
    """
    process raw prediction in result_pred_table -> models.FactorBacktestEval Table for AI Score calculation
    """

    save_cache = False
    config_opt_columns = [x.name for x in models.FactorBacktestEval.config_opt_columns]
    config_define_columns = [x.name for x in models.FactorBacktestEval.train_config_define_columns] + \
                            [x.name for x in models.FactorBacktestEval.base_columns]

    def __init__(self, name_sql: str, processes: int):
        self.name_sql = name_sql
        self.weeks_to_expire = int(name_sql.split('_')[0][1:])
        self.processes = processes

    def write_db(self):
        """
        Returns
        -------
        eval_df = pd.DataFrame for [FactorBacktestEval] Table

        """

        with closing(mp.Pool(processes=self.processes, initializer=recreate_engine)) as pool:
            pred_df = cleanPrediction(name_sql=self.name_sql).get_prediction()
            subpillar_df = cleanSubpillar(start_date=pred_df["testing_period"].min(),
                                          weeks_to_expire=self.weeks_to_expire).get_subpillar()

            all_groups = load_eval_config(self.weeks_to_expire)

            eval_results = pool.starmap(partial(self._rank, pred_df=pred_df, subpillar_df=subpillar_df), all_groups)

        eval_df = pd.concat([e for e in eval_results if type(e) != type(None)], axis=0)  # df for each config evaluation results
        eval_df = eval_df.assign(name_sql=self.name_sql, updated=timestampNow())
        eval_df = self.__map_actual_factor_premium(eval_df=eval_df, pred_df=pred_df)

        if self.save_cache:
            eval_df.to_pickle(f"eval_{self.name_sql}.pkl")

        upsert_data_to_database(eval_df, models.FactorBacktestEval.__tablename__, how="ignore")

        return eval_df

    def _rank(self, *args, pred_df: pd.DataFrame = None, subpillar_df: pd.DataFrame = None):
        """
        rank based on config defined by each row in pred_config table
        """

        kwargs, = args

        sample_df = self.__filter_sample(df=pred_df, **kwargs)

        if len(sample_df) > 0:

            if kwargs["eval_removed_subpillar"]:
                sample_df = self.__map_remove_subpillar(df=sample_df, subpillar_df=subpillar_df, **kwargs)

            eval_df_new = self._eval_factor_all(df=sample_df, **kwargs)

            return eval_df_new

    def __filter_sample(self, df: pd.DataFrame, currency_code: str, pillar: str, **kwargs) -> pd.DataFrame:
        """
        filter pred table for sample for certain currency / pillar
        """

        if pillar != "cluster":
            sample_df = df.loc[(df["currency_code"] == currency_code) & (df["pillar"] == pillar)].copy(1)
        else:
            sample_df = df.loc[(df["currency_code"] == currency_code) & (df["pillar"].str.startswith("pillar"))].copy(1)

        return sample_df

    def __map_remove_subpillar(self, df: pd.DataFrame, subpillar_df: pd.DataFrame, pillar: str, **kwargs):
        """
        remove subpillar - same subpillar factors keep higher pred one
        """

        df = df.sort_values(by=["pred"])
        df = df.merge(subpillar_df, on=["testing_period", "currency_code", "factor_name"], how="left")
        df["subpillar"] = df["subpillar"].fillna(df["factor_name"])

        # for defined pillar to remove subpillar cross all pillar by keep top pred only
        # N/A for cluster pillar because subpillar will always in the same pillar, which will be removed after ranking later
        if pillar != "cluster":
            df = df.drop_duplicates(subset=["subpillar", "testing_period", "currency_code"] +
                                           self.config_define_columns + self.config_opt_columns,
                                    keep="last")

        return df

    # ----------------------------------- Add Rank & Evaluation Metrics ---------------------------------------------

    def _eval_factor_all(self, df, eval_q: float, eval_removed_subpillar: bool, **kwargs):
        """
        evaluate & rank different configuration;
        save backtest evaluation metrics -> backtest_eval_table
        """

        summary_cls = groupSummaryStats(eval_q=eval_q, eval_removed_subpillar=eval_removed_subpillar)

        df_eval = df.groupby(self.config_define_columns + self.config_opt_columns).apply(
            summary_cls.get_stats).reset_index()

        return df_eval.drop(columns=["group_idx"])

    def __map_actual_factor_premium(self, eval_df: pd.DataFrame, pred_df: pd.DataFrame):
        """
        map actual factor premiums to factor evaluation results
        """

        df_actual = pred_df.groupby(self.config_define_columns)[['actual']].mean()
        eval_df = eval_df.join(df_actual, on=self.config_define_columns, how='left')

        return eval_df