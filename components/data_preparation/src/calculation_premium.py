import gc

import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
import itertools
from functools import partial
from pandarallel import pandarallel

from typing import List
from sqlalchemy import select, join, and_, not_

from utils import (
    to_slack,
    err2slack,
    read_query,
    read_query_list,
    upsert_data_to_database,
    backdate_by_day,
    models,
    sys_logger,
    recreate_engine,
    timestampNow,
    check_memory,
    es_logger
)
from contextlib import closing

icb_num = 6

logger = sys_logger(__name__, "INFO")

factors_formula_table = models.FactorFormulaRatio.__table__.schema + '.' + models.FactorFormulaRatio.__table__.name
processed_ratio_table = models.FactorPreprocessRatio.__table__.schema + '.' + models.FactorPreprocessRatio.__table__.name
factor_premium_table = models.FactorPreprocessPremium.__tablename__


class CalcPremium:
    trim_outlier_ = False
    min_group_size = 3  # only calculate premium for group with as least 3 tickers

    def __init__(self,
                 weeks_to_expire: int,
                 weeks_to_offset: int = 1,
                 average_days_list: List[int] = (-7,),
                 currency_code_list: List[str] = None,
                 processes: int = 1,
                 factor_list: List[str] = (),
                 start_date: str = None,
                 end_date: str = None):
        """
        Parameters
        ----------
        weeks_to_expire:
            forward period for premium calculation
        weeks_to_offset
            weeks offset between samples (default=1 week),
            i.e. non-overlapping premiums <=> "weeks_to_offset"="weeks_to_expire"
        average_day_list:
            number of average days for the stock returns used to calculate premiums
        currency_code_list:
            currencies to calculate premiums (default=[USD])
        factor_list:
            factors to calculate premiums (default=[], i.e. calculate all
            active factors in Table [factors_formula_table])
        """
        logger.info(f'Groups: {" -> ".join(currency_code_list)}')
        logger.info(f'Will save to DB Table [{factor_premium_table}]')

        self.weeks_to_offset = weeks_to_offset
        self.currency_code_list = currency_code_list
        self.factor_list = factor_list if len(factor_list) > 0 \
            else self._factor_list_from_formula()  # if not
        self.y_col_list = [f'stock_return_y_w{weeks_to_expire}_d{x}'
                           for x in average_days_list]
        self._start_date = start_date or '1998-01-01'
        self._end_date = end_date or backdate_by_day(1)

        self._ratio_df = None
        self._current_currency_code = None
        pandarallel.initialize(progress_bar=True,
                               nb_workers=max(processes//2, 1))

    def write_all(self):
        """
        calculate premium for each group / factor and insert to Table [factor_processed_premium]

        [testing_period] is the start date of premium calculation, e.g. for premium with
        - weeks_to_expire = 4
        - average days = -7
        - testing_period = 2022-04-10

        This premium is in fact using average TRI up to 2022-05-08 (i.e. point-in-time TRI up to 2022-05-15).

        We match macro data according to 2022-05-08, because we assume future 7 days is unknown when prediction is available.
        """

        prem = None

        for cur in self.currency_code_list:
            self._current_currency_code = cur
            ratio_df = self._download_pivot_ratios()

            prem = ratio_df.groupby(['trading_day', 'field'])\
                .parallel_apply(self.get_premium)

            del ratio_df
            gc.collect()

            prem = prem.rename(columns={"trading_day": "testing_period"})
            prem["updated"] = timestampNow()
            prem["group"] = cur
            upsert_data_to_database(data=prem, table=factor_premium_table,
                                    how="update")

            to_slack("clair").message_to_slack(
                f"===  FINISH [update] DB [{factor_premium_table}] ===")

        return prem

    def _factor_list_from_formula(self):
        """
        get list of active factors defined and to update premium
        """

        logger.debug(f'=== Get {factors_formula_table} ===')
        formula_query = f"SELECT name FROM {factors_formula_table} WHERE is_active AND NOT(keep) "
        factor_list = read_query_list(formula_query)
        return factor_list

    def _download_pivot_ratios(self):
        """
        download ratio table calculated with calculation_ratio.py and pivot
        """

        logger.debug(f"=== Get ratios from {processed_ratio_table} ===")

        conditions = [
            models.Universe.is_active,
            models.Universe.currency_code == self._current_currency_code,
            ~models.FactorPreprocessRatio.ticker.like('.%%'),
            models.FactorPreprocessRatio.field.in_(self.factor_list +
                                                   self.y_col_list),
            models.FactorPreprocessRatio.trading_day >= self._start_date,
            models.FactorPreprocessRatio.trading_day <= self._end_date
        ]

        ratio_query = select(models.FactorPreprocessRatio)\
            .join(models.Universe)\
            .where(and_(*conditions))

        df = read_query(ratio_query,
                        index_cols=["ticker", "trading_day", "field"],
                        keep_index=True)["value"].unstack("field")
        return self.__reformat_ratios(df)

    def __reformat_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        convert ratio data to df suitable for groupby operation

        Returns
        -------
        df: pd.DataFrame
            Columns:
                value:              float64, field value
                *all stock_return_y cols:
                                    float64, return for that period
            Index:
                ticker:             str, ticker
                trading_day:        str, trading_day
                field:              str, field for all factors (i.e. not *_y_*)
        """
        df_factor = df[self.factor_list].stack().rename("value").to_frame()
        df_factor = df_factor.merge(df[self.y_col_list],
                                    left_on=["ticker", "trading_day"],
                                    right_index=True, how="left")
        return df_factor

    # @err2slack("factor")
    def get_premium(self, g: pd.DataFrame):
        """
        Calculate: premium for certain group and factor (e.g. EUR + roic) in 1 process
        """

        kwargs = {"group": self._current_currency_code,
                  "factor": g.index.get_level_values("field")[0]}

        prem_all = []
        for y_col in g.filter(regex="stock_return_y_*").columns:
            try:
                kwargs["y_col"] = y_col
                logger.info(f'=== Calculate premium for ({kwargs}) ===')

                df = g.droplevel("field")[["value", y_col]].reset_index()
                df = self.__resample_df_by_interval(df)
                df = self._filter_factor_df(df=df, **kwargs)
                prem = self._qcut_factor_df(df=df, **kwargs)
                prem = self._clean_prem_df(df=prem, **kwargs)
                prem_all.append(prem)

                es_logger.info({"finished": True, **kwargs})
            except Exception as e:
                es_logger.error({"error": "others: " + str(e), **kwargs})
                # raise e

        if len(prem_all) > 0:
            return pd.concat(prem_all, axis=0)
        else:
            return pd.DataFrame()

    def _filter_factor_df(self, df: pd.DataFrame, factor: str, y_col: str,
                          **kwargs):
        """
        Data processing: filter complete ratio pd.DataFrame for certain Y &
        Factor only
        """

        df = df.dropna(subset=[y_col])
        assert len(df) > 0, f"[{y_col}] for all ticker is missing"

        df = df.dropna(subset=["value"])
        assert len(df) > 0, f"[{factor}] for all ticker is missing"

        if self.trim_outlier_:
            df[y_col] = self.__trim_outlier(df[y_col], prc=.05)

        return df

    def _qcut_factor_df(self, df: pd.DataFrame, factor: str, y_col: str,
                        **kwargs):
        """
        Calculate quantile (0, 1, 2) average return for each factor
        """
        df[['quantile', "pct"]] = df.groupby(['trading_day'])["value"]\
            .apply(self.__qcut)
        df = df.dropna(subset=['quantile']).copy()
        df['quantile'] = df['quantile'].astype(int)
        prem = df.groupby(['trading_day', 'pct', 'quantile'])[
            ["value", y_col]].mean()\
            .rename(columns={"value": "ratio", y_col: "return"})\
            .unstack("quantile")
        prem.columns = ["_".join([str(i) for i in x]) for x in prem.columns]
        prem_count = df.groupby(['trading_day'])[y_col].count()\
            .rename("n_sample")
        prem = prem.reset_index("pct").join(prem_count)

        assert np.all(prem["ratio_0"] < prem["ratio_2"]), \
            f"[{factor}] ratio to rank class 0 must < 2."

        return prem

    def _clean_prem_df(self, df: pd.DataFrame, factor: str, y_col: str,
                       **kwargs):
        """
        premium = the difference of average returns for smaller factor value
        groups and larger factor value groups

        i.e. Calculate: prem = bottom quantile - top quantile
        """
        df["value"] = (df["return_0"] - df["return_2"]).dropna()

        static_info = {"field": factor,
                       "weeks_to_expire": int(y_col.split('_')[-2][1:]),
                       "average_days": int(y_col.split('_')[-1][1:])}
        df = df.assign(**static_info).reset_index()

        if self.trim_outlier_:
            df['field'] = 'trim_' + df['field']

        return df

    def __resample_df_by_interval(self, df):
        """
        Data processing:
        resample df (days ->weeks) for premium calculation
        dates every (n=weeks_to_offset) since the most recent period
        """

        end_date = pd.to_datetime(pd.date_range(
            end=self._end_date, freq=f"W-MON", periods=1)[0]).date()
        periods = (end_date - df["trading_day"].min()).days // (
                    7 * self.weeks_to_offset) + 1
        date_list = pd.date_range(end=end_date,
                                  freq=f"{self.weeks_to_offset}W-SUN",
                                  periods=periods)
        date_list = list(date_list)
        df = df.loc[pd.to_datetime(df["trading_day"]).isin(date_list)]

        return df

    def __trim_outlier(self, df, prc: float = 0):
        """
        assign a max value for the 99% percentile to replace
        """

        df_nan = df.replace([np.inf, -np.inf], np.nan)
        pmax = df_nan.quantile(q=(1 - prc))
        pmin = df_nan.quantile(q=prc)
        df = df.mask(df > pmax, pmax)
        df = df.mask(df < pmin, pmin)

        return df

    def __qcut(self, series) -> pd.DataFrame:
        """
        For each factor at certain period, use qcut to calculate factor premium
        as Top 30%/20% - Bottom 30%/20%
        """

        try:
            series_fillinf = series.replace([-np.inf, np.inf], np.nan)
            nonnull_count = series_fillinf.notnull().sum()
            if nonnull_count < self.min_group_size:
                return series.map(lambda _: np.nan)
            elif nonnull_count > 65:
                prc = [0, 0.2, 0.8, 1]
            else:
                prc = [0, 0.3, 0.7, 1]

            q = pd.qcut(series_fillinf, prc, duplicates='drop')

            n_cat = len(q.cat.categories)
            if n_cat == 3:
                q.cat.categories = range(3)
            elif n_cat == 2:
                q.cat.categories = [0, 2]
            else:
                return series.map(lambda _: np.nan)
            q[series_fillinf == np.inf] = 2
            q[series_fillinf == -np.inf] = 0
            return q.rename("value").to_frame().assign(pct=prc[1])
        except ValueError as e:
            logger.debug(f'Premium not calculated: {e}')
            return series.map(lambda _: np.nan)
