import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
import itertools
from functools import partial

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
    check_memory
)
from contextlib import closing

icb_num = 6

logger = sys_logger(__name__, "INFO")

factors_formula_table = models.FactorFormulaRatio.__table__.schema + '.' + models.FactorFormulaRatio.__table__.name
processed_ratio_table = models.FactorPreprocessRatio.__table__.schema + '.' + models.FactorPreprocessRatio.__table__.name
factor_premium_table = models.FactorPreprocessPremium.__tablename__


class CalcPremium:
    start_date = '1998-01-01'
    trim_outlier_ = False
    min_group_size = 3  # only calculate premium for group with as least 3 tickers

    def __init__(self,
                 weeks_to_expire: int, weeks_to_offset: int = 1,
                 average_days_list: List[int] = (-7,),
                 currency_code_list: List[str] = None,
                 processes: int = 1, factor_list: List[str] = ()):
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
            factors to calculate premiums (default=[], i.e. calculate all active factors in Table [factors_formula_table])
        """
        logger.info(f'Groups: {" -> ".join(currency_code_list)}')
        logger.info(f'Will save to DB Table [{factor_premium_table}]')

        self.weeks_to_offset = weeks_to_offset
        self.processes = processes
        self.currency_code_list = currency_code_list
        self.factor_list = factor_list if len(factor_list) > 0 \
            else self._factor_list_from_formula()  # if not
        self.y_col_list = [f'stock_return_y_w{weeks_to_expire}_d{x}'
                           for x in average_days_list]

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

        with closing(mp.Pool(processes=self.processes,
                             initializer=recreate_engine)) as pool:
            ratio_df = self._download_pivot_ratios()
            all_groups = itertools.product(self.currency_code_list,
                                           self.factor_list, self.y_col_list)
            all_groups = [tuple(e) for e in all_groups]
            prem = pool.starmap(partial(self.get_premium, ratio_df=ratio_df),
                                all_groups)

        prem = pd.concat([x for x in prem if type(x) != type(None)],
                         axis=0).sort_values(by=['group', 'trading_day'])
        prem = prem.rename(columns={"trading_day": "testing_period"})

        prem["updated"] = timestampNow()
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

        ratio_query = f'''
            SELECT r.*, u.currency_code FROM {processed_ratio_table} r
            INNER JOIN (
                SELECT ticker, currency_code FROM universe
                WHERE is_active 
                    AND currency_code in {tuple(self.currency_code_list)}
                    AND ticker not like '.%%'
            ) u ON r.ticker=u.ticker
            WHERE field in {tuple(self.factor_list + self.y_col_list)} 
                AND trading_day>='{self.start_date}'
        '''.replace(",)", ")")

        df = read_query(ratio_query)
        df = df.pivot(index=["ticker", "trading_day", "currency_code"],
                      columns=["field"], values='value').reset_index()

        return df

    # @err2slack("factor")
    def get_premium(self, *args, ratio_df=None):
        """
        Calculate: premium for certain group and factor (e.g. EUR + roic) in 1 process
        """

        logger.debug(f'=== Calculate premium for ({args}) ===')
        group, factor, y_col = args
        kwargs = {"group": group,
                  "factor": factor,
                  "y_col": y_col}

        df = self._filter_factor_df(ratio_df=ratio_df, **kwargs)
        df = self._qcut_factor_df(df=df, **kwargs)
        prem = self._clean_prem_df(prem=df, **kwargs)

        return prem

    def _filter_factor_df(self, group, factor, y_col, ratio_df=None):
        """
        Data processing: filter complete ratio pd.DataFrame for certain Y & Factor only 
        """

        df = ratio_df.loc[
            ratio_df['currency_code'] == group, ['ticker', 'trading_day', y_col,
                                                 factor]].copy()
        df = self.__clean_missing_y_row(df, y_col, group)
        df = self.__resample_df_by_interval(df)
        df = self.__clean_missing_factor_row(df, factor, group)

        if self.trim_outlier_:
            df[y_col] = self.__trim_outlier(df[y_col], prc=.05)

        return df

    def _qcut_factor_df(self, group, factor, y_col, df=None):
        df['quantile_train_currency'] = df.groupby(['trading_day'])[
            factor].transform(self.__qcut)
        df = df.dropna(subset=['quantile_train_currency']).copy()
        df['quantile_train_currency'] = df['quantile_train_currency'].astype(
            int)
        prem = df.groupby(['trading_day', 'quantile_train_currency'])[
            y_col].mean().unstack()

        return prem

    def _clean_prem_df(self, group, factor, y_col, prem=None):
        """
        premium = small group average returns - big groups  ?????????? 
        Calculate: prem = top quantile - bottom quantile? 
        """

        prem = (prem[0] - prem[2]).dropna().rename('value').reset_index()

        static_info = {"group": group,
                       "field": factor,
                       "weeks_to_expire": int(y_col.split('_')[-2][1:]),
                       "average_days": int(y_col.split('_')[-1][1:])}
        prem = prem.assign(**static_info)

        if self.trim_outlier_:
            prem['field'] = 'trim_' + prem['field']

        return prem

    def __resample_df_by_interval(self, df):
        """
        Data processing:
        resample df (days ->weeks) for premium calculation
        dates every (n=weeks_to_offset) since the most recent period
        """

        end_date = pd.to_datetime(
            pd.date_range(end=backdate_by_day(1), freq=f"W-MON", periods=1)[
                0]).date()
        periods = (end_date - df["trading_day"].min()).days // (
                    7 * self.weeks_to_offset) + 1
        date_list = pd.date_range(end=end_date,
                                  freq=f"{self.weeks_to_offset}W-SUN",
                                  periods=periods)
        date_list = list(date_list)
        df = df.loc[pd.to_datetime(df["trading_day"]).isin(date_list)]

        return df

    def __clean_missing_y_row(self, df, y_col, group):
        df = df.dropna(subset=['ticker', 'trading_day', y_col], how='any')
        if len(df) == 0:
            raise Exception(
                f"[{y_col}] for all ticker in group '{group}' is missing")
        return df

    def __clean_missing_factor_row(self, df, factor, group):
        df = df.dropna(subset=[factor], how='any')
        if len(df) == 0:
            raise Exception(
                f"[{factor}] for all ticker in group '{group}' is missing")
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

    def __qcut(self, series):
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
            return q
        except ValueError as e:
            logger.debug(f'Premium not calculated: {e}')
            return series.map(lambda _: np.nan)
