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
    upsert_data_to_database,
    models,
    sys_logger
)
from contextlib import closing

from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION, INTEGER
import gc

icb_num = 6

logger = sys_logger(__name__, "DEBUG")

factors_formula_table = models.FormulaRatio.__table__.schema + '.' + models.FormulaRatio.__table__.name
processed_ratio_table = models.PreprocessRatio.__table__.schema + '.' + models.PreprocessRatio.__table__.name
factor_premium_table = models.PreprocessPremium.__table__.name


class calcPremium:

    trim_outlier_ = False
    start_date = '1998-01-01'

    def __init__(self,
                 weeks_to_expire: int, weeks_to_offset: int = 1,
                 average_days: List[int] = (-7,), processes: int = 12,
                 all_train_currencys: List[str] = None, factor_list: List[str] = ()):
        """  calculate factor premium for different configurations and write to DB Table [factor_premium_table]

        Parameters
        ----------
        weeks_to_expire (Int):
            forward period for premium calculation
        weeks_to_offset (Int, Optional):
            weeks offset between samples (default=1 week),
            i.e. if calculating non-duplicated premiums, should set "weeks_to_offset"="weeks_to_expire"
        average_days (Int, Optional):
            number of average days for the stock returns used to calculate premiums
        trim_outlier_ (Bool, Optional):
            if True, use trimmed returns for top/bottom stocks
        processes (Int, Optional):
            multiprocess threads (default=12)
        all_train_currencys (List[Str], Optional):
            currencies to calculate premiums (default=[USD])
        factor_list (List[Str], Optional):
            factors to calculate premiums (default=[], i.e. calculate all active factors in Table [factors_formula_table])
        """
        logger.info(f'Groups: {" -> ".join(all_train_currencys)}')
        logger.info(f'Will save to DB Table [{factor_premium_table}]')

        self.weeks_to_offset = weeks_to_offset
        self.processes = processes
        self.all_train_currencys = all_train_currencys
        self.factor_list = factor_list if len(factor_list) > 0 else self._factor_list_from_formula
        self.y_col = [f'stock_return_y_w{weeks_to_expire}_d{x}' for x in average_days]
        self.df = None

    def get_all(self):

        with closing(mp.Pool(processes=self.processes)) as pool:
            ratio_df = self._download_pivot_ratios()
            all_groups = itertools.product(self.all_train_currencys, self.factor_list, self.y_col)
            all_groups = [tuple(e) for e in all_groups]
            prem = pool.starmap(partial(self._insert_prem_for_train_currency, ratio_df=ratio_df), all_groups)

        prem = pd.concat(prem, axis=0).sort_values(by=['group', 'trading_day'])
        upsert_data_to_database(data=prem, table=factor_premium_table, how="update")
        to_slack("clair").message_to_slack(f"===  FINISH [update] DB [{factor_premium_table}] ===")

        return prem

    @property
    def _factor_list_from_formula(self):
        logger.debug(f'=== Get {factors_formula_table} ===')
        formula_query = f"SELECT * FROM {factors_formula_table} WHERE is_active AND NOT(keep) "
        formula = read_query(formula_query)
        factor_list = formula['name'].to_list()  # default factor = all variabales
        return factor_list

    def _download_pivot_ratios(self):
        logger.debug(f"=== Get ratios from {processed_ratio_table} ===")

        ratio_query = f'''
            SELECT r.*, u.currency_code FROM {processed_ratio_table} r
            INNER JOIN (
                SELECT ticker, currency_code FROM universe
                WHERE is_active 
                    AND currency_code in {tuple(self.all_train_currencys)}
                    AND ticker not like '.%%'
            ) u ON r.ticker=u.ticker
            WHERE field in {tuple(self.factor_list + self.y_col)} 
                AND trading_day>='{self.start_date}'
        '''.replace(",)", ")")

        df = read_query(ratio_query)
        df = df.pivot(index=["ticker", "trading_day", "currency_code"], columns=["field"], values='value').reset_index()

        return df

    @err2slack("clair")
    def _insert_prem_for_train_currency(self, *args, ratio_df=None):
        """ calculate premium for each group / factor and insert to Table [factor_processed_premium] """

        group, factor, y_col = args
        weeks_to_expire, average_days = int(y_col.split('_')[-2][1:]), int(y_col.split('_')[-1][1:])

        logger.debug(f'=== Calculate premium for ({group}, {factor}, {y_col}) ===')
        df = ratio_df.loc[ratio_df['currency_code'] == group, ['ticker', 'trading_day', y_col, factor]].copy()
        df = self.__clean_missing_y_row(df, y_col, group)
        df = self.__resample_df_by_interval(df)
        df = self.__clean_missing_factor_row(df, factor, group)

        if self.trim_outlier_:
            df[y_col] = self.__trim_outlier(df[y_col], prc=.05)

        df['quantile_train_currency'] = df.groupby(['trading_day'])[factor].transform(self.__qcut)
        df = df.dropna(subset=['quantile_train_currency']).copy()
        df['quantile_train_currency'] = df['quantile_train_currency'].astype(int)
        prem = df.groupby(['trading_day', 'quantile_train_currency'])[y_col].mean().unstack()

        # Calculate small minus big
        prem = (prem[0] - prem[2]).dropna().rename('value').reset_index()
        prem = prem.assign({"group": group, "field": factor, "weeks_to_expire": weeks_to_expire, "average_days": average_days})
        if self.trim_outlier_:
            prem['field'] = 'trim_'+prem['field']

        return prem

    def __clean_missing_y_row(self, df, y_col, group):
        df = df.dropna(subset=['ticker', 'trading_day', y_col], how='any')
        if len(df) == 0:
            raise Exception(f"[{y_col}] for all ticker in group '{group}' is missing")
        return df

    def __resample_df_by_interval(self, df):
        """
        resample df for premium calculation dates every (n=weeks_to_offset) since the most recent period
        """
        date_list = reversed(df["trading_day"].unique())
        date_list = [x for i, x in enumerate(date_list) if (i % self.weeks_to_offset == 0)]
        df = df.loc[df["trading_day"].isin(date_list)]
        return df

    def __clean_missing_factor_row(self, df, factor, group):
        df = df.dropna(subset=[factor], how='any')
        if len(df) == 0:
            raise Exception(f"[{factor}] for all ticker in group '{group}' is missing")
        return df

    def __trim_outlier(self, df, prc=0):
        """ assign a max value for the 99% percentile to replace  """

        df_nan = df.replace([np.inf, -np.inf], np.nan)
        pmax = df_nan.quantile(q=(1 - prc))
        pmin = df_nan.quantile(q=prc)
        df = df.mask(df > pmax, pmax)
        df = df.mask(df < pmin, pmin)

        return df

    def __qcut(self, series):
        """ assign a max value for the 99% percentile to replace  """

        try:
            series_fillinf = series.replace([-np.inf, np.inf], np.nan)
            nonnull_count = series_fillinf.notnull().sum()
            if nonnull_count < 3:
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


if __name__ == "__main__":

    last_update = datetime.now()

    calcPremium(weeks_to_expire=8, average_days=[-7], weeks_to_offset=4, processes=10,
                     all_train_currencys=["HKD", "CNY", "USD", "EUR"])
    # calcPremium(weeks_to_expire=26, average_days=-7, weeks_to_offset=4, processes=12,
    #                  all_train_currencys=["HKD", "CNY", "USD", "EUR"], start_date='2020-02-02')
    # stock_return_map = {4: [-7]}
    # start = datetime.now()
    # for fwd_weeks, avg_days in stock_return_map.items():
    #     for d in avg_days:
    #         calcPremium(weeks_to_expire=fwd_weeks, average_days=d, weeks_to_offset=1,
    #                          all_train_currencys=['CNY'], processes=10)
    # end = datetime.now()
    #
    # logger.debug(f'Time elapsed: {(end - start).total_seconds():.2f} s')
    # write_local_csv_to_db()
