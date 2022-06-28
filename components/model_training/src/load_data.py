import numpy as np
import pandas as pd
import datetime as dt
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sqlalchemy import select, and_, not_, func
from typing import List, Union, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
import math

from utils import (
    sys_logger,
    read_query,
    read_query_list,
    dateNow,
    read_table,
    upsert_data_to_database,
    backdate_by_day,
    models
)

logger = sys_logger(__name__, "DEBUG")

macro_data_table = models.DataMacro.__table__.schema + '.' + models.DataMacro.__table__.name
vix_data_table = models.DataVix.__table__.schema + '.' + models.DataVix.__table__.name
fred_data_table = models.DataFred.__table__.schema + '.' + models.DataFred.__table__.name
processed_ratio_table = models.FactorPreprocessRatio.__table__.schema + '.' + models.FactorPreprocessRatio.__table__.name
factor_premium_table = models.FactorPreprocessPremium.__table__.schema + '.' + models.FactorPreprocessPremium.__table__.name
factors_formula_table = models.FactorFormulaRatio.__table__.schema + '.' + models.FactorFormulaRatio.__table__.name


class calcTestingPeriod:

    def __init__(self, weeks_to_expire: int,
                 sample_interval: int,
                 backtest_period: int,
                 currency_code: str = None,
                 restart: bool = None):
        self.weeks_to_expire = weeks_to_expire
        self.sample_interval = sample_interval
        self.backtest_period = backtest_period
        self.restart = restart
        if currency_code:
            self.currency_code_list = [currency_code]

    @property
    def _testing_period_list(self) -> pd.DatetimeIndex:
        """
        testing_period matched with testing period calculation for in data_preparation/calculation_premium.py
        """

        if self.restart:
            end_date = pd.to_datetime(pd.date_range(end=self._restart_iteration_first_running_date,
                                                    freq=f"W-MON", periods=1)[0])
        else:
            end_date = pd.to_datetime(pd.date_range(end=backdate_by_day(1), freq=f"W-MON", periods=1)[0])

        period_list = pd.date_range(end=end_date, freq=f"{self.sample_interval}W-SUN", periods=1500//self.sample_interval)
        return period_list

    @property
    def _restart_iteration_first_running_date(self) -> dt.date:
        """
        for restart iteration find training start date with uid
        """
        query = select(func.min(models.FactorResultScore.uid)).where(models.FactorResultScore.name_sql == self.restart)
        restart_iter_running_date = read_query_list(query)[0]
        restart_iter_running_date = pd.to_datetime(restart_iter_running_date[:8], format="YYYYMMDD").date()
        return restart_iter_running_date


class cleanMacros(calcTestingPeriod):
    """
    load clean macro as independent variables for training
    """

    def __init__(self,
                 weeks_to_expire: int,
                 sample_interval: int,
                 backtest_period: int,
                 currency_code: str = None,
                 restart: bool = None):

        super().__init__(weeks_to_expire, sample_interval, backtest_period, currency_code, restart)
        self.period_list = self._testing_period_list
        self.weeks_to_expire = weeks_to_expire

    def _download_clean_macro(self):
        """
        download from [data_macro] (macro data from dsws, e.g. GDP, interest rate, ...)
        """
        macros = read_query(select(models.DataMacro))[["field", "trading_day", "value"]]
        macros['trading_day'] = pd.to_datetime(macros['trading_day'], format='%Y-%m-%d')
        macros = self.__calc_yoy(macros)

        return macros

    def __calc_yoy(self, macros: pd.DataFrame, yoy_trh: int = 80):
        """
        Use YoY instead of original value for Macro Index for fields (e.g. GDP)
        """

        yoy_field = macros.groupby(["field"])["value"].mean()
        yoy_field = list(yoy_field[yoy_field > yoy_trh].index)

        macros_1yb = macros.loc[macros["field"].isin(yoy_field)].copy()
        macros_1yb["trading_day"] = macros_1yb["trading_day"].apply(lambda x: x + relativedelta(years = 1))
        macros = macros.merge(macros_1yb, on = ["field", "trading_day"], how="left", suffixes=("", "_1yb"))
        macros["value_1yb"] = macros.sort_values(by="trading_day").groupby("field")["value_1yb"].fillna(method='ffill')
        macros.loc[macros["field"].isin(yoy_field), "value"] = macros["value"]/macros["value_1yb"] - 1

        return macros.drop(columns=["value_1yb"])

    def _download_vix(self):
        """
        download from [data_vix] (vix index for different market)
        """
        vix = read_query(select(models.DataVix))
        vix = vix.rename(columns={"vix_id": "field", "vix_value": "value"})
        vix['trading_day'] = pd.to_datetime(vix['trading_day'], format='%Y-%m-%d')
        return vix[["field", "trading_day", "value"]]

    def _download_fred(self):
        """
        download from [data_fred] (ICE BofA BB US High Yield Index Effective Yield)
        """
        fred = read_query(select(models.DataFred))
        fred['field'] = "fred_data"
        fred['trading_day'] = pd.to_datetime(fred['trading_day'], format = '%Y-%m-%d')
        return fred[["field", "trading_day", "value"]]

    def _download_index_return(self):
        """
        download from factor preprocessed ratio table (index returns)
        """
        index_col = ['stock_return_r12_7', 'stock_return_r1_0', 'stock_return_r6_2']
        index_ticker = ['.SPX', '.CSI300', '.SXXGR', '.HSI']
        conditions = [
            models.FactorPreprocessRatio.ticker.in_(tuple(index_ticker)),
            models.FactorPreprocessRatio.field.in_(tuple(index_col)),
        ]
        index_query = select(models.FactorPreprocessRatio).where(and_(*conditions))
        index_ret = read_query(index_query)

        index_ret['trading_day'] = pd.to_datetime(index_ret['trading_day'])
        index_ret['field'] += "_" + index_ret["ticker"]
        index_ret['trading_day'] = pd.to_datetime(index_ret['trading_day'], format='%Y-%m-%d')
        return index_ret[["field", "trading_day", "value"]]

    def get_all_macros(self):
        """
        combine all macros and resample to testing periods matched with premium table dates
        """
        logger.debug(f'=== Download macro data ===')

        macros = self._download_clean_macro()
        vix = self._download_vix()
        fred = self._download_fred()
        index = self._download_index_return()

        df = pd.concat([macros, vix, fred, index], axis=0).dropna(how="any")
        df = df.pivot(index=["trading_day"], columns=["field"], values="value").reset_index()
        df = self._resample_macros_to_testing_period(df)

        return df

    def _resample_macros_to_testing_period(self, df):
        """
        Use last available macro data for each testing_period.
        [testing_period] should match with [testing_period] in factor preprocessed premium table.
        """

        df["testing_period"] = df["trading_day"] - pd.tseries.offsets.DateOffset(weeks=self.weeks_to_expire)

        df = df.set_index("testing_period")
        combine_index = df.index
        combine_index = combine_index.append(self.period_list).drop_duplicates()

        df = df.reindex(combine_index)
        df = df.sort_index().ffill()
        reindex_df = df.reindex(self.period_list).reset_index().rename(columns={"index": "testing_period"})

        return reindex_df.drop(columns=["trading_day"])


class combineData(cleanMacros):
    """
    combine all raw premium + macro data as inputs / outputs
    """

    currency_code_list = ["HKD", "CNY", "USD", "EUR"]
    trim = False

    def __init__(self,
                 weeks_to_expire: int,
                 sample_interval: int,
                 backtest_period: int,
                 currency_code: str = None,
                 restart: bool = None):

        super().__init__(weeks_to_expire, sample_interval, backtest_period, currency_code, restart)
        self.weeks_to_expire = weeks_to_expire
        self.sample_interval = sample_interval
        self.backtest_period = backtest_period
        self.restart = restart
        if currency_code:
            self.currency_code_list = [currency_code]

    def get_raw_data(self):
        """
        Returns
        -------
        pd.DataFrame(columns=["group", "testing_period", "average_days"] + factor_premium_columns + macro_input_columns)

        1. "testing_period" with equal interval as any other consecutive rows
        2. sort by ["group", "testing_period"]

        """
        df = self._download_premium()
        df = self._remove_high_missing_samples(df)
        df = self._add_macros_inputs(df)

        return df.sort_values(by=["group", "testing_period"])

    def _add_macros_inputs(self, premium):
        """
        Combine macros and premium inputs
        """

        macros = self.get_all_macros()         # TODO: remove after debug
        macros.to_pickle("factor_macros.pkl")

        macros = pd.read_pickle("factor_macros.pkl")
        macros_premium = premium.merge(macros, on=["testing_period"])

        assert len(macros_premium) == len(premium)

        return macros_premium

    def _download_premium(self):
        """
        download from factor processed premium table
        """
        conditions = [
            models.FactorPreprocessPremium.weeks_to_expire == self.weeks_to_expire,
            models.FactorPreprocessPremium.group.in_(tuple(self.currency_code_list)),
            models.FactorPreprocessPremium.testing_period.in_(tuple(self._testing_period_list))
        ]
        if self.trim:
            conditions.append(models.FactorPreprocessPremium.field.like("trim_%%"))
        else:
            conditions.append(not_(models.FactorPreprocessPremium.field.like("trim_%%")))

        query = select(models.FactorPreprocessPremium).where(and_(*conditions))
        df = read_query(query)
        df = df.pivot(index=['testing_period', 'group', 'average_days'], columns=['field'], values="value").reset_index()
        df['testing_period'] = pd.to_datetime(df['testing_period'], format = '%Y-%m-%d')

        return df

    def _remove_high_missing_samples(self, df: pd.DataFrame, trh: float = 0.5) -> pd.DataFrame:
        """
        remove first few samples with high missing
        """
        max_missing_cols = (df.shape[1] - 3) * trh
        low_missing_samples = df.loc[df.isnull().sum(axis = 1) < max_missing_cols]
        low_missing_first_period = low_missing_samples["testing_period"].min()
        df = df.loc[df["testing_period"] >= low_missing_first_period]

        return df


class loadData:
    """
    main function:
        1. split train + valid + tests -> sample set
        2. convert x with standardization, y with qcut
    """

    train_lookback_year = 21                # 1-year buffer for Y calculation
    lasso_reverse_lookback_window = 13      # period (time-span varied depending on sample_interval at combineData)
    lasso_reverse_alpha = 1e-5
    convert_y_use_median = True
    convert_y_cut_bins = None               # [List] instead of qcut -> use defined bins (e.g. [-inf, 0, 1, inf])
    convert_x_ar_period = []
    convert_x_ma_period = {3: [], 12: []}
    convert_x_macro_pca = 0.6
    all_factor_list = read_query_list(select(models.FactorFormulaRatio.name).where(models.FactorFormulaRatio.is_active))

    def __init__(self,
                 weeks_to_expire: int,
                 train_currency: str,
                 pred_currency: str,
                 testing_period: dt.datetime,
                 average_days: int,
                 factor_list: List[str],
                 factor_reverse: int,
                 y_qcut: int,
                 factor_pca: float,
                 valid_pct: float,
                 valid_method: Union[str, int],
                 **kwargs):
        """
        Parameters
        ----------
        valid_pct (Float):
            must be < 1, use _valid_pct% of training as validation.
        valid_method (Str / Int):
            if "cv", cross validation between groups;
            if "chron", use last 2 years / % of training as validation;
            if Int (must be year after 2008), use valid_pct% sample from year indicated by valid_method.
        """

        self.weeks_to_expire = weeks_to_expire
        self.testing_period = testing_period
        self.average_days = average_days
        self.factor_list = factor_list
        self.factor_reverse = factor_reverse
        self.y_qcut = y_qcut
        self.factor_pca = factor_pca
        self.valid_pct = valid_pct
        self.valid_method = valid_method

        if train_currency == 'currency':
            self.train_currency = ['CNY', 'HKD', 'EUR', 'USD']
        else:
            self.train_currency = train_currency.split(',')

        self.pred_currency = pred_currency.split(',')

    def split_all(self, main_df) -> (List[Dict[str, pd.DataFrame]], list, list):
        """
        work through cleansing process

        Returns
        ----------
        tuple
            sample_sets: dict of [train / valid / test] set dataframes of
                - x: inputs consisting of premium + macros data
                - y: original predicted premium for [evaluation]
                - y_cut: qcut & convert to median premium for [training]
                - * All dataframes with pd.MultiIndex([group, testing_period])
            neg_factor: list of factor reverse in calculation
            cut_bins: list of threshold of cut_bins
        """

        sample_df = self._filter_sample(main_df=main_df)
        sample_df, neg_factor = self._convert_sample_neg_factor(sample_df=sample_df)

        df_train_cut, df_test_cut, df_train_org, df_test_org, cut_bins = self._get_y(sample_df=sample_df)
        df_train_pca, df_test_pca = self._get_x(sample_df=sample_df)

        gkf = self._split_valid(df_train_pca)
        sample_sets = self._get_sample_sets(
            train_x=df_train_pca,
            train_y=df_train_org,
            train_y_cut=df_train_cut,
            test_x=df_test_pca,
            test_y=df_test_org,
            test_y_cut=df_test_cut,
            gkf=gkf
        )

        return sample_sets, neg_factor, cut_bins

    def _filter_sample(self, main_df) -> pd.DataFrame:
        """
        filter main premium_df for data frame with sample needed
        """

        conditions = \
            (main_df["group"].isin(self.train_currency + self.pred_currency)) & \
            (main_df["average_days"] == self.average_days) & \
            (main_df["testing_period"] >= (self.testing_period - relativedelta(years=self.train_lookback_year))) & \
            (main_df["testing_period"] <= (self.testing_period + relativedelta(weeks=self.weeks_to_expire)))

        sample_df = main_df.loc[conditions].copy()
        return sample_df.drop(columns=["average_days"]).set_index(["group", "testing_period"])

    def __train_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        training sample from (n = lookback) year ahead to (n = weeks_to_expire) ahead (i.e. to avoid data snooping in Y)
        """
        start_date = self.testing_period - relativedelta(years=self.train_lookback_year)
        end_date = self.testing_period - relativedelta(weeks=self.weeks_to_expire)\

        if not set(self.train_currency).issubset(set(df.index.get_level_values("group").unique())):
            raise Exception("train_currency not in raw_df. Please check combineData().")

        train_df = df.loc[(start_date <= df.index.get_level_values("testing_period")) &
                          (df.index.get_level_values("testing_period") <= end_date) &
                          (df.index.get_level_values("group").isin(self.train_currency))]
        assert len(train_df) > 0
        return train_df

    def __test_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        testing sample is all testing as at testing period
        """
        test_df = df.loc[(df.index.get_level_values("testing_period") == self.testing_period) &
                         (df.index.get_level_values("group").isin(self.pred_currency))]
        if len(test_df) > 0:
            return test_df
        else:
            return pd.DataFrame(
                index=pd.MultiIndex.from_product([[self.testing_period], self.pred_currency],
                                                 names=["testing_period", "group"]),
                columns=df.columns.to_list())

    def _convert_sample_neg_factor(self, sample_df) -> (pd.DataFrame, list):
        """
        convert consistently negative premium factor to positive
        refer to: https://loratechai.atlassian.net/wiki/spaces/SEAR/pages/994803719/AI+Score+Brainstorms+2022-01-28

        Parameters
        ----------
        factor_reverse (Int, Optional):
            if 1, use training period average;
            if 2, use lasso;
            if 0, no conversion.
        n_years (Int, Optional):
            lasso training period length (default = 7) years.
        lasso_alpha (Float, Optional):
            lasso training alpha lasso_alpha (default = 1e-5).
        """

        if self.factor_reverse == 0:
            neg_factor = []
        elif self.factor_reverse == 1:  # using average of training period -> next period +/-
            neg_factor = self._factor_reverse_on_average(sample_df)
        else:
            neg_factor = self._factor_reverse_on_lasso(sample_df)

        sample_df[neg_factor] *= -1
        return sample_df, neg_factor

    def _factor_reverse_on_average(self, sample_df):
        """
        reverse if factor premium on training sets on average < 0
        """

        sample_df_mean = self.__train_sample(sample_df).mean(axis=0)
        neg_factor = list(sample_df_mean[sample_df_mean < 0].index)

        return neg_factor

    def __lasso_get_window_total(self, sample_df):
        """
        total sample window for lasso = lookback period + forward prediction period
        """

        testing_period_list = sample_df.index.get_level_values("testing_period").values
        sample_interval = (testing_period_list[-1] - testing_period_list[-2])/np.timedelta64(1, 'D')
        window_forward = int(self.weeks_to_expire*7 // sample_interval)
        window_total = int(self.lasso_reverse_lookback_window + window_forward)

        return window_total

    def _factor_reverse_on_lasso(self, sample_df) -> list:
        """
        reverse if lasso predicted next period (i.e. Y) factor premium < 0
        """

        neg_factor = []
        lasso_train = self.__train_sample(sample_df).groupby(level="testing_period")[self.factor_list].mean().fillna(0)
        window_total = self.__lasso_get_window_total(sample_df)

        for col in self.factor_list:
            X_windows = lasso_train[col].rolling(window=window_total)
            window_arr = np.array([window.to_list() for window in X_windows if len(window) == window_total])

            clf = Lasso(alpha=self.lasso_reverse_alpha)
            clf.fit(X=window_arr[:, :self.lasso_reverse_lookback_window],
                    y=window_arr[:, -1])
            pred = clf.predict(window_arr[[-1], -self.lasso_reverse_lookback_window:])

            if pred < 0:
                neg_factor.append(col)

        return neg_factor

    def _get_y(self, sample_df) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list):
        """
        convert Y with qcut and group median
        """

        df_y = self.__y_convert_testing_period(sample_df=sample_df)
        df_train = self.__train_sample(df_y).copy().dropna(how='all').fillna(0)
        df_test = self.__test_sample(df_y).copy().dropna(how='all').fillna(0)

        if type(self.convert_y_cut_bins) != type(None):
            df_train_cut, df_test_cut, cut_bins = self.__y_cut_with_defined_cut_bins(df_train, df_test)
        elif self.y_qcut > 0:
            df_train_cut, df_test_cut, cut_bins = self.__y_qcut_with_flatten_train(df_train, df_test)
        elif self.y_qcut == 0:
            df_train_cut, df_test_cut, cut_bins = df_train, df_test, []
        else:
            raise Exception("Wrong [y_qcut] config!")

        return df_train_cut, df_test_cut, df_train, df_test, cut_bins

    def __y_convert_testing_period(self, sample_df):
        """
        testing_period of y = testing_period of X + weeks_to_expire to avoid data snooping
        """

        df_y = sample_df[self.factor_list].copy()

        df_y = df_y.reset_index()
        df_y["testing_period"] = df_y['testing_period'] - pd.tseries.offsets.DateOffset(weeks=self.weeks_to_expire)
        df_y = df_y.set_index(["group", "testing_period"])

        return df_y

    def __y_replace_median(self, train_org, train_cut, test_cut) -> (pd.DataFrame, pd.DataFrame):
        """
        convert qcut results (e.g. 012) to the median of each group to remove noise for regression problem
        """

        df = pd.DataFrame(np.vstack((train_org.values.flatten(), train_cut.values.flatten()))).T
        median_map = df.groupby([1])[0].median().to_dict()

        train_cut = train_cut.replace(median_map)
        test_cut = test_cut.replace(median_map)

        return train_cut, test_cut

    def __y_cut_with_defined_cut_bins(self, train, test) -> (pd.DataFrame, pd.DataFrame, list):
        """ qcut all factors predict together with defined cut_bins """

        train_cut = train.apply(pd.cut, bins=self.convert_y_cut_bins, labels=False)
        test_cut = test.apply(pd.cut, bins=self.convert_y_cut_bins, labels=False)

        if self.convert_y_use_median:
            train_cut, test_cut = self.__y_replace_median(train_org=train, train_cut=train_cut, test_cut=test_cut)

        return train_cut, test_cut, self.convert_y_cut_bins

    def __y_qcut_with_flatten_train(self, train, test) -> (pd.DataFrame, pd.DataFrame, list):
        """
        Flatten all training factors to qcut all together
        """

        arr = train.values.flatten()
        assert len(arr) > 0

        arr_cut, cut_bins = pd.qcut(arr, q=self.y_qcut, retbins=True, labels=False)
        cut_bins[0], cut_bins[-1] = [-np.inf, np.inf]

        arr_cut_reshape = np.reshape(arr_cut, train.shape, order='C')
        train_cut = pd.DataFrame(arr_cut_reshape, index=train.index, columns=train.columns)

        test_cut = self.__test_sample(test).apply(pd.cut, bins=cut_bins, labels=False)

        if self.convert_y_use_median:
            train_cut, test_cut = self.__y_replace_median(train_org=train, train_cut=train_cut, test_cut=test_cut)

        return train_cut, test_cut, cut_bins

    def _get_x(self, sample_df):
        """
        convert X with standardization and PCA
        """

        input_cols = sample_df.columns.to_list()
        input_factor_cols = list(set(self.all_factor_list) & set(input_cols))
        input_macro_cols = list(set(input_cols) - set(self.all_factor_list))

        # sample_df = self._x_convert_ar(sample_df, arma_col=input_cols)
        # sample_df = self._x_convert_ma(sample_df, arma_col=input_cols, ma_avg_period=3)
        # sample_df = self._x_convert_ma(sample_df, arma_col=input_cols, ma_avg_period=12)

        df_train = self.__train_sample(sample_df).copy().fillna(0)
        df_test = self.__test_sample(sample_df).copy().fillna(0)
        assert len(df_test) > 0

        df_train_factor, df_test_factor = self.__x_standardize_pca(df_train, df_test,
                                                                   pca_cols=input_factor_cols,
                                                                   n_components=self.factor_pca,
                                                                   prefix="factor")
        df_train_macro, df_test_macro = self.__x_standardize_pca(df_train, df_test,
                                                                 pca_cols=input_macro_cols,
                                                                 n_components=self.convert_x_macro_pca,
                                                                 prefix="macro")

        df_train_pca = df_train_factor.merge(df_train_macro, left_index=True, right_index=True)
        df_test_pca = df_test_factor.merge(df_test_macro, left_index=True, right_index=True)

        return df_train_pca, df_test_pca

    def _x_convert_ar(self, sample_df, arma_col: list):
        """
        (Obsolete) Calculate the time_series history for predicted Y (use 1/2/12 based on ARIMA results)
        """

        for i in self.convert_x_ar_period:
            ar_col = [f"ar_{x}_{i}m" for x in arma_col]
            sample_df[ar_col] = sample_df.groupby(level='group')[arma_col].shift(i)
        return sample_df

    def _x_convert_ma(self, sample_df, arma_col: list, ma_avg_period: int):
        """
        (Obsolete) Calculate the moving average for predicted Y
        """
        ma_q_col = [f"ma{ma_avg_period}_{x}_q" for x in arma_col]
        sample_df[ma_q_col] = sample_df.groupby(level='group')[arma_col].rolling(ma_avg_period, min_periods=1).mean()

        for i in self.convert_x_ma_period[ma_avg_period]:
            ma_col = [f'{x}{i}' for x in ma_q_col]
            sample_df[ma_col] = sample_df.groupby(level='group')[ma_q_col].shift(i)

        return sample_df

    def __x_standardize_pca(self, train, test, pca_cols: List[str], n_components: float = None, prefix: str = None):
        """
        standardize x + PCA applied to x with train_x fit
        """

        assert n_components < 1  # always use explanation ratios

        if n_components > 0:
            scaler = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_components))])
        else:
            scaler = StandardScaler()

        scaler.fit(train[pca_cols].values)
        x_train = scaler.transform(train[pca_cols].values)
        x_test = scaler.transform(test[pca_cols].values)

        pca_train = self.__x_after_pca_rename(train[pca_cols], x_train, prefix=prefix)
        pca_test = self.__x_after_pca_rename(test[pca_cols], x_test, prefix=prefix)

        return pca_train, pca_test

    def __x_after_pca_rename(self, df_org, arr_pca, prefix: str = None):
        """
        rename after PCA column names
        """

        if df_org.shape[1] == arr_pca.shape[1]:
            new_columns = df_org.columns.to_list()
        else:
            new_columns = [f"{prefix}_{i}" for i in range(arr_pca.shape[1])]

        df_pca = pd.DataFrame(arr_pca, index=df_org.index, columns=new_columns)
        return df_pca

    def _split_valid(self, train_x) -> List[tuple]:
        """
        split for training and validation set indices
        """

        assert self.valid_pct < 1

        if self.valid_method == "cv":
            gkf = self.__split_valid_cv(train_x)
        elif self.valid_method == "chron":
            gkf = self.__split_valid_chron(train_x)
        elif isinstance(self.valid_method, int):
            gkf = self.__split_valid_from_year(train_x)
        else:
            raise ValueError(f"Invalid 'valid_method'. "
                             f"Expecting 'cv' or 'chron' or integer of year (e.g. 2010) got {self.valid_method}")

        return gkf

    def __split_valid_cv(self, train_x) -> list:
        """ 
        (obsolete) split validation set by cross-validation 5 split (if.valid_pct = .2)
        """
        n_splits = int(round(1 / self.valid_pct))
        gkf = GroupShuffleSplit(n_splits=n_splits).split(train_x, groups=train_x.index.get_level_values("group"))
        return gkf
    
    def __split_valid_chron(self, train_x) -> list:
        """ 
        (obsolete) split validation set by chronological order
        """
        gkf = []
        train = train_x.reset_index().copy()
        valid_len = (train['trading_day'].max() - train['trading_day'].min()) * self.valid_pct
        valid_period = train['trading_day'].max() - valid_len
        valid_index = train.loc[train['trading_day'] >= valid_period].index.to_list()
        train_index = train.loc[train['trading_day'] < valid_period].index.to_list()
        gkf.append((train_index, valid_index))
        
        return gkf
    
    def __split_valid_from_year(self, train_x) -> list:
        """       
        valid_method can be year name (e.g. 2010) -> use valid_pct% amount of data since 2010-01-01 as valid sets
        """

        valid_len = (train_x.index.get_level_values('testing_period').max() -
                     train_x.index.get_level_values('testing_period').min()) * self.valid_pct
        valid_start = dt.datetime(self.valid_method, 1, 1, 0, 0, 0)
        valid_end = valid_start + valid_len

        valid_index = train_x.loc[(train_x.index.get_level_values('testing_period') >= valid_start) &
                                  (train_x.index.get_level_values('testing_period') < valid_end)].index.to_list()

        # half of valid sample have data leak from training sets
        train_index = train_x.loc[(train_x.index.get_level_values('testing_period') < (valid_start - valid_len / 2)) |
                                  (train_x.index.get_level_values('testing_period') >= valid_end)].index.to_list()
        gkf = [(train_index, valid_index)]
        return gkf

    def _get_sample_sets(self, train_x, train_y, train_y_cut, test_x, test_y, test_y_cut, gkf) -> List[Dict[str, pd.DataFrame]]:
        """
        cut sample into actual train / valid sets
        """

        sample_set = []

        for train_index, valid_index in gkf:

            new_train_index = self.__remove_nan_y_from_train_index(train_index, train_y)
            new_valid_index = self.__remove_nan_y_from_train_index(valid_index, train_y)

            df_dict = {
                "train_x":       train_x.loc[new_train_index],
                "train_y":       train_y.loc[new_train_index],
                "train_y_final": train_y_cut.loc[new_train_index],
                "valid_x":       train_x.loc[new_valid_index],
                "valid_y":       train_y.loc[new_valid_index],
                "valid_y_final": train_y_cut.loc[new_valid_index],
                "test_x":        test_x,
                "test_y":        test_y.reindex(test_x.index),              # may have NaN -> production latest prediction
                "test_y_final":  test_y_cut.reindex(test_x.index),          # may have NaN -> production latest prediction
            }

            assert all([len(x) > 0 for x in df_dict.values()])
            assert all([v.isnull().sum().sum() == 0 for k, v in df_dict.items() if k not in ["test_y", "test_y_final"]])

            sample_set.append(df_dict)

        return sample_set

    def __remove_nan_y_from_train_index(self, index, train_y: pd.DataFrame) -> pd.MultiIndex:
        """
        remove non-complete y sample for train_index / valid_index -> remove nan Y
        """

        new_index = pd.MultiIndex.from_tuples(set(index) & set(train_y.index))
        return new_index


