import warnings

from scipy.stats import skew
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sqlalchemy import select, and_, not_, func

from utils import (
    read_query,
    models,
    sys_logger,
)

logger = sys_logger(__name__, "DEBUG")


class scaleFundamentalScore:

    sample_interval = 1

    use_cache = True

    def __init__(self, start_date: str = '2016-01-01'):
        self.start_date = start_date

    def get(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame for fundamental ratios
            columns:
                ticker (primary_key):               str, e.g. AAPL.O
                trading_day (primary_key):          dt.date, e.g. 2022-01-01
                currency_code (primary_key):        str, e.g. HKD
                + factor ratios (...):
                        float64, ratios scaled to 0 - 10 as fundamental scores for AI score calculation
                + return (e.g. stock_return_y_...):
                        float64, returns to calculate evaluation results
        """

        if self.use_cache:
            try:
                df = self.__load_cache_fundamental_score()
            except Exception as e:
                df = self._calc_fundamentals_score()
        else:
            df = self._calc_fundamentals_score()

        df = df.reset_index()
        df["trading_day"] = pd.to_datetime(df["trading_day"])

        return df

    def __get_trading_day_list(self):
        """
        Returns
        -------
        trading_day_list: List
            list of Sundays since evaluation start date - now to get ratios
        """

        trading_day_list = pd.date_range(self.start_date, dt.datetime.now(), freq=f'{self.sample_interval}W-SUN')
        trading_day_list = [x.strftime('%Y-%m-%d') for x in trading_day_list]

        return trading_day_list

    def __load_cache_fundamental_score(self):
        """
        load saved fundamental score if use cache
        """

        fundamentals_score = pd.read_pickle(f'cached_fundamental_score_{self.start_date}.pkl')
        return fundamentals_score

    def _calc_fundamentals_score(self) -> pd.DataFrame:
        """
        Download ratios and convert to 0-10 scores
        """

        df = self.__download_fundamentals_score()

        df = df.replace([np.inf, -np.inf], np.nan).copy()
        adj_df = df.groupby(level=['currency_code', 'trading_day']).apply(self._scale_fundamental_scores)

        if self.use_cache:
            adj_df.to_pickle(f'cached_fundamental_score_{self.start_date}.pkl')

        return adj_df

    def __download_fundamentals_score(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame for fundamental ratios
            columns:
                factor ratios (...)
            index:
                pd.MultiIndex([ticker, trading_day, currency_code])
        """

        conditions = [
            models.FactorPreprocessRatio.trading_day.in_(self.__get_trading_day_list()),
            not_(models.FactorPreprocessRatio.ticker.like(".%%")),
        ]

        query = select(*models.FactorPreprocessRatio.__table__.columns, models.Universe.currency_code)\
            .join_from(models.FactorPreprocessRatio, models.Universe)\
            .where(and_(*conditions))

        fundamentals_score = read_query(query)
        fundamentals_score = fundamentals_score.pivot(index=["ticker", "trading_day", "currency_code"],
                                                      columns=["field"], values="value")

        return fundamentals_score

    def _scale_fundamental_scores(self, df) -> pd.DataFrame:
        """
        calculate score (i.e. scale ratio to 0-10) for single currency / pillar within same (currency, trading_day)
        """

        return_col = df.filter(regex='^stock_return_y_').columns.to_list()
        non_return_col = [x for x in df if x not in return_col]

        adj_df = self.__transform_non_return_col(df[non_return_col])

        adj_df = adj_df.fillna(adj_df.mean(axis=1))
        adj_df[return_col] = df[return_col]                  # return cols just keep original value for return calculation

        return adj_df

    def __transform_non_return_col(self, non_return_df) -> pd.DataFrame:
        """
        transform original score -> 0 - 10 scale
        """

        pipe = Pipeline(steps=[
            ('std', StandardScaler()),
            ('trim', TrimOutlierTransformer(skew_trh=5, std_trh=2)),
            ('robust', RobustScaler()),
            ('minmax', MinMaxScaler(feature_range=(0, 10)))])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            adj_fundamentals = 10 - pipe.fit_transform(non_return_df)    # we use 10 - minmax(0, 10) because factor model is S-L

        # fillna for scores with average for that factor
        adj_fundamentals = pd.DataFrame(adj_fundamentals, index=non_return_df.index, columns=non_return_df.columns)

        return adj_fundamentals


class TrimOutlierTransformer(BaseEstimator, TransformerMixin):
    """
    trim outliers by
    - clip with (1%, 99%) range
    - PowerTransformer for ratio with skewness > 5
    """

    def __init__(self, skew_trh=5, std_trh=2):
        self.skew_trh = skew_trh
        self.std_trh = std_trh

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # clip for (.01 - .99)
        X = np.apply_along_axis(lambda s: s.clip(np.nanpercentile(s, .01), np.nanpercentile(s, .99)), axis=1, arr=X)
        power_X = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(X)

        # if skewness > 5 then do power_transform
        s = skew(X, nan_policy='omit')
        X = np.where((s < -self.skew_trh) | (s > self.skew_trh), power_X, X)

        # clip be (mean +/- 2*std)
        X = np.apply_along_axis(
            lambda x: np.clip(x, np.nanmean(X, axis=0) - self.std_trh * np.nanstd(X, axis=0),
                                 np.nanmean(X, axis=0) + self.std_trh * np.nanstd(X, axis=0)), axis=1, arr=X)

        return X


# Download: DataFrame for [factor_formula]
# factor_formula = read_table(global_vars.factors_formula_table, global_vars.db_url_alibaba_prod)
# factor_formula = factor_formula.set_index(['name'])
# calculate_column = list(factor_formula.loc[factor_formula['scaler'].notnull()].index)
# calculate_column = sorted(set(calculate_column) & set(fundamentals_score.columns))
#
# # filter [fundamentals_score] for calculation scores
# label_col = ['ticker', 'trading_day', 'currency_code'] + fundamentals_score.filter(
#     regex='^stock_return_y_').columns.to_list()
# fundamentals = fundamentals_score[label_col + calculate_column]