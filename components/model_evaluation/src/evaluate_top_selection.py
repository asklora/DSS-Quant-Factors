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
from typing import List, Dict

from utils import (
    read_query,
    models,
    sys_logger,
    err2slack,
    recreate_engine,
    upsert_data_to_database,
)
from .calculation_backtest_score import scaleFundamentalScore
from .load_eval_configs import load_eval_config

logger = sys_logger(__name__, "DEBUG")


class cleanEval:
    """
    download eval Table for top ticker evaluation
    """

    eval_start_date = dt.datetime(2000, 1, 1)

    def __init__(self, name_sql: str):
        self.name_sql = name_sql

    def get(self):
        """
        download eval Table for top ticker evaluation
        """

        conditions = [
            models.FactorBacktestEval.name_sql == self.name_sql,
            models.FactorBacktestEval.updated > self.eval_start_date,
        ]

        query = select(models.FactorBacktestEval).where(and_(*conditions))
        eval_df = read_query(query)

        return eval_df


def get_industry_name_map() -> Dict[str, str]:
    """
    industry_name_map = e.g. {AAPL.O (ticker): Technology (industry_name)}
    """

    query = select(models.Universe.ticker,
                   models.ICBCodeExplanation.name_4).join_from(models.Universe,
                                                               models.ICBCodeExplanation)
    industry_name_map = read_query(query).set_index(["ticker"])[
        "name_4"].to_dict()

    return industry_name_map


class EvalTop:
    base_score = 5
    average_days = -7

    eval_config_opt_columns = \
        [x.name for x in models.FactorBacktestEval.config_opt_columns]

    # select evaluation factors within each group
    eval_config_define_columns = \
        [x.name for x in models.FactorBacktestEval.train_config_define_columns] + \
        [x.name for x in models.FactorBacktestEval.base_columns] + \
        [x.name for x in models.FactorBacktestEval.eval_config_define_columns]
    top_config_define_columns = \
        [x.name for x in models.FactorFormulaEvalConfig.eval_top_config_columns]

    save_cache = True

    n_top_ticker_list = [-10, -50, 50, 10, 3]
    industry_name_map = get_industry_name_map()

    def __init__(self,
                 name_sql: str,
                 processes: int,
                 eval_df: pd.DataFrame = None):
        self.name_sql = name_sql
        self.weeks_to_expire = int(name_sql.split('_')[0][1:])
        self.processes = processes

        logger.debug("load clean eval")
        if eval_df is None:
            self._eval_df = cleanEval(name_sql=self.name_sql).get()
        else:
            self._eval_df = eval_df

        logger.debug("load eval config")
        self._all_groups = load_eval_config(self.weeks_to_expire)
        self._score_df = None

    def write_top_select_eval(self):
        """
        Used within multiprocess to
        - calculate selected factors
        - calculate backtest AI scores with factors selected
        """

        logger.debug("get scaleFundamentalScore")
        # fundamental ratio scores
        self._score_df = scaleFundamentalScore(
            # start_date=self._eval_df["testing_period"].min()).get()
            start_date='2017-01-01').get()

        # with closing(mp.Pool(processes=self.processes,
        #                      initializer=recreate_engine)) as pool:
        results = []
        for group in self._all_groups:
            results.append(self._select_and_score(*group))
            # results = pool.starmap(self._select_and_score, self._all_groups)

        # df for all AI scores
        pillar_score_df = pd.concat([e for e in results if e is not None],
                                    axis=0)

        final_score_df = self.__calculate_final_score(
            pillar_score_df).reset_index()
        top_eval_df = self._final_score_eval(final_score_df)

        top_eval_df["name_sql"] = self.name_sql
        top_eval_df = top_eval_df[
            [x.name for x in models.FactorBacktestTop.__table__.columns]]

        if self.save_cache:
            top_eval_df.to_pickle(f"top_eval_{self.name_sql}.pkl")

        upsert_data_to_database(top_eval_df,
                                models.FactorBacktestTop.__tablename__,
                                how='update')

        return top_eval_df, final_score_df

    def write_latest_select(self):
        """
        combine results in table [FactorBacktestEval] for different configurations

        Returns
        -------
        select: pd.DataFrame
            Columns:
                weeks_to_expire:        int, [FactorResultSelect] primary key, e.g. 4
                currency_code:          str, [FactorResultSelect] primary key, e.g. HKD
                pillar:                 str, [FactorResultSelect] primary key, e.g. cluster
                max/min_factor:         list, list of factors select as good / bad considering all configs
                max/min_factor_trh:     float64, threshold used to select factors
                max/min_factor_extra:   list, list of factors select as extra good / bad considering all configs
                updated:                Timestamp, [FactorResultSelectHistory] primary key
        """

        with closing(mp.Pool(processes=self.processes,
                             initializer=recreate_engine)) as pool:
            results = pool.starmap(self._select, self._all_groups)

        # df for all AI scores
        select_df = pd.concat([e for e in results if e is not None], axis=0)
        select_df = select_df.groupby(["currency_code"], as_index=False).apply(
            lambda x: x.loc[x["trading_day"] == x["trading_day"].max()])

        select_df["updated"] = dt.datetime.now()
        select_df = select_df[
            [x.name for x in models.FactorResultSelect.__table__.columns]]

        # update [FactorResultSelect]
        upsert_data_to_database(select_df,
                                models.FactorResultSelect.__tablename__,
                                how='update')

        # update [FactorResultSelectHistory]
        upsert_data_to_database(select_df,
                                models.FactorResultSelectHistory.__tablename__,
                                how='update')

        return select_df

    def _select_and_score(self, kwargs):
        """
        Used within multiprocess to
        - calculate selected factors
        - calculate backtest AI scores with factors selected
        """

        logger.debug(f"select_and_score: {kwargs}")

        sample_df = self.__filter_sample(df=self._eval_df, **kwargs)

        # i.e. configuration use data not trained by current iteration
        if len(sample_df) == 0:
            return

        select_df = self._get_minmax_factors_df(sample_df,
                                                **kwargs).reset_index()
        select_df = self.__convert_trading_day(select_df)

        # select factors + fundamental score -> final scores
        pillar_df = self.__calculate_pillar_score(select_df, **kwargs)
        pillar_df = pillar_df.assign(**kwargs)

        return pillar_df

    def _select(self, kwargs):
        """
        Used within multiprocess to
        - calculate selected factors only
        """

        sample_df = self.__filter_sample(df=self._eval_df, **kwargs)

        # i.e. configuration use data not trained by current iteration
        if len(sample_df) == 0:
            return

        select_df = self._get_minmax_factors_df(sample_df,
                                                **kwargs).reset_index()
        select_df = self.__convert_trading_day(select_df)
        select_df["eval_metric"] = kwargs["eval_top_metric"]

        return select_df

    def __filter_sample(self, df: pd.DataFrame, pillar: str,
                        currency_code: str, **kwargs) -> pd.DataFrame:
        """
        filter eval table for sample for certain currency / pillar;

        Returns
        -------
        sample_df: pd.DataFrame
            filter all evaluation results for samples with defined [currency_code] & [pillar]
        """

        # all cluster pillar will calculate together
        if pillar != "cluster":
            sample_df = df.loc[(df["currency_code"] == currency_code) & (
                        df["pillar"] == pillar)].copy(1)
        else:
            sample_df = df.loc[(df["currency_code"] == currency_code) & (
                df["pillar"].str.startswith("pillar"))].copy(1)
            sample_df["pillar"] = pillar

        return sample_df

    def __convert_trading_day(self, sample_df) -> pd.DataFrame:
        """
        convert trading_day (date to select factor = date of fundamental scores) = testing_period + n weeks
        """
 
        sample_df["trading_day"] = \
            pd.to_datetime(sample_df["testing_period"]) + \
            pd.tseries.offsets.DateOffset(weeks=(self.weeks_to_expire))
        sample_df ['trading_day'] = sample_df['trading_day'] - sample_df['average_days'].apply(lambda x: pd.Timedelta(x,unit='D'))
        sample_df = sample_df.drop(columns=["testing_period"])

        return sample_df

    def _get_minmax_factors_df(self, df: pd.DataFrame, eval_top_metric: str,
                               **kwargs) -> pd.DataFrame:
        """
        return DataFrame with (trading_day, list of good / bad factor on trading_day) for certain pillar

        Returns
        -------
        df_select_agg
            Columns:
                max/min_factor:         list, list of factors select as good / bad considering all configs
                max/min_factor_trh:     float64, threshold used to select factors 
                max/min_factor_extra:   list, list of factors select as extra good / bad considering all configs

        """

        df_agg = self.__calc_agg_returns(df)
        df_select = self.__filter_best_configs_by_history(
            df_agg, eval_top_metric=eval_top_metric, **kwargs)

        if eval_top_metric == "max_ret":
            df_select_agg = self.__select_final_factor(df_select, "max_factor")
            df_select_agg = df_select_agg.assign(min_factor=np.nan,
                                                 min_factor_trh=np.nan,
                                                 min_factor_extra=np.nan)
        else:
            df_select_max = self.__select_final_factor(df_select, "max_factor")
            df_select_min = self.__select_final_factor(df_select, "min_factor")
            df_select_agg = pd.concat([df_select_max, df_select_min], axis=1)

        return df_select_agg

    def __select_final_factor(self, df: pd.DataFrame,
                               select_col: str) -> pd.DataFrame:
        """
        Returns
        -------
        factor_select: pd.DataFrame
            Columns:
                [select_col]:         list, list of factors always selected most time of configs
                [select_col]_trh:     float64, min_occur_pct used in selecting factors
                [select_col]_extra:   list, list of factors always selected among configs
            Index:
                MultiIndex with eval_config_define_columns
        """

        factor_all = df.groupby(self.eval_config_define_columns)[
            select_col].agg(["sum", "count"])
        factor_select = factor_all.apply(self.__select_each_row_try_pct, axis=1)
        factor_select.columns = [x.replace("*", select_col) for x in
                                 factor_select.columns]

        return factor_select

    def __calc_agg_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        combine factor selection / average returns for different pillars if uses [cluster]

        i.e. if input DataFrame
            | pillar   | max_factor            | max_ret |
            |----------|-----------------------|---------|
            | pillar_0 | ["roic", "asset_1yr"] | 0.80    |
            | pillar_1 | ["vol_0_30"]          | 0.40    |

        then output DataFrame
            | pillar   | max_factor                        | max_ret |
            |----------|-----------------------------------|---------|
            | cluster  | ["roic", "asset_1yr", "vol_0_30"] | 0.60    |

        Returns
        -------
        df: pd.DataFrame
            Columns:
                max_factor:     list, all good factors selected in group
                min_factor:     list, all bad factors selected in group
                max_ret:        float64, average factor premiums for good factors
                net_ret:        float64, average premiums for good factors - bad factors
                avg_ret:        float64, average of max_ret & net_ret
        """

        df['net_ret'] = df['max_ret'] - df['min_ret']
        df['avg_ret'] = (df['max_ret'] + df['net_ret']) / 2
        agg_func = {'max_factor': 'sum', 'min_factor': 'sum', 'max_ret': 'mean',
                    'net_ret': 'mean', 'avg_ret': 'mean'}
        df_agg = df.groupby(
            self.eval_config_define_columns + self.eval_config_opt_columns).agg(
            agg_func).reset_index()

        return df_agg

    def __filter_best_configs_by_history(self, df: pd.DataFrame,
                                         eval_top_metric: str,
                                         eval_top_backtest_period: int,
                                         eval_top_n_configs: int,
                                         **kwargs) -> pd.DataFrame:
        """
        Only keep records (i.e. configurations) within top [eval_top_n_configs] with average [eval_top_metric]
        over past [eval_top_backtest_period];

        e.g. training tried 32 configurations on defined group (testing_period = 2022-04-24, currency = USD,
        pillar = value, weeks_to_expire = 8)

        * configurations are defined in model_training > loadTrainConfig > _auto_select_options.

        Among 32 configurations, we will
        - If [eval_top_backtest_period = 12], backtest_period = 2021-05-02 ~ 2022-03-27 (e.g. sample interval = 4)
        - If [eval_top_metric = net_ret], calculate R as average over backtest periods
        - If [eval_top_n_configs = 0.2], calculate R-thresholds for good configuration with 0.8 quantile R
        - keep configurations with R > R-thresholds

        """

        # calculate different config rolling average return
        rolling_groupby = list(
            set(self.eval_config_define_columns + self.eval_config_opt_columns) - {
                "testing_period"})

        df = df.sort_values(
            by=rolling_groupby + ["testing_period"]).reset_index(drop=True)
        df['rolling_ret'] = df.groupby(rolling_groupby)[
            eval_top_metric].rolling(
            eval_top_backtest_period, closed='left').mean().values
        df = df.dropna(subset=[
            "rolling_ret"])  # remove sample don't have enough history in training

        # filter for configuration with rolling_ret (historical) > quantile
        df['trh'] = df.groupby(self.eval_config_define_columns)[
            'rolling_ret'].transform(
            lambda x: x.quantile(1 - eval_top_n_configs)).values

        df_select = df.loc[df['rolling_ret'] >= df['trh']]

        logger.debug(f"df_agg_select shape: {df.shape} -> {df_select.shape}")

        return df_select

    def __select_each_row_try_pct(self, row: pd.Series,
                                  init_min_occur_pct: float = 0.8):
        """
        when no factor can be selected with [init_min_occur_pct], try with min_occur_pct=-0.1
        """

        min_occur_pct = init_min_occur_pct
        while min_occur_pct > 0:
            results = self.__select_each_row_by_pct(row=row,
                                                    min_occur_pct=min_occur_pct)
            if type(results) != type(None):
                return results
            min_occur_pct -= 0.1

        raise Exception(f"No factor selected for row {row}!")

    def __select_each_row_by_pct(self, row: pd.Series,
                                 min_occur_pct: float = 0.8,
                                 min_select_pct: float = 0.33,
                                 min_extra_occur_pct: float = 0.9) -> pd.Series:
        """
        Parameters
        ----------
        min_occur_pct :
            among n selected configs, select factors occurs over 80% of all times
        min_extra_occur_pct :
            among n selected configs for extra factors
        min_select_pct :
            among n factors, selection is only valid when select over 50% of the factors
        """

        # count occurrence of factors each testing_period
        count_occur = dict(Counter(row["sum"]))
        n_min_occur = min_occur_pct * row["count"]
        n_min_extra_occur = min_extra_occur_pct * row["count"]
        n_min_select = min_select_pct * len(set(row["sum"]))

        # keep factor with (e.g. > .5 occurrence in selected configs)
        results = {
            "*": [k for k, v in count_occur.items() if v >= n_min_occur],
            "*_trh": round(min_occur_pct, 2),
            "*_extra": [k for k, v in count_occur.items() if
                        v >= n_min_extra_occur]
            # extra if all config select factor [X]
        }

        if len(results["*"]) >= n_min_select:
            return pd.Series(results)

    def __calculate_pillar_score(self, 
                                 df: pd.DataFrame, 
                                 pillar: str, 
                                 currency_code: str,
                                 eval_top_metric: str, **kwargs):

        final_score_df_list = []

        for idx, r in df.iterrows():

            # we use * to label heuristically reversed factors
            heuristic_neg_cols = [x.strip('*')
                                  for i in ["max_factor", "min_factor"]
                                  if isinstance(r[i], list)
                                  for x in r[i] if x[-1] == "*"]
            heuristic_neg_cols = self._score_df.filter(heuristic_neg_cols)\
                .columns.to_list()

            g = self.__filter_sample_score_df(score_df=self._score_df,
                                              trading_day=r["trading_day"],
                                              currency_code=currency_code,
                                              reverse_cols=heuristic_neg_cols)

            if len(g) == 0:
                continue

            # make sure all factor use raw name for ratio data
            for i in ["max_factor", "min_factor",
                      "max_factor_extra", "min_factor_extra"]:
                if isinstance(r[i], list):
                    r[i] = [x.strip('*') for x in r[i]]

            for suffix in ["", "_extra"]:
                if eval_top_metric == "max_ret":
                    g[f'{pillar}_score{suffix}'] = g[
                        r[f'max_factor{suffix}']].mean(axis=1)
                elif eval_top_metric == "avg_ret":
                    g[f'{pillar}_score{suffix}'] = \
                        self.base_score * 0.5 + \
                        g[r[f'max_factor{suffix}']].mean(axis=1) - \
                        g[r[f'min_factor{suffix}']].mean(axis=1) * 0.5
                elif eval_top_metric == "net_ret":
                    g[f'{pillar}_score{suffix}'] = \
                        self.base_score + \
                        g[r[f'max_factor{suffix}']].mean(axis=1) - \
                        g[r[f'min_factor{suffix}']].mean(axis=1)
                else:
                    raise Exception(
                        f"Wrong evaluation metric: {eval_top_metric}!")

            g_score = g[["trading_day", "ticker", "return", f"{pillar}_score",
                         f"{pillar}_score_extra"]]

            if g_score[f"{pillar}_score"].isnull().sum():
                raise Exception(f"Found NaN in backtest [{pillar}_scores]!")

            # define config used for factors selected when calculating scores
            g_score = g_score.assign(
                **{k: v for k, v in r.to_dict().items() if "factor" not in k})
            final_score_df_list.append(g_score)

        if len(final_score_df_list) > 0:
            final_score_df = pd.concat(final_score_df_list, axis=0)
            return final_score_df

    def __filter_sample_score_df(self, score_df: pd.DataFrame,
                                 trading_day: dt.datetime,
                                 currency_code: str,
                                 reverse_cols: List[str],
                                 missing_penalty_pct: float = 0.9):
        """
        filter fundamental score table for sample for certain
        trading_day / currency;

        Parameters
        ----------
        missing_penalty_pct :
            fillna with average for all tickers * missing_penalty_pct
            (default = 0.9)

        Returns
        -------
        sample_df: pd.DataFrame
            filter all evaluation results for samples with defined
            [currency_code] & [trading_day]

        """
        g = score_df.loc[(score_df['trading_day'] == trading_day) &
                         (score_df['currency_code'] == currency_code)].copy(1)
        if len(reverse_cols) > 0:
            g[reverse_cols] = (10 - g[reverse_cols]).values

        col = f'stock_return_y_w{self.weeks_to_expire}_d{self.average_days}'
        g['return'] = g[col].copy(1)
        g = g.dropna(subset=["return"])
        g = g.fillna(g.mean(numeric_only=True) * missing_penalty_pct)

        return g

    def __calculate_quantile_return_for_each_trading_day(self, df_agg):
        """This function takes in df_agg and calculate quantile return for each testing_period and and currency_code. Steps include:
        1. [Quantile]

        Args:
            df_agg (pd.DataFrame): The aggregated dataframe which contains the final AI score for each ticker in each currency code for each testing period. 
        """
        quantile_without_label =df_agg.reset_index().set_index(['weeks_to_expire','currency_code','trading_day']).groupby(['weeks_to_expire','currency_code','trading_day'])['ai_score'].transform(lambda x : pd.qcut(x, q=10, labels=False, duplicates='drop'))
        quantile_with_label = df_agg.reset_index().set_index(['weeks_to_expire','currency_code','trading_day']).groupby(['weeks_to_expire','currency_code','trading_day'])['ai_score'].transform(lambda x : pd.qcut(x,q=10, duplicates='drop'))
        quantile_with_label=quantile_with_label.reset_index().drop_duplicates().set_index(['weeks_to_expire','currency_code','trading_day']).sort_index(level=['currency_code'])
        quantile_without_label=quantile_without_label.reset_index().drop_duplicates().set_index(['weeks_to_expire','currency_code','trading_day']).sort_index(level=['currency_code'])
        
        ret = []
        quantiles = []
        weeks_to_expire = []
        currency_code = []
        trading_day=[]
        name_sql = []

        for idx in quantile_with_label.index.unique():
            data = df_agg.reset_index()
            data = data.loc[(data['weeks_to_expire']==idx[0])&(data['currency_code']==idx[1])&(data['trading_day']==idx[2])]
            
            for q in quantile_with_label.loc[(quantile_with_label.index.get_level_values(0)==idx[0])&(quantile_with_label.index.get_level_values(1)==idx[1])&(quantile_with_label.index.get_level_values(2)==idx[2])]['ai_score']:
                
                qtl = quantile_without_label.loc[(quantile_with_label['ai_score']==q)& \
                (quantile_with_label.index.get_level_values(0)==idx[0])& \
                (quantile_with_label.index.get_level_values(1)==idx[1])& \
                (quantile_with_label.index.get_level_values(2)==idx[2])]['ai_score'].unique()[0]

                retn = data.loc[data['ai_score'].between(left=q.left,right=q.right)]['return'].mean()
                ret.append(retn)
                name_sql.append(self.name_sql)
                quantiles.append(qtl)
                weeks_to_expire.append(idx[0])
                currency_code.append(idx[1])
                trading_day.append(idx[2])

        summary = pd.DataFrame({'name_sql':name_sql,'average_return':ret,'quantiles':quantiles,'weeks_to_expire':weeks_to_expire,'currency_code':currency_code,'trading_day':trading_day})
        summary = summary.sort_values(['currency_code','quantiles'],ascending=[False,False])

        upsert_data_to_database(summary,table=models.FactorBacktestQuantile.__tablename__,how="update", set_index_pk=True)

    def __calculate_final_score(self, df):
        """
        combine different pillar row and calculate final score;

        calculate final scores / extra scores with average of all pillar scores;

        e.g. input dataframe:
            | value_score | momentum_score | quality_score |
            |-------------|----------------|---------------|
            | 2           |                |               |
            |             | 3              |               |
            |             |                | 4             |

        output dataframe:
            | value_score | momentum_score | quality_score | ai_score |
            |-------------|----------------|---------------|----------|
            | 2           | 3              | 4             | 3        |

        We combine all defined training / testing configuration to combine different pillar with different configs;
        """

        groupby_col = ["ticker", "weeks_to_expire", "currency_code",
                       "trading_day"]
        df_agg = df.groupby(groupby_col).mean()
        df_agg["extra_score"] = df_agg.filter(regex='_score_extra$').mean(
            axis=1)
        df_agg['ai_score'] = df_agg.filter(regex='_score$').mean(axis=1)
        
        self.__calculate_quantile_return_for_each_trading_day(df_agg)

        return df_agg

    def _final_score_eval(self, df):
        """
        Evaluate: calculate return for top 10 score / mode industry

        Returns
        -------
        pd.DataFrame() for top backtest selected ticker returns

        """

        groupby_col = ["weeks_to_expire", "currency_code", "trading_day"]

        eval_df_list = []
        df["industry_name"] = df["ticker"].map(self.industry_name_map)
        for i in self.n_top_ticker_list:
            eval_n_df = df.groupby(groupby_col).apply(self.__eval_best,
                                                      best_n=i).reset_index()
            eval_n_df["top_n"] = i
            eval_n_df = eval_n_df.drop(columns=["group_idx"])
            eval_df_list.append(eval_n_df)

        # concat different top (n) selection ticker results
        eval_df = pd.concat(eval_df_list, axis=0)

        return self.__clean_eval_df(eval_df)

    def __eval_best(self, g_all, best_n=10, best_col='ai_score'):
        """
        evaluate score history with top 10 score return & industry
        """

        top_ret = {}
        if best_n > 0:
            g = g_all.set_index('ticker').nlargest(best_n, columns=[best_col],
                                                   keep='all')
        else:
            g = g_all.set_index('ticker').nsmallest(-best_n, columns=[best_col],
                                                    keep='all')

        top_ret["ticker_count"] = g.shape[0]
        top_ret["ret"] = g['return'].mean()
        top_ret["mode"] = g[f"industry_name"].mode()[0]
        top_ret["mode_count"] = np.sum(
            g[f"industry_name"] == top_ret["mode"]).item()
        top_ret["pos_pct"] = np.sum(g['return'] > 0) / len(g)
        top_ret["bm_ret"] = g_all['return'].mean()
        top_ret["bm_pos_pct"] = np.sum(g_all['return'] > 0) / len(g_all)
        top_ret["tickers"] = list(g.index)

        top_ret = pd.Series(top_ret).to_frame().T
        top_ret.index.name = "group_idx"

        return top_ret

    def __clean_eval_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        change data types for DB writing
        """

        df['trading_day'] = pd.to_datetime(df['trading_day'])
        df['ret'] = pd.to_numeric(df['ret']).round(4) * 100
        df['pos_pct'] = np.where(df['ret'].isnull(), None,
                                 pd.to_numeric(df['pos_pct']).round(2) * 100)
        df['bm_ret'] = pd.to_numeric(df['bm_ret']).round(4) * 100
        df['bm_pos_pct'] = np.where(df['bm_ret'].isnull(), None,
                                    pd.to_numeric(df['bm_pos_pct']).round(
                                        2) * 100)
        df["trading_day"] = df["trading_day"].dt.date
        df["updated"] = dt.datetime.now()

        return df
