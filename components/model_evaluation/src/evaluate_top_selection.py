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
)

logger = sys_logger(__name__, "DEBUG")


class cleanEval:
    """
    download eval Table for top ticker evaluation directly
    i.e. when only --eval_top without --eval_factor
    """

    def __init__(self, weeks_to_expire):
        self.weeks_to_expire = weeks_to_expire

    def _download_eval(self, name_sql, weeks_to_expire):
        """
        TODO: rewrite after write eval table again
        download eval Table for top ticker evaluation
        """

        # read config eval history
        query = select(models.FactorBacktestEval).where(models.FactorBacktestEval == self.weeks_to_expire)
        eval_df = read_query(query)

        # for production: only use history with same data_configs as production
        if "debug" in name_sql:
            query_conf = f"SELECT * FROM {config_train_table} WHERE is_active AND weeks_to_expire = {weeks_to_expire}"

            # remove [pred_currency] columns because [config_train] table concatenate all available pred_currency
            prod_train_config = read_query(query_conf).drop(columns=["is_active", "last_finish", "pred_currency"])
            prod_train_config_col = [f"_{x}" for x in prod_train_config]
            prod_train_config.columns = prod_train_config_col

            logger.debug(f'eval_df columns: {eval_df.columns.to_list()}')
            logger.debug(f'prod_train_config columns: {prod_train_config.columns.to_list()}')

            logger.debug(f'before eval_df shape: {eval_df.shape}')
            eval_df_noncluster = eval_df.loc[eval_df["_pillar"] != "cluster"].merge(
                prod_train_config, on=prod_train_config_col, how="left")
            eval_df_cluster = eval_df.loc[eval_df["_pillar"] == "cluster"].merge(
                prod_train_config.drop(columns=['_pillar']), on=[x for x in prod_train_config_col if x != "_pillar"], how="left")
            eval_df = eval_df_noncluster.append(eval_df_cluster)
            logger.debug(f'after eval_df shape: {eval_df.shape}')

        return eval_df


class cleanFundamentalScore:

    # if calculate backtest score & eval top selections
    if self.eval_top:
        try:
            adj_fundamentals = pd.read_pickle("adj_fundamentals.pkl")  # TODO: remove in production
        except Exception as e:
            print(e)
            fundamentals, factor_formula = get_fundamental_scores(start_date=pred_start_testing_period,
                                                                  sample_interval=1)
            adj_fundamentals = fundamentals.groupby(['currency_code', 'trading_day']).apply(scale_fundamental_scores)
            adj_fundamentals.to_pickle("adj_fundamentals.pkl")

        self.adj_fundamentals = adj_fundamentals.reset_index()
        # logger.debug(f"fundamental from: {adj_fundamentals['trading_day'].min()} to {adj_fundamentals['trading_day'].max()}")


class evalTop:

    def __init__(self):

        if kwargs["pillar"] == "cluster":
            eval_df = self.eval_df_history.loc[(self.eval_df_history["_pred_currency"] == kwargs["pred_currency"]) &
                                               (self.eval_df_history["_pillar"].str.startswith("pillar"))].copy(1)
        else:
            eval_df = self.eval_df_history.loc[(self.eval_df_history["_pred_currency"] == kwargs["pred_currency"]) &
                                               (self.eval_df_history["_pillar"] == kwargs["pillar"])].copy(1)


        # Based on evaluation df calculate DataFrame for list of selected factors (pillar & extra)
        select_history_df, select_df = self.__get_minmax_factors(eval_df, **kwargs)
        breakpoint()

        if self.eval_top:
            score_df = self.score_(df=select_history_df, **kwargs)
        else:
            score_df = pd.DataFrame()


    # ------------------------------------ Calculate good/bad factor -------------------------------------------

    def __get_minmax_factors(self, df, pillar, weeks_to_expire, eval_top_metric, eval_top_n_configs,
                             eval_top_backtest_period, **kwargs):
        """ return DataFrame with (trading_day, list of good / bad factor on trading_day) for certain pillar """

        factor_eval = df.copy(1)

        # testing_period = start date of tests sets (i.e. data cutoff should be 1 period later / last return = 2p later)
        factor_eval["trading_day"] = pd.to_datetime(factor_eval["_testing_period"]) + pd.tseries.offsets.DateOffset(
            weeks=weeks_to_expire)
        factor_eval = factor_eval.drop(columns=["_testing_period"])
        factor_eval["_pillar"] = pillar  # for cluster pillar: change "_pillar" -> "cluster" i.e. combine all cluster

        factor_eval['net_ret'] = factor_eval['max_ret'] - factor_eval['min_ret']
        factor_eval['avg_ret'] = (factor_eval['max_ret'] + factor_eval['net_ret']) / 2
        logger.debug(f"factor_eval shape: {factor_eval.shape}")

        # get config cols: group = select within each group; select = config to select top n
        group_config_col = [x for x in factor_eval.filter(regex='^_[a-z]').columns.to_list() if x!='_name_sql']
        logger.debug(f"group_config_col: {group_config_col}")

        select_config_col = factor_eval.filter(regex='^__').columns.to_list()
        logger.debug(f"select_config_col: {select_config_col}")

        n_config = factor_eval[select_config_col].drop_duplicates().shape[0]
        n_select_config = np.ceil(n_config * eval_top_n_configs)
        logger.debug(factor_eval.isnull())

        # in case of cluster pillar combine different cluster selection
        agg_func = {'max_factor': 'sum', 'min_factor': 'sum', 'max_ret': 'mean', 'net_ret': 'mean', 'avg_ret': 'mean'}
        factor_eval_agg = factor_eval.groupby(group_config_col + select_config_col + ["trading_day"]).agg(
            agg_func).reset_index()
        logger.debug(f"factor_eval_agg shape: {factor_eval_agg.shape}")

        n_select_factor = factor_eval_agg['max_factor'].apply(lambda x: len(x)).mean() * 0.5

        # calculate different config rolling average return
        factor_eval_agg['rolling_ret'] = factor_eval_agg.groupby(group_config_col + select_config_col)[
            eval_top_metric].rolling(eval_top_backtest_period, closed='left').mean().values

        # filter for configuration with rolling_ret (historical) > quantile
        factor_eval_agg['trh'] = factor_eval_agg.groupby(group_config_col + ["trading_day"])['rolling_ret'].transform(
            lambda x: np.quantile(x, 1 - eval_top_n_configs)).values
        factor_eval_agg_select = factor_eval_agg.loc[factor_eval_agg['rolling_ret'] >= factor_eval_agg['trh']]
        logger.debug(f"factor_eval_agg_select shape: {factor_eval_agg_select.shape}")

        # count occurrence of factors each testing_period & keep factor with (e.g. > .5 occurrence in selected configs)
        if eval_top_metric == "max_ret":
            select_col = ['max_factor']
        else:
            select_col = ['max_factor', 'min_factor']  # i.e. net return
        period_agg = factor_eval_agg_select.groupby(group_config_col + ["trading_day"])[select_col].sum()
        logger.debug(period_agg.columns.to_list())
        period_agg_filter_counter = period_agg[select_col].applymap(lambda x: dict(Counter(x)))
        logger.debug(period_agg_filter_counter)

        # create final filter factor list table
        period_agg_filter = pd.DataFrame(index=period_agg.index,
                                         columns=select_col + [x + i for x in select_col for i in ["_trh", "_extra"]])
        for col in select_col:
            min_occur_pct = 0.8
            while any(period_agg_filter[col + "_trh"].isnull()) and min_occur_pct > 0:

                # min/max factor for pillar (i.e. value / momentum / quality / cluster)
                min_occur = n_select_config * min_occur_pct
                temp = period_agg[col].apply(lambda x: [k for k, v in dict(Counter(x)).items() if v >= min_occur])
                temp = pd.DataFrame(
                    temp[temp.apply(lambda x: len(x) >= n_select_factor)])
                temp[col + "_trh"] = round(min_occur_pct, 2)
                period_agg_filter = period_agg_filter.fillna(temp)

                # min/max factor for [extra] -> later will be combined across pillars
                temp2 = period_agg[col].apply(
                    lambda x: [k for k, v in dict(Counter(x)).items() if v == n_select_config])
                temp2.name = f"{col}_extra"
                period_agg_filter = period_agg_filter.fillna(pd.DataFrame(temp2))

                min_occur_pct -= 0.1

        period_agg_filter[select_col] = period_agg_filter[select_col].applymap(lambda x: [] if type(x)!=type([]) else x)
        period_agg_count = period_agg_filter[select_col].applymap(lambda x: len(x))
        logger.debug(period_agg_count)

        period_agg_filter = period_agg_filter.reset_index()
        logger.debug(f"period_agg_filter columns: {period_agg_filter.columns.to_list()}")
        period_agg_filter['_eval_top_metric'] = eval_top_metric
        period_agg_filter['_eval_top_n_configs'] = eval_top_n_configs
        period_agg_filter['_eval_top_backtest_period'] = eval_top_backtest_period

        period_agg_filter_current = period_agg_filter.sort_values(by=["trading_day"]).groupby(
            group_config_col, as_index=False).last()

        return period_agg_filter, period_agg_filter_current

        # TODO (LATER): rewrite add_factor_penalty -> [factor_rank] past period factor prediction
        # if self.add_factor_penalty:
        #     factor_rank['z'] = factor_rank['actual_z'] / factor_rank['pred_z']
        #     factor_rank['z_1'] = factor_rank.sort_values(['testing_period']).groupby(['group', 'factor_name'])['z'].shift(
        #         1).fillna(1)
        #     factor_rank['pred_z'] = factor_rank['pred_z'] * np.clip(factor_rank['z_1'], 0, 2)
        #     factor_rank['factor_weight_adj'] = factor_rank.groupby(by=['group', 'testing_period'])['pred_z'].transform(
        #         lambda x: pd.qcut(x, q=3, labels=False, duplicates='drop')).astype(int)
        #     factor_rank['factor_weight_org'] = factor_rank['factor_weight'].copy()
        #     conditions = [
        #         (factor_rank['factor_weight'] == factor_rank['factor_weight'].max())
        #         & (factor_rank['factor_weight_adj'] == factor_rank['factor_weight_adj'].max()),
        #         (factor_rank['factor_weight'] == 0) & (factor_rank['factor_weight_adj'] == 0)
        #     ]
        #     factor_rank['factor_weight'] = np.select(conditions, [2, 0], 1)
        #
        #     factor_rank.to_csv('factor_rank_adj.csv', index=False)

    # ---------------------------------- Calculate backtest scores --------------------------------------------

    def score_(self, df, pillar=None, pred_currency=None, eval_top_metric='max_ret', **kwargs):
        """ convert eval table to top table """

        score_df_list = []

        for idx, r in df.iterrows():
            try:
                g = self.adj_fundamentals.loc[(self.adj_fundamentals['trading_day'] == r["trading_day"]) &
                                              (self.adj_fundamentals['currency_code'] == pred_currency)].copy()
                g['return'] = g[f'stock_return_y_w{r["_weeks_to_expire"]}_d{r["_average_days"]}']
                for suffix in ["", "_extra"]:
                    if eval_top_metric == "max_ret":
                        g[f'{pillar}_score{suffix}'] = self.base_score + g[r[f'max_factor{suffix}']].mean(axis=1)
                    elif eval_top_metric == "avg_ret":
                        g[f'{pillar}_score{suffix}'] = self.base_score + g[r[f'max_factor{suffix}']].mean(axis=1) - \
                                                       0.5*g[r[f'min_factor{suffix}']].mean(axis=1)
                    else:
                        g[f'{pillar}_score{suffix}'] = self.base_score + g[r[f'max_factor{suffix}']].mean(axis=1) - \
                                                       g[r[f'min_factor{suffix}']].mean(axis=1)
            except Exception as e:
                print(e)
            g_score = g[["trading_day", "ticker", "industry_name", "return", f"{pillar}_score", f"{pillar}_score_extra"]]

            # define config used for factors selected when calculating scores
            for k, v in r.to_dict().items():
                if k[0] == "_":
                    g_score[k] = v
            score_df_list.append(g_score)

        score_df = pd.concat(score_df_list, axis=0)
        score_df["_eval_top_metric"] = eval_top_metric

        return score_df

    def score_top_eval_(self, score_df):
        """ combine all pillar score calculated and calculate backtest ai_score for evaluation

        Returns
        -------
        pd.DataFrame() for top backtest selected ticker returns

        """

        # config_col = [x for x in score_df.filter(regex="^_").columns.to_list() if x not in ['_pillar', '_eval_top_metric']]
        config_col = [x for x in score_df.filter(regex="^_").columns.to_list() if x not in ['_pillar']]
        score_df_comb = score_df.groupby(['trading_day', 'ticker', 'industry_name'] + config_col).mean().reset_index()
        score_df_comb["extra_score"] = score_df_comb.filter(regex='_score_extra$').mean(axis=1)
        score_df_comb['ai_score'] = score_df_comb.filter(regex='_score$').mean(axis=1)

        # Evaluate: calculate return for top 10 score / mode industry
        eval_best_all = {}  # calculate score
        n_top_ticker_list = [-10, -50, 50, 10, 3]
        for i in n_top_ticker_list:
            for idx, g_score in score_df_comb.groupby(['trading_day'] + config_col):
                new_idx = tuple([i] + list(idx))
                eval_best_all[new_idx] = self.__eval_best(g_score, best_n=i).copy()

        # concat different trading_day top n selection ticker results
        df = pd.DataFrame(eval_best_all).transpose()
        df.index.names = tuple(["top_n", "trading_day"] + config_col)
        df = df.reset_index().rename(columns={"_pred_currency": "currency_code", "_weeks_to_expire": "weeks_to_expire"})
        df['trading_day'] = pd.to_datetime(df['trading_day'])

        # change data presentation for DB reading
        df['ret'] = pd.to_numeric(df['ret']).round(4) * 100
        df['pos_pct'] = np.where(df['ret'].isnull(), None, pd.to_numeric(df['pos_pct']).round(2) * 100)
        df['bm_ret'] = pd.to_numeric(df['bm_ret']).round(4) * 100
        df['bm_pos_pct'] = np.where(df['bm_ret'].isnull(), None,  pd.to_numeric(df['bm_pos_pct']).round(2) * 100)
        df["trading_day"] = df["trading_day"].dt.date
        df["updated"] = dt.datetime.now()

        return df

    def __eval_best(self, g_all, best_n=10, best_col='ai_score'):
        """ evaluate score history with top 10 score return & industry """

        top_ret = {}
        if best_n > 0:
            g = g_all.set_index('ticker').nlargest(best_n, columns=[best_col], keep='all')
        else:
            g = g_all.set_index('ticker').nsmallest(-best_n, columns=[best_col], keep='all')

        top_ret["ticker_count"] = g.shape[0]
        top_ret["ret"] = g['return'].mean()
        top_ret["mode"] = g[f"industry_name"].mode()[0]
        top_ret["mode_count"] = np.sum(g[f"industry_name"] == top_ret["mode"]).item()
        top_ret["pos_pct"] = np.sum(g['return'] > 0) / len(g)
        top_ret["bm_ret"] = g_all['return'].mean()
        top_ret["bm_pos_pct"] = np.sum(g_all['return'] > 0) / len(g_all)
        top_ret["tickers"] = list(g.index)

        return top_ret