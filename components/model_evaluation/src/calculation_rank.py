import pandas as pd
import numpy as np
import datetime as dt
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial
from global_vars import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from general.sql.sql_process import (
    read_query,
)
from results_analysis.calculation_backtest_score import get_fundamental_scores, scale_fundamental_scores
from collections import Counter

logger = logger(__name__, LOGGER_LEVEL)

def apply_parallel(grouped, func):
    """ (obsolete) parallel run groupby """
    g_list = Parallel(n_jobs=mp.cpu_count())(delayed(func)(group) for name, group in grouped)
    return pd.concat(g_list)


def weight_qcut(x, q_):
    """ qcut within groups """
    return pd.qcut(x, q=q_, labels=False, duplicates='drop')


class calculate_rank_pred:
    """ process raw prediction in result_pred_table -> production_rank_table for AI Score calculation """

    eval_col = ['max_ret', 'r2', 'mae', 'mse']
    if_combine_pillar = False       # combine cluster pillars
    base_score = 5

    def __init__(self, name_sql, pred_pillar=None, pred_start_testing_period='2000-01-01',
                 pred_start_uid='200000000000000000', pass_eval=False, pass_eval_top=False, fix_config_col=None):
        """
        Parameters
        ----------
        q (Float):
            If q > 1 (e.g. 5), top q factors used as best factors;
            If q < 1 (e.g. 1/3), top (q * total No. factors predicted) factors am used as best factors;
        name_sql (Str):
            name_sql to evaluate
        pillar (List[Str], Optional):
            pillar to evaluate;
        pred_start_testing_period (Str, Optional):
            String in "%Y-%m-%d" format for the start date to download prediction;
        pred_start_uid (Str, Optional):
            String in "%Y%m%d%H%M%S%f" format to filter factor_model records based on training start time
        eval_current (Boolean, Optional):
            if True, use only current name_sql
        # TODO: complete docstring
        """

        self.name_sql = name_sql
        self.weeks_to_expire = int(name_sql.split('_')[0][1:])
        self.pass_eval = pass_eval
        self.pass_eval_top = pass_eval_top
        self.fix_config_col = fix_config_col + ["testing_period"]

        if not self.pass_eval:
            # 1. Download subpillar table (if removed)
            self.subpillar_df = self._download_pillar_cluster_subpillar(pred_start_testing_period, self.weeks_to_expire)

            # 2. Download & merge all prediction from iteration
            self.pred = self._download_prediction(name_sql, pred_pillar, pred_start_testing_period, pred_start_uid)
            self.pred['uid_hpot'] = self.pred['uid'].str[:20]
            self.pred = self.__get_neg_factor_all(self.pred)

            if self.if_combine_pillar:
                self.pred['pillar'] = "combine"

        self.eval_df_history = self._download_eval(name_sql, self.weeks_to_expire)

        # if calculate backtest score & eval top selections
        if not self.pass_eval_top:
            try:
                adj_fundamentals = pd.read_pickle("adj_fundamentals.pkl")       # TODO: remove in production
            except Exception as e:
                print(e)
                fundamentals, factor_formula = get_fundamental_scores(start_date=pred_start_testing_period, sample_interval=1)
                adj_fundamentals = fundamentals.groupby(['currency_code', 'trading_day']).apply(scale_fundamental_scores)
                adj_fundamentals.to_pickle("adj_fundamentals.pkl")

            self.adj_fundamentals = adj_fundamentals.reset_index()
            # logger.debug(f"fundamental from: {adj_fundamentals['trading_day'].min()} to {adj_fundamentals['trading_day'].max()}")

    def rank_(self, *args):
        """ rank based on config defined by each row in pred_config table  """

        kwargs, = args
        if type(kwargs) != type({}):
            kwargs = kwargs[0]
        logger.debug(f"=== Evaluation for [{kwargs}]===")

        # if not pass_eval = read eval table from DB and calculate top ticker table directly
        if not self.pass_eval:

            # filter pred table
            df = self.pred.copy(1)
            if kwargs["pillar"] != "cluster":
                df = df.loc[(df["currency_code"] == kwargs["pred_currency"]) & (df["pillar"] == kwargs["pillar"])]
            else:
                df = df.loc[(df["currency_code"] == kwargs["pred_currency"]) & (df["pillar"].str.startswith("pillar"))]

            # 1. remove subpillar - same subpillar factors keep higher pred one
            if kwargs["eval_removed_subpillar"]:
                subpillar_df_idx = ["testing_period", "currency_code"]      # weeks_to_expire defined when read table
                df = df.merge(self.subpillar_df, on=subpillar_df_idx + ["factor_name"], how="left")
                df["subpillar"] = df["subpillar"].fillna(df["factor_name"])

                # for defined pillar to remove subpillar cross all pillar by keep top pred only
                if "pillar" not in kwargs["pillar"]:
                    df = df.sort_values(by=["pred"]).drop_duplicates(
                        subset=['subpillar'] + subpillar_df_idx + self.select_config_col, keep="last")

            # 2. save backtest evaluation metrics to DB Table [backtest_eval]
            # Change [currency_code] in pred_table to [pred_currency] because we define index based on data_configs
            df = df.drop(columns=["pred_currency"]).rename(columns={"currency_code": "pred_currency"})
            eval_df_new = self.__backtest_save_eval_metrics(df, **kwargs)
        else:
            eval_df_new = pd.DataFrame()

        if kwargs["pillar"] == "cluster":
            eval_df = self.eval_df_history.loc[(self.eval_df_history["_pred_currency"] == kwargs["pred_currency"]) &
                                               (self.eval_df_history["_pillar"].str.startswith("pillar"))].copy(1)
        else:
            eval_df = self.eval_df_history.loc[(self.eval_df_history["_pred_currency"] == kwargs["pred_currency"]) &
                                               (self.eval_df_history["_pillar"] == kwargs["pillar"])].copy(1)

        eval_df = eval_df.append(eval_df_new)
        logger.info(f'eval_df shape: {eval_df.shape}')

        # Based on evaluation df calculate DataFrame for list of selected factors (pillar & extra)
        select_history_df, select_df = self.__get_minmax_factors(eval_df, **kwargs)
        breakpoint()

        if not self.pass_eval_top:
            score_df = self.score_(df=select_history_df, **kwargs)
        else:
            score_df = pd.DataFrame()

        return eval_df_new, score_df, select_df

    def __get_neg_factor_all(self, pred):
        """ get all neg factors for all (testing_period, group) """

        neg_factor = pred.groupby(['uid_hpot'])['neg_factor'].first().reset_index().copy()

        arr_len = neg_factor['neg_factor'].str.len().values
        arr_info = np.repeat(neg_factor["uid_hpot"].values, arr_len, axis=0)
        arr_factor = np.array([e for x in neg_factor["neg_factor"].to_list() for e in x])[:, np.newaxis]

        idx = pd.Series(arr_info, name='uid_hpot')
        neg_factor_new = pd.DataFrame(arr_factor, index=idx, columns=["factor_name"]).reset_index()
        neg_factor_new['neg_factor'] = True

        pred = pred.drop(columns="neg_factor").merge(neg_factor_new, on=['uid_hpot', 'factor_name'], how='left')
        pred['neg_factor'] = pred['neg_factor'].fillna(False)

        pred.loc[pred['neg_factor'], ["actual", "pred"]] *= -1

        return pred

    # -------------------------------- Download tables for ranking (if restart) ------------------------------------

    def _download_pillar_cluster_subpillar(self, pred_start_testing_period, weeks_to_expire):
        """ download pillar cluster table """

        query = f"SELECT * FROM {pillar_cluster_table} WHERE pillar like 'subpillar_%%' " \
                f"AND testing_period>='{pred_start_testing_period}' AND weeks_to_expire={weeks_to_expire}"
        subpillar = read_query(query)
        subpillar['testing_period'] = pd.to_datetime(subpillar['testing_period'])

        arr_len = subpillar['factor_list'].str.len().values
        arr_info = np.repeat(subpillar[["testing_period", "currency_code", "pillar"]].values, arr_len, axis=0)
        arr_factor = np.array([e for x in subpillar["factor_list"].to_list() for e in x])[:, np.newaxis]

        idx = pd.MultiIndex.from_arrays(arr_info.T, names=["testing_period", "currency_code", "subpillar"])
        df_new = pd.DataFrame(arr_factor, index=idx, columns=["factor_name"]).reset_index()
        return df_new

    def _download_prediction(self, name_sql, pred_pillar, pred_start_testing_period, pred_start_uid):
        """ merge factor_stock & factor_model_stock """

        try:
            pred = pd.read_pickle(f"pred_{name_sql}.pkl")       # TODO: remove
            logger.info(f'=== Load local prediction history on name_sql=[{name_sql}] ===')
            # pred = pred.rename(columns={"trading_day": "testing_period"})
        except Exception as e:
            logger.info(e)
            logger.info(f'=== Download prediction history on name_sql=[{name_sql}] ===')
            conditions = [f"name_sql='{name_sql}'",
                          f"testing_period>='{pred_start_testing_period}'",
                          f"to_timestamp(left(uid, 20), 'YYYYMMDDHH24MISSUS') > "
                          f"to_timestamp('{pred_start_uid}', 'YYYYMMDDHH24MISSUS')",
                          ]
            if pred_pillar:
                conditions.append(f"pillar in {tuple(pred_pillar)}")

            query = f'''SELECT P.currency_code, P.factor_name, P.actual, P.pred, S.* 
                        FROM (SELECT * FROM {result_score_table} WHERE {' AND '.join(conditions)}) S  
                        INNER JOIN (SELECT * FROM {result_pred_table}) P ON S.uid=P.uid
                        ORDER BY S.uid
                        '''
            pred = read_query(query.replace(",)", ")")).fillna(0)

            if len(pred) > 0:
                pred.to_pickle(f'pred_{name_sql}.pkl')
            else:
                raise Exception(f"ERROR: No prediction download from DB with name_sql: [{name_sql}]")

        self.select_config_col = pred.filter(regex='^_[a-z]').columns.to_list()
        pred["testing_period"] = pd.to_datetime(pred["testing_period"])
        pred = pred.loc[pred['testing_period'] >= dt.datetime.strptime(pred_start_testing_period, "%Y-%m-%d")]
        return pred

    def _download_eval(self, name_sql, weeks_to_expire):
        """ download eval Table directly for top ticker evaluation """

        # read config eval history
        query = f"SELECT * FROM {backtest_eval_table} WHERE _weeks_to_expire={weeks_to_expire}"
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

    # ----------------------------------- Add Rank & Evaluation Metrics ---------------------------------------------

    def __backtest_save_eval_metrics(self, df, **kwargs):
        """ evaluate & rank different configuration;
            save backtest evaluation metrics -> backtest_eval_table """

        # 2.4.1. calculate group statistic
        df_actual = df.groupby(self.fix_config_col)[['actual']].mean()      # actual factor premiums
        df_eval = df.groupby(self.fix_config_col + self.select_config_col).apply(
            partial(self.__get_summary_stats_in_train_currency, **kwargs)).reset_index()
        df_eval = df_eval.loc[df_eval['testing_period'] < df_eval['testing_period'].max()]

        df_eval.to_pickle("cached_df_eval.pkl")   # TODO: remove after debug
        # df_eval = pd.read_pickle("cached_df_eval.pkl")

        df_eval[self.eval_col] = df_eval[self.eval_col].astype(float)
        df_eval = df_eval.join(df_actual, on=self.fix_config_col, how='left')

        # 2.4.2. create DataFrame for eval results to DB
        col_rename = {k: "_" + k for k in self.fix_config_col + self.select_config_col}
        df_eval = df_eval.rename(columns=col_rename)
        df_eval["_eval_q"] = kwargs["eval_q"]
        df_eval["_eval_removed_subpillar"] = kwargs["eval_removed_subpillar"]

        return df_eval

    def __get_summary_stats_in_train_currency(self, g, eval_q, eval_removed_subpillar, **kwargs):
        """ Calculate basic evaluation metrics for factors """

        ret_dict = {}
        g = g.dropna(how='any').copy()
        logger.debug(g)

        if len(g) > 1:
            max_g = g.loc[g['pred'] > np.quantile(g['pred'], 1 - eval_q)].sort_values(by=["pred"], ascending=False)
            min_g = g.loc[g['pred'] < np.quantile(g['pred'], eval_q)].sort_values(by=["pred"])
            if eval_removed_subpillar:
                max_g = max_g.drop_duplicates(subset=['subpillar'], keep="first")
                min_g = min_g.drop_duplicates(subset=['subpillar'], keep="first")
        # for cluster pillar in case only 1 factor in cluster
        elif len(g) == 1:
            max_g = g.loc[g['pred'] > .005]     # b/c 75% quantile pred is usually 0.005+
            min_g = g.loc[g['pred'] < -.005]
        else:
            return None

        ret_dict['max_factor'] = max_g['factor_name'].tolist()
        ret_dict['min_factor'] = min_g['factor_name'].tolist()
        ret_dict['max_factor_pred'] = max_g.groupby('factor_name')['pred'].mean().to_dict()
        ret_dict['min_factor_pred'] = max_g.groupby('factor_name')['pred'].mean().to_dict()
        ret_dict['max_factor_actual'] = max_g.groupby(['factor_name'])['actual'].mean().to_dict()
        ret_dict['min_factor_actual'] = min_g.groupby(['factor_name'])['actual'].mean().to_dict()
        ret_dict['max_ret'] = max_g['actual'].mean()
        ret_dict['min_ret'] = min_g['actual'].mean()
        if len(g) > 1:
            ret_dict['mae'] = mean_absolute_error(g['pred'], g['actual'])
            ret_dict['mse'] = mean_squared_error(g['pred'], g['actual'])
            ret_dict['r2'] = r2_score(g['pred'], g['actual'])
        else:
            ret_dict['mae'] = np.nan
            ret_dict['mse'] = np.nan
            ret_dict['r2'] = np.nan
        return pd.Series(ret_dict)

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
        logging.debug(period_agg_filter_counter)

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
        logging.debug(period_agg_count)

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


if __name__ == "__main__":
    # download_prediction('w4_d-7_20220310130330_debug')
    download_prediction('w4_d-7_20220312222718_debug')
    # download_prediction('w4_d-7_20220317005620_debug')    # adj_mse (1)
    # download_prediction('w4_d-7_20220317125729_debug')    # adj_mse (2)
    # download_prediction('w4_d-7_20220321173435_debug')      # adj_mse (3) long history
    # download_prediction('w4_d-7_20220324031027_debug')      # cluster pillar * 3
    exit(1)