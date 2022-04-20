import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import argparse
import json
import ast
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial
from general.send_slack import to_slack
from general.utils import to_excel
import global_vars
from global_vars import *
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import DATE, DOUBLE_PRECISION, TEXT, INTEGER, BOOLEAN, TIMESTAMP, JSON
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from general.sql_process import (
    read_query,
    upsert_data_to_database,
    uid_maker,
    trucncate_table_in_database,
    delete_data_on_database,
)
from results_analysis.calculation_backtest_from_eval import calculate_backtest_score
from collections import Counter

# define dtypes for factor_rank (current/history/backtest) tables when writing to DB

rank_dtypes = dict(
    group=TEXT,
    factor_name=TEXT,
    weeks_to_expire=INTEGER,
    factor_weight=INTEGER,
    pred=DOUBLE_PRECISION,
    long_large=BOOLEAN,
    last_update=TIMESTAMP
)

backtest_eval_dtypes = dict(_name_sql=TEXT, _group=TEXT, _group_code=TEXT, _pillar=TEXT, _testing_period=DATE,
                            max_factor=JSON, min_factor=JSON,
                            max_factor_pred=JSON, min_factor_pred=JSON,
                            max_factor_actual=JSON, min_factor_actual=JSON,
                            max_ret=DOUBLE_PRECISION(precision=53), min_ret=DOUBLE_PRECISION(precision=53),
                            mae=DOUBLE_PRECISION(precision=53), mse=DOUBLE_PRECISION(precision=53),
                            r2=DOUBLE_PRECISION(precision=53), actual=DOUBLE_PRECISION(precision=53),
                            _weeks_to_expire=INTEGER, _is_remove_subpillar=BOOLEAN,
                            last_update=TIMESTAMP, __use_average=BOOLEAN, __tree_type=TEXT, __use_pca=DOUBLE_PRECISION,
                            __qcut_q=INTEGER, __n_splits=DOUBLE_PRECISION, __valid_method=INTEGER, __down_mkt_pct=DOUBLE_PRECISION)

# diff_config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', 'use_average', 'down_mkt_pct']

logging.info(f" ---> result_score_table: [{result_score_table}]")
logging.info(f" ---> production_factor_rank_backtest_eval_table: [{production_factor_rank_backtest_eval_table}]")
logging.info(f" ---> production_factor_rank_backtest_top_table: [{production_factor_rank_backtest_top_table}]")


def apply_parallel(grouped, func):
    """ (obsolete) parallel run groupby """
    g_list = Parallel(n_jobs=mp.cpu_count())(delayed(func)(group) for name, group in grouped)
    return pd.concat(g_list)


def weight_qcut(x, q_):
    """ qcut within groups """
    return pd.qcut(x, q=q_, labels=False, duplicates='drop')


class calculate_rank_pred:
    """ process raw prediction in result_pred_table -> production_factor_rank_table for AI Score calculation """

    eval_col = ['max_ret', 'r2', 'mae', 'mse']
    # label_col = ["currency_code", "pillar", "testing_period"]
    if_plot = False
    if_combine_pillar = False       # combine cluster pillars
    if_eval_top = True
    if_return_rank = False  # TODO: change to True for prod

    def __init__(self, name_sql, pred_pillar=None, pred_start_testing_period='2000-01-01',
                 pred_start_uid='200000000000000000', pass_eval=False):
        """
        Parameters
        ----------
        q (Float):
            If q > 1 (e.g. 5), top q factors used as best factors;
            If q < 1 (e.g. 1/3), top (q * total No. factors predicted) factors am used as best factors;
        name_sql (Str, Optional):
            if not None (by default), name_sql to evaluate (This overwrites "weeks_to_expire" & "average_days" set)
        pillar (List[Str], Optional):
            pillar to evaluate;
        pred_start_testing_period (Str, Optional):
            String in "%Y-%m-%d" format for the start date to download prediction;
        pred_start_uid (Str, Optional):
            String in "%Y%m%d%H%M%S%f" format to filter factor_model records based on training start time
        eval_current (Boolean, Optional):
            if True, use only current name_sql
        # TODO: complete
        """

        self.name_sql = name_sql
        self.weeks_to_expire = int(name_sql.split('_')[0][1:])
        self.average_days = int(name_sql.split('_')[1][1:])
        self.pass_eval = pass_eval

        if not self.pass_eval:
            # 1. Download subpillar table (if removed)
            self.subpillar_df = self._download_pillar_cluster_subpillar(pred_start_testing_period)

            # 2. Download & merge all prediction from iteration
            self.pred = self._download_prediction(name_sql, pred_pillar, pred_start_testing_period, pred_start_uid)
            self.pred['uid_hpot'] = self.pred['uid'].str[:20]
            self.pred = self.__get_neg_factor_all(self.pred)

            if self.if_combine_pillar:
                self.pred['pillar'] = "combine"
        else:
            self.eval_df_all = self._download_eval(name_sql)

        if self.if_eval_top:
            self.top_eval_cls = calculate_backtest_score()

    def rank_all(self, *args):
        """ rank based on config defined by each row in pred_config table  """
        kwargs, = args

        if not self.pass_eval:
            df = self.pred.copy(1)
            df = df.rename(columns={"group": "currency_code"})      # TODO: remove after changing all table columns
            if kwargs["pillar"] != "cluster":
                df = df.loc[(df["currency_code"] == kwargs["pred_currency"]) & (df["pillar"] == kwargs["pillar"])]
            else:
                df = df.loc[(df["currency_code"] == kwargs["pred_currency"]) & (df["pillar"].str.startswith("pillar"))]

            logging.info(f'=== Generate rank [n={df.shape}] for [{kwargs}] ===')

            # 1. remove subpillar - same subpillar factors keep higher pred one
            if kwargs["eval_removed_subpillar"]:
                df = df.merge(self.subpillar_df, on=["testing_period", "currency_code", "factor_name"], how="left")
                df["subpillar"] = df["subpillar"].fillna(df["factor_name"])

                # for defined pillar to remove subpillar cross all pillar by keep top pred only
                if "pillar" not in kwargs["pillar"]:
                    df = df.sort_values(by=["pred"]).drop_duplicates(
                        subset=["testing_period", "currency_code", 'subpillar'] + self.select_config_col, keep="last")

            # 2.4. save backtest evaluation metrics to DB Table [backtest_eval]
            eval_df = self.__backtest_save_eval_metrics(df, **kwargs)
        else:
            eval_df = self.eval_df_all.loc[(self.eval_df_all["_currency_code"] == kwargs["pred_currency"]) &
                                           (self.eval_df_all["_pillar"] == kwargs["pillar"])].copy(1)

        if self.if_eval_top:
             top_eval_df = self.top_eval_cls.eval_to_top(eval_df, **kwargs)
        else:
            top_eval_df = pd.DataFrame()

        if self.if_return_rank:
            # TODO: add return rank code (for prod)
            # rank_df -> for current ranking
            pass
            # 2.6. rank for each testing_period
            # for (testing_period, group, group_code, pillar), g in df_cv.groupby(['testing_period', 'group', 'group_code', 'pillar']):
            #     best_config = dict(pillar=pillar, testing_period=testing_period, group=group)
            #
            #     # 2.5.1. use the best config prediction for this pillar
            #     # 2.5.2. use best [eval_top_config] config prediction for each (pillar, testing_period)
            #     logging.info(f'best config for [{pillar}]: top ({self.eval_top_config}) [{self.eval_metric}]')
            #     best_config_df = self.__backtest_find_calc_metric_avg(testing_period, group, group_code, pillar)
            #     if type(best_config_df) == type(None):
            #         continue
            #     eval = best_config_df[self.eval_col + ['net_ret']].mean().to_dict()
            #     best_config.update(eval)
            #
            #     # 2.5.2a. calculate rank for all config
            #     for i, row in best_config_df.iterrows():
            #         best_config.update(row[diff_config_col])
            #         g_best = g.loc[(g[diff_config_col] == row[diff_config_col]).all(axis=1)]
            #         rank_df = self.rank_each_testing_period(testing_period, g_best, group, group_code)
            #
            #         # 2.6.3. append to history / currency df list
            #         if testing_period == df_cv['testing_period'].max():  # if keep_all_history also write to prod table
            #             self.all_current.append({"info": best_config, "rank_df": rank_df.copy()})
            #         self.all_history.append({"info": best_config, "rank_df": rank_df.copy()})
        else:
            rank_df = pd.DataFrame()

        return eval_df, top_eval_df, rank_df


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

    def rank_each_testing_period(self, testing_period, g_best, group, group_code):
        """ (2.6) rank for each testing_period """

        logging.debug(f'calculate (neg_factor, factor_weight) for each config: {testing_period}')
        result_col = ['group', 'group_code', 'testing_period', 'factor_name', 'pred', 'factor_weight', 'actual']
        df = g_best.loc[g_best['testing_period'] == testing_period, result_col].reset_index(drop=True)

        # 2.6.1. record basic info
        df['last_update'] = dt.datetime.now()

        # 2.6.2. record neg_factor
        # original premium is "small - big" = short_large -> those marked neg_factor = long_large
        # df['long_large'] = False
        # neg_factor = self.neg_factor[pd.Timestamp(testing_period)][f'{group_code}_{group}']
        # for k, v in neg_factor.items():  # write neg_factor i.e. label factors
        #     if type(v) == type([]):
        #         df.loc[df['factor_name'].isin(v), 'long_large'] = True

        return df.sort_values(['group', 'group_code', 'pred'])

    # -------------------------------- Download tables for ranking (if restart) ------------------------------------

    def _download_pillar_cluster_subpillar(self, pred_start_testing_period):
        """ download pillar cluster table """

        query = f"SELECT * FROM {factors_pillar_cluster_table} WHERE pillar like 'subpillar_%%' " \
                f"AND testing_period>='{pred_start_testing_period}'"
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
            pred = pd.read_pickle(f"pred_{name_sql}.pkl")
            logging.info(f'=== Load local prediction history on name_sql=[{name_sql}] ===')
            pred = pred.rename(columns={"trading_day": "testing_period"})
        except Exception as e:
            logging.info(e)
            logging.info(f'=== Download prediction history on name_sql=[{name_sql}] ===')
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

    def _download_eval(self, name_sql):
        """ download eval Table directly for top ticker evaluation """

        query = f"SELECT * FROM {production_factor_rank_backtest_eval_table} WHERE name_sql='{name_sql}'"
        eval_df = read_query(query)
        return eval_df

    # ----------------------------------- Add Rank & Evaluation Metrics ---------------------------------------------

    def __backtest_save_eval_metrics(self, df, **kwargs):
        """ evaluate & rank different configuration;
            save backtest evaluation metrics -> production_factor_rank_backtest_eval_table """

        # 2.4.1. calculate group statistic
        logging.debug(f"=== Update [{production_factor_rank_backtest_eval_table}] ===")
        groupby_col = ["testing_period", "currency_code", "pillar"]

        # df = df.sort_values(by=groupby_col).head(200)       # TODO: remove after debug

        # actual factor premiums
        df_actual = df.groupby(groupby_col)[['actual']].mean()
        df_eval = df.groupby(groupby_col + self.select_config_col).apply(
            partial(self.__get_summary_stats_in_group, **kwargs)).reset_index()
        df_eval = df_eval.loc[df_eval['testing_period'] < df_eval['testing_period'].max()]

        # df_eval.to_pickle("cached_df_eval.pkl")   # TODO: remove after debug
        # df_eval = pd.read_pickle("cached_df_eval.pkl")

        df_eval[self.eval_col] = df_eval[self.eval_col].astype(float)
        df_eval = df_eval.join(df_actual, on=groupby_col, how='left')

        fix_config_col = groupby_col + ["eval_q", "eval_removed_subpillar"]
        df_eval["eval_q"] = kwargs["eval_q"]
        df_eval["eval_removed_subpillar"] = kwargs["eval_removed_subpillar"]

        # 2.4.2. create DataFrame for eval results to DB
        col_rename = {k: "_" + k for k in fix_config_col + self.select_config_col}
        df_eval = df_eval.rename(columns=col_rename)

        df_eval['is_valid'] = True
        df_eval['last_update'] = dt.datetime.now()

        # 2.4.3. save local plot for evaluation (TODO: decide whether keep)
        # if self.if_plot:
        #     rank_pred.__save_plot_backtest_ret(df_eval, diff_config_col, pillar)

        # 2.4.2. add config JSON column for configs
        # df_eval['config'] = df_eval[diff_config_col].to_dict(orient='records')
        # df_eval = df_eval.drop(columns=diff_config_col)
        # df['use_average'] = df['use_average'].replace(0, np.nan)

        return df_eval

    def __get_summary_stats_in_group(self, g, eval_q, eval_removed_subpillar, **kwargs):
        """ Calculate basic evaluation metrics for factors """

        ret_dict = {}
        g = g.dropna(how='any').copy()
        logging.debug(g)

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

    # def __backtest_find_calc_metric_avg(self, testing_period, group, group_code, pillar):
    #     """ Select Best Config (among other_group_col) """
    #
    #     conditions = [
    #         f"weeks_to_expire={self.weeks_to_expire}",
    #         f"pillar='{pillar}'",
    #         f"\"group\"='{group}'",
    #         f"group_code='{group_code}'",
    #         f"testing_period < '{testing_period}'",
    #         f"testing_period >= '{testing_period - relativedelta(weeks=self.weeks_to_expire * self.eval_config_select_period)}'",
    #         f"is_valid"
    #     ]
    #
    #     query = f"SELECT * FROM {production_factor_rank_backtest_eval_table} WHERE {' AND '.join(conditions)} "
    #     df = read_query(query, global_vars.db_url_alibaba_prod)
    #
    #     if len(df) > 0:
    #         df = pd.concat([df, pd.DataFrame(df['config'].to_list())], axis=1)
    #         df_mean = df.groupby(diff_config_col).mean().reset_index()  # average over testing_period
    #         df_mean['net_ret'] = df_mean['max_ret'] - df_mean['min_ret']
    #
    #         if self.eval_metric in ['max_ret', 'net_ret', 'r2']:
    #             best = df_mean.nlargest(self.eval_top_config, self.eval_metric, keep="all")
    #         elif self.eval_metric in ['mae', 'mse']:
    #             best = df_mean.nsmallest(self.eval_top_config, self.eval_metric, keep="all")
    #         else:
    #             raise Exception("ERROR: Wrong eval_metric")
    #         return best
    #     else:
    #         return None

    # --------------------------------------- Save Prod Table to DB -------------------------------------------------

    def write_backtest_rank_(self):
        """ write backtest factors: backtest rank -> production_factor_rank_backtest_table """

        if type(self.eval_top_config) == type(None):
            all_history = []
            for i in self.all_history:
                df_history = i["rank_df"]
                df_history["weeks_to_expire"] = self.weeks_to_expire
                df_info = pd.DataFrame(i["info"]).ffill(0)
                all_history.append({"rank_df": df_history.copy(), "info": df_info})
        else:
            all_history = pd.concat([x["rank_df"] for x in self.all_history], axis=0)
            all_history = all_history.groupby(['group', 'group_code', 'testing_period', 'factor_name']).mean().reset_index()
            all_history['factor_weight'] = all_history.groupby(by=['group', 'group_code', 'testing_period'])[
                'pred'].transform(partial(weight_qcut, q_=self.q_)).astype(int)
            all_history['last_update'] = dt.datetime.now()
            all_history["weeks_to_expire"] = self.weeks_to_expire

        return all_history

    def write_current_rank_(self):
        """write current use factors: current rank -> production_factor_rank_table / production_factor_rank_history"""

        df_current = pd.concat([x["rank_df"] for x in self.all_current], axis=0)
        df_current = df_current.groupby(['group', 'group_code', 'testing_period', 'factor_name']).mean().reset_index()
        df_current['factor_weight'] = df_current.groupby(by=['group', 'group_code', 'testing_period'])['pred'].transform(
            partial(weight_qcut, q_=self.q_)).astype(int)
        df_current['last_update'] = dt.datetime.now()
        df_current["weeks_to_expire"] = self.weeks_to_expire
        df_current = df_current.drop(columns=['actual_z'])

        # update [production_factor_rank_table]
        upsert_data_to_database(df_current, production_factor_rank_table,
                                primary_key=["group", "factor_name", "weeks_to_expire"],
                                db_url=db_url_write, how='update', dtype=rank_dtypes)

        # update [production_factor_rank_history_table]
        upsert_data_to_database(df_current, production_factor_rank_history_table,
                                primary_key=["group", "factor_name", "weeks_to_expire", "last_update"],
                                db_url=db_url_write, how='update', dtype=rank_dtypes)

    def write_to_db(self):
        """ concat rank current/history & write """

        all_history = self.write_backtest_rank_()
        self.write_current_rank_()
        return all_history

    # ---------------------------------- Save local Plot for evaluation --------------------------------------------

    @staticmethod
    def __save_plot_backtest_ret(result_all_comb, other_group_col, pillar):
        """ Save Plot for backtest average ret """

        logging.debug(f'=== Save Plot for backtest average ret ===')
        result_all_comb = result_all_comb.copy()
        result_all_comb['other_group'] = result_all_comb[other_group_col].astype(str).agg('-'.join, axis=1)
        result_all_comb['group'] = result_all_comb[['group', 'group_code']].astype(str).agg('-'.join, axis=1)

        num_group = len(result_all_comb['group'].unique())
        num_other_group = len(result_all_comb['other_group'].unique())

        # create figure for test & train boxplot
        fig = plt.figure(figsize=(num_group * 8, num_other_group * 4), dpi=120, constrained_layout=True)
        k = 1
        for (other_group, group), g in result_all_comb.groupby(['other_group', 'group']):
            ax = fig.add_subplot(num_other_group, num_group, k)
            g[['max_ret', 'actual', 'min_ret']] = (g[['max_ret', 'actual', 'min_ret']] + 1).cumprod(axis=0)
            plot_df = g.set_index(['testing_period'])[['max_ret', 'actual', 'min_ret']]
            ax.plot(plot_df)
            for i in range(3):
                ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
            if (k % num_group == 1) or (num_group == 1):
                ax.set_ylabel(other_group, fontsize=20)
            if k > (num_other_group - 1) * num_group:
                ax.set_xlabel(group, fontsize=20)
            if k == 1:
                plt.legend(['best', 'average', 'worse'])
            k += 1
        fig_name = f'#pred_{rank_pred.name_sql}_{pillar}.png'
        plt.suptitle(' - '.join(other_group_col), fontsize=20)
        plt.savefig(fig_name)
        plt.close()
        logging.debug(f'=== Saved [{fig_name}] for evaluation ===')


if __name__ == "__main__":
    # download_prediction('w4_d-7_20220310130330_debug')
    download_prediction('w4_d-7_20220312222718_debug')
    # download_prediction('w4_d-7_20220317005620_debug')    # adj_mse (1)
    # download_prediction('w4_d-7_20220317125729_debug')    # adj_mse (2)
    # download_prediction('w4_d-7_20220321173435_debug')      # adj_mse (3) long history
    # download_prediction('w4_d-7_20220324031027_debug')      # cluster pillar * 3
    exit(1)

    # linechart
    for factor, g in pred.groupby(['group', 'group_code', 'factor_name']):
        g1 = g.groupby(['testing_period'])[['actual', 'pred']].mean()
        plt.plot(g1)
        plt.title(factor)
        plt.show()

    # TODO: factor selection (heatmap)
    import seaborn as sns

    for (pillar, group, group_code), g in pred.groupby(['pillar', 'group', 'group_code']):
        try:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

            g_pred_agg = g.groupby(['group', 'group_code', 'testing_period', 'factor_name'])[
                'pred_weight'].mean().unstack().fillna(1)
            sns.heatmap(g_pred_agg, ax=ax[0])
            ax[0].set_xlabel('pred_weight')

            g_actual_agg = g.groupby(['group', 'group_code', 'testing_period', 'factor_name'])[
                'actual_weight'].mean().unstack().fillna(1)
            sns.heatmap(g_actual_agg, ax=ax[1])
            ax[1].set_xlabel('actual_weight')

            plt.suptitle('{}-{}-{}'.format(pillar, group, group_code))
            plt.savefig('{}-{}-{}.png'.format(pillar, group, group_code))

        except Exception as e:
            print(e)

    exit(1)
