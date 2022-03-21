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
from collections import Counter

# define dtypes for factor_rank (current/history/backtest) tables when writing to DB

rank_dtypes = dict(
    group=TEXT,
    factor_name=TEXT,
    weeks_to_expire=INTEGER,
    factor_weight=INTEGER,
    pred_z=DOUBLE_PRECISION,
    long_large=BOOLEAN,
    last_update=TIMESTAMP
)

backtest_eval_dtypes = dict(name_sql=TEXT(), group=TEXT(), testing_period=DATE,
                            max_factor_count=JSON, min_factor_count=JSON,
                            max_factor_pred=JSON, min_factor_pred=JSON,
                            max_factor_actual=JSON, min_factor_actual=JSON,
                            max_ret=DOUBLE_PRECISION(precision=53), min_ret=DOUBLE_PRECISION(precision=53),
                            mae=DOUBLE_PRECISION(precision=53), mse=DOUBLE_PRECISION(precision=53),
                            r2=DOUBLE_PRECISION(precision=53), actual=DOUBLE_PRECISION(precision=53), y_type=TEXT(),
                            weeks_to_expire=INTEGER(), config=JSON(astext_type=TEXT()), last_update=TIMESTAMP)

diff_config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', 'use_average', 'down_mkt_pct']


def apply_parallel(grouped, func):
    """ (obsolete) parallel run groupby """
    g_list = Parallel(n_jobs=mp.cpu_count())(delayed(func)(group) for name, group in grouped)
    return pd.concat(g_list)


def weight_qcut(x, q_):
    """ qcut within groups """
    return pd.qcut(x, q=q_, labels=False, duplicates='drop')


def download_prediction(name_sql, pred_y_type=None, pred_start_testing_period='2000-01-01',
                        pred_start_uid='200000000000000000'):
    """ download joined table factor_model / *_stock and save csv """

    conditions = [f"name_sql='{name_sql}'",
                  f"testing_period>='{pred_start_testing_period}'",
                  f"to_timestamp(left(uid, 20), 'YYYYMMDDHH24MISSUS') > "
                  f"to_timestamp('{pred_start_uid}', 'YYYYMMDDHH24MISSUS')",
                  ]
    if pred_y_type:
        conditions.append(f"y_type in {tuple(pred_y_type)}")

    global diff_config_col
    pred_col = ["uid", "pred", "actual", "factor_name", "\"group\""]
    label_col = ['name_sql', 'y_type', 'neg_factor', 'testing_period', 'cv_number', 'uid', 'group_code'] + diff_config_col
    query = f'''SELECT P.\"group\", P.factor_name, P.actual, P.pred, S.* 
                FROM (SELECT {', '.join(label_col)} FROM {result_score_table} WHERE {' AND '.join(conditions)}) S  
                INNER JOIN (SELECT {', '.join(pred_col)} FROM {result_pred_table}) P
                  ON S.uid=P.uid
                ORDER BY S.uid'''
    pred = read_query(query.replace(",)", ")")).fillna(0)
    if len(pred) > 0:
        # pred.to_json(f'pred_{name_sql}.json')
        pred.to_pickle(f'pred_{name_sql}.pkl')
    else:
        raise Exception(f"ERROR: No prediction download from DB with name_sql: [{name_sql}]")
    return pred


class rank_pred:
    """ process raw prediction in result_pred_table -> production_factor_rank_table for AI Score calculation """

    q_ = None
    eval_col = ['max_ret', 'r2', 'mae', 'mse']
    iter_unique_col = ['name_sql', 'group', 'group_code', 'testing_period', 'factor_name']
    diff_config_col = diff_config_col
    if_plot = False

    def __init__(self, q, name_sql,
                 pred_y_type=None, pred_start_testing_period='2000-01-01', pred_start_uid='200000000000000000',
                 eval_current=True, eval_top_config=10, eval_metric='net_ret', eval_config_select_period=12):
        """
        Parameters
        ----------
        q (Float):
            If q > 1 (e.g. 5), top q factors used as best factors;
            If q < 1 (e.g. 1/3), top (q * total No. factors predicted) factors am used as best factors;
        name_sql (Str, Optional):
            if not None (by default), name_sql to evaluate (This overwrites "weeks_to_expire" & "average_days" set)
        y_type (List[Str], Optional):
            y_type to evaluate;
        pred_start_testing_period (Str, Optional):
            String in "%Y-%m-%d" format for the start date to download prediction;
        pred_start_uid (Str, Optional):
            String in "%Y%m%d%H%M%S%f" format to filter factor_model records based on training start time
        eval_current (Boolean, Optional):
            if True, use only current name_sql
        # TODO: complete
        """

        self.q = q
        self.pred_start_testing_period = pred_start_testing_period
        self.name_sql = name_sql
        self.eval_current = eval_current
        self.eval_top_config = eval_top_config
        self.eval_metric = eval_metric
        self.eval_config_select_period = eval_config_select_period
        self.weeks_to_expire = int(name_sql.split('_')[0][1:])
        self.average_days = int(name_sql.split('_')[1][1:])

        # 1. Download & merge all prediction from iteration
        pred = self._download_prediction(name_sql, pred_y_type, pred_start_testing_period, pred_start_uid)
        pred['uid_hpot'] = pred['uid'].str[:20]
        pred = self.__get_neg_factor_all(pred)

        # pred.to_pickle('pred_cache.pkl')

        # 2. Process separately for each y_type (i.e. momentum/value/quality/all)
        self.all_current = []
        self.all_history = []
        for y_type, g in pred.groupby('y_type'):
            self.y_type = y_type
            self.rank_each_y_type(q, y_type, g, name_sql)

    def rank_each_y_type(self, q, y_type, df, name_sql):
        """ rank for each y_type """

        logging.info(f'=== Generate rank for [{y_type}] with results={df.shape}) ===')

        # 2.1. remove duplicate samples from running twice when testing
        # df = df.drop_duplicates(subset=['uid'] + self.iter_unique_col + self.diff_config_col, keep='last')
        # df['testing_period'] = pd.to_datetime(df['testing_period'])

        # 2.2. use average predictions from different validation sets
        df_cv_avg = df.groupby(['uid_hpot'] + self.iter_unique_col + self.diff_config_col)[['pred', 'actual', 'actual_s']].mean()

        # 2.3. calculate 'pred_z'/'factor_weight' by ranking
        q = q / len(df['factor_name'].unique()) if isinstance(q, int) else q  # convert q(int) -> ratio
        self.q_ = [0., q, 1. - q, 1.]
        df_cv_avg = rank_pred.__calc_z_weight(self.q_, df_cv_avg, self.diff_config_col)

        # 2.4. save backtest evaluation metrics to DB Table [backtest_eval]
        self.__backtest_save_eval_metrics(df_cv_avg, y_type, name_sql)

        # 2.6. rank for each testing_period
        for (testing_period, group, group_code), g in df_cv_avg.groupby(['testing_period', 'group', 'group_code']):
            best_config = dict(y_type=self.y_type, testing_period=testing_period, group=group)

            # 2.5.1. use the best config prediction for this y_type
            # 2.5.2. use best [eval_top_config] config prediction for each (y_type, testing_period)
            logging.info(f'best config for [{y_type}]: top ({self.eval_top_config}) [{self.eval_metric}]')
            best_config_df = self.__backtest_find_calc_metric_avg(testing_period, group, group_code)
            if type(best_config_df) == type(None):
                continue
            eval = best_config_df[self.eval_col + ['net_ret']].mean().to_dict()
            best_config.update(eval)

            # 2.5.2a. calculate rank for all config
            for i, row in best_config_df.iterrows():
                best_config.update(row[self.diff_config_col])
                g_best = g.loc[(g[self.diff_config_col] == row[self.diff_config_col]).all(axis=1)]
                rank_df = self.rank_each_testing_period(testing_period, g_best, group, group_code)

                # 2.6.3. append to history / currency df list
                if testing_period == df_cv_avg['testing_period'].max():  # if keep_all_history also write to prod table
                    self.all_current.append({"info": best_config, "rank_df": rank_df.copy()})
                self.all_history.append({"info": best_config, "rank_df": rank_df.copy()})

            # break

    @staticmethod
    def __calc_z_weight(q_, df, diff_config_col):
        """ calculate 'pred_z'/'factor_weight' by ranking within current testing_period """

        # 2.3.1. calculate factor_weight with qcut
        logging.info("---> Calculate factor_weight")
        # groupby_keys = ['uid_hpot', 'group', 'group_code', 'testing_period'] + diff_config_col

        # df['factor_weight'] = apply_parallel(df.groupby(by=groupby_keys)['pred'], partial(weight_qcut, q_=q_))
        df['factor_weight'] = df.groupby(by='uid_hpot')['pred'].transform(partial(weight_qcut, q_=q_))
        df = df.reset_index()

        # 2.3.2. count rank for debugging
        # rank_count = df.groupby(by='uid_hpot')['factor_weight'].apply(pd.value_counts)
        # rank_count = rank_count.unstack().fillna(0)
        # logging.info(f'n_factor used:\n{rank_count}')

        # 2.3.3b. pred_z = original pred
        logging.info("---> Calculate pred_z")
        df['pred_z'] = df['pred']
        df['actual_z'] = df['actual']

        # 2.3.3a. calculate pred_z using mean & std of all predictions in entire testing history
        # df['pred_z'] = df.groupby(by=['group'] + diff_config_col)['pred'].apply(lambda x: (x - np.mean(x)) / np.std(x))
        # df['actual_z'] = df.groupby(by=['group'] + diff_config_col)['actual'].apply(lambda x: (x - np.mean(x)) / np.std(x))

        return df

    def __get_neg_factor_all(self, pred):
        """ get all neg factors for all (testing_period, group) """

        new_pred = []
        pred['actual_s'] = pred['actual'].copy()
        for uid_hpot, g in pred.groupby(['uid_hpot']):
            neg_factor = g['neg_factor'].to_list()[0]
            if len(neg_factor) > 0:
                try:
                    g.loc[g['factor_name'].isin(neg_factor), 'actual'] *= -1
                    g['factor_name'] = [f'{x} (L)' if x in neg_factor else f'{x} (S)' for x in g['factor_name']]
                    new_pred.append(g)
                except Exception as e:
                    to_slack("clair").message(f"*neg_factor ERROR*: {e.args}")

        new_pred = pd.concat(new_pred, axis=0)

        return new_pred

    def rank_each_testing_period(self, testing_period, g_best, group, group_code):
        """ (2.6) rank for each testing_period """

        logging.debug(f'calculate (neg_factor, factor_weight) for each config: {testing_period}')
        result_col = ['group', 'group_code', 'testing_period', 'factor_name', 'pred_z', 'factor_weight', 'actual_z']
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

        return df.sort_values(['group', 'group_code', 'pred_z'])

    # --------------------------------------- Download Prediction -------------------------------------------------

    def _download_prediction(self, name_sql, pred_y_type, pred_start_testing_period, pred_start_uid):
        """ merge factor_stock & factor_model_stock """

        try:
            pred = pd.read_pickle(f"pred_{name_sql}.pkl")
            logging.info(f'=== Load local prediction history on name_sql=[{name_sql}] ===')
            pred = pred.rename(columns={"trading_day": "testing_period"})

        except Exception as e:
            print(e)
            logging.info(f'=== Download prediction history on name_sql=[{name_sql}] ===')
            download_prediction(name_sql, pred_y_type, pred_start_testing_period, pred_start_uid)

        pred["testing_period"] = pd.to_datetime(pred["testing_period"])
        pred = pred.loc[pred['testing_period'] >= dt.datetime.strptime(pred_start_testing_period, "%Y-%m-%d")]

        # TODO: fix bug for missing CNY 2021-09-12 in eval table
        # pred = pred.loc[(pred['group_code'] == 'CNY') & (pred['testing_period'] == dt.date(2021, 9, 12))]

        # fix wrong "actual" premium in factor_model_stock (1)
        premium = read_query(f"SELECT * FROM {factor_premium_table} "
                             f"WHERE weeks_to_expire={self.weeks_to_expire} and average_days={self.average_days}")
        premium['trading_day'] = premium['trading_day'] - pd.tseries.offsets.DateOffset(weeks=self.weeks_to_expire)
        premium = premium.set_index(['trading_day', "group", 'field'])['value']
        pred = pred.join(premium, on=['testing_period', 'group', 'factor_name'])
        pred = pred.drop(columns=['actual']).rename(columns={"value": 'actual'})

        return pred

    # ----------------------------------- Add Rank & Evaluation Metrics ---------------------------------------------

    def __backtest_save_eval_metrics(self, result_all, y_type, name_sql):
        """ evaluate & rank different configuration;
            save backtest evaluation metrics -> production_factor_rank_backtest_eval_table """

        # 2.4.1. calculate group statistic
        logging.debug(f"=== Update [{production_factor_rank_backtest_eval_table}] ===")

        # actual factor premiums
        result_all_avg = result_all.groupby(['name_sql', 'group', 'group_code', 'testing_period'])[['actual', 'actual_s']].mean()
        result_all_comb = result_all.groupby(['name_sql', 'group', 'group_code', 'testing_period'] + self.diff_config_col
                                             ).apply(self.__get_summary_stats_in_group)
        result_all_comb = result_all_comb.loc[result_all_comb.index.get_level_values('testing_period') <
                                              result_all_comb.index.get_level_values(
                                                  'testing_period').max()].reset_index()
        result_all_comb[self.eval_col] = result_all_comb[self.eval_col].astype(float)
        result_all_comb = result_all_comb.join(result_all_avg, on=['name_sql', 'group', 'group_code', 'testing_period'], how='left')

        # 2.4.2. create DataFrame for eval results to DB
        result_all_comb["y_type"] = y_type
        result_all_comb["weeks_to_expire"] = self.weeks_to_expire
        # primary_keys = ["name_sql", "group", "group_code", "testing_period", "y_type"] + self.diff_config_col
        # result_all_comb = uid_maker(result_all_comb, primary_key=primary_keys)

        # 2.4.3. save local plot for evaluation
        if self.if_plot:
            rank_pred.__save_plot_backtest_ret(result_all_comb, self.diff_config_col, y_type, name_sql)

        # 2.4.2. add config JSON column for configs
        result_all_comb['config'] = result_all_comb[self.diff_config_col].to_dict(orient='records')
        result_all_comb = result_all_comb.drop(columns=self.diff_config_col)
        result_all_comb['is_valid'] = True
        result_all_comb['last_update'] = dt.datetime.now()
        result_all_comb['q'] = self.q
        upsert_data_to_database(result_all_comb, production_factor_rank_backtest_eval_table, primary_key=["uid"],
                                db_url=db_url_write, how="append", dtype=backtest_eval_dtypes)

    def __get_summary_stats_in_group(self, g):
        """ Calculate basic evaluation metrics for factors """

        ret_dict = {}
        g = g.dropna(how='any').copy()
        if len(g) > 0:
            max_g = g[g['factor_weight'] == g['factor_weight'].max()]
            min_g = g[g['factor_weight'] == 0]
            ret_dict['max_factor_count'] = dict(Counter(max_g['factor_name'].tolist()))
            ret_dict['min_factor_count'] = dict(Counter(min_g['factor_name'].tolist()))
            ret_dict['max_factor_pred'] = max_g.groupby('factor_name')['pred'].mean().to_dict()
            ret_dict['min_factor_pred'] = max_g.groupby('factor_name')['pred'].mean().to_dict()
            ret_dict['max_factor_actual'] = max_g.groupby(['factor_name'])['actual'].mean().to_dict()
            ret_dict['min_factor_actual'] = min_g.groupby(['factor_name'])['actual'].mean().to_dict()
            ret_dict['max_ret'] = max_g['actual'].mean()
            ret_dict['min_ret'] = min_g['actual'].mean()
            ret_dict['mae'] = mean_absolute_error(g['pred'], g['actual'])
            ret_dict['mse'] = mean_squared_error(g['pred'], g['actual'])
            ret_dict['r2'] = r2_score(g['pred'], g['actual'])
            return pd.Series(ret_dict)

    def __backtest_find_calc_metric_avg(self, testing_period, group, group_code):
        """ Select Best Config (among other_group_col) """

        conditions = [
            f"weeks_to_expire={self.weeks_to_expire}",
            f"y_type='{self.y_type}'",
            f"\"group\"='{group}'",
            f"group_code='{group_code}'",
            f"testing_period < '{testing_period}'",
            f"testing_period >= '{testing_period - relativedelta(weeks=self.weeks_to_expire * self.eval_config_select_period)}'",
            f"is_valid"
        ]
        if self.eval_current:
            conditions.append(f"name_sql='{self.name_sql}'")

        query = f"SELECT * FROM {production_factor_rank_backtest_eval_table} WHERE {' AND '.join(conditions)} "
        df = read_query(query, global_vars.db_url_alibaba_prod)

        if len(df) > 0:
            df = pd.concat([df, pd.DataFrame(df['config'].to_list())], axis=1)
            df_mean = df.groupby(self.diff_config_col).mean().reset_index()  # average over testing_period
            df_mean['net_ret'] = df_mean['max_ret'] - df_mean['min_ret']

            if self.eval_metric in ['max_ret', 'net_ret', 'r2']:
                best = df_mean.nlargest(self.eval_top_config, self.eval_metric, keep="all")
            elif self.eval_metric in ['mae', 'mse']:
                best = df_mean.nsmallest(self.eval_top_config, self.eval_metric, keep="all")
            else:
                raise Exception("ERROR: Wrong eval_metric")
            return best
        else:
            return None

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
                'pred_z'].transform(partial(weight_qcut, q_=self.q_)).astype(int)
            all_history['last_update'] = dt.datetime.now()
            all_history["weeks_to_expire"] = self.weeks_to_expire

        return all_history

    def write_current_rank_(self):
        """write current use factors: current rank -> production_factor_rank_table / production_factor_rank_history"""

        df_current = pd.concat([x["rank_df"] for x in self.all_current], axis=0)
        df_current = df_current.groupby(['group', 'group_code', 'testing_period', 'factor_name']).mean().reset_index()
        df_current['factor_weight'] = df_current.groupby(by=['group', 'group_code', 'testing_period'])['pred_z'].transform(
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
        # if not DEBUG:
        #     self.write_current_rank_()
        #     logging.INFO("calculation_rank will not write to [current] in DEBUG mode")
        return all_history

    # ---------------------------------- Save local Plot for evaluation --------------------------------------------

    @staticmethod
    def __save_plot_backtest_ret(result_all_comb, other_group_col, y_type, name_sql):
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
        fig_name = f'#pred_{name_sql}_{y_type}.png'
        plt.suptitle(' - '.join(other_group_col), fontsize=20)
        plt.savefig(fig_name)
        plt.close()
        logging.debug(f'=== Saved [{fig_name}] for evaluation ===')


if __name__ == "__main__":
    # download_prediction('w4_d-7_20220310130330_debug')
    # download_prediction('w4_d-7_20220312222718_debug')
    # # download_prediction('w4_d-7_20220312222632_debug')
    # exit(1)

    # linechart
    for factor, g in pred.groupby(['group', 'group_code', 'factor_name']):
        g1 = g.groupby(['testing_period'])[['actual', 'pred']].mean()
        plt.plot(g1)
        plt.title(factor)
        plt.show()

    # TODO: factor selection (heatmap)
    import seaborn as sns

    for (y_type, group, group_code), g in pred.groupby(['y_type', 'group', 'group_code']):
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

            plt.suptitle('{}-{}-{}'.format(y_type, group, group_code))
            plt.savefig('{}-{}-{}.png'.format(y_type, group, group_code))

        except Exception as e:
            print(e)

    exit(1)
