import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import argparse
import json
import ast

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

backtest_eval_dtypes = dict(uid=TEXT(), name_sql=TEXT(), group=TEXT(), trading_day=TIMESTAMP(),
                            max_factor=JSON(astext_type=TEXT()), min_factor=JSON(astext_type=TEXT()),
                            max_ret=DOUBLE_PRECISION(precision=53), min_ret=DOUBLE_PRECISION(precision=53),
                            mae=DOUBLE_PRECISION(precision=53), mse=DOUBLE_PRECISION(precision=53),
                            r2=DOUBLE_PRECISION(precision=53), actual=DOUBLE_PRECISION(precision=53), y_type=TEXT(),
                            weeks_to_expire=INTEGER(), config=JSON(astext_type=TEXT()))


class rank_pred:
    """ process raw prediction in result_pred_table -> production_factor_rank_table for AI Score calculation """

    q_ = None
    eval_col = ['max_ret', 'net_ret', 'r2', 'mae', 'mse']
    iter_unique_col = ['name_sql', 'group', 'trading_day', 'factor_name']
    diff_config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', 'use_average', 'down_mkt_pct']

    def __init__(self, q, name_sql,
                 pred_y_type=None, pred_start_testing_period='2000-01-01', pred_start_uid='200000000000000000',
                 eval_current=True, eval_top_config=None, eval_metrics='net_ret', eval_config_select_period=12):
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
        """

        self.q = q
        self.pred_start_testing_period = pred_start_testing_period
        self.name_sql = name_sql
        self.eval_current = eval_current
        self.eval_top_config = eval_top_config
        self.eval_metrics = eval_metrics
        self.eval_config_select_period = eval_config_select_period
        self.weeks_to_expire = int(name_sql.split('_')[0][1:])

        # 1. Download & merge all prediction from iteration
        pred = self._download_prediction(name_sql, pred_y_type, pred_start_testing_period, pred_start_uid)
        pred['uid_hpot'] = pred['uid'].str[:20]
        self.neg_factor = rank_pred.__get_neg_factor_all(pred)

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
        # df['trading_day'] = pd.to_datetime(df['trading_day'])

        # 2.2. use average predictions from different validation sets
        df_cv_avg = df.groupby(['uid_hpot'] + self.iter_unique_col + self.diff_config_col)[['pred', 'actual']].mean()

        # 2.3. calculate 'pred_z'/'factor_weight' by ranking
        q = q / len(df['factor_name'].unique()) if isinstance(q, int) else q  # convert q(int) -> ratio
        self.q_ = [0., q, 1. - q, 1.]
        df_cv_avg = rank_pred.__calc_z_weight(self.q_, df_cv_avg, self.diff_config_col)

        # 2.4. save backtest evaluation metrics to DB Table [backtest_eval]
        self.__backtest_save_eval_metrics(df_cv_avg, y_type, name_sql)

        # 2.6. rank for each trading_day
        for (trading_day, group), g in df.groupby(['trading_day', 'group']):
            best_config = dict(y_type=self.y_type, trading_day=trading_day, group=group)

            # 2.5.1. use the best config prediction for this y_type
            # 2.5.2. use best [eval_top_config] config prediction for each (y_type, trading_day)
            logging.info(f'best config for [{y_type}]: top ({self.eval_top_config}) [{self.eval_metrics}]')
            best_config_df = self.__backtest_find_calc_metric_avg(trading_day, group)
            eval = best_config_df[self.eval_col].mean().to_dict()
            best_config.update(eval)

            # 2.5.2a. calculate rank for all config
            for i, row in best_config_df.iterrows():
                best_config.update(row['config'])
                g_best = g.loc[(g[self.diff_config_col] == pd.Series(best_config)).all(axis=1)]
                rank_df = self.rank_each_trading_day(trading_day, g_best)

                # 2.6.3. append to history / currency df list
                if trading_day == df['trading_day'].max():  # if keep_all_history also write to prod table
                    self.all_current.append({"info": best_config, "rank_df": rank_df.copy()})
                self.all_history.append({"info": best_config, "rank_df": rank_df.copy()})

    @staticmethod
    def __calc_z_weight(q_, df, diff_config_col):
        """ calculate 'pred_z'/'factor_weight' by ranking within current testing_period """

        # 2.3.1. calculate factor_weight with qcut
        logging.info("Calculate factor_weight")
        groupby_keys = ['uid_hpot', 'group', 'trading_day'] + diff_config_col
        df['factor_weight'] = df.groupby(by=groupby_keys)['pred'].transform(
            lambda x: pd.qcut(x, q=q_, labels=False, duplicates='drop'))
        df = df.reset_index()

        # 2.3.2. count rank for debugging
        rank_count = df.groupby(groupby_keys)['factor_weight'].apply(pd.value_counts)
        rank_count = rank_count.unstack().fillna(0)
        logging.info(f'n_factor used:\n{rank_count}')

        # 2.3.3. calculate pred_z using mean & std of all predictions in entire testing history
        logging.info("Calculate pred_z")
        df['pred_z'] = df.groupby(by=['group'] + diff_config_col)['pred'].apply(lambda x: (x - np.mean(x)) / np.std(x))
        return df

    @staticmethod
    def __get_neg_factor_all(pred):
        """ get all neg factors for all (testing_period, group) """
        pred_unique = pred.groupby(['group', 'trading_day'])[['neg_factor']].first()
        if type(pred_unique["neg_factor"].to_list()[0]) != type([]):
            pred_unique["neg_factor"] = pred_unique["neg_factor"].apply(ast.literal_eval)
        neg_factor = pred_unique['neg_factor'].unstack().to_dict()
        return neg_factor

    def rank_each_trading_day(self, trading_day, g_best):
        """ (2.6) rank for each trading_day """

        logging.debug(f'calculate (neg_factor, factor_weight): {trading_day}')
        result_col = ['group', 'trading_day', 'factor_name', 'pred_z', 'factor_weight']
        df = g_best.loc[g_best['trading_day'] == trading_day, result_col].reset_index(drop=True)

        # merge same config iterations -> recalc factor_weight
        df = df.groupby(['group', 'trading_day', 'factor_name'])['pred_z'].mean().reset_index()
        df['factor_weight'] = df.groupby(by=['group', 'trading_day'])['pred_z'].transform(
            lambda x: pd.qcut(x, q=self.q_, labels=False, duplicates='drop'))

        # 2.6.1. record basic info
        df['factor_weight'] = df['factor_weight'].astype(int)
        df['last_update'] = dt.datetime.now()

        # 2.6.2. record neg_factor
        # original premium is "small - big" = short_large -> those marked neg_factor = long_large        
        df['long_large'] = False
        neg_factor = self.neg_factor[pd.Timestamp(trading_day)]
        for k, v in neg_factor.items():  # write neg_factor i.e. label factors
            if type(v) == type([]):
                df.loc[(df['group'] == k) & (df['factor_name'].isin(v)), 'long_large'] = True

        return df.sort_values(['group', 'pred_z'])

    # --------------------------------------- Download Prediction -------------------------------------------------

    def _download_prediction(self, name_sql, pred_y_type, pred_start_testing_period, pred_start_uid):
        """ merge factor_stock & factor_model_stock """

        try:
            logging.info(f'=== Load local prediction history on name_sql=[{name_sql}] ===')
            pred = pd.read_csv(f'pred_{name_sql}.csv')
        except Exception as e:
            logging.info(f'=== Download prediction history on name_sql=[{name_sql}] ===')

            conditions = [f"S.name_sql='{name_sql}'",
                          f"S.testing_period>='{pred_start_testing_period}'",
                          f"to_timestamp(left(S.uid, 20), 'YYYYMMDDHH24MISSUS') > "
                          f"to_timestamp('{pred_start_uid}', 'YYYYMMDDHH24MISSUS')"
                          ]
            if pred_y_type:
                conditions.append(f"S.y_type in {tuple(pred_y_type)}")

            label_col = ['name_sql', 'y_type', 'neg_factor', 'testing_period', 'cv_number',
                         'uid'] + self.diff_config_col
            query = text(f"""
                    SELECT P.pred, P.actual, P.factor_name, P.group, {', '.join(['S.' + x for x in label_col])} 
                    FROM {result_pred_table} P 
                    INNER JOIN (SELECT * FROM {result_score_table} WHERE group_code<>'currency') S ON (S.uid=P.uid) 
                    WHERE {' AND '.join(conditions)}
                    ORDER BY S.{uid_col}""".replace(",)", ")"))
            pred = read_query(query, db_url_read).rename(columns={"testing_period": "trading_day"}).fillna(0)
            if len(pred) > 0:
                pred.to_csv(f'pred_{name_sql}.csv', index=False)
            else:
                raise Exception(f"ERROR: No prediction download from DB with name_sql: [{name_sql}]")

        pred["trading_day"] = pd.to_datetime(pred["trading_day"])
        pred = pred.loc[pred['trading_day'] >= dt.strptime(pred_start_testing_period, "%Y-%m-%d")]
        return pred

    # ----------------------------------- Add Rank & Evaluation Metrics ---------------------------------------------

    def __backtest_save_eval_metrics(self, result_all, y_type, name_sql):
        """ evaluate & rank different configuration;
            save backtest evaluation metrics -> production_factor_rank_backtest_eval_table """

        # 2.4.1. calculate group statistic
        logging.debug("=== Update testing set results ===")
        
        # actual factor premiums
        result_all_avg = result_all.groupby(['name_sql', 'group', 'trading_day'])['actual'].mean()  
        result_all_comb = result_all.groupby(['name_sql', 'group', 'trading_day'] + self.diff_config_col
                                             ).apply(self.__get_summary_stats_in_group)
        result_all_comb = result_all_comb.loc[result_all_comb.index.get_level_values('trading_day') <
                                              result_all_comb.index.get_level_values('trading_day').max()].reset_index()
        result_all_comb[self.eval_col] = result_all_comb[self.eval_col].astype(float)
        result_all_comb = result_all_comb.join(result_all_avg, on=['name_sql', 'group', 'trading_day'], how='left')

        # 2.4.2. create DataFrame for eval results to DB
        result_all_comb["y_type"] = y_type
        result_all_comb["weeks_to_expire"] = self.weeks_to_expire
        primary_keys = ["name_sql", "group", "trading_day", "y_type"] + self.diff_config_col
        result_all_comb = uid_maker(result_all_comb, primary_key=primary_keys)

        # 2.4.3. save local plot for evaluation (when DEBUG)    # TODO: change to plot for debug
        # if DEBUG:
        #     rank_pred.__save_plot_backtest_ret(result_all_comb, self.diff_config_col, y_type, name_sql)

        # 2.4.2. add config JSON column for configs
        result_all_comb['config'] = result_all_comb[self.diff_config_col].to_dict(orient='records')
        # result_all_comb['config_mean_mse'] = result_all_comb.groupby(['group'] + self.diff_config_col)['mse'].transform('mean')
        # result_all_comb['config_mean_max_ret'] = result_all_comb.groupby(['group'] + self.diff_config_col)['max_ret'].transform('mean')
        result_all_comb = result_all_comb.drop(columns=self.diff_config_col)
        upsert_data_to_database(result_all_comb, production_factor_rank_backtest_eval_table, primary_key=["uid"],
                                db_url=db_url_write, how="update", dtype=backtest_eval_dtypes)


    def __get_summary_stats_in_group(self, g):
        """ Calculate basic evaluation metrics for factors """

        ret_dict = {}
        g = g.copy()
        c_date = g["trading_day"].to_list()[0]
        c_group = g["group"].to_list()[0]
        g['factor_name'] = [f'{x} (L)' if x in self.neg_factor[c_date][c_group] else f'{x} (S)' for x in
                            g['factor_name']]
        max_g = g[g['factor_weight'] == 2]
        min_g = g[g['factor_weight'] == 0]
        ret_dict['max_factor'] = dict(Counter(max_g['factor_name'].tolist()))
        ret_dict['min_factor'] = dict(Counter(min_g['factor_name'].tolist()))
        ret_dict['max_ret'] = max_g['actual'].mean()
        ret_dict['min_ret'] = min_g['actual'].mean()
        ret_dict['mae'] = mean_absolute_error(g['pred'], g['actual'])
        ret_dict['mse'] = mean_squared_error(g['pred'], g['actual'])
        ret_dict['r2'] = r2_score(g['pred'], g['actual'])
        return pd.Series(ret_dict)

    def __backtest_find_calc_metric_avg(self, trading_day, group):
        """ Select Best Config (among other_group_col) """

        conditions = [f"weeks_to_expire={self.weeks_to_expire}",
                      f"y_type='{self.y_type}'",
                      f"\"group\"='{group}'",
                      f"trading_day < '{trading_day}'",
                      f"trading_day >= '{trading_day - relativedelta(weeks=self.eval_config_select_period)}'"]
        if self.eval_current:
            conditions.append(f"name_sql='{self.name_sql}'")

        query = f"SELECT * FROM {production_factor_rank_backtest_eval_table} WHERE {' AND '.join(conditions)} "
        df = read_query(query, global_vars.db_url_alibaba_prod)

        df = pd.concat([df, pd.DataFrame(df['config'].to_list())], axis=1)
        df_mean = df.groupby(['group'] + self.diff_config_col).mean().reset_index()  # average over trading_day
        df_mean['net_ret'] = df_mean['max_ret'] - df_mean['min_ret']

        if self.eval_metric in ['max_ret', 'net_ret', 'r2']:
            best = df_mean.groupby('group').apply(lambda x: x.nlargest(self.eval_top_config, self.eval_metric, keep="all"))
        elif self.eval_metric in ['mae', 'mse']:
            best = df_mean.groupby('group').apply(lambda x: x.nsmallest(self.eval_top_config, self.eval_metric, keep="all"))
        else:
            best = None
        # logging.info(f'best_iter:\n{best}')
        return best

    # --------------------------------------- Save Prod Table to DB -------------------------------------------------

    def write_backtest_rank_(self, upsert_how=False):
        """ write backtest factors: backtest rank -> production_factor_rank_backtest_table """
        tbl_name_backtest = production_factor_rank_backtest_table

        if type(self.eval_top_config) == type(None):
            all_history = []
            for i in self.all_history:
                df_history = i["rank_df"]
                df_history["weeks_to_expire"] = self.weeks_to_expire
                df_info = pd.DataFrame(i["info"]).ffill(0)
                all_history.append({"rank_df": df_history.copy(), "info": df_info})
        else:
            all_history = pd.concat([x["rank_df"] for x in self.all_history], axis=0)
            all_history = all_history.groupby(['group', 'trading_day', 'factor_name']).mean().reset_index()
            all_history['factor_weight'] = all_history.groupby(by=['group', 'trading_day'])['pred_z'].transform(
                lambda x: pd.qcut(x, q=self.q_, labels=False, duplicates='drop'))
            all_history['last_update'] = dt.datetime.now()
            all_history["weeks_to_expire"] = self.weeks_to_expire
            all_history = all_history.drop_duplicates(subset=["group", "trading_day", "factor_name", "weeks_to_expire"],
                                                      keep="last")

        if upsert_how == False:
            return all_history
        # elif upsert_how=="append":
        #     delete_data_on_database(tbl_name_backtest, db_url_write, query=f"weeks_to_expire='{self.weeks_to_expire}'")
        #     upsert_data_to_database(all_history, tbl_name_backtest,
        #                             primary_key=["group", "trading_day", "factor_name", "weeks_to_expire"],
        #                             db_url=db_url_write, how="append", dtype=rank_dtypes)
        # else:
        #     upsert_data_to_database(all_history, tbl_name_backtest,
        #                             primary_key=["group", "trading_day", "factor_name", "weeks_to_expire"],
        #                             db_url=db_url_write, how=upsert_how, dtype=rank_dtypes)
        return all_history

    def write_current_rank_(self):
        """write current use factors: current rank -> production_factor_rank_table / production_factor_rank_history"""

        tbl_name_current = production_factor_rank_table
        tbl_name_history = production_factor_rank_history_table

        df_current = pd.concat([x["rank_df"] for x in self.all_current], axis=0)
        df_current = df_current.groupby(['group', 'trading_day', 'factor_name']).mean().reset_index()
        df_current['factor_weight'] = df_current.groupby(by=['group', 'trading_day'])['pred_z'].transform(
            lambda x: pd.qcut(x, q=self.q_, labels=False, duplicates='drop'))
        df_current['last_update'] = dt.datetime.now()
        df_current["weeks_to_expire"] = self.weeks_to_expire
        df_current = df_current.drop_duplicates(subset=["group", "trading_day", "factor_name", "weeks_to_expire"],
                                                keep="last")

        # update [production_factor_rank_table]
        upsert_data_to_database(df_current, tbl_name_current,
                                primary_key=["group", "factor_name", "weeks_to_expire"],
                                db_url=db_url_write, how='update', dtype=rank_dtypes)

        # update [production_factor_rank_history_table]
        df_current = df_current.drop(columns=["trading_day"])
        upsert_data_to_database(df_current, tbl_name_history,
                                primary_key=["group", "factor_name", "weeks_to_expire", "last_update"],
                                db_url=db_url_write, how='update', dtype=rank_dtypes)

    def write_to_db(self):
        """ concat rank current/history & write """

        if not DEBUG:
            all_history = self.write_backtest_rank_()
            self.write_current_rank_()
            return all_history
        else:
            logging.INFO("calculation_rank will not return [all_history] in DEBUG mode")
            return False

    # ---------------------------------- Save local Plot for evaluation --------------------------------------------

    @staticmethod
    def __save_plot_backtest_ret(result_all_comb, other_group_col, y_type, name_sql):
        """ Save Plot for backtest average ret """

        logging.debug(f'=== Save Plot for backtest average ret ===')
        result_all_comb = result_all_comb.copy()
        result_all_comb['other_group'] = result_all_comb[other_group_col].astype(str).agg('-'.join, axis=1)
        num_group = len(result_all_comb['group'].unique())
        num_other_group = len(result_all_comb['other_group'].unique())

        # create figure for test & train boxplot
        fig = plt.figure(figsize=(num_group * 8, num_other_group * 4), dpi=120, constrained_layout=True)
        k = 1
        for name, g in result_all_comb.groupby(['other_group', 'group']):
            ax = fig.add_subplot(num_other_group, num_group, k)
            g[['max_ret', 'actual', 'min_ret']] = (g[['max_ret', 'actual', 'min_ret']] + 1).cumprod(axis=0)
            plot_df = g.set_index(['trading_day'])[['max_ret', 'actual', 'min_ret']]
            ax.plot(plot_df)
            for i in range(3):
                ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
            if (k % num_group == 1) or (num_group == 1):
                ax.set_ylabel(name[0], fontsize=20)
            if k > (num_other_group - 1) * num_group:
                ax.set_xlabel(name[1], fontsize=20)
            if k == 1:
                plt.legend(['best', 'average', 'worse'])
            k += 1
        fig_name = f'#pred_{name_sql}_{y_type}.png'
        plt.suptitle(' - '.join(other_group_col), fontsize=20)
        plt.savefig(fig_name)
        plt.close()
        logging.debug(f'=== Saved [{fig_name}] for evaluation ===')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=float, default=1 / 3)
    parser.add_argument('--weeks_to_expire', type=str, default='%%')
    parser.add_argument('--average_days', type=str, default='%%')
    args = parser.parse_args()

    # Example
    # rank_pred(1/3, name_sql='w26_d7_20220207153438_debug', pred_start_testing_period=None, y_type=[]).write_to_db()
    rank_pred(1 / 3, name_sql='w4_d7_official', pred_start_testing_period=None, y_type=[], eval_top_config=10).write_to_db()
    rank_pred(1 / 3, name_sql='w8_d7_official', pred_start_testing_period=None, y_type=[], eval_top_config=10).write_to_db()
    rank_pred(1 / 3, name_sql='w13_d7_official', pred_start_testing_period=None, y_type=[], eval_top_config=10).write_to_db()
    # calc_rank = rank_pred(1/3, name_sql='w26_d7_official', pred_start_testing_period=None, eval_top_config=10).write_to_db()

    # rank_pred(1/3, weeks_to_expire=1, average_days=1, pred_start_testing_period=None, y_type=[]).write_to_db()
    # rank_pred(1/3, weeks_to_expire=26, pred_start_testing_period=None, y_type=[], pred_start_uid='20220128000000389209').write_to_db()

    # from results_analysis.score_backtest import score_history
    # score_history(self.weeks_to_expire)
