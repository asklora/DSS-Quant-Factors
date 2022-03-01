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
from sqlalchemy.dialects.postgresql.base import DATE, DOUBLE_PRECISION, TEXT, INTEGER, BOOLEAN, TIMESTAMP
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

rank_history_dtypes = dict(
    group=TEXT,
    factor_name=TEXT,
    weeks_to_expire=INTEGER,
    factor_weight=INTEGER,
    pred_z=DOUBLE_PRECISION,
    long_large=BOOLEAN,
    last_update=TIMESTAMP
)

rank_dtypes = rank_history_dtypes.update({"trading_day": DATE})

class rank_pred:
    ''' process raw prediction in result_pred_table -> production_factor_rank_table for AI Score calculation '''

    def __init__(self, q, weeks_to_expire='%%', average_days='%%', name_sql=None, y_type=None, eval_start_date=None,
                 start_uid=None, top_config=None):
        '''
        Parameters
        ----------
        q (Float):
            If q > 1 (e.g. 5), top q factors used as best factors;
            If q < 1 (e.g. 1/3), top (q * total No. factors predicted) factors am used as best factors;
        weeks_to_expire (Int, Optional):
            number of weeks to predict (default evaluate all model)
        average_days (Int, Optional):
            number of average days when calculate premiums (default evaluate all model)
        name_sql (Str, Optional):
            if not None (by default), name_sql to evaluate (This overwrites "weeks_to_expire" & "average_days" set)
        y_type (List[Str], Optional):
            y_type to evaluate;
        eval_start_date (Str, Optional):
            String in "%Y-%m-%d" format for the start date to download prediction;
        start_uid (Str, Optional):
            String in "%Y%m%d%H%M%S%f" format to filter factor_model records based on training start time
        '''

        self.q = q
        self.eval_start_date = eval_start_date
        self.name_sql = name_sql
        self.top_config = top_config
        if weeks_to_expire=="%%":
            self.weeks_to_expire = int(name_sql.split('_')[0][1:])
        else:
            self.weeks_to_expire = int(weeks_to_expire)

        # keep 'cv_number' in last one for averaging
        self.iter_unique_col = ['name_sql', 'group', 'trading_day', 'factor_name', 'cv_number']
        self.diff_config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', 'use_average', 'down_mkt_pct']
        # self.diff_config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', 'use_average']

        # 1. Download & merge all prediction from iteration
        pred = self._download_prediction(weeks_to_expire, average_days, name_sql, eval_start_date, y_type, start_uid)

        # best = {'tree_type': 'rf1', 'use_pca': 0, 'qcut_q': 10, 'n_splits': 0.1,
        #         'valid_method': 2012, 'use_average': False, 'down_mkt_pct': 0.6,
        #         "trading_day": dt.datetime(2021, 6, 20, 0, 0, 0), "group": "USD"}
        # pred_best = pred.loc[(pred[list(best.keys())]==pd.Series(best)).all(axis=1)].sort_values(by=["y_type", "pred"])

        pred['uid_hpot'] = pred['uid'].str[:20]

        for name_sql, pred_name_sql in pred.groupby(["name_sql"]):
            self.neg_factor = rank_pred.__get_neg_factor_all(pred_name_sql)

            # 2. Process separately for each y_type (i.e. momentum/value/quality/all)
            self.all_current = []
            self.all_history = []
            for y_type, g in pred_name_sql.groupby('y_type'):
                self.y_type = y_type
                self.rank_each_y_type(q, y_type, g, name_sql)

    def rank_each_y_type(self, q, y_type, df, name_sql):
        ''' rank for each y_type '''

        logging.info(f'=== Generate rank for [{y_type}] with results={df.shape}) ===')

        # 2.1. remove duplicate samples from running twice when testing
        df = df.drop_duplicates(subset=['uid'] + self.iter_unique_col+self.diff_config_col, keep='last')
        df['trading_day'] = pd.to_datetime(df['trading_day'])

        # 2.2. use average predictions from different validation sets
        df_cv_avg = df.groupby(['uid_hpot']+self.iter_unique_col[:-1]+self.diff_config_col)[['pred', 'actual']].mean()

        # 2.3. calculate 'pred_z'/'factor_weight' by ranking
        q = q/len(df['factor_name'].unique()) if isinstance(q, int) else q  # convert q(int) -> ratio
        self.q_ = [0., q, 1.-q, 1.]
        df_cv_avg = rank_pred.__calc_z_weight(self.q_, df_cv_avg, self.diff_config_col)

        # 2.4. save backtest evaluation metrics
        eval_metrics = self.__backtest_save_eval_metrics(df_cv_avg, y_type, name_sql)

        # 2.5.1. use best config prediction for this y_type
        eval_metrics_avg = self.__backtest_find_calc_metric_avg(df=eval_metrics)

        # 2.5.2. use best config prediction for this y_type  (need to replace rank part)
        if type(self.top_config)!=type(None):
            logging.info(f'best config for [{y_type}]: top ({self.top_config})')
            eval_metrics_avg = self.__backtest_find_best_config(df_mean=eval_metrics_avg, eval_metric='mse')

        # 2.5.2a. calculate rank for all config
        eval_col = ['max_ret', 'net_ret', 'r2', 'mae', 'mse']
        for configs, config_g in eval_metrics_avg.groupby(self.diff_config_col):
            best_config = dict(zip(self.diff_config_col, configs))
            eval = config_g.set_index(['group'])[eval_col].to_dict()

            logging.info(f'best config for [{y_type}]: {best_config}')
            df_cv_avg_best = []
            for name, g in df_cv_avg.groupby(['group']):
                if name in config_g['group'].to_list():
                    g_best = g.loc[(g[self.diff_config_col]==pd.Series(best_config)).all(axis=1)]
                    df_cv_avg_best.append(g_best)
            df_cv_avg_best = pd.concat(df_cv_avg_best, axis=0)

            best_config.update(eval)
            best_config["y_type"] = self.y_type

            # 2.6. rank for each trading_day
            rank_df_history = []
            for period in df_cv_avg_best['trading_day'].unique():
                rank_df = self.rank_each_trading_day(period, df_cv_avg_best)

                # 2.6.3. append to history / currency df list
                rank_df_history.append(rank_df)
                if (period == df['trading_day'].max()):  # if keep_all_history also write to prod table
                    self.all_current.append({"info": best_config, "rank_df": rank_df.copy()})

            rank_df_history = pd.concat(rank_df_history, axis=0).sort_values(by=['trading_day', 'group', 'factor_name'])
            self.all_history.append({"info": best_config, "rank_df": rank_df_history.copy()})

    @staticmethod
    def __calc_z_weight(q_, df, diff_config_col):
        ''' calculate 'pred_z'/'factor_weight' by ranking within current testing_period '''

        # 2.3.1. calculate factor_weight with qcut
        groupby_keys = ['uid_hpot', 'group', 'trading_day'] + diff_config_col
        df['factor_weight'] = df.groupby(by=groupby_keys)['pred'].transform(
            lambda x: pd.qcut(x, q=q_, labels=False, duplicates='drop'))
        df = df.reset_index()

        # 2.3.2. count rank for debugging
        rank_count = df.groupby(groupby_keys)['factor_weight'].apply(pd.value_counts)
        rank_count = rank_count.unstack().fillna(0)
        logging.info(f'n_factor used:\n{rank_count}')

        # 2.3.3. calculate pred_z using mean & std of all predictions in entire testing history
        df['pred_z'] = df.groupby(by=['group']+diff_config_col)['pred'].apply(lambda x: (x-np.mean(x))/np.std(x))
        return df

    @staticmethod
    def __get_neg_factor_all(pred):
        ''' get all neg factors for all (testing_period, group) '''
        if type(pred["neg_factor"].to_list()[0])!=type([]):
            pred["neg_factor"] = pred["neg_factor"].apply(ast.literal_eval)
        pred_unique = pred.drop_duplicates(subset=['group', 'trading_day'])
        neg_factor = pred_unique.set_index(['group', 'trading_day'])['neg_factor'].unstack().to_dict()
        return neg_factor

    def rank_each_trading_day(self, period, result_all):
        ''' (2.6) rank for each trading_day '''

        logging.debug(f'calculate (neg_factor, factor_weight): {period}')
        result_col = ['group', 'trading_day', 'factor_name', 'pred_z', 'factor_weight']
        df = result_all.loc[result_all['trading_day'] == period, result_col].reset_index(drop=True)

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
        neg_factor = self.neg_factor[pd.Timestamp(period)]
        for k, v in neg_factor.items():  # write neg_factor i.e. label factors
            if type(v)==type([]):
                df.loc[(df['group'] == k) & (df['factor_name'].isin(v)), 'long_large'] = True

        return df.sort_values(['group', 'pred_z'])

    # --------------------------------------- Download Prediction -------------------------------------------------

    def _download_prediction(self, weeks_to_expire, average_days, name_sql, eval_start_date, y_type, start_uid):
        ''' merge factor_stock & factor_model_stock '''

        try:
            logging.info('=== Load local prediction history ===')
            pred = pd.read_csv(f'pred_{name_sql}.csv')
        except:
            logging.info('=== Download prediction history ===')
            if name_sql:
                conditions = [f"S.name_sql='{name_sql}'"]
                logging.warning(f"=== Will update factor rank tables based on name_sql=[{name_sql}] ===")
            else:
                conditions = [f"S.name_sql like 'w{weeks_to_expire}_d{average_days}_%%'"]
            uid_col = "uid"
            if eval_start_date:
                conditions.append(f"S.testing_period>='{eval_start_date}'")
            if y_type:
                conditions.append(f"S.y_type in {tuple(y_type)}")
            if start_uid:
                conditions.append(f"to_timestamp(left(S.uid, 20), 'YYYYMMDDHH24MISSUS') > "
                                  f"to_timestamp('{start_uid}', 'YYYYMMDDHH24MISSUS')")

            label_col = ['name_sql', 'y_type', 'neg_factor', 'testing_period', 'cv_number', 'uid'] + self.diff_config_col
            query = text(f'''
                    SELECT P.pred, P.actual, P.factor_name, P.group, {', '.join(['S.'+x for x in label_col])} 
                    FROM {result_pred_table} P 
                    INNER JOIN (SELECT * FROM {result_score_table} WHERE group_code<>'currency') S ON ((S.{uid_col}=P.{uid_col})) 
                    WHERE {' AND '.join(conditions)}
                    ORDER BY S.{uid_col}'''.replace(",)", ")"))
            pred = read_query(query, db_url_read).rename(columns={"testing_period": "trading_day"}).fillna(0)
            pred.to_csv(f'pred_{name_sql}.csv', index=False)
        pred["trading_day"] = pd.to_datetime(pred["trading_day"])

        # x = pred.groupby(['trading_day','group']+self.diff_config_col)['pred'].count().sort_values()
        # x = pred.groupby(['trading_day','group'])['pred'].count()
        # x = x.unstack()
        # x = x.reset_index()
        # x = x.sort_values(by=['trading_day'])
        # x.to_csv('debug_pred.csv', index=False)

        return pred

    # ----------------------------------- Add Rank & Evaluation Metrics ---------------------------------------------

    def __backtest_save_eval_metrics(self, result_all, y_type, name_sql):
        ''' evaluate & rank different configuration;
            save backtest evaluation metrics -> production_factor_rank_backtest_eval_table '''

        # 2.4.1. calculate group statistic
        logging.debug("=== Update testing set results ===")
        result_all_avg = result_all.groupby(['name_sql', 'group', 'trading_day'])['actual'].mean()  # actual factor premiums
        result_all_comb = result_all.groupby(['name_sql', 'group', 'trading_day']+self.diff_config_col
                                             ).apply(self.__get_summary_stats_in_group)
        result_all_comb = result_all_comb.loc[result_all_comb.index.get_level_values('trading_day') <
                                              result_all_comb.index.get_level_values('trading_day').max()].reset_index()
        result_all_comb[['max_ret','min_ret','mae','mse','r2']] = result_all_comb[['max_ret','min_ret','mae','mse','r2']].astype(float)
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
        result_all_comb['config_mean_mse'] = result_all_comb.groupby(['group'] + self.diff_config_col)['mse'].transform('mean')
        result_all_comb['config_mean_max_ret'] = result_all_comb.groupby(['group'] + self.diff_config_col)['max_ret'].transform('mean')
        result_all_comb = result_all_comb.drop(columns=self.diff_config_col)
        upsert_data_to_database(result_all_comb, production_factor_rank_backtest_eval_table, primary_key=["uid"],
                                db_url=db_url_write, how="update", dtype=rank_history_dtypes)

        return result_all_comb

    def __get_summary_stats_in_group(self, g):
        ''' Calculate basic evaluation metrics for factors '''

        ret_dict = {}
        g = g.copy()
        c_date = g["trading_day"].to_list()[0]
        c_group = g["group"].to_list()[0]
        g['factor_name'] = [f'{x} (L)' if x in self.neg_factor[c_date][c_group] else f'{x} (S)' for x in g['factor_name']]
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

    def __backtest_find_calc_metric_avg(self, df=None):
        ''' Select Best Config (among other_group_col) '''

        if type(df)==type(None):
            query = f"SELECT * FROM {production_factor_rank_backtest_eval_table} " \
                    f"WHERE weeks_to_expire={self.weeks_to_expire} AND y_type='{self.y_type}'"
            if self.name_sql:
                query += f" AND name_sql='{self.name_sql}'"
            df = read_query(query, global_vars.db_url_alibaba_prod)
        df = pd.concat([df, pd.DataFrame(df['config'].to_list())], axis=1)
        df_mean = df.groupby(['group'] + self.diff_config_col).mean().reset_index()     # average over trading_day
        df_mean['net_ret'] = df_mean['max_ret'] - df_mean['min_ret']

        return df_mean # Return: best = {Currency -> [Config -> {Config dict}, Metrics -> {Metrics dict}]}

    def __backtest_find_best_config(self, df_mean, eval_metric='mse'):
        ''' '''
        if eval_metric in ['max_ret', 'net_ret', 'r2']:
            best = df_mean.groupby('group').apply(lambda x: x.nlargest(self.top_config, eval_metric, keep="all"))
        elif eval_metric in ['mae', 'mse']:
            best = df_mean.groupby('group').apply(lambda x: x.nsmallest(self.top_config, eval_metric, keep="all"))
        logging.info(f'best_iter:\n{best}')
        return best

    # --------------------------------------- Save Prod Table to DB -------------------------------------------------

    def write_backtest_rank_(self, upsert_how="append"):
        ''' write backtest factors: backtest rank -> production_factor_rank_backtest_table '''
        tbl_name_backtest = production_factor_rank_backtest_table

        if type(self.top_config)==type(None):
            all_history = []
            for i in self.all_history:
                df_history = i["rank_df"]
                df_history["weeks_to_expire"] = self.weeks_to_expire
                df_info = pd.DataFrame(i["info"]).ffill(0)
                all_history.append({"rank_df": df_history.copy(), "info": df_info})
        else:
            all_history = pd.concat([x["rank_df"] for x in self.all_history], axis=0)
            all_history = all_history.groupby(['group','trading_day','factor_name']).mean().reset_index()
            all_history['factor_weight'] = all_history.groupby(by=['group','trading_day'])['pred_z'].transform(
                lambda x: pd.qcut(x, q=self.q_, labels=False, duplicates='drop'))
            all_history['last_update'] = dt.datetime.now()
            all_history["weeks_to_expire"] = self.weeks_to_expire
            all_history = all_history.drop_duplicates(subset=["group", "trading_day", "factor_name", "weeks_to_expire"], keep="last")

        if upsert_how==False:
            return all_history
        elif upsert_how=="append":
            # trucncate_table_in_database(tbl_name_backtest, db_url_write)
            upsert_data_to_database(all_history, tbl_name_backtest,
                                    primary_key=["group", "trading_day", "factor_name", "weeks_to_expire"],
                                    db_url=db_url_write, how="append", dtype=rank_dtypes)
        else:
            upsert_data_to_database(all_history, tbl_name_backtest,
                                    primary_key=["group", "trading_day", "factor_name", "weeks_to_expire"],
                                    db_url=db_url_write, how=upsert_how, dtype=rank_dtypes)

    def write_current_rank_(self):
        '''write current use factors: current rank -> production_factor_rank_table / production_factor_rank_history'''

        tbl_name_current = production_factor_rank_table
        tbl_name_history = production_factor_rank_history_table
        
        df_current = pd.concat([x["rank_df"] for x in self.all_current], axis=0)
        df_current = df_current.groupby(['group', 'trading_day', 'factor_name']).mean().reset_index()
        df_current['factor_weight'] = df_current.groupby(by=['group', 'trading_day'])['pred_z'].transform(
            lambda x: pd.qcut(x, q=self.q_, labels=False, duplicates='drop'))
        df_current['last_update'] = dt.datetime.now()
        df_current["weeks_to_expire"] = self.weeks_to_expire
        df_current = df_current.drop_duplicates(subset=["group", "trading_day", "factor_name", "weeks_to_expire"], keep="last")

        # update [production_factor_rank_table]
        upsert_data_to_database(df_current, tbl_name_current,
                                primary_key=["group", "factor_name", "weeks_to_expire"],
                                db_url=db_url_write, how='update', dtype=rank_dtypes)

        # update [production_factor_rank_history_table]
        df_current = df_current.drop(columns=["trading_day"])
        upsert_data_to_database(df_current, tbl_name_history,
                                primary_key=["group", "factor_name", "weeks_to_expire", "last_update"],
                                db_url=db_url_write, how='update', dtype=rank_history_dtypes)

    def write_to_db(self):
        ''' concat rank current/history & write '''

        if not DEBUG:
            self.write_backtest_rank_()
            self.write_current_rank_()

    # ---------------------------------- Save local Plot for evaluation --------------------------------------------

    @staticmethod
    def __save_plot_backtest_ret(result_all_comb, other_group_col, y_type, name_sql):
        ''' Save Plot for backtest average ret '''

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
            if (k % num_group == 1) or (num_group==1):
                ax.set_ylabel(name[0], fontsize=20)
            if k > (num_other_group - 1) * num_group:
                ax.set_xlabel(name[1], fontsize=20)
            if k == 1:
                plt.legend(['best', 'average', 'worse'])
            k += 1
        fig_name = f'#pred_{name_sql}_{y_type}.png'
        plt.suptitle(' - '.join(other_group_col),  fontsize=20)
        plt.savefig(fig_name)
        plt.close()
        logging.debug(f'=== Saved [{fig_name}] for evaluation ===')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=float, default=1/3)
    parser.add_argument('--weeks_to_expire', type=str, default='%%')
    parser.add_argument('--average_days', type=str, default='%%')
    args = parser.parse_args()

    # Example
    # rank_pred(1/3, name_sql='w26_d7_20220207153438_debug', eval_start_date=None, y_type=[]).write_to_db()
    rank_pred(1/3, name_sql='w4_d7_official', eval_start_date=None, y_type=[], top_config=10).write_to_db()
    rank_pred(1/3, name_sql='w8_d7_official', eval_start_date=None, y_type=[], top_config=10).write_to_db()
    rank_pred(1/3, name_sql='w13_d7_official', eval_start_date=None, y_type=[], top_config=10).write_to_db()
    rank_pred(1/3, name_sql='w26_d7_official', eval_start_date=None, y_type=[], top_config=10).write_to_db()

    # rank_pred(1/3, weeks_to_expire=1, average_days=1, eval_start_date=None, y_type=[]).write_to_db()
    # rank_pred(1/3, weeks_to_expire=26, eval_start_date=None, y_type=[], start_uid='20220128000000389209').write_to_db()

    # from results_analysis.score_backtest import score_history
    # score_history(self.weeks_to_expire)

