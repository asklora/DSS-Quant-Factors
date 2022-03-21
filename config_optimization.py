import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import gc

import global_vars
from general.send_slack import to_slack
from general.sql_process import read_query, read_table, trucncate_table_in_database, upsert_data_to_database

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from lightgbm import cv, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, auc, precision_score
from itertools import product

cls_lgbm_space = reg_lgbm_space = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [128, 256, 512]),
    'num_leaves': hp.quniform('num_leaves', 50, 300, 50),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [1]),
    'feature_fraction': hp.quniform('feature_fraction', 0.4, 0.8, 0.2),
    'bagging_fraction': hp.quniform('bagging_fraction', 0.4, 0.8, 0.2),
    'bagging_freq': hp.quniform('bagging_freq', 2, 8, 2),
    'min_gain_to_split': 0,
    'lambda_l1': 0,
    'lambda_l2': hp.quniform('lambda_l2', 0, 40, 20),
    'verbose': 1,
    'random_state': 0,
}

eval_metric_cls = ["multiclass", "multi_error", "auc_mu"]


def get_timestamp_now_str():
    """ return timestamp in form of string of numbers """
    return str(dt.datetime.now()).replace('.', '').replace(':', '').replace('-', '').replace(' ', '')


class HPOT:
    """ use hyperopt on each set """

    eval_metric = 'accuracy'

    def __init__(self, sql_result, max_evals, sample_set, feature_names=None, sample_names=None):
        self.sql_result = sql_result
        self.sample_set = sample_set
        self.feature_names = feature_names
        self.sample_names = sample_names
        self.hpot_start = get_timestamp_now_str()  # start time for entire traing (for all output)
        self.max_evals = max_evals

        trials = Trials()
        kwargs = {"algo": tpe.suggest, "max_evals": self.max_evals, "trials": trials}
        self.__reset_result_dict(reg=True)
        best = fmin(fn=self.__eval, space=reg_lgbm_space, **kwargs)

        self.__save_feature_importance()
        # self.__save_test_prediction()
        self.__save_model()

    # ============================================== Evaluation =====================================================

    def __eval(self, space):
        """ train & evaluate LightGBM on given rf_space by hyperopt trials with Regression model """

        self.sql_result['uid'] = self.hpot_start + get_timestamp_now_str()

        # Training
        params = self.__int_params_to_int(space)
        cvboosters = self.__lgbm_train(params, self.sample_set)

        cv_score = []
        all_prediction = []
        cv_number = 0
        for model in cvboosters:
            self.sql_result['uid'] = self.sql_result['uid'][:40] + str(cv_number)

            result = {}
            for i in ['train', 'test']:
                pred = model.predict(self.sample_set[f'{i}_x'])
                for k, func in {"accuracy": accuracy_score, "auc": auc, "precision": precision_score}.items():
                    result[f"{k}_{i}"] = func(self.sample_set[f'{i}_y'], pred)

            cv_score.append(result[self.eval_metric])
            self.sql_result.update(result)  # update result of model
            self.hpot['all_results'].append(self.sql_result.copy())

            feature_importance_df = pd.DataFrame(model.feature_importance_, index=self.feature_names)
            feature_importance_df = feature_importance_df.sort_values('split', ascending=False).reset_index()
            feature_importance_df['uid'] = self.sql_result['uid']
            self.hpot['all_importance'].append(feature_importance_df.copy())

            pred_prob = model.predict_proba(self.sample_set[f'{i}_x'])
            all_prediction.append(pred_prob)

            cv_number += 1

        cv_score_mean = cv_score.mean()
        if cv_score_mean > self.hpot['best_score']:  # update best_mae to the lowest value for Hyperopt
            self.hpot['best_score'] = cv_score_mean
            self.hpot['best_prediction'] = all_prediction
        gc.collect()

        return cv_score_mean

    # ============================================ Training =======================================================

    def __lgbm_train(self, params, sample_set, plot_eval=True, nfold=5):
        """ train lightgbm booster based on training / validaton set -> give predictions of Y """

        train_set = Dataset(sample_set['train_x'], sample_set['train_y'],
                            weight=sample_set['class_weight'],
                            # group=sample_set['group'],
                            feature_name=self.feature_names)

        eval_hist, cvboosters = cv(params=params,
                                   train_set=train_set,
                                   metrics=self.sql_result['objective'],
                                   nfold=nfold,
                                   num_boost_round=10,
                                   early_stopping_rounds=150,
                                   eval_train_metric=eval_metric_cls,
                                   return_cvbooster=True,
                                   )

        if plot_eval:
            self.__plot_eval_results(eval_hist)

        return cvboosters.boosters

    # ========================================== Results Saving ===================================================

    @staticmethod
    def __plot_eval_results(eval_results):
        """ plot eval results """

        n_metrics = len(eval_metric_cls)
        fig, ax = plt.subplots(nrows=1, ncols=n_metrics, figsize=(n_metrics * 5, 5))

        for k, v in eval_results.items():
            i = 0
            for k1, v1 in v.items():
                ax[i].plot(v1, label=k)
                ax[i].set_title(k1)
                i += 1
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'plot/trial2_{self.start_time}.png')

    def __save_feature_importance(self):
        """ save feature importance to DB """

        model_table = global_vars.factor_config_importance_table
        df = pd.concat(self.hpot['all_importance'], axis=0)
        upsert_data_to_database(df, model_table, how="append", verbose=-1)

    def __save_test_prediction(self, multioutput=False):
        """ save testing set prediction to DB """
        # TODO: change
        pass
        # df = pd.DataFrame(self.hpot['best_stock_df'], index=self.sample_names)
        # if multioutput:
        #     df.columns = labels
        #     df = df.unstack().reset_index()
        #     df = df.rename(columns={0: "value", "level_0": "uid", "level_1": "ticker"})
        #     df["uid"] = self.hpot['best_uid'] + df["uid"]  # Hyperopt trial start time
        # else:
        #     df = df.reset_index().rename(columns={0: "value", "index": "ticker"})
        #     df["uid"] = self.hpot['best_uid']
        # upsert_data_to_database(df, model_stock_table, db_url=global_vars.db_url_alibaba, how="append", verbose=1)

    def __save_model(self):
        """ save model config & score to DB """

        model_table = global_vars.factor_config_score_table
        df = pd.DataFrame(self.hpot['all_results'])
        upsert_data_to_database(df, model_table, primary_key="uid", how="append", verbose=1)

    # ========================================== Utils Funcs ===================================================

    def __reset_result_dict(self, reg=True):
        """ initiate reset results space """
        if reg:
            self.hpot = {'all_results': [], 'best_score': 10000}  # initial MAE/MSE Score = 10000
        else:
            self.hpot = {'all_results': [], 'best_score': 0}  # initial accuracy_score = 0
        self.sql_result = {"uid": get_timestamp_now_str()}
        self.sql_result.update(self.config)

    @staticmethod
    def __int_params_to_int(space):
        """ reset results space -> {} -> need to be done after each Hyperopt """
        params = space.copy()
        for k, v in params.items():
            if type(v) == float:
                if v.is_integer():
                    params[k] = int(v)
        return params


class load_date:

    def __init__(self, df):
        """ create DataFrame for x, y for all testing periods """

        # x = all configurations
        df_x = pd.DataFrame(df['config'].to_list()).drop(columns=['tree_type'])
        if list(df['group'].unique()) != ['USD']:
            df_x['is_usd'] = (df['group_code'] == "USD")
        df_x['q'] = df['q'].copy()
        df_x['actual'] = df['actual'].copy()
        df_x['testing_period'] = df['testing_period'].copy()
        self.feature_names = df.columns.to_list()

        # add whether using [max_ret] column to x
        df_x['max_ret'] = True
        df_x_copy = df_x.copy()
        df_x_copy['max_ret'] = False
        self.df_x = df_x.append(df_x_copy).reset_index(drop=True)

        # y = max_ret + net_ret
        df_y_max = df['max_ret'] - df['actual']
        df_y_net = df['max_ret'] - df['min_ret']
        self.df_y = df_y_max.append(df_y_net).reset_index(drop=True)

    def split_all(self, testing_period, qcut_q=3):
        # [x]: split train / test
        X_train = self.df_x.loc[self.df_x["testing_period"] < testing_period]
        X_train['testing_period'] -= testing_period
        X_test = self.df_x.loc[self.df_x["testing_period"] == testing_period]
        X_test['testing_period'] = 0

        # [y]: split train / test + qcut
        y_train = self.df_y.loc[self.df_x["testing_period"] < testing_period]
        y_train, cut_bins = pd.qcut(y_train, q=qcut_q, retbins=True, labels=False)
        y_test = self.df_y.loc[self.df_x["testing_period"] == testing_period]
        y_test = pd.cut(y_test, bins=cut_bins, labels=False)

        # [x]: apply standard scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train.values)
        X_test = scaler.transform(X_test.values)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # --------------------------------- Parser --------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='multiclass')
    parser.add_argument('--name_sql', default='w4_d-7_20220312222718_debug')
    parser.add_argument('--qcut_q', type=int, default=3)
    args = parser.parse_args()

    # --------------------------------- Load Data -----------------------------------------------

    df = read_query(f"SELECT * FROM {global_vars.production_factor_rank_backtest_eval_table} "
                    f"WHERE name_sql='{args.name_sql}'")
    df.to_pickle(f'eval_{args.name_sql}.pkl')

    sql_result = vars(args).copy()  # data write to DB TABLE lightgbm_results
    sql_result['name_sql2'] = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')  # name_sql for config opt
    sql_result['class_weight'] = {i: 1 for i in range(args.qcut_q)}

    # df = pd.read_pickle(args.name_sql + '.pkl')
    print(df)

    testing_period = np.sort(df['testing_period'].unique())[-12:]

    for (group, y_type), g in df.groupby(['group', 'y_type']):
        data = load_date(g)
        for t in testing_period:
            X_train, X_test, y_train, y_test = data.split_all(t, qcut_q=args.qcut_q)
            HPOT(sql_result=sql_result,
                 max_evals=10,
                 sample_set={"train_x": X_train, "test_x": X_test, "train_y": y_train, "test_y": y_test},
                 feature_names=data.feature_names)