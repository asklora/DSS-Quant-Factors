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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
from itertools import product
from functools import partial

cls_lgbm_space = {
    'objective': 'multiclass',
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
    'verbose': 2,
    'random_state': 0,
}

count_plot = 0

def get_timestamp_now_str():
    """ return timestamp in form of string of numbers """
    return str(dt.datetime.now()).replace('.', '').replace(':', '').replace('-', '').replace(' ', '')


class HPOT:
    """ use hyperopt on each set """

    def __init__(self, sql_result, sample_set, max_evals=10, feature_names=None, sample_names=None):

        self.sql_result = sql_result
        self.hpot = {'all_results': [], 'all_importance': [], 'best_score': 10000}  # initial accuracy_score = 0
        self.sample_set = sample_set
        self.feature_names = feature_names
        self.sample_names = sample_names
        self.hpot_start = get_timestamp_now_str()  # start time for entire training (for all output)
        self.max_evals = max_evals

        trials = Trials()
        kwargs = {"algo": tpe.suggest, "max_evals": self.max_evals, "trials": trials}
        best = fmin(fn=self.__eval, space=cls_lgbm_space, **kwargs)

        self.__save_feature_importance()
        # self.__save_test_prediction()
        self.__save_model()

    # ============================================== Evaluation =====================================================

    def __eval(self, space, eval_metric='logloss_valid'):
        """ train & evaluate LightGBM on given rf_space by hyperopt trials with Regression model """

        self.sql_result['uid'] = self.hpot_start + get_timestamp_now_str()
        self.sql_result['hpot_metric'] = eval_metric

        # Training
        params = self.__int_params_to_int(space)
        self.sql_result.update(params.copy())
        cvboosters, eval_valid_score, eval_train_score = \
            self.__lgbm_train(params, self.sample_set, nfold=self.sql_result['nfold'])

        cv_score = []
        all_prediction = []
        cv_number = 0
        for model in cvboosters:
            self.sql_result['uid'] = self.sql_result['uid'][:40] + str(cv_number)

            result = {"logloss_train": eval_train_score, "logloss_valid": eval_valid_score}
            for i in ['train', 'test']:
                pred_prob = model.predict(self.sample_set[f'{i}_x'])
                result[f"len_{i}"] = pred_prob.shape[0]

                # eval top class prob distribution
                result[f"top_class_prob_med"] = np.median(pred_prob[:, -1])
                for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    result[f"top_class_prob_{q*100}_{i}"] = np.quantile(pred_prob[:, -1], q=q)

                for pct in [0.2, 0.25, 0.33, 0.5]:
                    result[f"top_class_{pct*100}_ret_{i}"] = self.sample_set[f'{i}_y'][pred_prob[:, -1] > pct].mean()

                all_prediction.append(pred_prob)
                pred = pred_prob.argmax(axis=1)
                # TODO: maybe add back -> "auc": partial(roc_auc_score, multi_class='ovo')
                for k, func in {"accuracy": accuracy_score, "precision": partial(precision_score, average='weighted')}.items():
                    result[f"{k}_{i}"] = func(self.sample_set[f'{i}_y_cut'], pred)

            # TODO: maybe change to real 'valid' set
            cv_score.append(result[eval_metric])
            self.sql_result.update(result)  # update result of model
            self.hpot['all_results'].append(self.sql_result.copy())

            # if i == 'train':
            feature_importance_df = pd.DataFrame(model.feature_importance(), index=self.feature_names, columns=['split'])
            feature_importance_df = feature_importance_df.sort_values('split', ascending=False).reset_index()
            feature_importance_df['uid'] = self.sql_result['uid']
            self.hpot['all_importance'].append(feature_importance_df.copy())

            cv_number += 1

        cv_score_mean = np.mean(cv_score)
        if cv_score_mean < self.hpot['best_score']:  # update best_mae to the lowest value for Hyperopt
            self.hpot['best_score'] = cv_score_mean
            self.hpot['best_prediction'] = all_prediction
        gc.collect()

        return cv_score_mean

    # ============================================ Training =======================================================

    def __lgbm_train(self, params, sample_set, plot_eval=True, nfold=5):
        """ train lightgbm booster based on training / validaton set -> give predictions of Y """

        train_set = Dataset(sample_set['train_x'], sample_set['train_y_cut'],
                            weight=[self.sql_result['class_weight'][x] for x in sample_set['train_y_cut']],
                            # group=sample_set['group'],
                            feature_name=self.feature_names)

        params['num_class'] = self.sql_result['qcut_q']
        params['num_thread'] = args.process

        results = cv(params=params,
                     train_set=train_set,
                     # metrics=self.sql_result['objective'],
                     nfold=nfold,
                     num_boost_round=1000,
                     early_stopping_rounds=150,
                     eval_train_metric=True,
                     return_cvbooster=True,
                     )
        eval_valid_score = results['valid multi_logloss-mean'][-1]
        eval_train_score = results['train multi_logloss-mean'][-1]

        global count_plot
        if (plot_eval) & (count_plot == 0):
            self.__plot_eval_results(results)
            count_plot += 1

        return results['cvbooster'].boosters, eval_valid_score, eval_train_score

    # ========================================== Results Saving ===================================================

    def __plot_eval_results(self, results):
        """ plot eval results """

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].plot(results['train multi_logloss-mean'], label='train', c='k')
        ax[0].plot(results['valid multi_logloss-mean'], label='valid', c='r')
        ax[0].set_title('multi_logloss-mean')

        ax[1].plot(results['train multi_logloss-stdv'], label='train', c='k')
        ax[1].plot(results['valid multi_logloss-stdv'], label='valid', c='r')
        ax[1].set_title('multi_logloss-stdv')

        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'config_opt_{self.hpot_start}.png')

    def __save_feature_importance(self):
        """ save feature importance to DB """

        table_name = global_vars.factor_config_importance_table
        df = pd.concat(self.hpot['all_importance'], axis=0)
        upsert_data_to_database(df, table_name, how="append", verbose=-1)

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

        table_name = global_vars.factor_config_score_table
        df = pd.DataFrame(self.hpot['all_results'])
        upsert_data_to_database(df, table_name, primary_key=["uid"], how="update", verbose=1)

    # ========================================== Utils Funcs ===================================================

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
            df_x['is_usd'] = (df['group_code'] == "USD").values
        df_x['q'] = df['q'].values
        for i in range(1, 4):
            df_x[f'actual_{i}'] = df[f'actual_{i}'].values
        df_x['testing_period'] = df['testing_period'].values

        # add whether using [max_ret] column to x
        df_x['max_ret'] = True
        df_x_copy = df_x.copy()
        df_x_copy['max_ret'] = False
        self.df_x = pd.concat([df_x, df_x_copy], axis=0).reset_index(drop=True)

        # y = max_ret + net_ret
        df_y_max = df['max_ret'].copy()
        df_y_net = df['max_ret'] - df['min_ret']
        self.df_y = pd.concat([df_y_max, df_y_net], axis=0).reset_index(drop=True)

    def split_all(self, testing_period, qcut_q=3):
        # [x]: split train / test
        X_train = self.df_x.loc[self.df_x["testing_period"] < testing_period].drop(columns=['testing_period'])
        # X_train['testing_period'] = (X_train['testing_period'] - testing_period).dt.days
        X_test = self.df_x.loc[self.df_x["testing_period"] == testing_period].drop(columns=['testing_period'])
        # X_test['testing_period'] = 0
        self.feature_names = X_train.columns.to_list()

        # [y]: split train / test + qcut
        y_train = self.df_y.loc[self.df_x["testing_period"] < testing_period]
        y_train_cut, self.cut_bins = pd.qcut(y_train, q=qcut_q, retbins=True, labels=False)
        y_test = self.df_y.loc[self.df_x["testing_period"] == testing_period]
        cut_bins = self.cut_bins.copy()
        cut_bins[0], cut_bins[-1] = -np.inf, np.inf
        y_test_cut = pd.cut(y_test, bins=self.cut_bins, labels=False)

        # from collections import Counter
        # a = Counter(y_train)
        # b = Counter(y_test)

        # [x]: apply standard scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train.values)
        X_test = scaler.transform(X_test.values)

        return X_train, X_test, y_train, y_test, y_train_cut.astype(int), y_test_cut.astype(int)


if __name__ == "__main__":

    # --------------------------------- Parser --------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='multiclass')
    parser.add_argument('--name_sql', default='w4_d-7_20220312222718_debug')
    parser.add_argument('--qcut_q', type=int, default=5)
    parser.add_argument('--nfold', type=int, default=5)
    parser.add_argument('--process', type=int, default=12)
    args = parser.parse_args()

    # --------------------------------- Load Data -----------------------------------------------

    # df = read_query(f"SELECT * FROM {global_vars.production_factor_rank_backtest_eval_table} "
    #                 f"WHERE name_sql='{args.name_sql}'")
    # df.to_pickle(f'eval_{args.name_sql}.pkl')

    df = pd.read_pickle(f'eval_{args.name_sql}.pkl')
    df = df.sort_values(by=['testing_period'])
    for i in range(1, 4):
        df[f'actual_{i}'] = df.groupby(['testing_period'])['actual_s'].shift(i)
    df = df.dropna(how='any')

    sql_result = vars(args).copy()  # data write to DB TABLE lightgbm_results
    sql_result['name_sql2'] = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')  # name_sql for config opt
    sql_result['class_weight'] = {i: 1 for i in range(args.qcut_q)}

    # df = pd.read_pickle(args.name_sql + '.pkl')
    print(df)

    testing_period = np.sort(df['testing_period'].unique())[-12:]

    for (group, y_type), g in df.groupby(['group', 'y_type']):
        data = load_date(g)
        for t in testing_period:
            X_train, X_test, y_train, y_test, y_train_cut, y_test_cut = data.split_all(t, qcut_q=args.qcut_q)
            sql_result['cut_bins'] = list(data.cut_bins)
            HPOT(sql_result=sql_result,
                 sample_set={"train_x": X_train, "test_x": X_test,
                             "train_y": y_train, "test_y": y_test,
                             "train_y_cut": y_train_cut, "test_y_cut": y_test_cut},
                 feature_names=data.feature_names)