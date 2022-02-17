import datetime as dt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
import pandas as pd
import gc
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from general.sql_process import upsert_data_to_database
from global_vars import *

def get_timestamp_now_str():
    ''' return timestamp in form of string of numbers '''
    return str(dt.datetime.now()).replace('.', '').replace(':', '').replace('-', '').replace(' ', '')

class rf_HPOT:
    ''' use hyperopt on each set '''

    def __init__(self, max_evals, sql_result, sample_set, x_col, y_col, group_index):
        
        self.hpot = {}
        self.hpot['all_results'] = []
        self.hpot['best_score'] = 10000  # record best training (min mae_valid) in each hyperopt
        
        self.sql_result = sql_result
        self.sample_set = sample_set
        self.x_col = x_col
        self.y_col = y_col
        self.group_index = group_index
        self.hpot_start = get_timestamp_now_str()

        rf_space = {
            # 'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
            'n_estimators': hp.choice('n_estimators', [15, 50, 100]),
            'max_depth': hp.choice('max_depth', [8, 32, 64]),
            'min_samples_split': hp.choice('min_samples_split', [5, 10, 50]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [1, 5, 20]),
            'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0, 1e-2, 1e-1]),
            'max_features': hp.choice('max_features', [0.5, 0.7, 0.9]),
            'min_impurity_decrease': 0,
            'max_samples': hp.choice('max_samples',[0.7, 0.9]),
            'ccp_alpha': hp.choice('ccp_alpha', [0, 1e-3]),
            # 'random_state': 666
        }

        trials = Trials()
        best = fmin(fn=self.eval_regressor, space=rf_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    def write_db(self):
        ''' write score/prediction/feature to DB '''

        # update results
        upsert_data_to_database(self.hpot['best_stock_df'], result_pred_table, db_url=db_url_write, how="append", verbose=-1)
        upsert_data_to_database(pd.DataFrame(self.hpot['all_results']), result_score_table, primary_key=["uid"],
                                db_url=db_url_write, how="ignore", verbose=-1)

    def rf_train(self, space):
        ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

        params = space.copy()
        for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
            params[k] = int(params[k])
        self.sql_result.update(params)
        params['bootstrap'] = False
        params['n_jobs'] = 1

        if 'extra' in self.sql_result['tree_type']:
            regr = ExtraTreesRegressor(criterion=self.sql_result['objective'], **params)
        elif 'rf' in self.sql_result['tree_type']:
            regr = RandomForestRegressor(criterion=self.sql_result['objective'], **params)

        regr.fit(self.sample_set['train_xx'], self.sample_set['train_yy_final'],
                 sample_weight=self.sample_set['train_yy_weight'])

        # prediction on all sets
        Y_train_pred = regr.predict(self.sample_set['train_xx'])
        self.sql_result['train_pred_std'] = np.std(Y_train_pred, axis=0).mean()

        Y_valid_pred = regr.predict(self.sample_set['valid_x'])
        Y_test_pred = regr.predict(self.sample_set['test_x'])
        logging.debug(f'Y_train_pred: \n{Y_train_pred[:5]}')

        self.sql_result['feature_importance'], feature_importance_df = self.to_list_importance(regr)

        return Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df

    def _eval_test_return(self, actual, pred):
        ''' test return based on test / train set quantile bins '''

        q_ = [0., 1/3, 2/3, 1.]
        pred_qcut = pd.qcut(pred.flatten(), q=q_, labels=False, duplicates='drop').reshape(pred.shape)
        ret = actual[pred_qcut==2].mean()
        best_factor = np.array([x[2:] for x in self.y_col])[pred_qcut[0,:]==2]
        return ret, best_factor

    def eval_regressor(self, rf_space):
        ''' train & evaluate LightGBM on given rf_space by hyperopt trials with Regression model
        -------------------------------------------------
        This part haven't been modified for multi-label questions purpose
        '''

        self.sql_result['uid'] = self.hpot_start + get_timestamp_now_str()

        self.sample_set['train_yy_pred'], self.sample_set['valid_y_pred'], self.sample_set['test_y_pred'], \
            feature_importance_df = self.rf_train(rf_space)

        if len(self.sample_set['test_y']) == 0:  # for the actual prediction iteration
            self.sample_set['test_y'] = np.zeros(Y_test_pred.shape)

        ret, best_factor = self._eval_test_return(self.sample_set['test_y'], self.sample_set['test_y_pred'])
        result = {'net_ret': ret}
        for k, func in {"mae": mean_absolute_error, "r2": r2_score, "mse": mean_squared_error}.items():
            for i in ['train_yy', 'valid_y', 'test_y']:
                score = func(self.sample_set[i].T, self.sample_set[i+'_pred'].T, multioutput='raw_values')
                result[f"{k}_{i.split('_')[0]}"] = np.median(score)
        logging.debug(f"R2 train: {result['r2_train']}")
        logging.debug(f"R2 valid: {result['r2_valid']}")
        logging.debug(f"R2 test: {result['r2_test']}")

        self.sql_result.update(result)  # update result of model
        self.hpot['all_results'].append(self.sql_result.copy())

        if (result['mae_valid'] < self.hpot['best_score']):  # update best_mae to the lowest value for Hyperopt
            self.hpot['best_score'] = result['mae_valid']
            self.hpot['best_stock_df'] = self.to_sql_prediction(self.sample_set['test_y_pred'])
            self.hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

        gc.collect()
        logging.info(f"HPOT --> {str(result['mse_valid'] * 100)[:6]}, {str(result['mse_test'] * 100)[:6]}, "
                     f"{str(result['net_ret'])[:6]}, {best_factor}")
        return result['mse_valid']

    def to_sql_prediction(self, Y_test_pred):
        ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

        df = pd.DataFrame(Y_test_pred, index=self.group_index, columns=[x[2:] for x in self.y_col])
        df = df.unstack().reset_index(drop=False)
        df.columns = ['factor_name', 'group', 'pred']
        df['actual'] = self.sample_set['test_y_final'].flatten(order='F')  # also write actual qcut to BD
        df['uid'] = [self.sql_result['uid']] * len(df)  # use finish time to distinguish dup pred
        return df

    def to_list_importance(self, rf):
        ''' based on rf model -> records feature importance in DataFrame to be uploaded to DB '''

        df = pd.DataFrame()
        df['name'] = self.x_col  # column names
        df['split'] = rf.feature_importances_
        df['uid'] = [self.sql_result['uid']] * len(df)  # use finish time to distinguish dup pred
        return df.sort_values(by=['split'], ascending=False)['name'].to_list(), df

