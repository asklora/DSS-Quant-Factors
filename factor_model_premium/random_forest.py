import datetime as dt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
import pandas as pd
import gc
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from general.sql_output import upsert_data_to_database
import global_vars


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

        rf_space = {
            'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
            # 'n_estimators': hp.choice('n_estimators', [15, 50, 100]),
            'max_depth': hp.choice('max_depth', [8, 32, 64]),
            'min_samples_split': hp.choice('min_samples_split', [5, 10, 50, 100]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [5, 10, 50]),
            'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0, 1e-2, 1e-1]),
            'max_features': hp.choice('max_features', [0.5, 0.7, 0.9]),
            'min_impurity_decrease': 0,
            # 'max_samples': hp.choice('max_samples',[0.7, 0.9]),
            'ccp_alpha': hp.choice('ccp_alpha', [0, 1e-3]),
            # 'random_state': 666
        }

        trials = Trials()
        best = fmin(fn=self.eval_regressor, space=rf_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    def write_db(self, tbl_suffix = ''):
        ''' write score/prediction/feature to DB '''

        # update results
        upsert_data_to_database(self.hpot['best_stock_df'], f"{global_vars.result_pred_table}{tbl_suffix}",
                                db_url=global_vars.db_url_alibaba_prod, how="append", verbose=-1)
        upsert_data_to_database(pd.DataFrame(self.hpot['all_results']), f"{global_vars.result_score_table}{tbl_suffix}",
                                db_url=global_vars.db_url_alibaba_prod, how="append", verbose=-1)
        # upsert_data_to_database(self.hpot['best_stock_feature'], f"{global_vars.feature_importance_table}{tbl_suffix}",
        #                         db_url=global_vars.db_url_alibaba_prod, how="append", verbose=-1)

    def rf_train(self, space, rerun):
        ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

        params = space.copy()
        for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
            params[k] = int(params[k])
        self.sql_result.update(params)
        params['bootstrap'] = False
        params['n_jobs'] = self.sql_result['n_jobs']

        if 'extra' in self.sql_result['tree_type']:
            regr = ExtraTreesRegressor(criterion=self.sql_result['objective'], **params)
        elif 'rf' in self.sql_result['tree_type']:
            regr = RandomForestRegressor(criterion=self.sql_result['objective'], **params)

        if rerun:
            regr.fit(self.sample_set['train_x'], self.sample_set['train_y_final'])
        else:
            regr.fit(self.sample_set['train_xx'], self.sample_set['train_yy_final'])

        # prediction on all sets
        if rerun:
            Y_train_pred = regr.predict(self.sample_set['train_x'])
            Y_valid_pred = regr.predict(self.sample_set['valid_x'])
            Y_test_pred = regr.predict(self.sample_set['test_x'])
        else:
            Y_train_pred = regr.predict(self.sample_set['train_xx'])
            Y_valid_pred = regr.predict(self.sample_set['valid_x'])
            Y_test_pred = regr.predict(self.sample_set['test_x'])

        self.sql_result['feature_importance'], feature_importance_df = self.to_list_importance(regr)

        return Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df

    def _eval_test_return(self, actual, pred, Y_train_pred=[]):
        ''' test return based on test / train set quantile bins '''

        p = np.linspace(0, 1, 4)
        if len(Y_train_pred) > 0:
            bins = np.quantile(Y_train_pred, p)
        else:
            bins = np.quantile(pred, p)

        ret = []
        factor_name = []
        for i in range(3):
            ret.append(np.mean(actual[(pred >= bins[i]) & (pred <= bins[i + 1])]))
            f = np.array([x[2:] for x in self.y_col])
            f_mask = np.reshape((pred >= bins[i]) & (pred < bins[i + 1]), (len(f),))
            factor_name.append(f[f_mask])

        return ret, factor_name[2]

    def eval_regressor(self, rf_space, rerun=False):
        ''' train & evaluate LightGBM on given rf_space by hyperopt trials with Regressiong model
        -------------------------------------------------
        This part haven't been modified for multi-label questions purpose
        '''

        self.sql_result['finish_timing'] = dt.datetime.now()

        Y_train_pred, Y_valid_pred, Y_test_pred, feature_importance_df = self.rf_train(rf_space, rerun)

        if rerun:  # save prediction bins for training as well
            p = np.linspace(0, 1, 10)
            self.sql_result['train_bins'] = list(np.quantile(Y_train_pred, p))
            self.sql_result['train_mean'] = np.nanmean(Y_train_pred.flatten())
            self.sql_result['train_std'] = np.nanstd(Y_train_pred.flatten())

        if len(self.sample_set['test_y']) == 0:  # for the actual prediction iteration
            self.sample_set['test_y'] = np.zeros(Y_test_pred)

        ret, best_factor = self._eval_test_return(self.sample_set['test_y'], Y_test_pred, Y_test_pred)
        if rerun:
            result = {'mae_train': mean_absolute_error(self.sample_set['train_y'], Y_train_pred),
                      'mae_valid': mean_absolute_error(self.sample_set['valid_y'], Y_valid_pred),
                      'mse_train': mean_squared_error(self.sample_set['train_y'], Y_train_pred),
                      'mse_valid': mean_squared_error(self.sample_set['valid_y'], Y_valid_pred),
                      'r2_train': r2_score(self.sample_set['train_y'], Y_train_pred),
                      'r2_valid': r2_score(self.sample_set['valid_y'], Y_valid_pred),
                      'mae_test': mean_absolute_error(self.sample_set['test_y'], Y_test_pred),
                      'mse_test': mean_squared_error(self.sample_set['test_y'], Y_test_pred),
                      'net_ret': ret[2]
                      }
        else:
            result = {'mae_train': mean_absolute_error(self.sample_set['train_yy'], Y_train_pred),
                      'mae_valid': mean_absolute_error(self.sample_set['valid_y'], Y_valid_pred),
                      'mse_train': mean_squared_error(self.sample_set['train_yy'], Y_train_pred),
                      'mse_valid': mean_squared_error(self.sample_set['valid_y'], Y_valid_pred),
                      'r2_train': r2_score(self.sample_set['train_yy'], Y_train_pred),
                      'r2_valid': r2_score(self.sample_set['valid_y'], Y_valid_pred),
                      'mae_test': mean_absolute_error(self.sample_set['test_y'], Y_test_pred),
                      'mse_test': mean_squared_error(self.sample_set['test_y'], Y_test_pred),
                      'net_ret': ret[2]
                      }

        self.sql_result.update(result)  # update result of model
        self.hpot['all_results'].append(self.sql_result.copy())
        self.hpot['all_results'][-1].pop('n_jobs')

        if (result['mae_valid'] < self.hpot['best_score']) or (rerun):  # update best_mae to the lowest value for Hyperopt
            self.hpot['best_score'] = result['mae_valid']
            self.hpot['best_stock_df'] = self.to_sql_prediction(Y_test_pred)
            self.hpot['best_stock_feature'] = feature_importance_df.sort_values('split', ascending=False)

        gc.collect()

        if rerun:
            print(
                f"RERUN --> {str(result['mse_train'] * 100)[:6]}, {str(result['mse_test'] * 100)[:6]}, "
                f"{str(result['net_ret'])[:6]}, {best_factor}")
            return result['mse_train']
        else:
            print(
                f"HPOT --> {str(result['mse_valid'] * 100)[:6]}, {str(result['mse_test'] * 100)[:6]}, "
                f"{str(result['net_ret'])[:6]}, {best_factor}")
            return result['mse_valid']

    def to_sql_prediction(self, Y_test_pred):
        ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

        self.sql_result['y_type'] = [x[2:] for x in self.y_col]
        df = pd.DataFrame(Y_test_pred, index=self.group_index, columns=self.sql_result['y_type'])
        df = df.unstack().reset_index(drop=False)
        df.columns = ['y_type', 'group', 'pred']
        df['actual'] = self.sample_set['test_y_final'].flatten(order='F')  # also write actual qcut to BD
        df['finish_timing'] = [self.sql_result['finish_timing']] * len(df)  # use finish time to distinguish dup pred
        return df

    def to_list_importance(self, rf):
        ''' based on rf model -> records feature importance in DataFrame to be uploaded to DB '''

        df = pd.DataFrame()
        df['name'] = self.x_col  # column names
        df['split'] = rf.feature_importances_
        df['finish_timing'] = [self.sql_result['finish_timing']] * len(df)  # use finish time to distinguish dup pred
        return ','.join(df.sort_values(by=['split'], ascending=False)['name'].to_list()), df
