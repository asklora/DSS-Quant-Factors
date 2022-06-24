import datetime as dt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
import pandas as pd
import gc
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from utils import (
    upsert_data_to_database,
    to_slack,
    err2slack,
    sys_logger,
    models
)
from functools import partial
from typing import Dict

logger = sys_logger(__name__, "DEBUG")

result_pred_table = models.FactorResultPrediction.__tablename__
result_score_table = models.FactorResultScore.__tablename__
feature_importance_table = models.FactorResultImportance.__tablename__


def get_timestamp_now_str():
    """ return timestamp in form of string of numbers """
    return str(dt.datetime.now()).replace('.', '').replace(':', '').replace('-', '').replace(' ', '')


def adj_mse_score(actual: np.array, pred: np.array, multioutput=False):
    """
    adjusted metrics:
    when rank(actual) (ranked by row) < rank(pred) (we overestimate premium) use mse
    when rank(actual) > rank(pred) (we underestimate premium) use mae
    """

    actual_sort = actual.argsort().argsort()
    pred_sort = pred.argsort().argsort()
    # rank_r2 = r2_score(actual_sort, pred_sort, multioutput='uniform_average')
    # rank_mse = mean_squared_error(actual_sort, pred_sort, multioutput='uniform_average')
    # rank_mae = mean_absolute_error(actual_sort, pred_sort, multioutput='uniform_average')
    diff = actual_sort - pred_sort
    diff2 = np.where(diff > 0, abs(diff), diff**2)  # if actual > pred
    adj_mse = np.mean(diff2)
    return adj_mse


class rf_HPOT:
    """
    use hyperopt on each set
    """

    if_write_feature_imp = True
    rf_space = {
        # 'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
        'n_estimators'            : hp.choice('n_estimators', [15, 50, 100]),
        'max_depth'               : hp.choice('max_depth', [8, 32, 64]),
        'min_samples_split'       : hp.choice('min_samples_split', [5, 10, 50]),
        'min_samples_leaf'        : hp.choice('min_samples_leaf', [1, 5, 20]),
        'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0, 1e-2, 1e-1]),
        'max_features'            : hp.choice('max_features', [0.5, 0.7, 0.9]),
        'min_impurity_decrease'   : 0,
        'max_samples'             : hp.choice('max_samples', [0.7, 0.9]),
        'ccp_alpha'               : hp.choice('ccp_alpha', [0]),
    }
    score_map = {"mae": mean_absolute_error, "r2": r2_score, "mse": mean_squared_error, "adj_mse": adj_mse_score}

    def __init__(self, max_evals: int,
                 down_mkt_pct: float,
                 tree_type: str,
                 objective: str,
                 sql_result: dict,
                 hpot_eval_metric: str,
                 **kwargs):
        self.max_evals = max_evals
        self.down_mkt_pct = down_mkt_pct
        self.tree_type = tree_type
        self.objective = objective
        self.hpot_eval_metric = hpot_eval_metric
        self.sql_result = sql_result

        self.hpot = {'all_results': [], 'best_score': 10000}
        self.hpot_start = get_timestamp_now_str()

    def train_and_write(self, sample_set: Dict[str, pd.DataFrame]):
        """ write score/prediction/feature to DB """

        trials = Trials()
        best = fmin(fn=partial(self._eval_reg, sample_set=sample_set),
                    space=self.rf_space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
        print(best)

        self.__write_score_db()
        self.__write_prediction_db()
        self.__write_importance_db()
        return True

    @err2slack("clair")
    def __write_score_db(self):
        upsert_data_to_database(pd.DataFrame(self.hpot['all_results']), result_score_table, how="append", add_primary_key=True)
        return True

    @err2slack("clair")
    def __write_prediction_db(self):
        upsert_data_to_database(self.hpot['best_stock_df'], result_pred_table, how="append", add_primary_key=True)
        return True

    @err2slack("clair")
    def __write_importance_db(self):
        upsert_data_to_database(self.hpot['best_stock_feature'], feature_importance_table, how="append", add_primary_key=True)
        return True

    def _regr_train(self, space: dict, sample_set: dict) -> object:
        """ train lightgbm booster based on training / validation set -> give predictions of Y """

        params = space.copy()
        params = self.__clean_rf_params(params)

        if 'extra' in self.tree_type:
            regr = ExtraTreesRegressor(criterion=self.objective, **params)
        elif 'rf' in self.tree_type:
            regr = RandomForestRegressor(criterion=self.objective, **params)
        else:
            raise Exception(f"Except tree_type = 'extra' or 'rf', get [{self.tree_type}]")

        regr.fit(X=sample_set['train_x'].values,
                 y=sample_set['train_y_final'].values,
                 sample_weight=self.__sample_weight(sample_set['train_y']))

        return regr

    def __clean_rf_params(self, params):
        for k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
            params[k] = int(params[k])
        self.sql_result.update({f"{k}": v for k, v in params.items()})

        params['bootstrap'] = True
        params['n_jobs'] = 1
        return params

    def __sample_weight(self, train_y: pd.DataFrame) -> np.array:
        sample_weight = np.where(train_y.mean(axis=1) < 0, self.down_mkt_pct, 1 - self.down_mkt_pct)
        return sample_weight

    def _regr_pred(self, regr: object, sample_set: dict):
        """
        add [.._y_pred] pd.DataFrame to sample set
        """

        for i in ["train", "valid", "test"]:
            sample_set[f"{i}_y_pred"] = pd.DataFrame(
                regr.predict(sample_set[f"{i}_x"].values),
                index=sample_set[f"{i}_y"].index,
                columns=sample_set[f"{i}_y"].columns,
            )
            if i == "train":
                self.sql_result['train_pred_std'] = sample_set[f"{i}_y_pred"].std(axis=0).mean()
                logger.debug(f'Y_train_pred: \n{sample_set[f"{i}_y_pred"][:5]}')

        return sample_set

    def _regr_importance(self, regr: object, feature_names: list) -> pd.DataFrame:
        """
        based on rf model -> records feature importance in DataFrame to be uploaded to DB
        """

        df = pd.DataFrame(regr.feature_importances_,
                          index=pd.Index(feature_names, name="name"),
                          columns=["split"]).reset_index()
        df['uid'] = str(self.sql_result['uid'])
        df = df.sort_values('split', ascending=False)
        self.sql_result['feature_importance'] = df['name'].to_list()

        return df

    def __calc_pred_eval_scores(self, sample_set: dict):
        """
        Calculate evaluation scores
        """

        result = {}
        for set_name in ["train", "valid", "test"]:
            for score_name, func in self.score_map.items():
                if sample_set[f"{set_name}_y"].notnull().sum().sum() == 0:
                    logger.warning(f"[warning] can't calculate eval ({score_name}, {set_name}) because no actual Y.")
                else:
                    result[f"{score_name}_{set_name}"] = func(sample_set[f"{set_name}_y"].values,
                                                              sample_set[f"{set_name}_y_pred"].values,
                                                              multioutput='uniform_average')

        return result

    def __get_eval_metric(self, test_y: pd.DataFrame, default_metric: str = 'mse_valid'):
        """
        if multi-output (i.e. multi factors) -> [hpot_eval_metric] as defined;
        if only 1 factor in pillar           -> default = [mse_valid]
        """

        if test_y.shape[1] > 1:
            return self.hpot_eval_metric
        else:
            return default_metric

    def _eval_reg(self, rf_space: dict, sample_set: dict):
        """
        train & evaluate LightGBM on given rf_space by hyperopt trials with Regression model
        """

        self.sql_result['uid'] = self.hpot_start + get_timestamp_now_str()

        regr = self._regr_train(space=rf_space, sample_set=sample_set)
        sample_set = self._regr_pred(regr=regr, sample_set=sample_set)
        feature_importance_df = self._regr_importance(regr=regr, feature_names=sample_set["train_x"].columns.to_list())

        result = self.__calc_pred_eval_scores(sample_set=sample_set)
        self.sql_result.update(result)  # update result of model
        self.hpot['all_results'].append(self.sql_result.copy())

        eval_metric = self.__get_eval_metric(sample_set["test_y"])
        if result[eval_metric] < self.hpot['best_score']:  # update best_mae to the lowest value for Hyperopt
            self.hpot['best_score'] = result[eval_metric]
            self.hpot['best_stock_df'] = self.__to_sql_prediction(sample_set['test_y'], sample_set['test_y_pred'])
            self.hpot['best_stock_feature'] = feature_importance_df

        return result[eval_metric]

    def __to_sql_prediction(self, test_y: pd.DataFrame, test_y_pred: pd.DataFrame) -> pd.DataFrame:
        """
        prepare array Y_test_pred to DataFrame ready to write to SQL
        """

        assert len(test_y.index.get_level_values("testing_period").unique()) == 1
        assert test_y.index.names == ("group", "testing_period")

        df = pd.concat([test_y.stack(), test_y_pred.stack()], axis=1)
        df.columns = ["actual", "pred"]
        df.index.names = ("currency_code", "testing_period", "factor_name")
        df = df.reset_index().drop(columns=["testing_period"])
        df['uid'] = str(self.sql_result['uid'])

        return df
