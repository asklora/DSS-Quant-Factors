from hyperopt import hp

space_lgbm_reg = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': 'gbdt',
    'max_bin': hp.choice('max_bin', [256, 512]),
    'num_leaves': hp.choice('num_leaves', [100, 200, 300]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [10, 30, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.4, 0.6, 0.8]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [2, 4, 8]),
    'min_gain_to_split': 0,
    'lambda_l1': 0,
    'lambda_l2': hp.choice('lambda_l2', [0, 5]),
    }

space_lgbm_class = {}

space_lgbm_class['industry'] = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': 'dart',
    'max_bin': hp.choice('max_bin', [256, 512]),
    'num_leaves': hp.choice('num_leaves', [100, 200, 300]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [10, 50, 150]),
    'feature_fraction': hp.choice('feature_fraction', [0.4, 0.6]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [4, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [1e-2, 1e-1]),
    'lambda_l1': hp.choice('lambda_l1', [0, 10, 20]),
    'lambda_l2': hp.choice('lambda_l2', [0, 10, 100]),
}

space_lgbm_class['currency'] = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': 'gbdt',     'max_bin': hp.choice('max_bin', [256, 512]),
    'num_leaves': hp.choice('num_leaves', [100, 200, 300]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.4, 0.6]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [4, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 1e-2]),
    'lambda_l1': hp.choice('lambda_l1', [0, 10, 20]),
    'lambda_l2': hp.choice('lambda_l2', [0, 10, 100]),
}

def find_hyperspace(sql_result):            # Later maybe add different Hyperspace for Industry / Currency

    if sql_result['objective'] in ['regression_l1', 'regression_l2']:
        return space_lgbm_reg
    else:
        return space_lgbm_class[sql_result['group_code']]

if __name__ == '__main__':
    sql_result = {'y_type':'rev_yoy', 'group_code':'CNY'}
    # space = find_hyperspace(sql_result)
    print(space_lgbm_class.keys())