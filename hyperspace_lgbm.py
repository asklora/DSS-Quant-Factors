from hyperopt import hp

space_lgbm_reg = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': 'gbdt',
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.quniform('num_leaves', 100, 300, 100),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [10, 30, 50]),
    'feature_fraction': hp.quniform('feature_fraction', 0.4, 0.8, 0.2),
    'bagging_fraction': hp.quniform('bagging_fraction', 0.4, 0.8, 0.2),
    'bagging_freq': hp.quniform('bagging_freq', 2, 8, 2),
    'min_gain_to_split': 0,
    'lambda_l1': hp.quniform('lambda_l1', 10, 40, 20),
    'lambda_l2': hp.quniform('lambda_l2', 0, 40, 20),
    }

space_lgbm_class = {}

space_lgbm_class['industry'] = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': 'gbdt',
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.quniform('num_leaves', 100, 300, 100),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 50, 20),
    'feature_fraction': hp.quniform('feature_fraction', 0.4, 0.6, 0.2),
    'bagging_fraction': hp.quniform('bagging_fraction', 0.6, 0.8, 0.2),
    'bagging_freq': hp.choice('bagging_freq', [2, 8]),
    # 'min_gain_to_split': 0,
    'lambda_l1': hp.quniform('lambda_l1', 10, 40, 15),
    'lambda_l2': hp.quniform('lambda_l2', 10, 40, 15),
}

space_lgbm_class['currency'] = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': 'gbdt',
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.quniform('num_leaves', 100, 300, 100),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 50, 20),
    'feature_fraction': hp.quniform('feature_fraction', 0.5, 0.8, 0.2),
    'bagging_fraction': hp.quniform('bagging_fraction', 0.6, 0.8, 0.2),
    'bagging_freq': hp.quniform('bagging_freq', 2, 8, 2),
    # 'min_gain_to_split': 0,
    'lambda_l1': hp.quniform('lambda_l1', 0, 40, 20),
    'lambda_l2': hp.quniform('lambda_l2', 0, 40, 20),
}

def find_hyperspace(sql_result):            # Later maybe add different Hyperspace for Industry / Currency

    if sql_result['objective'] in ['regression_l1', 'regression_l2']:
        return space_lgbm_reg[sql_result['group_code']]
    else:
        return space_lgbm_class[sql_result['group_code']]

if __name__ == '__main__':
    sql_result = {'y_type':'rev_yoy', 'group_code':'CNY'}
    # space = find_hyperspace(sql_result)
    print(space_lgbm_class.keys())