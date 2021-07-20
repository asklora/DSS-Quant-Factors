from hyperopt import hp

space = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.1]),
    'boosting_type': 'dart',
    # 'max_bin': hp.quniform('max_bin', 128, 256, 128),
    'max_bin': 256,
    'num_leaves': hp.quniform('num_leaves', 50, 300, 50),
    'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 2, 5, 1.5),
    'feature_fraction': hp.quniform('feature_fraction', 0.4, 0.8, 0.2),
    'bagging_fraction': hp.quniform('bagging_fraction', 0.4, 0.8, 0.2),
    'bagging_freq': hp.quniform('bagging_freq', 2, 8, 2),
    'min_gain_to_split': 0,
    'lambda_l1': 0,
    'lambda_l2': hp.quniform('lambda_l2', 0, 40, 10),
    }


def find_hyperspace(sql_result):            # Later maybe add different Hyperspace for Industry / Currency
    return space

if __name__ == '__main__':
    sql_result = {'y_type':'rev_yoy', 'group_code':'CNY'}
    # space = find_hyperspace(sql_result)
    print(space.keys())