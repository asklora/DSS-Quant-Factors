# def test_merge_groups_df():
#     from components.model_training.src.load_train_configs import loadTrainConfig
#     df = loadTrainConfig(weeks_to_expire=4,).merge_groups_df()
#
#     assert len(df) > 0
#     assert "testing_period" in df.columns.to_list()
#
#
# def test__restart_finished_configs():
#     from components.model_training.src.load_train_configs import loadTrainConfig
#     df = loadTrainConfig(weeks_to_expire=4, restart='test')._restart_finished_configs()
#
#     assert len(df) > 0
#
#
# def test_loadTrainConfig_HKD():
#     from components.model_training.src.load_train_configs import loadTrainConfig
#     lst = loadTrainConfig(weeks_to_expire=8, currency_code="HKD", backtest_period=60).get_all_groups()
#
#     assert len(lst) > 0


def test_loadTrainConfig():
    from components.model_training.src.load_train_configs import loadTrainConfig
    lst = loadTrainConfig(weeks_to_expire=4).get_all_groups()

    assert len(lst) > 0