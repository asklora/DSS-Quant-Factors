from components.model_training.src.load_data import combineData, loadData
from components.model_training.src.load_train_configs import loadTrainConfig
from components.model_training.main import start

processes = 1
weeks_to_expire = 26
sample_interval = 4
backtest_period = 50
restart = None
currency_code = "USD"
end_date = '2022-11-01'


def test_rf_train():
    sql_result = {"name_sql": "test1"}

    raw_df = combineData(weeks_to_expire=weeks_to_expire,
                         sample_interval=sample_interval,
                         backtest_period=backtest_period,
                         currency_code=None,  # raw_df should get all
                         restart=restart,
                         end_date=end_date).get_raw_data()

    all_groups = loadTrainConfig(weeks_to_expire=weeks_to_expire,
                                 sample_interval=sample_interval,
                                 backtest_period=backtest_period,
                                 restart=restart,
                                 currency_code=currency_code,
                                 end_date=end_date) \
        .get_all_groups()

    # test on first group
    start(all_groups[0][0], raw_df=raw_df, sql_result=sql_result.copy())

