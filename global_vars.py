from sqlalchemy.dialects.postgresql import DATE, DOUBLE_PRECISION, TEXT, INTEGER, BOOLEAN, TIMESTAMP, JSON, NUMERIC

db_url_aws_read = "postgres://postgres:ml2021#LORA@droid-v2-production-cluster.cluster-ro-cy4dofwtnffp.ap-east-1.rds.amazonaws.com:5432/postgres"  # AWS Read url
db_url_alibaba = "postgresql://asklora:AskLORAv2@pgm-3nse9b275d7vr3u18o.pg.rds.aliyuncs.com:1921/postgres"
db_url_alibaba_prod = "postgresql://asklora:AskLORAv2@pgm-3nscoa6v8c876g5xlo.pg.rds.aliyuncs.com:1924/postgres"

db_url_alibaba_temp = "postgresql://loratech:loraTECH123@pgm-3ns7dw6lqemk36rgpo.pg.rds.aliyuncs.com:5432/postgres"
db_url_local = "postgresql://postgres:AskLORAv2@localhost:5432/postgres"
db_url_local_pc1 = "postgresql://postgres:AskLORAv2@localhost:15432/postgres"

db_url_read = db_url_alibaba_prod
db_url_write = db_url_alibaba_prod

# TABLE names - preprocess formula
factors_formula_table = "factor_formula_ratios"
config_train_table = "factor_formula_config_train"
config_eval_table = "factor_formula_config_eval"
pillar_defined_table = "factor_formula_pillar_defined"
pillar_cluster_table = "factor_formula_pillar_cluster"

# TABLE names - factor model results
result_pred_table = "factor_model_stock"  # + "_lgbm"/"_rf" + "_reg/class"
result_score_table = "factor_model"  # cluster pillar
feature_importance_table = "factor_result_importance"
production_rank_table = "factor_result_select"
production_rank_history_table = production_rank_table + "_history"
backtest_eval_table = "factor_result_rank_backtest_eval"
backtest_top_table = "factor_result_rank_backtest_top"  # updated version for non-peeking backtest

# TABLE name - factor config optimization results
# factor_config_score_table = 'factor_config_model'
# factor_config_prediction_table = 'factor_config_model_stock'
# factor_config_importance_table = 'factor_config_importance'

# TABLE names - raw data
universe_table = "universe"
worldscope_data_table = "data_worldscope"
ibes_data_table = "data_ibes"
macro_data_table = "data_macro_key"
vix_data_table = "data_vix"
fred_data_table = "data_fred"
stock_data_table_ohlcv = "data_ohlcv"
stock_data_table_tri = "data_tri"
latest_mktcap_data_table = "data_latest_mktcap"

eikon_fx_table = "data_factor_eikon_fx"
currency_history_table = "currency_price_history"
ingestion_name_table = "ingestion_name"
ingestion_name_macro_table = "ingestion_name_macro"
update_time_table = "ingestion_update_time"  # all table update time record in this table

# TABLE names - preprocessed data
processed_ratio_table = "factor_processed_ratio"
factor_premium_table = "factor_processed_premium"

# TABLE names - universe rating results
production_score_current = "universe_rating"  # in DROID v2 DB
production_score_current_history = "universe_rating_history"

# Add Logging
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.DEBUG)

# Define dtypes for tables
backtest_eval_dtypes = dict(
    _name_sql=TEXT,
    _testing_period=DATE,
    _currency_code=TEXT,
    _pillar=TEXT,
    _eval_q=NUMERIC(2, 2),
    _eval_removed_subpillar=BOOLEAN,
    __tree_type=TEXT,
    __factor_pca=NUMERIC(2, 2),
    __valid_pct=NUMERIC(2, 2),
    __valid_method=TEXT,
    __y_qcut=INTEGER,
    __factor_reverse=BOOLEAN,
    __down_mkt_pct=NUMERIC(2, 2),
    max_factor=JSON,
    min_factor=JSON,
    max_factor_pred=JSON,
    min_factor_pred=JSON,
    max_factor_actual=JSON,
    min_factor_actual=JSON,
    max_ret=DOUBLE_PRECISION,
    min_ret=DOUBLE_PRECISION,
    mae=DOUBLE_PRECISION,
    mse=DOUBLE_PRECISION,
    r2=DOUBLE_PRECISION,
    actual=DOUBLE_PRECISION,
    is_valid=BOOLEAN,
    updated=TIMESTAMP,
)

backtest_top_dtypes = dict(
    n_top=INTEGER,
    currency_code=TEXT,
    trading_day=DATE,
    mode=TEXT,
    mode_count=INTEGER,
    pos_pct=INTEGER,
    ret=NUMERIC(5, 2),
    bm_pos_pct=INTEGER,
    bm_ret=NUMERIC(5, 2),
    tickers=JSON,
    updated=TIMESTAMP,
)

rank_dtypes = dict(
    weeks_to_expire=INTEGER,
    currency_code=TEXT,
    pillar=TEXT,
    max_factor=JSON,
    min_factor=JSON,
    max_factor_extra=JSON,
    min_factor_extra=JSON,
    max_factor_trh=NUMERIC(2, 2),
    min_factor_trh=NUMERIC(2, 2),
)
