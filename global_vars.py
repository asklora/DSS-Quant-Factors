db_url_aws_read = "postgres://postgres:ml2021#LORA@droid-v2-production-cluster.cluster-ro-cy4dofwtnffp.ap-east-1.rds.amazonaws.com:5432/postgres" # AWS Read url
db_url_alibaba = "postgresql://asklora:AskLORAv2@pgm-3nse9b275d7vr3u18o.pg.rds.aliyuncs.com:1921/postgres"
db_url_alibaba_prod = "postgresql://asklora:AskLORAv2@pgm-3nscoa6v8c876g5xlo.pg.rds.aliyuncs.com:1924/postgres"

db_url_alibaba_temp = "postgresql://loratech:loraTECH123@pgm-3ns7dw6lqemk36rgpo.pg.rds.aliyuncs.com:5432/postgres"
db_url_local = "postgresql://postgres:AskLORAv2@localhost:5432/postgres"
db_url_local_pc1 = "postgresql://postgres:AskLORAv2@localhost:15432/postgres"

db_url_read = db_url_alibaba_prod
db_url_write = db_url_alibaba_prod

# TABLE names - factor model results
result_pred_table = "factor.factor_model_stock"     # + "_lgbm"/"_rf" + "_reg/class"
result_score_table = "factor.factor_model5"    # cluster pillar
feature_importance_table = "factor.factor_result_importance"
production_factor_rank_table = "factor_result_rank"
# production_factor_rank_ratio_table = "factor_result_rank_ratio"
# production_factor_rank_backtest_table = "factor_result_rank_backtest"
production_factor_rank_backtest_eval_table = "factor_result_rank_backtest_eval6"
production_factor_rank_backtest_top_table = "factor_result_rank_backtest_top6"  # updated version for non-peeking backtest
production_factor_rank_history_table = "factor_result_rank_history"

# TABLE name - factor config optimization results
factor_config_score_table = 'factor_config_model'
factor_config_prediction_table = 'factor_config_model_stock'
factor_config_importance_table = 'factor_config_importance'

# TABLE names - universe rating results
production_score_current = "universe_rating" # in DROID v2 DB
production_score_current_history = "universe_rating_history"

# TABLE names - raw data
universe_table = "universe"
worldscope_data_table = "data_worldscope"
ibes_data_table = "data_ibes"
macro_data_table = "data_macro_key"
vix_data_table = "data_vix"
fred_data_table = "data_fred"
stock_data_table_ohlc = "data_ohlcv"
stock_data_table_tri = "data_tri"
anchor_table_mkt_cap = "data_dsws_addition"

eikon_fx_table = "data_factor_eikon_fx"
currency_history_table = "currency_price_history"
ingestion_name_table = "ingestion_name"
ingestion_name_macro_table = "ingestion_name_macro"

# TABLE names - preprocessed data
processed_ratio_table = "factor_processed_ratio"
factor_premium_table = "factor_processed_premium"

# TABLE names - preprocess formula
formula_factors_table_prod = "factor_formula_ratios_prod"
factor_formula_config_train_prod = "factor_formula_config_train_prod"
factor_formula_config_eval_prod = "factor_formula_config_eval_prod"
factors_pillar_defined_table = "factor_formula_pillar_defined"
factors_pillar_cluster_table = "factor_formula_pillar_cluster"
update_time_table = "ingestion_update_time"     # all table update time record in this table

# # Set DEBUG status
# import sys
# gettrace = getattr(sys, 'gettrace', None)
# DEBUG = gettrace() is not None
# print('DEBUG: ', DEBUG)

# Add Logging
import logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.DEBUG)
