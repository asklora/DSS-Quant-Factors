db_url_aws_read = "postgres://postgres:ml2021#LORA@droid-v2-production-cluster.cluster-ro-cy4dofwtnffp.ap-east-1.rds.amazonaws.com:5432/postgres" # AWS Read url
db_url_alibaba = "postgres://asklora:AskLORAv2@pgm-3nse9b275d7vr3u18o.pg.rds.aliyuncs.com:1921/postgres"

# db_url_alibaba_prod = "postgres://asklora:AskLORAv2@pgm-3nscoa6v8c876g5xlo.pg.rds.aliyuncs.com:1924/postgres"
db_url_alibaba_prod = "postgres://asklora:AskLORAv2@pgm-3nse9b275d7vr3u18o.pg.rds.aliyuncs.com:1921/postgres"

# TABLE names - factor model results
result_pred_table = "factor_model"     # + "_lgbm"/"_rf" + "_reg/class"
result_score_table = "factor_model_stock"
feature_importance_table = "factor_result_importance"
production_factor_rank_table = "factor_result_rank"
production_factor_rank_history_table = "factor_result_rank_history"

# TABLE names - universe rating results
production_score_current = "universe_rating" # in DROID v2 DB
production_score_current_history = "universe_rating_history"


# TABLE names - raw data
universe_table = "universe"
worldscope_quarter_summary_table = "data_worldscope"
ibes_data_table = "data_ibes"
macro_data_table = "data_macro"
stock_data_table_ohlc = "data_ohlcv"
stock_data_table_tri = "data_tri"
anchor_table_mkt_cap = "data_dsws_addition"

eikon_price_table = "data_factor_eikon_price"           # Django Managed here
eikon_report_date_table = "data_factor_eikon_date"      # Django Managed here
eikon_fx_table = "data_factor_eikon_fx"                 # Django Managed here
currency_history_table = "currency_price_history"
ingestion_name_table = "ingestion_name"

# TABLE names - preprocessed data
processed_ratio_table = "factor_processed_ratio"
factor_premium_table = "factor_processed_factor_premium"

# TABLE names - preprocess formula
formula_factors_table_prod = "factor_formula_ratios_prod"
update_time_table = "ingestion_update_time"     # all table update time record in this table

# TABLE names - descriptive preprocess formula
descriptive_formula_factors_table = "factor_formula_ratios_descriptive"
