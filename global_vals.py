from sqlalchemy import create_engine

db_url_prod = "postgres://postgres:ml2021#LORA@droid-v2-prod-instance.cy4dofwtnffp.ap-east-1.rds.amazonaws.com:5432/postgres"
db_url_droid = "postgres://postgres:ml2021#LORA@droid-v2-prod-instance.cy4dofwtnffp.ap-east-1.rds.amazonaws.com:5432/postgres" # currently using
db_clair_local = 'postgresql://localhost:5432/postgres'
db_url_alibaba = "postgres://asklora:AskLORAv2@pgm-3nse9b275d7vr3u18o.pg.rds.aliyuncs.com:1921/postgres"

# TABLE names - results
membership_table = "factor_membership"
result_pred_table = "factor_result_pred"     # + "_lgbm"/"_rf" + "_reg/class"
result_score_table = "factor_result_score"
feature_importance_table = "factor_result_importance"
production_factor_rank_table = "factor_result_pred_prod"
production_score_history = "data_fundamental_score_history_testing"
production_score_current = "universe_rating" # in DROID v2 DB

# TABLE names - raw data
dl_value_universe_table = "universe"
worldscope_quarter_summary_table = "data_worldscope_summary"
ibes_data_table = "data_ibes_monthly"
macro_data_table = "data_macro_monthly"
stock_data_table_ohlc = "data_dss"
stock_data_table_tri = "data_dsws"
eikon_mktcap_table = "data_factor_eikon_mktcap"
eikon_other_table = "data_factor_eikon_others"
eikon_vix_table = "data_factor_eikon_vix"
eikon_price_table = "data_factor_eikon_price_daily_final"
eikon_report_date_table = "data_factor_eikon_others_date"      # ALIBABA DB

# TABLE names - preprocessed data
processed_ratio_table = "processed_ratio"
processed_stock_table = "processed_stock_weekavg"
factor_premium_table = "processed_factor_premium"
processed_group_ratio_table = "processed_group_ratio"
processed_cutbins_table = "processed_cutbins"
processed_pca_table = 'processed_pca'

# TABLE names - preprocess formula
formula_factors_table = "factor_formula_ratios"

# COLUMN names - preprocess
ticker_column = "ticker"
date_column = "period_end"
icb_column = "icb_code"
index_column = "currency_code"

engine = create_engine(db_url_droid, max_overflow=-1, isolation_level="AUTOCOMMIT")              # APP production DB
engine_prod = create_engine(db_url_prod, max_overflow=-1, isolation_level="AUTOCOMMIT")          # production version
engine_ali = create_engine(db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")        # research DB

