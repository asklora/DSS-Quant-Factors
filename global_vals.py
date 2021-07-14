from sqlalchemy import create_engine

db_url_droid = "postgres://postgres:ml2021#LORA@droid-v2-prod-cluster.cluster-ro-cy4dofwtnffp.ap-east-1.rds.amazonaws.com:5432/postgres" # currently using
db_clair_local = 'postgresql://localhost:5432/postgres'

# TABLE names - results
membership_table = "ai_factor_membership"
factor_premium_table = "ai_factor_factor_premium"

# TABLE names - preprocess data
dl_value_universe_table = "universe"
worldscope_quarter_summary_table = "data_worldscope_summary_test"
ibes_data_table = "data_ibes_monthly"
macro_data_table = "data_macro_monthly"
stock_data_table = "master_ohlcvtr"
eikon_data_table = "data_eikon_ai_factor"

# TABLE names - preprocess formula
formula_factors_table = "ai_factor_formula_ratios"

# COLUMN names - preprocess
ticker_column = "ticker"
date_column = "period_end"
icb_column = "icb_code"
index_column = "currency_code"

engine = create_engine(db_url_droid, max_overflow=-1, isolation_level="AUTOCOMMIT")          # production version

