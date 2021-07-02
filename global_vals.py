from sqlalchemy import create_engine

db_url_prod = "postgres://postgres:ml2021#LORA@droid-v2-prod-instance.cy4dofwtnffp.ap-east-1.rds.amazonaws.com:5432/postgres"
db_url_droid = "postgres://postgres:ml2021#LORA@droid-v2-prod-cluster.cluster-ro-cy4dofwtnffp.ap-east-1.rds.amazonaws.com:5432/postgres" # currently using
db_url_hkpolyu = "postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres"

# TABLE names - results
test_score_results_table = "ai_value_lgbm_score"          # table store all running information
test_pred_results_table = "ai_value_lgbm_pred"     # table store prediction for each ticker
test_pred_eval_table = "ai_value_lgbm_eval"         # table store prediction backtest evaluation
final_pred_results_table = "ai_value_lgbm_pred_final"      # table for consolidated pred after consolidation
final_eps_pred_results_table = "ai_value_lgbm_pred_final_eps"      # table with the best ratio pred -> EPS format

# TABLE names - preprocess data
dl_value_universe_table = "universe"
worldscope_quarter_summary_table = "data_worldscope_summary"
ibes_data_table = "data_ibes_monthly"
macro_data_table = "data_macro_monthly"
stock_data_table = "master_ohlcvtr"

# TABLE names - preprocess formula
formula_ratios_table = "ai_value_formula_ratios"
formula_factors_table = "ai_value_formula_factors"

# COLUMN names - preprocess
ticker_column = "ticker"
date_column = "period_end"
icb_column = "icb_code"
index_column = "currency_code"

engine = create_engine(db_url_droid, max_overflow=-1, isolation_level="AUTOCOMMIT")          # production version

