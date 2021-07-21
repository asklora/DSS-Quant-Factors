import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION

def move_factor_only_table():
    membership_table = "ai_factor_membership"
    factor_premium_table = "ai_factor_factor_premium"
    eikon_mktcap_table = "data_factor_eikon_mktcap"
    eikon_other_table = "data_factor_eikon_others"
    formula_factors_table = "ai_factor_formula_ratios"

    with global_vals.engine.connect() as conn:
        final_member_df = pd.read_sql(f'SELECT * FROM {global_vals.membership_table}', conn)
        final_results_df = pd.read_sql(f'SELECT * FROM {global_vals.factor_premium_table}', conn)
        mktcap = pd.read_sql(f'SELECT * FROM {global_vals.eikon_mktcap_table}', conn)
        others = pd.read_sql(f'SELECT * FROM {global_vals.eikon_other_table}', conn)
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)
    global_vals.engine.dispose()

    mem_dtypes = {}
    for i in list(final_member_df.columns):
        mem_dtypes[i] = DOUBLE_PRECISION
    mem_dtypes['period_end'] = DATE
    mem_dtypes['group']=TEXT
    mem_dtypes['ticker']=TEXT

    results_dtypes = {}
    for i in list(final_results_df.columns):
        results_dtypes[i] = DOUBLE_PRECISION
    results_dtypes['period_end'] = DATE
    results_dtypes['group']=TEXT

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 1000}
        mktcap.to_sql(global_vals.eikon_mktcap_table, **extra)
        others.to_sql(global_vals.eikon_other_table, **extra)
        formula.to_sql(global_vals.formula_factors_table, **extra)
        final_results_df.to_sql(global_vals.factor_premium_table, **extra, dtype=results_dtypes)
        final_member_df.to_sql(global_vals.membership_table, **extra, dtype=mem_dtypes)
    global_vals.engine.dispose()

def move_general(table_name):
    with global_vals.engine.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    global_vals.engine.dispose()

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 1000}
        df.to_sql(table_name, **extra)
    global_vals.engine_ali.dispose()

if __name__ == "__main__":

    # dl_value_universe_table = "universe"
    # worldscope_quarter_summary_table = "data_worldscope_summary_test"
    # ibes_data_table = "data_ibes_monthly"
    # macro_data_table = "data_macro_monthly"

    test_score_results_table = "ai_value_lgbm_score"  # table store all running information
    test_pred_results_table = "ai_value_lgbm_pred"  # table store prediction for each ticker
    test_pred_eval_table = "ai_value_lgbm_eval"  # table store prediction backtest evaluation
    final_pred_results_table = "ai_value_lgbm_pred_final"  # table for consolidated pred after consolidation
    final_eps_pred_results_table = "ai_value_lgbm_pred_final_eps"  # table with the best ratio pred -> EPS format
    formula_ratios_table = "ai_value_formula_ratios"

    for f in [test_score_results_table, test_pred_results_table, test_pred_eval_table, final_pred_results_table, final_eps_pred_results_table, formula_ratios_table]:
        move_general(f)
        print('finish moving ', f)