import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION

if __name__ == "__main__":
    membership_table = "ai_factor_membership"
    factor_premium_table = "ai_factor_factor_premium"
    eikon_mktcap_table = "data_factor_eikon_mktcap"
    eikon_other_table = "data_factor_eikon_others"
    formula_factors_table = "ai_factor_formula_ratios"

    with global_vals.engine.connect() as conn:
        final_member_df = pd.read_sql(f'SELECT * FROM {global_vals.membership_table}', conn)
        final_results_df = pd.read_sql(f'SELECT * FROM {global_vals.factor_premium_table}', conn)
        # mktcap = pd.read_sql(f'SELECT * FROM {global_vals.eikon_mktcap_table}', conn)
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
        # mktcap.to_sql(global_vals.eikon_mktcap_table, **extra)
        others.to_sql(global_vals.eikon_other_table, **extra)
        formula.to_sql(global_vals.formula_factors_table, **extra)
        final_results_df.to_sql(global_vals.factor_premium_table, **extra, dtype=results_dtypes)
        print(f'      ------------------------> Finish writing factor premium table ')
        final_member_df.to_sql(global_vals.membership_table, **extra, dtype=mem_dtypes)
        print(f'      ------------------------> Finish writing factor membership table ')
    global_vals.engine.dispose()