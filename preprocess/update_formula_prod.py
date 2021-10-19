#TODO: Update prod formula to include new "ww" & "quality" factors
import global_vals
import pandas as pd

with global_vals.engine_ali.connect() as conn:
    formula = pd.read_sql('SELECT * FROM factor_formula_ratios WHERE x_col', conn)
    formula_prod = pd.read_sql('SELECT * FROM factor_formula_ratios_prod', conn)
global_vals.engine_ali.dispose()

r = set(formula['name']) - set(formula_prod['name'])
p = set(formula_prod['name']) - set(formula['name'])

formula = formula.loc[formula['name'].isin(r)]
formula['new_col'] = True

formula_prod = pd.concat([formula, formula_prod], axis=0)

with global_vals.engine_ali.connect() as conn:
    extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
    formula_prod.to_sql('factor_formula_ratios_prod_test', **extra)
global_vals.engine_ali.dispose()

print(r)
print(p)