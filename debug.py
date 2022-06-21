from components.data_preparation.src.calculation_ratio import calc_factor_variables_multi

df = calc_factor_variables_multi(currency_codes=["USD"], processes=10)

assert len(df) > 0