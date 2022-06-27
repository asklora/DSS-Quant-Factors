# Data Preparation
Read raw data from quant DB Tables and write preprocessed Tables to quant DB. 

## Overview
Run main.py with different argparse to perform data preprocessing and write results to DB.

Preprocessing contains 3 main parts:
1. `--recalc_ratio`: calculate _ratio_ `factor_processed_ratio` from _raw data_
2. `--recalc_premium`: calculate _premium_ `factor_processed_premium` from _ratio(1)_
3. `--recalc_subpillar`: calculate _pillar/subpillar_ `factor_formula_pillar_cluster` from _ratio(1)_

## Change Logs
### 0.1
- Use utils.sql to read / write to DB.
