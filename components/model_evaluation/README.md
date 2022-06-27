# Model Evaluation
Evaluate model prediction results on re-training.

## Overview
Run main.py to perform evaluation of given training iteration label by `name_sql` from Table [factor_model]. 

Evaluation contains 3 main parts:
- `--eval_factor`: calculate average premium for selected factors
- `--eval_top`: calculate top selection returns with backtest AI score calculated from selected factors
- `--eval_select`: rewrite factor selection table with good/bad factor selected by name_sql


## ChangeLog
