# AI Score Model
1. [Principle](#principle)
2. [Factor Model](#factor-model)
    1. [Data Preparation](#data-preparation)
    2. [Prediction](#prediction)
3. [Ingestion](#ingestion)
4. [Database](#database)

# Principle

AI Score calculation consists of two major parts, factors selection and calculate the score based on the selected factors.

Formula of AI Score: AI Score = avg(Value Score + Quality Score + Momentum Score + Extra Score or ESG Score)

Where 

Value Score = avg(Best value factors), Best value factors = top 33% value factors predicted by random forest (factor model)

Quality Score = avg(Quality factor 1 score), Best quality factors = top 33% quality factors predicted by random forest (factor model)

Momentum Score = avg(Momentum factor 1 score), Best momentum factors = top 33% momentum factors predicted by random forest (factor model)

Extra Score = avg(Best factors), Best factors = all factors in factor_result_rank table where pred_z > 1 (which means the factor is exceptionally profiting)

*for each factor* pred_z value = (predicted premium - avg(all predicted premium))/std(all predicted premium)

See further on [confluence](https://loratechai.atlassian.net/wiki/spaces/SEAR/pages/880738405/AI+Score)

# Factor Model

This repository is the factor model reposition, which selects "good"/"bad" factors for each period on different horizons (1-week, 4-week, 8-week, 26-week).

## Contents
```
DSS-Quant-Factors/
┣ cron/
┃ ┣ LogFile/
┃ ┣ factor_monthly.sh
┃ ┗ factor_weekly.sh
┣ general/
┃ ┣ send_email.py
┃ ┣ send_slack.py
┃ ┣ sql_process.py
┃ ┗ utils.py
┣ images/
┃ ┗ factormodel.png
┣ preprocess/
┃ ┣ calculation_ratio.py
┃ ┣ calculation_premium.py
┃ ┣ model.py
┃ ┗ analysis_premium.py
┣ results_analysis/
┃ ┣ calculation_rank.py
┃ ┣ analysis_backtest_eval.py
┃ ┣ analysis_runtime_eval.py
┃ ┣ analysis_score_backtest.py
┃ ┣ analysis_score_backtest.py
┃ ┗ analysis_universe_rating_history.py
┣ .gitignore
┣ README.md
┣ global_vars.py
┣ main.py
┣ Makefile
┣ random_forest.py
┗ requirements.txt
```
---
## **cron/**

### **LogFile/**
This folder contains *.log files for weekly / monthly training of models.
The training is currently scheduled on PC1 with the following cron jobs.
```
10 14 * * 0 /home/loratech/PycharmProjects/factors/cron/factor_monthly.sh 2>&1 > /home/loratech/PycharmProjects/factors/cron/factor_monthly.log 2>&1
10 13 * * 0 /home/loratech/PycharmProjects/factors/cron/factor_weekly.sh 2>&1 > /home/loratech/PycharmProjects/factors/cron/factor_weekly.log 2>&1
```
In [main.py](main.py), we add extra constrains for the starting time of above jobs. 
1. Monthly training will only start on the first Sunday of each month.
2. Monthly training will only start after weekly training finished. 
3. Weekly training will only start after weekly ingestion of Worldscope/IBES/Macros Data finished. 

### **factor_monthly.sh**
For monthly training schedule. 

### **factor_weekly.sh**
For weekly training schedule.

---
## **general/**

### **send_email.py**
Send emails from [asklora@loratechai.com](asklora@loratechai.com).

### **send_slack.py**
Send message / pd.Series / pd.DataFrames / Files to slack. 
Factor model slack channel: [#dss-quant-factors-message](#dss-quant-factors-message).

### **sql_process.py**
For read / write to SQL DB. Refer to [global_vars.py](global_vars.py) for db_url_read / db_url_write. 
Production should set both as ALIBABA Prod DB URL.

### **utils.py**
Other general utility functions (e.g. save to excel).

---
## **images/**

### **factormodel.png**
Flowchart for training process for [README.md](README.md).

---
## **preprocess/**
Preprocessing raw data ingested to expected format for training / prediction.

### **calculation_ratio.py**
Using raw data ingestion to calculate weekly factor ratios of each ticker and write to Table `factor_processed_ratio`. 
Full list of factors calculated can refer to Table `factor_formula_ratios_prod`.

Ratio table will be updated weekly before training for recent 3-month ratios. 

### **calculation_premium.py**
Using factor ratios calculated with [calculation_ratio.py](calculation_ratio.py) to calculate premiums in each group (currency_code / indsutry_code) and write to `factor_processed_premium`.

Premium table will be updated weekly before training for recent 3-month ratios. 


### **load_data.py**

### **model_log.py**

### **analysis_premium.py**


## Data Preparation

Model input are factor premiums of the factors in one pillar, defined in this [page](https://loratechai.atlassian.net/wiki/spaces/SEAR/pages/858685974/Story+2021-08-20)

Recap: 4 pillars = Value, Quality, Momentum, Extra

Output are the factor premiums of the next period.

Test data set are the factor premiums in backtest period x week before today

Validation data set are the factor premiums in 2 years before the test set

Training data set are all the factor premiums before the validation set

## Prediction

#### Factor selection and ranking process is as 

![factormodel](images/factormodel.png)

use `python3 main.py --option *sth*` to build, test, run model by a signle script.
Default 1 period = 1 week
| Option | Explanation | Input |
|--------|-------------|------------|
| recalc_premium | recalculate stock premiums | None |
| recalc_ratio | recalculate stock to vector ratios | None |
| weeks_to_expire | how many weeks for this prediction to expire | n (int) |
| processes | create how many parallel process (multiprocessing) to run the script | n (int) |
| backtest_period | use how many weeks as test dataset | n (int) |
| n_splits | split validation set into how many sets | n (int) |
| trim | trim outliers (top & bottom 5% of each dataset) | True/False |
| debug | run script in dev mode (will not affect prod database) | True/False |


Model is not saved as time required for the whole process is short (depends on settings, at most few hours)

Run main.py without --debug will cause the script to wait until the next database update (returns) to execute



# Ingestion
### *ingestion repository*

Ingestion is to update the ai_score of each ticker based on the current factor settings (prediction results given by the factor model)

use the script ingestion/universe_rating.py to execute the above operation
like `python3 universe_rating.py`

# Database

universe_rating → current displayed ai_score

universe_rating_history → ai_score history

universe_rating_detail_history → field and values used for ai_score calculation

factor_model → each prediction model run details

factor_model_stock → predicted return for each factor compared with actual return

factor_result_rank → predicted z score for each factors and its rank currently used

factor_result_rank_history → predicted z score for each factors and its rank currently in the past