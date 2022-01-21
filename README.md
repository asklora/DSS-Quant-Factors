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

Extra Score = avg(Best factors), Best factors = all factors in factor_result_rank table where pred_z > 1

See further on [confluence](https://loratechai.atlassian.net/wiki/spaces/SEAR/pages/880738405/AI+Score)

# Factor Model

### *factor repository*

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

