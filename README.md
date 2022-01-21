# AI Score Model
1. [Principle](#principle)
2. [Factor Model](#factor-model)
    1. [Data Preparation](#data-preparation)
    2. [Prediction](#prediction)
3. [Ingestion](#ingestion)
4. [Database Structure](#databse-structure)

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

## Data Preparation

Model input are factor premiums of the factors in one pillar, defined in this [page](https://loratechai.atlassian.net/wiki/spaces/SEAR/pages/858685974/Story+2021-08-20)

Recap: 4 pillars = Value, Quality, Momentum, Extra

Output are the factor premiums of the next period.

Test data set are the factor premiums in backtest period x week before today

Validation data set are the factor premiums in 2 years before the test set

Training data set are all the factor premiums before the validation set

## Prediction

Factor selection and ranking process is as 
![factormodel](images/factormodel.png)

use `python3 main.py` to build, test, run model by a signle script.
Default 1 period = 1 week
| Option | Explanation | Param type |
|--------|-------------|------------|
| recalc_premium | recalculate stock premiums | None |


Model is not saved as time required for the whole process is short (depends on settings, at most few hours)

# Ingestion

# Database Structure
