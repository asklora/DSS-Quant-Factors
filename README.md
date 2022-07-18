# AI Score Model

AI Score calculation consists of two major parts, factors selection and calculate the score based on the selected factors.

Formula of AI Score: AI Score = avg(Value Score + Quality Score + Momentum Score + Extra Score or ESG Score)

Where 

Value Score = avg(Best value factors), Best value factors = top 33% value factors predicted by random forest (factor model)

Quality Score = avg(Quality factor 1 score), Best quality factors = top 33% quality factors predicted by random forest (factor model)

Momentum Score = avg(Momentum factor 1 score), Best momentum factors = top 33% momentum factors predicted by random forest (factor model)

Extra Score = avg(Best factors), Best factors = all factors in factor_result_rank table where pred_z > 1 (which means the factor is exceptionally profitiable)

*for each factor* pred_z value = (predicted premium - avg(all predicted premium))/std(all predicted premium)

See further on [confluence](https://loratechai.atlassian.net/wiki/spaces/DS/pages/1061421057/Factor+Model+Summary)

