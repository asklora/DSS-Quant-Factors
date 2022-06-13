# DSS-Quant-ML-Template
Proposed template for DSS repos centered the training &amp; deployment of ML models 


# Ideas

1. versioning for data
   1. Goal: avoid situation where we are unable to replicate results due to change to input data
   2. Methods: 
      1. dump pre-training data to local -> (X) large storage needed
      2. save key statistic for the pre-training data (e.g. number / start date / end ...) -> (X) debugging may require unsaved statistic
