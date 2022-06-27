# Model Training
This Step train model based on preprocessed data.

## Overview
Run main.py with different argparse to train factor model for different `weeks_to_expire` (i.e. prediction period).

It performs the following 3 step-by-step:
1. combineData(): get all raw data for premium + macros
2. loadTrainConfig(): get all configuration to try
3. start(): Multiprocess to run training on each configuration by
   1. loadData(): convert raw data to train/valid/test sample sets
   2. rf_HPOT(): hyperopt + random forest training + prediction

## ChangeLog

## Research Documentation
This component should be supplemented with research documentation on why we eventually decide to use current model.