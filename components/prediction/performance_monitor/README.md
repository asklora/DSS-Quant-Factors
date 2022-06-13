# Performance Monitor
This steps evaluate prediction results based on prediction.

This component differ from [model_evaluation](components/model/model_evaluation) based on:
1. performance_monitor will perform evaluation everytime after prediction, while model_evaluation is only ran after retraining.
E.g. AI Score actual returns.

In case where we only do prediction once right after retraining (e.g. DSS-Quant-AIvalue). This component is not applicable.

## Overview

## ChangeLog
