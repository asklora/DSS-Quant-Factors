#!/bin/sh
cd /home/loratech/PycharmProjects/factors;
python3 main.py --backtest_period 12 --recalc_premium --debug

cd /home/loratech/PycharmProjects/DROID_V2.1
/home/loratech/droid2env/bin/python3 /home/loratech/PycharmProjects/DROID_V2.1/ingestion/data_from_dsws.py update_fundamentals_quality_value

#cd /home/loratech/PycharmProjects/factors;
#python3 score_evaluate.py