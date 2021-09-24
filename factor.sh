#!/bin/sh
cd /home/loratech/PycharmProjects/factors;
python3 main.py --backtest_period 12

cd /home/loratech/PycharmProjects/DROID_V2.1
/home/loratech/droid2env/bin/python3 /home/loratech/PycharmProjects/DROID_V2.1/manage.py main --settings=config.production --fundamentals_rating True

cd /home/loratech/PycharmProjects/factors;
python3 score_evaluate.py