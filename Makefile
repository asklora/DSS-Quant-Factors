.PHONY: build

cd /home/loratech/PycharmProjects/factors;
python3 main.py --backtest_period 48 --debug --recalc_premium --tbl_suffix _monthly1 --processes 8
python3 main.py --backtest_period 210 --debug --recalc_premium --tbl_suffix _monthly1 --processes 8

cd /home/loratech/PycharmProjects/DROID_V2.1
/home/loratech/droid2env/bin/python3 /home/loratech/PycharmProjects/DROID_V2.1/manage.py main --settings=config.production --fundamentals_rating True


factor_monthly:
	@sudo /home/loratech/droid2env/bin/python3 /home/loratech/PycharmProjects/DROID_V2.1/manage.py backtest --settings=config.settings.production --training True
