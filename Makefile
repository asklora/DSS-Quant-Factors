.PHONY: build

factor_monthly:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 48 --recalc_premium --weeks_to_expire 4 --processes 32

factor_weekly:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 210 --recalc_ratio --recalc_premium --weeks_to_expire 1 --processes 32
