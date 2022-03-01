.PHONY: build

factor_monthly:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 4 --average_days 7 --processes 30 --recalc_ratio --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 6 --weeks_to_expire 8 --average_days 7 --processes 30 --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 4 --weeks_to_expire 13 --average_days 7 --processes 30 --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 2 --weeks_to_expire 26 -average_days 7 --processes 30 --recalc_premium

factor_weekly:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 210 --recalc_ratio --recalc_premium --weeks_to_expire 1 --processes 32
