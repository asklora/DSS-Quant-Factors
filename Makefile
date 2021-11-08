.PHONY: build

factor_monthly:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 48 --recalc_premium --tbl_suffix _monthly1 --processes 12

factor_weekly:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 210 --recalc_premium --tbl_suffix _weekly1 --processes 12

factor_eval_current:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/score_evaluate.py --slack
