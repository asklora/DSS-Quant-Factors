.PHONY: build

factor_monthly:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 4 --sample_interval 4 --average_days 7 --processes 30 --recalc_ratio --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 8 --sample_interval 4 --average_days 7 --processes 30 --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 13 --sample_interval 4 --average_days 7 --processes 30 --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 26 --sample_interval 4 -average_days 7 --processes 30 --recalc_premium

factor_weekly:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 210 --recalc_ratio --recalc_premium --weeks_to_expire 1 --processes 32

eval_eval:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 1 --debug --eval_metric max_ret --eval_n_configs 10 --eval_backtest_period 36 \
		--eval_removed_subpillar  --restart w4_d-7_20220312222718_debug --restart_eval
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 1 --debug --eval_metric max_ret --eval_n_configs 10 --eval_backtest_period 36 \
		--restart w4_d-7_20220312222718_debug --restart_eval

eval_top:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 1 --debug --eval_metric max_ret --eval_n_configs 10 --eval_backtest_period 36 \
		--eval_removed_subpillar  --restart w4_d-7_20220329120327_debug --restart_eval --restart_eval_top

trial:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 4 --sample_interval 4 --average_days -7 --processes 30 --debug \
		--group_code HKD,CNY,currency --y_type cluster --hpot_eval_metric adj_mse_valid
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 4 --sample_interval 4 --average_days -7 --processes 30 --debug \
		--group_code USD,currency --y_type momentum,value,quality --hpot_eval_metric mse_valid
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 4 --sample_interval 4 --average_days -7 --processes 30 --debug \
		--group_code EUR,currency --y_type momentum,value,quality --hpot_eval_metric adj_mse_valid
