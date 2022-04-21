.PHONY: build

4w:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 4 --sample_interval 4 --average_days 7 --processes 30 --recalc_premium --recalc_ratio

8w:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 8 --sample_interval 4 --average_days 7 --processes 30 --recalc_premium

13w:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 13 --sample_interval 4 --average_days 7 --processes 30 --recalc_premium

26w:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py --backtest_period 12 --weeks_to_expire 26 --sample_interval 4 --average_days 7 --processes 30 --recalc_premium


eval_eval:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 1 --debug --eval_metric max_ret --eval_n_configs 10 --eval_backtest_period 36 \
		--eval_removed_subpillar  --restart w4_d-7_20220408122219_debug --restart_eval
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 1 --debug --eval_metric max_ret --eval_n_configs 10 --eval_backtest_period 36 \
		--restart w4_d-7_20220408122219_debug --restart_eval

eval_top:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 10 --debug --eval_removed_subpillar  --restart w4_d-7_20220324031027_debug --restart_eval --restart_eval_top
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 10 --debug  --restart w4_d-7_20220324031027_debug --restart_eval --restart_eval_top

trial:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 8 --sample_interval 4 --processes 30 --recalc_subpillar --debug --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 26 --sample_interval 4 --processes 30 --recalc_subpillar --debug --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 52 --sample_interval 4 --processes 30 --recalc_subpillar --debug --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 4 --sample_interval 4 --processes 30 --recalc_subpillar --debug --recalc_premium