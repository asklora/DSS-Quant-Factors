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
		--processes 1 --debug --restart w26_20220425095800_debug --pass_train

eval_top:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 12 --debug --restart w26_20220425095800_debug --pass_train --pass_eval
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 12 --debug --restart w8_20220422100952_debug --pass_train --pass_eval

write_select:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 1 --restart w26_20220425095800_debug --pass_train --pass_eval --pass_eval_top
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--processes 1 --restart w8_20220422100952_debug --pass_train --pass_eval --pass_eval_top

trial:
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 8 --sample_interval 4 --processes 10 --recalc_subpillar --debug
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 26 --sample_interval 4 --processes 30 --recalc_subpillar --debug --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 52 --sample_interval 4 --processes 30 --recalc_subpillar --debug --recalc_premium
	@sudo /home/loratech/PycharmProjects/factors/venv/bin/python3 /home/loratech/PycharmProjects/factors/main.py \
		--backtest_period 80 --weeks_to_expire 4 --sample_interval 4 --processes 30 --recalc_subpillar --debug --recalc_premium