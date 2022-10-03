from dataclasses import dataclass

@dataclass
class LOGGER_LEVELS:
    CALCULATION_BACKTEST_SCORE: str = "INFO"
    EVALUATION_FACTOR_PREMIUM:  str = "INFO"
    EVALUATE_TOP_SELECTION:     str = "INFO"
    LOAD_EVAL_CONFIGS:          str = "INFO"
    MAIN:                       str = "INFO"
