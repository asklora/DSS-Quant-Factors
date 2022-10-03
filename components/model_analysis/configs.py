from dataclasses import dataclass

@dataclass
class LOGGER_LEVELS:
    ANALYSIS_CONFIG_OPTIMIZATION_REG: str = "INFO"
    ANALYSIS_CONFIG_OPTIMIZATION:     str = "INFO"
    ANALYSIS_FACTOR_CONFIG:          str = "INFO"
    ANALYSIS_RUNTIME_EVAL:            str = "INFO"
    ANALYSIS_SCORE_BACKTEST_EVAL:     str = "INFO"
    ANALYSIS_SCORE_BACKTEST_EVAL2:    str = "INFO"
    ANALYSIS_SCORE_BACKTEST_EVAL3:    str = "INFO"
    ANALYSIS_SCORE_BACKTEST:          str = "INFO"
    MAIN:                             str = "INFO"
    ANALYSIS_UNIVERSE_RATING_HISTORY: str = "INFO"
    ANALYSIS_BEST_LASSO_PERIOD:       str = "INFO"
    ANALYSIS_RANK_FACTOR_ROTATION:    str = "INFO"
    ANALYSIS_UNIVERSE_RATING:         str = "INFO"