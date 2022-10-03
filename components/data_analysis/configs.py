from dataclasses import dataclass

@dataclass
class LOGGER_LEVELS:
    ANALYSIS_PREMIUM_LEGACY:  str = "INFO"
    ANALYSIS_PREMIUM:         str = "INFO"
    ANALYSIS_RATIO:           str = "INFO"
    ANALYSIS_VOLATILITY:      str = "INFO"
    ANALYSIS_AVERAGE_PREMIUM: str = "INFO"
