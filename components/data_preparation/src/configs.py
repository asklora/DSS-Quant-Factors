from dataclasses import dataclass


@dataclass
class LOGGER_LEVELS:
    CALCULATION_PREMIUM:        str = "INFO"
    CALCULATION_RATIO:          str = "INFO"
    CALCULATION_PILLAR_CLUSTER: str = "INFO"
    MAIN:                       str = "INFO"