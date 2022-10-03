from dataclasses import dataclass

@dataclass
class LOGGER_LEVELS:
    MAIN:               str = "WARNING"
    LOAD_DATA:          str = "WARNING"
    LOAD_TRAIN_CONFIGS: str = "WARNING"
    RANDOM_FOREST:      str = "WARNING"
