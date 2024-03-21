from configs.config import Config


class EnvironmentConfig(Config):

    DETECTOR_EXTENSION = 100  # int -> distance (m); string with 's' at the end -> time (s)
    USE_GUI = False
    DEBUG = False

