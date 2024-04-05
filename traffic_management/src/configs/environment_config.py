from configs.config import Config


class EnvironmentConfig(Config):

    DETECTOR_EXTENSION = 100  # int -> distance (m); string with 's' at the end -> time (s)
    USE_GUI = False
    DEBUG = False
    DETECTOR_ROAD_TYPE = [
        "highway.primary",
        "highway.primary_link",
        "highway.secondary",
        "highway.secondary_link",
        "highway.tertiary",
        "highway.tertiary_link",
        "highway.residential"
    ]

