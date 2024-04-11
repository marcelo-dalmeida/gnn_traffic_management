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
    ACCIDENT_GEN_FAILED_TRIES = 10
    ACCIDENT_GEN_WARMUP = 1800                                      # -1 deactivates accident generation
    ACCIDENT_GEN_COOLDOWN = 1800
    ACCIDENT_GEN_ROAD_TYPE_PROB = [
        (0.45, ("highway.primary", "highway.primary_link",)),
        (0.30, ("highway.secondary", "highway.secondary_link",)),
        (0.20, ("highway.tertiary", "highway.tertiary_link",)),
        (0.05, ("highway.residential",))
    ]
    ACCIDENT_GEN_DURATION_PROB = [
        (0.20, 30),
        (0.40, 60),
        (0.30, 90),
        (0.10, 120)
    ]
    ACCIDENT_GEN_LANES_BLOCKED_PROB = [
        (0.40, 1),
        (0.60, 2)
    ]
    SLOW_DOWN_GEN_WARMUP = 60                                      # -1 deactivates slow down generation
    SLOW_DOWN_GEN_COOLDOWN = 10
    SLOW_DOWN_GEN_VEHICLE_PERCENTAGE = 0.1
    SLOW_DOWN_GEN_DURATION_PROB = [
        (0.20, 3),
        (0.20, 4),
        (0.20, 5),
        (0.20, 6),
        (0.20, 7)
    ]
