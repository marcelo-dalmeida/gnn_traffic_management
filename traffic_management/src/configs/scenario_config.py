
from configs.config import Config

MISSING_FILE = "___MISSING_FILE___"


class ScenarioConfig(Config):

    HAS_BUSES = False
    HAS_PASSENGERS = False

    CONFIGURATION_FILE = MISSING_FILE
    NET_FILE = MISSING_FILE
    MULTI_INTERSECTION_TL_FILE = MISSING_FILE

    CAR_TRIPS_FILE = MISSING_FILE

    BUS_STOPS_FILE = MISSING_FILE
    BUS_TRIPS_FILE = MISSING_FILE
    BUS_LINES_FILE = MISSING_FILE

    PASSENGER_TRIPS_FILE = MISSING_FILE

    @classmethod
    def warn_missing_configuration(cls, k, v):
        return
