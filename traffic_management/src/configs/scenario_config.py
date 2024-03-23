import json
import os
import warnings

import config
from configs.config import Config
from utils import collections_util

MISSING_FILE = "___MISSING_FILE___"
TO_COMPUTE = "___TO_COMPUTE___"


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

    MULTI_INTERSECTION_CONFIG = TO_COMPUTE

    @classmethod
    def update_globals(cls, modifications):

        super().update_globals(modifications)

        try:
            with open(os.path.join(
                    config.ROOT_DIR, config.PATH_TO_DATA, ScenarioConfig.MULTI_INTERSECTION_TL_FILE), 'r') as file:
                ScenarioConfig.MULTI_INTERSECTION_CONFIG = collections_util.HashableDict(json.load(file))
        except Exception as e:
            warnings.warn("No multi intersection tl file present")
            ScenarioConfig.MULTI_INTERSECTION_CONFIG = collections_util.HashableDict()

    @classmethod
    def warn_missing_configuration(cls, k, v):
        return
