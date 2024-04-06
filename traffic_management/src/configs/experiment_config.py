import uuid
import time
import warnings

from configs.config import Config


class ExperimentConfig(Config):

    SCENARIO_FOLDER = '0_regular-intersection'
    SCENARIO_NAME = '0_regular-intersection__regular'

    __name_base = f"{SCENARIO_FOLDER}/{SCENARIO_NAME}"
    __unique_id = str(uuid.uuid4())
    __suffix = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())) + '__' + __unique_id
    __NAME = __name_base + "___" + __suffix
    LABEL = "Gman"

    NAME = ""
    BASELINE = ""
    COPY_DATA_FROM = ""

    DATA_GENERATION_RUN_COUNTS = 3600 * 24
    MODEL_NAME = "Gman"
    DEBUG = False
    TIME = "00:00:00"

    __MISSING_CONFIG_WARNING_SUPPRESSION = [
        "NAME",
        "BASELINE",
        "COPY_DATA_FROM"
    ]

    @classmethod
    def update_globals(cls, modifications):

        super().update_globals(modifications)

        if 'NAME' not in modifications:
            if 'SCENARIO_FOLDER' in modifications or 'SCENARIO_NAME' in modifications:
                name_base = f"{ExperimentConfig.SCENARIO_FOLDER}/{ExperimentConfig.SCENARIO_NAME}"
                ExperimentConfig.NAME = name_base + "___" + ExperimentConfig.__suffix
            else:
                ExperimentConfig.NAME = cls.__NAME

            print(f"New experiment -> \n{ExperimentConfig.NAME}")

        else:
            warnings.warn(f"Using existing experiment -> \n{ExperimentConfig.NAME}")

    @classmethod
    def warn_missing_configuration(cls, k, v):

        if k in cls.__MISSING_CONFIG_WARNING_SUPPRESSION:
            return

        warnings.warn(f"Missing configuration. Using Default. ({cls.__name__})-> \n\tK:{k}\n\tV:{v}")
