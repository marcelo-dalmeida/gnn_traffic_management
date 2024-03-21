import os
import sys
import json
import shutil
import warnings
from os import path
from pathlib import Path

from multiprocessing import Lock

import load_envs

from configs.scenario_config import ScenarioConfig
from configs.experiment_config import ExperimentConfig
from configs.agent_config import AgentConfig
from configs.environment_config import EnvironmentConfig

load_envs.load()

VERSION = '0.1.0-dev'


class CONST:
    CAR = 'passenger'
    BUS = 'bus'
    EMERGENCY = 'emergency'
    BUS_PASSENGER = 'bus_passenger'
    PEDESTRIAN = 'pedestrian'

    AVAILABLE_VEHICLE_ACTORS = [CAR, BUS, EMERGENCY]
    AVAILABLE_PERSON_ACTORS = [BUS_PASSENGER]

    TRAFFIC_LIGHT_SYSTEM = 'TRAFFIC_LIGHT_SYSTEM'
    EXCLUSIVE_LANE_SYSTEM = 'EXCLUSIVE_LANE_SYSTEM'

    AVAILABLE_CONTROL_SYSTEMS = [TRAFFIC_LIGHT_SYSTEM, EXCLUSIVE_LANE_SYSTEM]

def _check_existing_experiment():

    experiment_config_filepath = os.path.join(CONFIG_DIR, 'experiment_config.json')
    with open(experiment_config_filepath, 'r') as experiment_config_file:
        config_data = json.load(experiment_config_file)

    if 'NAME' in config_data:

        experiment_name = config_data['NAME']

        if isinstance(experiment_name, list):
            if experiment_name[0] == "DELETE::":
                experiment_names = experiment_name[1:]

                for experiment_name in experiment_names:
                    delete_experiment(experiment_name)

                sys.exit()

            else:
                raise ValueError("Experiment name is a list and it doesn't have the delete directive")

        raw_experiment_name = experiment_name.replace('NEW::', '')

        path_to_config = os.path.join("config", raw_experiment_name)

        if os.path.isdir(os.path.join(ROOT_DIR, path_to_config)):

            if "NEW::" in config_data['NAME']:
                raise ValueError(f"This experiment already exists: {raw_experiment_name}")

            _set_experiment_name(experiment_name)

            return True, experiment_name
        else:

            if 'NEW::' not in config_data['NAME']:
                raise ValueError(f"This experiment does not exist: {raw_experiment_name}")

            return False, experiment_name

    return False, None


def delete_experiment(experiment_name):

    def onerror(function, path, excinfo):
        warnings.warn(f"Couldn't delete path {path}; {excinfo}")

    path_to_config = os.path.join(ROOT_DIR, "config", experiment_name)
    path_to_metric = os.path.join(ROOT_DIR, "metric", experiment_name)
    path_to_model = os.path.join(ROOT_DIR, "model", experiment_name)
    path_to_records = os.path.join(ROOT_DIR, "records", experiment_name)
    path_to_summary = os.path.join(ROOT_DIR, "summary", experiment_name)

    shutil.rmtree(path_to_config, onerror=onerror)
    shutil.rmtree(path_to_metric, onerror=onerror)
    shutil.rmtree(path_to_model, onerror=onerror)
    shutil.rmtree(path_to_records, onerror=onerror)
    shutil.rmtree(path_to_summary, onerror=onerror)


def _check_baseline():

    experiment_config_filepath = os.path.join(CONFIG_DIR, 'experiment_config.json')
    with open(experiment_config_filepath, 'r') as experiment_config_file:
        config_data = json.load(experiment_config_file)

    if 'BASELINE' in config_data:

        baseline_name = config_data['BASELINE']

        path_to_config = os.path.join("config", baseline_name)

        if os.path.isdir(os.path.join(ROOT_DIR, path_to_config)):
            return True
        else:
            raise ValueError(f"This baseline does not exist: {baseline_name}")

    return False


def _set_experiment_name(experiment_name):

    path_to_config = os.path.join("config", experiment_name)

    global experiment_config_filepath
    experiment_config_filepath = os.path.join(ROOT_DIR, path_to_config, 'experiment_config.json')

    global environment_config_filepath
    environment_config_filepath = os.path.join(ROOT_DIR, path_to_config, 'environment_config.json')

    global agent_config_filepath
    agent_config_filepath = os.path.join(ROOT_DIR, path_to_config, 'agent_config.json')

    global scenario_config_filepath
    scenario_config_filepath = os.path.join(ROOT_DIR, path_to_config, 'scenario_config.json')


def _copy_scenario_files():

    scenario_source_folder = \
        os.path.join(os.path.dirname(os.path.dirname(ROOT_DIR)), 'scenario', EXPERIMENT.SCENARIO_FOLDER)
    scenario_destination_folder = os.path.join(ROOT_DIR, PATH_TO_DATA)

    if not os.path.exists(scenario_destination_folder):
        shutil.copytree(scenario_source_folder, scenario_destination_folder)


def _load_configs(experiment_config_filepath=None, environment_config_filepath=None, agent_config_filepath=None):

    try:
        if experiment_config_filepath is None:
            experiment_config_filepath = os.path.join(CONFIG_DIR, 'experiment_config.json')
        with open(experiment_config_filepath, 'r') as experiment_config_file:
            config_data = json.load(experiment_config_file)
            _load_config(ExperimentConfig, config_data)
    except FileNotFoundError:
        warnings.warn("Using default experiment configuration")

    try:
        if environment_config_filepath is None:
            environment_config_filepath = os.path.join(CONFIG_DIR, 'environment_config.json')
        with open(environment_config_filepath, 'r') as environment_config_file:
            config_data = json.load(environment_config_file)
            _load_config(EnvironmentConfig, config_data)
    except FileNotFoundError:
        warnings.warn("Using default environment configuration")

    model_name = ExperimentConfig.MODEL_NAME
    try:
        if agent_config_filepath is None:
            agent_config_filepath = os.path.join(CONFIG_DIR, 'agent', f'{model_name.lower()}_config.json')
        with open(agent_config_filepath, 'r') as agent_config_file:
            config_data = json.load(agent_config_file)
            _load_config(AgentConfig.get_config(model_name), config_data)
    except FileNotFoundError:
        warnings.warn("Using default agent configuration")


def _load_scenario_config(scenario_config_filepath=None):
    _copy_scenario_files()

    overridable = True if scenario_config_filepath is None else False

    scenario_config_filepath = os.path.join(ROOT_DIR, PATH_TO_DATA, 'simulation', 'scenario_config.json')
    with open(scenario_config_filepath, 'r') as scenario_config_file:
        config_data = json.load(scenario_config_file)
        _load_config(ScenarioConfig, config_data)

    if overridable:
        scenario_config_filepath = os.path.join(CONFIG_DIR, 'scenario_config.json')
        with open(scenario_config_filepath, 'r') as scenario_config_file:
            config_data = json.load(scenario_config_file)
            _load_config(ScenarioConfig, config_data)


def _load_config(config_module, config_info):

    module_globals = config_module.get_globals()

    unknown_config_info = {k: v for k, v in config_info.items() if
                           k not in module_globals.keys()}

    for k, v in unknown_config_info.items():
        config_module.warn_unknown_configuration(k, v)

    missing_config_info = {k: v for k, v in module_globals.items() if
                           k not in config_info.keys()}

    for k, v in missing_config_info.items():
        config_module.warn_missing_configuration(k, v)

    config_info = {k: v for k, v in config_info.items() if
                   k in (module_globals.keys() & config_info.keys())}

    config_module.update_globals(config_info)


def _store_configs():

    Path(os.path.join(ROOT_DIR, PATH_TO_CONFIG)).mkdir(parents=True)

    with open(os.path.join(ROOT_DIR, PATH_TO_CONFIG, 'experiment_config.json'), 'w+') as experiment_config_file:
        config_data = ExperimentConfig.get_globals()
        json.dump(config_data, experiment_config_file, indent=4)

    with open(os.path.join(ROOT_DIR, PATH_TO_CONFIG, 'environment_config.json'), 'w+') as environment_config_file:
        config_data = EnvironmentConfig.get_globals()
        json.dump(config_data, environment_config_file, indent=4)

    model_name = ExperimentConfig.MODEL_NAME

    with open(os.path.join(ROOT_DIR, PATH_TO_CONFIG, 'agent_config.json'), 'w+') \
            as agent_config_file:
        config_data = AgentConfig.get_config(model_name).get_globals()
        json.dump(config_data, agent_config_file, indent=4)

    with open(os.path.join(ROOT_DIR, PATH_TO_CONFIG, 'scenario_config.json'), 'w+') as scenario_config_file:
        config_data = ScenarioConfig.get_globals()
        json.dump(config_data, scenario_config_file, indent=4)


ROOT_DIR = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'output')
CONFIG_DIR = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'config')

experiment_config_filepath = None
environment_config_filepath = None
agent_config_filepath = None
scenario_config_filepath = None

EXISTING_EXPERIMENT_FLAG, _experiment_name = _check_existing_experiment()
_check_baseline()

_load_configs(experiment_config_filepath=experiment_config_filepath,
              environment_config_filepath=environment_config_filepath,
              agent_config_filepath=agent_config_filepath)

EXPERIMENT = ExperimentConfig

if _experiment_name and 'NEW::' in _experiment_name:
    _experiment_name = _experiment_name.replace('NEW::', '')
    EXPERIMENT.NAME = _experiment_name

AGENT = AgentConfig.get_config(EXPERIMENT.MODEL_NAME)
ENVIRONMENT = EnvironmentConfig
SCENARIO = ScenarioConfig

PATH_TO_DATA = os.path.join(
    "data", EXPERIMENT.SCENARIO_FOLDER, EXPERIMENT.TIME.replace(':', '_'), EXPERIMENT.SCENARIO_FOLDER)
PATH_TO_MODEL = os.path.join("model", EXPERIMENT.NAME)
PATH_TO_RECORDS = os.path.join("records", EXPERIMENT.NAME)
PATH_TO_METRIC = os.path.join("metric", EXPERIMENT.NAME)
PATH_TO_SUMMARY = os.path.join("summary", EXPERIMENT.NAME)
PATH_TO_CONFIG = os.path.join("config", EXPERIMENT.NAME)

_load_scenario_config(scenario_config_filepath=scenario_config_filepath)

if not EXISTING_EXPERIMENT_FLAG:
    _store_configs()

print("Configuration finished")

PRINTING_LOCK = Lock()
