import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from utils import datetime_util, tqdm_util, xml_util
from utils.sumo import sumo_net_util


def run_simulation(env, simulation_warmup=False):

    path_to_log = os.path.join(config.PATH_TO_RECORDS, "data")

    if Path(os.path.join(config.ROOT_DIR, path_to_log, f"detector_logs.h5")).is_file():
        return

    env.setup(mode='test', path_to_log=path_to_log)

    execution_name = 'dataset_generation'
    env.reset(execution_name)

    time_in_seconds = datetime_util.convert_human_time_to_seconds(config.EXPERIMENT.TIME)
    if simulation_warmup:
        time_in_seconds = max(0, time_in_seconds - (1 * 60 * 60))

    parameters = {
        '--begin': time_in_seconds
    }

    if config.EXPERIMENT.DEBUG:
        parameters['--device.rerouting.deterministic'] = True

    env.start(parameters=parameters, with_gui=config.ENVIRONMENT.USE_GUI)

    step = 0
    if simulation_warmup:
        env.simulation_warmup = True
        with tqdm_util.std_out_err_redirect_tqdm() as orig_stdout:
            with tqdm(
                    desc=f"Dataset Generation (warmup), {time_in_seconds}",
                    total=config.EXPERIMENT.DATA_GENERATION_RUN_COUNTS,
                    position=0,
                    file=orig_stdout,
                    dynamic_ncols=True
            ) as pbar:
                while step < config.EXPERIMENT.DATA_GENERATION_RUN_COUNTS:
                    step, _ = env.step(pbar=pbar)
                    pbar.update()
            env.simulation_warmup = False
            time_in_seconds += config.EXPERIMENT.DATA_GENERATION_RUN_COUNTS

    warmup_step = step
    step = 0
    with tqdm_util.std_out_err_redirect_tqdm() as orig_stdout:
        with tqdm(
                desc=f"Dataset Generation , {time_in_seconds}",
                total=config.EXPERIMENT.DATA_GENERATION_RUN_COUNTS,
                position=0,
                file=orig_stdout,
                dynamic_ncols=True
        ) as pbar:
            while step < warmup_step + config.EXPERIMENT.DATA_GENERATION_RUN_COUNTS:
                step, _ = env.step(pbar=pbar)
                pbar.update()

    env.save_log()
    env.end()


def generate_dataset():

    filename = "dataset.h5"

    path_to_log = os.path.join(config.PATH_TO_RECORDS, "data")

    if Path(os.path.join(config.ROOT_DIR, path_to_log, filename)).is_file():
        return

    path_to_log_file = os.path.join(config.ROOT_DIR, path_to_log, f"detector_logs.h5")
    detector_logs = pd.read_hdf(path_to_log_file, key='data')

    detector_logs = detector_logs.resample('5T', origin='end').mean()

    path_to_dataset_file = os.path.join(config.ROOT_DIR, path_to_log, filename)
    detector_logs.to_hdf(path_to_dataset_file, key='data')


def generate_adjacency_graph(env):

    file = os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'Adj.txt')
    if Path(file).is_file():
        return

    net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.NET_FILE)
    net_xml = xml_util.parse_xml(net_file)

    detector_ids = env.detector_system.get_ids()

    adjacency_graph = sumo_net_util.generate_adjacency_graph(
        net_xml, detector_ids, config.SCENARIO.MULTI_INTERSECTION_CONFIG)

    adjacency_graph = {k: [{v: 1.0} for v in vs] for k, vs in adjacency_graph.items()}

    space_separated_values = [
        f"{k} {sk} {v}"
        for k, sub_list in adjacency_graph.items()
        for sub_dict in sub_list
        for sk, v in sub_dict.items()
    ]

    with open(file, 'w') as handle:
        handle.write('\n'.join(str(line) for line in space_separated_values))
