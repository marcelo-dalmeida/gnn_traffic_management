import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from utils import datetime_util, tqdm_util


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
