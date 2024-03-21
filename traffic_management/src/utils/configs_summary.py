import os
import argparse

import pandas as pd

from configs.agent.agent_config import AgentConfig
from configs.environment.environment_config import EnvironmentConfig
from configs.experiment.experiment_config import ExperimentConfig
from configs.scenario_config import ScenarioConfig

parser = argparse.ArgumentParser(
                    prog='config_collector',
                    description='Collect config from the experiments')

parser.add_argument('base_config_folder')
parser.add_argument('machine_name', default="__NO_INFORMATION__")
parser.add_argument('--default', action='store_true')
parser.add_argument('-age', '--agent', default=False)
parser.add_argument('-env', '--environment', default=True)
parser.add_argument('-exp', '--experiment', default=True)
parser.add_argument('-sce', '--scenario', default=False)

args = parser.parse_args()

config_tuples = [
    ('experiment', args.experiment, ExperimentConfig),
    ('environment', args.environment, EnvironmentConfig),
    ('scenario', args.scenario, ScenarioConfig),
    ('agent', args.agent, AgentConfig)
]

if args.default:

    print("Storing default config")

    final_config_df = pd.DataFrame()

    experiment_config_df = pd.DataFrame([{
        'experiment_name': '__DEFAULT__',
        'machine_name': args.machine_name
    }])

    for config_name, store, cls in config_tuples:

        if store:
            if config_name == 'agent':
                model_name = ExperimentConfig.MODEL_NAME

                config_df = pd.DataFrame([cls.get_config(model_name).get_globals()])
            else:
                config_df = pd.DataFrame([cls.get_globals()])

            config_df.columns = [f"{config_name}.{column}" for column in config_df.columns]

            experiment_config_df = pd.concat([experiment_config_df, config_df], axis=1)

    final_config_df = pd.concat([final_config_df, experiment_config_df], ignore_index=True)

    final_config_df = final_config_df.reindex(sorted(final_config_df.columns), axis=1)

    final_config_df.to_csv(f'{args.base_config_folder}/default_config.csv', index=False)

else:

    dirs = next(os.walk(args.base_config_folder))[1]

    final_config_df = pd.DataFrame()
    for dir_ in dirs:

        experiment_config_df = pd.DataFrame([{
            'experiment_name': dir_,
            'machine_name': args.machine_name
        }])

        for config_name, store, _ in config_tuples:

            if store:
                config_path = os.path.join(args.base_config_folder, dir_, f'{config_name}_config.json')

                config_df = pd.read_json(config_path, typ='series').to_frame().T
                config_df.columns = [f"{config_name}.{column}" for column in config_df.columns]

                experiment_config_df = pd.concat([experiment_config_df, config_df], axis=1)

        final_config_df = pd.concat([final_config_df, experiment_config_df], ignore_index=True)

        final_config_df = final_config_df.reindex(sorted(final_config_df.columns), axis=1)

        final_config_df.to_csv(f'{args.base_config_folder}/configs_summary.csv', index=False)
