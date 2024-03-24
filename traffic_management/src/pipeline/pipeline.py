import os
import time

import config
from environment.environment import Environment
from pipeline.pipeline_utils import dataset_generator
from utils import simulation_util

import agent.gman.train as gman_train
import agent.gman.test as gman_test


class Pipeline:

    def __init__(self):

        self.simulation_warmup = simulation_util.prepare_warmup()
        self.env = Environment()

    def run(self):

        # train
        start_t = time.time()
        gman_train.train()

        end_t = time.time()
        with open(os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, "timing.txt"), "a+") as file:
            file.write(f"{end_t - start_t}\n")
        print(f"training: {end_t - start_t}\n")

    def run_test(self):

        # test
        start_t = time.time()
        gman_test.test()

        end_t = time.time()
        with open(os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, "timing.txt"), "a+") as file:
            file.write(f"{end_t - start_t}\n")
        print(f"testing: {end_t - start_t}\n")

    def generate_dataset(self):

        dataset_generator.run_simulation(self.env, simulation_warmup=False)
        dataset_generator.generate_dataset()
        dataset_generator.generate_adjacency_graph(self.env)
        dataset_generator.generate_static_embedding()


