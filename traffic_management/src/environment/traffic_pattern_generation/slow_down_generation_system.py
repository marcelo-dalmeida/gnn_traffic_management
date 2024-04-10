import math
import os
import random
from itertools import cycle

import traci.constants as tc

import config
from environment.traffic_pattern_generation.slow_down import SlowDown
from environment.simulation_data_subscriber import SIMULATION
from utils import xml_util


class SlowDownGenerationSystem:

    _SIMULATION_VARIABLES_TO_SUBSCRIBE = set([
        tc.LAST_STEP_VEHICLE_ID_LIST
    ])

    def __init__(self):

        net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_SCENARIO, config.SCENARIO.NET_FILE)
        self._net_xml = xml_util.parse_xml(net_file)

    def setup(self, data_subscription):
        self._data_subscription = data_subscription
        self.update_subscription_features()

        self._randomizer = random.Random()
        self._randomizer.seed(2154)

    def update_subscription_features(self):

        self._data_subscription.update_subscription_variables(
            SIMULATION, self._SIMULATION_VARIABLES_TO_SUBSCRIBE)

    def reset(self, execution_name):

        self._execution_name = execution_name

        self.slow_downs = {}
        self._slow_downs_counter = 0

    def start(self):
        self._slow_down_gen_cooldown = config.ENVIRONMENT.SLOW_DOWN_GEN_WARMUP + 1

    def step_environment(self):
        remove_stops = []
        for slow_down_id, slow_down in self.slow_downs.items():
            keep = slow_down.step()
            if not keep:
                remove_stops.append(slow_down_id)

        for slow_down_id in remove_stops:
            self.slow_downs.pop(slow_down_id)

        if self._slow_down_gen_cooldown > 0:
            self._slow_down_gen_cooldown -= 1
        if self._slow_down_gen_cooldown == 0:
            self.generate_slow_downs()
            self._slow_down_gen_cooldown = config.ENVIRONMENT.SLOW_DOWN_GEN_COOLDOWN

    def get_ids(self):
        return list(self.slow_downs.keys())

    # I don't know if we can just say something like 0.1% of the vehicles will stop for 3-7 seconds every 10s
    def generate_slow_downs(self):
        vehicle_ids, durations = self._sample_slow_downs_attributes()

        for vehicle_id, duration in zip(vehicle_ids, durations):
            self._slow_downs_counter += 1
            slow_down_id = self._slow_downs_counter

            slow_down = SlowDown(
                slow_down_id, vehicle_id, duration,
                randomizer=self._randomizer, data_subscription=self._data_subscription, execution_name=self._execution_name)
            self.slow_downs[slow_down_id] = slow_down

    def _sample_slow_downs_attributes(self):

        all_vehicle_ids = self._data_subscription[SIMULATION][tc.LAST_STEP_VEHICLE_ID_LIST]
        vehicle_ids_to_sample = zip(cycle([1]), all_vehicle_ids)
        vehicle_sample_size = math.floor(len(all_vehicle_ids) * config.ENVIRONMENT.SLOW_DOWN_GEN_VEHICLE_PERCENTAGE)

        vehicle_ids = self._sample(vehicle_ids_to_sample, k=vehicle_sample_size)
        duration = self._sample(config.ENVIRONMENT.SLOW_DOWN_GEN_DURATION_PROB, k=vehicle_sample_size)

        return vehicle_ids, duration

    def _sample(self, attribute_probs, k=1):

        probabilities, items = zip(*attribute_probs)
        sampled_item = self._randomizer.choices(items, weights=probabilities, k=k)

        return sampled_item
