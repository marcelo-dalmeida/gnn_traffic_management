import os
import random

import traci.constants as tc

import config
from environment.accident_generation.accident import Accident
from environment.simulation_data_subscriber import EDGE, SIMULATION, VEHICLE, LANE
from utils import xml_util
from utils.sumo import sumo_net_util


class AccidentGenerationSystem:

    _EDGE_VARIABLES_TO_SUBSCRIBE = set([
        tc.VAR_LANES
    ])

    _LANE_VARIABLES_TO_SUBSCRIBE = set([
        tc.LAST_STEP_VEHICLE_ID_LIST
    ])

    _VEHICLE_VARIABLES_TO_SUBSCRIBE = set([
        tc.VAR_ROAD_ID,
        tc.VAR_POSITION,
        tc.VAR_ROUTE_ID,
        tc.VAR_ROUTE_INDEX,
        tc.VAR_EDGES,
        tc.VAR_LANEPOSITION,
        tc.VAR_LANE_INDEX
    ])

    def __init__(self):

        net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_SCENARIO, config.SCENARIO.NET_FILE)
        self._net_xml = xml_util.parse_xml(net_file)

    def setup(self, data_subscription):
        self._data_subscription = data_subscription
        self.update_subscription_features()

        self._randomizer = random.Random()
        self._randomizer.seed(4471)

    def update_subscription_features(self):

        self._data_subscription.update_subscription_variables(
            EDGE, self._EDGE_VARIABLES_TO_SUBSCRIBE)
        self._data_subscription.update_subscription_variables(
            LANE, self._LANE_VARIABLES_TO_SUBSCRIBE)
        self._data_subscription.update_subscription_variables(
            VEHICLE, self._VEHICLE_VARIABLES_TO_SUBSCRIBE)

    def reset(self, execution_name):

        self._execution_name = execution_name

        self.accidents = {}
        self._accident_counter = 0

    def start(self):
        self._accident_gen_cooldown = config.ENVIRONMENT.ACCIDENT_GEN_WARMUP + 1

    def step_environment(self):
        remove_accidents = []
        for accident_id, accident in self.accidents.items():
            keep = accident.step()
            if not keep:
                remove_accidents.append(accident_id)

        for accident_id in remove_accidents:
            self.accidents.pop(accident_id)

        self._accident_gen_cooldown -= 1
        if self._accident_gen_cooldown <= 0:
            self.generate_accident()
            self._accident_gen_cooldown = config.ENVIRONMENT.ACCIDENT_GEN_COOLDOWN

    def get_ids(self):
        return list(self.accidents.keys())

    def generate_accident(self):
        road_type, duration, lanes_blocked = self._sample_accident_attributes()

        vehicle_id = edge_id = None

        successful = False
        tries = 0
        while not successful and tries < config.ENVIRONMENT.ACCIDENT_GEN_FAILED_TRIES:
            try:
                edge_id = self._pick_random_edge(road_type)
                vehicle_id = self._pick_random_vehicle(edge_id)
                successful = True
            except IndexError:
                tries += 1

        if not successful:
            print(f'Failed to generate accidents after {config.ENVIRONMENT.ACCIDENT_GEN_FAILED_TRIES} tries')
            all_vehicle_ids = self._data_subscription[SIMULATION][tc.LAST_STEP_VEHICLE_ID_LIST]
            if all_vehicle_ids:
                print(f'Picking a random vehicle to crash')
                vehicle_id = self._randomizer.choice(all_vehicle_ids)
                edge_id = self._data_subscription[VEHICLE][vehicle_id][tc.VAR_ROAD_ID]
                road_type = None
            else:
                print(f'No vehicle available')
                return

        self._accident_counter += 1
        accident_id = self._accident_counter

        accident = Accident(
            accident_id, vehicle_id, edge_id, road_type, duration, lanes_blocked,
            randomizer=self._randomizer, data_subscription=self._data_subscription, execution_name=self._execution_name)
        self.accidents[accident_id] = accident

    def _sample_accident_attributes(self):
        road_type = self._sample(config.ENVIRONMENT.ACCIDENT_GEN_ROAD_TYPE_PROB)
        duration = self._sample(config.ENVIRONMENT.ACCIDENT_GEN_DURATION_PROB)
        lanes_blocked = self._sample(config.ENVIRONMENT.ACCIDENT_GEN_LANES_BLOCKED_PROB)

        return road_type, duration, lanes_blocked

    def _pick_random_edge(self, road_types):
        edges = [
            edge_id
            for type_ in road_types
            for edge_id in sumo_net_util.get_edges_by_road_type(self._net_xml)[type_]
        ]
        return self._randomizer.choice(edges)

    def _pick_random_vehicle(self, edge_id):
        vehicle_ids = self._data_subscription[EDGE][edge_id][tc.LAST_STEP_VEHICLE_ID_LIST]
        return self._randomizer.choice(vehicle_ids)

    def _sample(self, attribute_probs):

        probabilities, items = zip(*attribute_probs)
        sampled_item = self._randomizer.choices(items, weights=probabilities)[0]

        return sampled_item
