import os
import collections

import numpy as np

from environment.simulation_data_subscriber import VEHICLE, SIMULATION, EDGE

import traci.constants as tc


import config
from utils import xml_util
from utils.sumo import sumo_traci_util, sumo_util, sumo_net_util


class IntersectionTrafficDetector:

    def __init__(self, intersection_id, multi_intersection,
                 evaluate_metrics=False, include_analysis_data=False, execution_name=None):

        net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.NET_FILE)
        self._net_xml = xml_util.parse_xml(net_file)

        self.intersection_id = intersection_id
        self._multi_intersection_config = multi_intersection

        self.entering_edges = sumo_net_util.get_intersection_edges(
            self._net_xml,
            self.intersection_id,
            self._multi_intersection_config,
            edge_type='incoming'
        )

        self.exiting_edges = sumo_net_util.get_intersection_edges(
            self._net_xml,
            self.intersection_id,
            self._multi_intersection_config,
            edge_type='outgoing'
        )

        self.edges = self.entering_edges + self.exiting_edges
        self.edge_ids = [edge.get('id') for edge in self.edges]

        maximum_detection_length, maximum_detection_time = (
            self._get_maximum_detector_length_and_time(config.ENVIRONMENT.DETECTOR_EXTENSION))

        self.subscription_extension = (
            sumo_util.get_edge_based_subscription_extension(
                self._net_xml, self.entering_edges, self.exiting_edges, maximum_detection_length, maximum_detection_time
            ))

        self.expanded_edge_ids = np.unique([
            edge_tuple[0]
            for edge_tuples in self.subscription_extension.values()
            for edge_tuple in edge_tuples
        ]).tolist()

        self.detector_additional_info = None

        self.reset(execution_name)

    def setup(self, data_subscription, evaluate_metrics=False, include_analysis_data=False):

        self._data_subscription = data_subscription

    def reset(self, execution_name):
        self._execution_name = execution_name

        self._detector_logs = collections.defaultdict(list)

        self._detector_vehicle_subscription = collections.defaultdict(list)

        self._detector_vehicle_tracker = collections.defaultdict(set)
        self._detector_vehicle_count = collections.defaultdict(int)
        self._detector_vehicle_speed = collections.defaultdict(int)

        self._build_detector_additional_info()

    def start(self):
        self._update_data_subscriptions()

    def step(self):
        self._update_data_subscriptions()
        self.log()

    def _update_data_subscriptions(self):

        self._update_vehicle_subscription()
        self._update_detector_vehicle_count()
        self._update_detector_vehicle_speed()

    def _update_vehicle_subscription(self):

        self._detector_vehicle_subscription = collections.defaultdict(list)
        for detector_id, edge_tuples in self.subscription_extension.items():
            for edge_id, _, _, _, _ in edge_tuples:

                edge_additional_info = self.detector_additional_info[detector_id][edge_id]

                start_position = edge_additional_info[sumo_traci_util.VAR_EDGE_START_POSITION]
                end_position = edge_additional_info[sumo_traci_util.VAR_EDGE_END_POSITION]
                is_partial_detector = edge_additional_info[sumo_traci_util.VAR_IS_PARTIAL_DETECTOR]

                edge_vehicle_ids = self._data_subscription[EDGE][edge_id][tc.LAST_STEP_VEHICLE_ID_LIST]

                for vehicle_id in edge_vehicle_ids:
                    edge_position = self._data_subscription[VEHICLE][vehicle_id][tc.VAR_LANEPOSITION]

                    if not is_partial_detector or (start_position < edge_position <= end_position):
                        self._detector_vehicle_subscription[detector_id].append(vehicle_id)

    def _update_detector_vehicle_count(self):

        previous_tracking = self._detector_vehicle_tracker
        current_tracking = collections.defaultdict(set)

        for detector_id, vehicles in self._detector_vehicle_subscription.items():
            current_tracking[detector_id].update(vehicles)
            self._detector_vehicle_count[detector_id] = len(
                current_tracking[detector_id] - previous_tracking[detector_id])

    def _update_detector_vehicle_speed(self):

        for detector_id, vehicles in self._detector_vehicle_subscription.items():
            vehicle_data_subscription = [self._data_subscription[VEHICLE][vehicle_id] for vehicle_id in vehicles]
            self._detector_vehicle_speed[detector_id] = sumo_traci_util.get_average_speed(vehicle_data_subscription)

    def log(self):

        current_time = self._data_subscription[SIMULATION][tc.VAR_TIME]
        if current_time % 60 != 0:
            return

        data = {
            "volume": {
                detector_id: self._detector_vehicle_count[detector_id]
                for detector_id in self.subscription_extension.keys()
            },
            "speed": {
                detector_id: self._detector_vehicle_speed[detector_id]
                for detector_id in self.subscription_extension.keys()
            }
        }

        for detector_id in self.subscription_extension.keys():
            self._detector_logs[current_time].append({
                detector_id: {
                    "volume": data["volume"][detector_id],
                    "speed": data["speed"][detector_id]
                }
            })

    def _build_detector_additional_info(self):
        self.detector_additional_info = collections.defaultdict(dict)

        # get vehicle list
        for detector_id, edge_tuples in self.subscription_extension.items():

            for edge_id, edge_length, accumulated_detection_length, detection_length, edge_type in edge_tuples:

                edge_information = self.detector_additional_info[detector_id][edge_id] = {}

                start_position = 0
                end_position = edge_length

                is_partial_detector = False
                if accumulated_detection_length > detection_length:
                    detector_partial_length = edge_length - (accumulated_detection_length - detection_length)
                    if edge_type == 'entering':
                        start_position = edge_length - detector_partial_length
                    elif edge_type == 'exiting':
                        end_position = detector_partial_length

                    is_partial_detector = True

                edge_information[sumo_traci_util.VAR_EDGE_START_POSITION] = start_position
                edge_information[sumo_traci_util.VAR_EDGE_END_POSITION] = end_position
                edge_information[sumo_traci_util.VAR_IS_PARTIAL_DETECTOR] = is_partial_detector

    @staticmethod
    def _get_maximum_detector_length_and_time(detector_extension):

        maximum_detector_length, maximum_detector_time = None, None

        if isinstance(detector_extension, str) and 's' in detector_extension:
            maximum_detector_time = float(detector_extension[:-1])
        else:
            maximum_detector_length = detector_extension

        return maximum_detector_length, maximum_detector_time

