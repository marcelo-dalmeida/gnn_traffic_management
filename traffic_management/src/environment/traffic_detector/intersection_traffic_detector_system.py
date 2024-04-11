import os
from pathlib import Path

import pandas as pd
import traci.constants as tc

import config
from environment.simulation_data_subscriber import VEHICLE, EDGE
from environment.traffic_detector.intersection_traffic_detector import IntersectionTrafficDetector
from utils import xml_util
from utils.sumo import sumo_net_util


class IntersectionTrafficDetectorSystem:

    _EDGE_VARIABLES_TO_SUBSCRIBE = set([
        tc.LAST_STEP_VEHICLE_ID_LIST
    ])

    _VEHICLE_VARIABLES_TO_SUBSCRIBE = set([
        tc.VAR_LANEPOSITION,
        tc.VAR_SPEED
    ])

    def __init__(self, evaluate_metrics=False, include_analysis_data=False):

        net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_SCENARIO, config.SCENARIO.NET_FILE)
        self._net_xml = xml_util.parse_xml(net_file)

        self.__setup(evaluate_metrics, include_analysis_data)

        self._traffic_detectors = self._setup_traffic_detectors()

        self._simulation_warmup = False

    @property
    def simulation_warmup(self):
        return self._simulation_warmup

    @simulation_warmup.setter
    def simulation_warmup(self, value):

        self._simulation_warmup = value

    def setup(self, data_subscription, evaluate_metrics=False, include_analysis_data=False):

        self._data_subscription = data_subscription
        self.update_subscription_features()

        self.__setup(evaluate_metrics, include_analysis_data)

        for intersection_id, detector in self._traffic_detectors.items():
            detector.setup(data_subscription, evaluate_metrics, include_analysis_data)

    def __setup(self, evaluate_metrics=False, include_analysis_data=False):

        self.do_evaluate_metrics = evaluate_metrics
        self.do_include_analysis_data = include_analysis_data

    def update_subscription_features(self):

        self._data_subscription.update_subscription_variables(
            (EDGE, self._EDGE_VARIABLES_TO_SUBSCRIBE),
            (VEHICLE, self._VEHICLE_VARIABLES_TO_SUBSCRIBE)
        )

    def reset(self, execution_name, traffic_pattern):

        self._traffic_pattern = traffic_pattern
        for intersection_id, detector in self._traffic_detectors.items():
            detector.reset(execution_name)

    def start(self):

        for intersection_id, detector in self._traffic_detectors.items():
            detector.start()

    def step_environment(self):
        for intersection_id, detector in self._traffic_detectors.items():
            detector.step()

    def get_ids(self):

        detector_ids = []
        for intersection_id, detector in self._traffic_detectors.items():
            detector_ids.extend(detector.detector_ids)

        return detector_ids

    def _setup_traffic_detectors(self):

        detectors = {}

        intersection_map = sumo_net_util.get_intersection_map(self._net_xml)
        border_intersection_map = sumo_net_util.get_border_intersection_map(self._net_xml)

        intersection_id_traffic_light_id_map = (
            sumo_net_util.get_intersection_id_to_traffic_light_id_map(self._net_xml, config.SCENARIO.MULTI_INTERSECTION_CONFIG))

        for intersection_id, _ in intersection_map.items():
            traffic_light = intersection_id_traffic_light_id_map.get(intersection_id, None)
            intersection_ids = intersection_id_traffic_light_id_map.inverse[traffic_light] if traffic_light else [intersection_id]
            intersection_id = ','.join(intersection_ids)

            is_intersection = sumo_net_util.is_intersection(
                self._net_xml, intersection_id, config.SCENARIO.MULTI_INTERSECTION_CONFIG)
            is_border = intersection_id in border_intersection_map

            if (is_intersection or is_border) and intersection_id not in detectors:

                edges = sumo_net_util.get_intersection_edges(
                    self._net_xml, intersection_id, config.SCENARIO.MULTI_INTERSECTION_CONFIG, _sorted=False)

                if all([edge.get('type') not in config.ENVIRONMENT.DETECTOR_ROAD_TYPE for edge in edges]):
                    continue

                detectors[intersection_id] = IntersectionTrafficDetector(
                    intersection_id, self.do_evaluate_metrics, self.do_include_analysis_data)

        return detectors

    def save_log(self):

        df_list = []
        for intersection_id, detector in self._traffic_detectors.items():
            for time, records in detector._detector_logs.items():
                for record in records:
                    d = [{
                        'time': time,
                        'detector_id': k,
                        **v
                    } for k, v in record.items()]
                    df_list.extend(d)

        log_df = pd.DataFrame(df_list)
        log_df = log_df.pivot_table(index='time', columns='detector_id', values=['volume', 'speed'])

        log_df.index = log_df.index.astype(int)
        log_df.index = pd.to_datetime(log_df.index, unit='s')

        log_df['traffic_pattern'] = self._traffic_pattern

        path_to_log_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA)
        Path(path_to_log_file).mkdir(parents=True, exist_ok=True)
        log_df.to_hdf(os.path.join(path_to_log_file, f"{self._traffic_pattern}_detector_logs.h5"), key='data')

        for _, detector in self._traffic_detectors.items():
            detector._detector_logs = []
