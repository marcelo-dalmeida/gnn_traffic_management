import os
import json
import pickle

import warnings
from pathlib import Path

import pandas as pd
import traci.constants as tc

import config
from environment.simulation_data_subscriber import VEHICLE, EDGE
from environment.intersection_traffic_detector import IntersectionTrafficDetector
from utils import xml_util, collections_util
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

        net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.NET_FILE)
        self._net_xml = xml_util.parse_xml(net_file)

        self.__setup(evaluate_metrics, include_analysis_data)

        try:
            with open(os.path.join(
                    config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.MULTI_INTERSECTION_TL_FILE), 'r') as file:
                self._multi_intersection_config = collections_util.HashableDict(json.load(file))
        except Exception as e:
            warnings.warn("No multi intersection tl file present")
            self._multi_intersection_config = collections_util.HashableDict()

        self._traffic_detectors = self._setup_traffic_detectors()

        self._simulation_warmup = False

    @property
    def simulation_warmup(self):
        return self._simulation_warmup

    @simulation_warmup.setter
    def simulation_warmup(self, value):

        self._simulation_warmup = value

    def setup(self, data_subscription, evaluate_metrics=False, include_analysis_data=False, path_to_log=None):

        self._data_subscription = data_subscription
        self.update_subscription_features()

        self.path_to_log = path_to_log
        if self.path_to_log:
            Path(os.path.join(config.ROOT_DIR, self.path_to_log)).mkdir(parents=True, exist_ok=True)

        self.__setup(evaluate_metrics, include_analysis_data)

        for intersection_id, detector in self._traffic_detectors.items():
            detector.setup(data_subscription, evaluate_metrics, include_analysis_data)

    def __setup(self, evaluate_metrics=False, include_analysis_data=False, path_to_log=None):

        self.do_evaluate_metrics = evaluate_metrics
        self.do_include_analysis_data = include_analysis_data

    def update_subscription_features(self):

        self._data_subscription.update_subscription_variables(
            EDGE, self._EDGE_VARIABLES_TO_SUBSCRIBE)
        self._data_subscription.update_subscription_variables(
            VEHICLE, self._VEHICLE_VARIABLES_TO_SUBSCRIBE)

    def reset(self, execution_name):

        for intersection_id, detector in self._traffic_detectors.items():
            detector.reset(execution_name)

    def start(self):

        for intersection_id, detector in self._traffic_detectors.items():
            detector.start()

    def step_environment(self):
        for intersection_id, detector in self._traffic_detectors.items():
            detector.step()

    def get_ids(self):
        return list(self._traffic_detectors.keys())

    def _setup_traffic_detectors(self):

        detectors = {}

        traffic_light_id_intersection_id_mapping = (
            sumo_net_util.get_traffic_light_id_intersection_id_map(self._net_xml, self._multi_intersection_config))

        for _, intersection_ids in traffic_light_id_intersection_id_mapping.items():
            intersection_id = ','.join(intersection_ids)

            detectors[intersection_id] = IntersectionTrafficDetector(
                intersection_id, self._multi_intersection_config,
                self.do_evaluate_metrics, self.do_include_analysis_data)

        return detectors

    def save_log(self):

        df_list = []
        for intersection_id, detector in self._traffic_detectors.items():
            df = pd.concat({time: pd.DataFrame(record) for time, records in detector._detector_logs.items() for record in records}, axis=1).T
            df = df.unstack()

            df_list.append(df)

            detector._detector_logs = []

        log_df = pd.concat(df_list, axis=1)

        log_df.index = log_df.index.astype(int)
        log_df.index = pd.to_datetime(log_df.index, unit='s')

        path_to_log_file = os.path.join(config.ROOT_DIR, self.path_to_log, f"detector_logs.h5")
        log_df.to_hdf(path_to_log_file, key='data')
