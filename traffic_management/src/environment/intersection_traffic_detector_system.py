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

        detector_ids = []
        for intersection_id, detector in self._traffic_detectors.items():
            detector_ids.extend(detector.detector_ids)

        return detector_ids

    def _setup_traffic_detectors(self):

        detectors = {}

        traffic_light_id_intersection_id_mapping = (
            sumo_net_util.get_traffic_light_id_intersection_id_map(self._net_xml, config.SCENARIO.MULTI_INTERSECTION_CONFIG))

        for _, intersection_ids in traffic_light_id_intersection_id_mapping.items():
            intersection_id = ','.join(intersection_ids)

            detectors[intersection_id] = IntersectionTrafficDetector(
                intersection_id, self.do_evaluate_metrics, self.do_include_analysis_data)

        return detectors

    def save_log(self):

        df_list = []
        for intersection_id, detector in self._traffic_detectors.items():

            l = []
            for time, records in detector._detector_logs.items():
                df = pd.concat({k: pd.DataFrame.from_dict(v, 'index').T for record in records for k, v in record.items()}, axis=1)
                df['time'] = time
                df.set_index('time', inplace=True)
                l.append(df)

            df = pd.concat(l, axis=0)
            df_list.append(df)

            detector._detector_logs = []

        log_df = pd.concat(df_list, axis=1)

        log_df.index = log_df.index.astype(int)
        log_df.index = pd.to_datetime(log_df.index, unit='s')
        log_df = log_df.swaplevel(axis=1)

        path_to_log_file = os.path.join(config.ROOT_DIR, self.path_to_log, f"detector_logs.h5")
        log_df.to_hdf(path_to_log_file, key='data')
