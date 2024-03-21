import os
import json
import warnings

import config
from environment.intersection_traffic_detector_system import IntersectionTrafficDetectorSystem
from environment.simulation_data_subscriber import SimulationDataSubscriber, PERSON, EDGE, LANE, \
    VEHICLE, SIMULATION

import traci
import traci.constants as tc

from utils import xml_util, collections_util
from utils.sumo import sumo_traci_util, sumo_util, sumo_net_util


class Environment:

    _SIMULATION_VARIABLES_TO_SUBSCRIBE = set([
        tc.VAR_DEPARTED_VEHICLES_IDS,
        tc.VAR_DEPARTED_PERSONS_IDS
    ])

    def __init__(self, evaluate_metrics=True, include_analysis_data=False, execution_name=None, *args, **kwargs):

        self.__setup(evaluate_metrics, include_analysis_data, *args, **kwargs)

        self.has_buses = config.SCENARIO.HAS_BUSES
        self.has_passengers = config.SCENARIO.HAS_PASSENGERS

        net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.NET_FILE)
        self._net_xml = xml_util.parse_xml(net_file)

        if self.has_buses:
            bus_stop_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.BUS_STOPS_FILE)
            self._bus_stop_xml = xml_util.parse_xml(bus_stop_file)
            bus_trips_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.BUS_TRIPS_FILE)
            self._bus_trips_xml = xml_util.parse_xml(bus_trips_file)

            self._bus_stops = sumo_net_util.get_bus_stop_ids(self._bus_stop_xml)
            self._bus_schedule = sumo_net_util.get_bus_schedules(self._bus_trips_xml)

        if self.has_passengers:
            self._passenger_trips_file = os.path.join(
                config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.PASSENGER_TRIPS_FILE)
            self._passenger_trips_xml = xml_util.parse_xml(self._passenger_trips_file)

        self.intersection_detector_system = IntersectionTrafficDetectorSystem(
            evaluate_metrics=evaluate_metrics, include_analysis_data=include_analysis_data)

        try:
            with open(os.path.join(
                    config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.MULTI_INTERSECTION_TL_FILE), 'r') as file:
                self._multi_intersection_config = collections_util.HashableDict(json.load(file))
        except Exception as e:
            warnings.warn("No multi intersection tl file present")
            self._multi_intersection_config = collections_util.HashableDict()

        # network attributes
        self._edges = (
                list(sumo_net_util.get_edge_map(self._net_xml).keys()) +
                list(sumo_net_util.get_internal_edge_map(self._net_xml).keys()))

        self._lanes = (
                list(sumo_net_util.get_lane_map(self._net_xml).keys()) +
                list(sumo_net_util.get_internal_lane_map(self._net_xml).keys()))

        self._edge_to_previous_intersection_mapping, self._edge_to_next_intersection_mapping = (
            sumo_net_util.get_edge_adjacent_intersections_mapping(self._net_xml))

        self._simulation_warmup = False

        self.reset(execution_name)

    @property
    def simulation_warmup(self):
        return self._simulation_warmup

    @simulation_warmup.setter
    def simulation_warmup(self, value):
        self._simulation_warmup = value

    def setup(self, evaluate_metrics=True, include_analysis_data=False, path_to_log=None, *args, **kwargs):

        self._data_subscription = SimulationDataSubscriber()
        self.__update_subscription_features(evaluate_metrics)

        self.__setup(evaluate_metrics, include_analysis_data, *args, **kwargs)

        self.intersection_detector_system.setup(
            self._data_subscription, evaluate_metrics=evaluate_metrics, include_analysis_data=include_analysis_data,
            path_to_log=path_to_log)

    def __setup(self, evaluate_metrics=True, include_analysis_data=False, *args, **kwargs):

        self.do_include_analysis_data = include_analysis_data

    def __update_subscription_features(self, evaluate_metrics):

        self._data_subscription.update_subscription_variables(
            SIMULATION, self._SIMULATION_VARIABLES_TO_SUBSCRIBE)

    def reset(self, execution_name=None):
        try:
            self._data_subscription.reset(execution_name)
        except AttributeError:
            pass

        self.__reset(execution_name)

    def __reset(self, execution_name=None):

        self._execution_name = execution_name

        # basic info
        self.step_ = 0

        self.intersection_detector_system.reset(execution_name)

    def start(self, parameters=None, with_gui=False):

        if parameters is None:
            parameters = {}

        sumocfg_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.CONFIGURATION_FILE)
        sumo_cmd_str = sumo_util.get_sumo_cmd(sumocfg_file, sumocfg_parameters=parameters, with_gui=with_gui)

        sumo_version = traci.start(sumo_cmd_str)
        print(f"Starting {sumo_version[1]}")

        self._start_up()

        self.intersection_detector_system.start()

    def _start_up(self):

        # start subscription
        self._data_subscription.subscribe(EDGE, self._edges)
        self._data_subscription.subscribe(LANE, self._lanes)
        self._data_subscription.subscribe(SIMULATION)
        # update measurements
        self._update_data_subscriptions()

    def end(self):
        sumo_traci_util.close_connection(self._execution_name)

    def step(self, *args, **kwargs):

        traci_connection = sumo_traci_util.get_traci_connection(self._execution_name)
        traci_connection.simulationStep()
        self.step_ += 1

        self._data_subscription.clear_subscriptions()
        self._update_data_subscriptions()

        self.intersection_detector_system.step_environment()

        return self.step_, tuple()

    def _update_data_subscriptions(self):

        recently_departed_vehicle_ids = self._data_subscription[SIMULATION][tc.VAR_DEPARTED_VEHICLES_IDS]
        recently_departed_persons_ids = self._data_subscription[SIMULATION][tc.VAR_DEPARTED_PERSONS_IDS]

        # update subscriptions
        self._data_subscription.subscribe(VEHICLE, recently_departed_vehicle_ids)
        self._data_subscription.subscribe(PERSON, recently_departed_persons_ids)

    def save_log(self):
        self.intersection_detector_system.save_log()
