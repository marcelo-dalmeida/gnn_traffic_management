import collections
from functools import lru_cache

import traci.constants as tc

import config
from utils.sumo import sumo_traci_util

PERSON = "person_variables"
EDGE = "edge_variables"
LANE = "lane_variables"
VEHICLE = "vehicle_variables"
SIMULATION = "simulation_variables"
TRAFFIC_LIGHT = "traffic_light_variables"
BUS_SPECIFIC = "bus_specific_variables"
EMV_SPECIFIC = "emv_specific_variables"
BUS_STOP = "bus_stop_variables"


class SimulationDataSubscriber:

    class _SubscriptionResult(dict):

        def __init__(self, subscriber, type_, *args, **kwargs):
            self._subscriber = subscriber
            self._type = type_
            self._kwargs = kwargs
            super().__init__(*args, **kwargs)

        def __missing__(self, key):
            value, is_subscribed = self._subscriber._compute_missing_variable(self._type, key, self._kwargs.get('_key'))

            try:
                value._kwargs['_key'] = key
            except AttributeError:
                pass

            if not is_subscribed:
                self[key] = value

            return value

    def __init__(self):
        self.setup()

    def __getitem__(self, type_):
        return self._SubscriptionResult(self, type_)

    def reset(self, execution_name):
        self._execution_name = execution_name
        self._traci = sumo_traci_util.get_traci_connection(self._execution_name)

        self.clear_subscriptions()

    def update_subscription_variables(self, *args):
        # args = (type, variables)

        for type_, variables in args:
            self._variables_to_subscribe[type_].update(variables)

    def subscribe(self, type_, ids=None):

        variables = list(self._variables_to_subscribe[type_])
        if not variables:
            return

        if not isinstance(ids, (list, tuple)):
            ids = [ids]

        if len(ids) == 0:
            return

        self._subscription_functions[type_](ids, variables)

    def unsubscribe(self, type_, ids=None):

        if not isinstance(ids, (list, tuple)):
            ids = [ids]

        if len(ids) == 0:
            return

        self._unsubscription_functions[type_](ids)

    def get_subscription_results(self, type_, ids=None):

        single_output = False
        if not isinstance(ids, (list, tuple)):
            ids = [ids]
            single_output = True

        results = {
            id_: self._get_subscription_results(type_, id_)
            for id_ in ids
        }

        if single_output:
            try:
                results = self._SubscriptionResult(self, type_, results[ids[0]])
            except IndexError:
                return {}
        else:
            results = self._SubscriptionResult(self, type_, results)

        return results

    def _compute_missing_variable(self, type_, key, id_):

        try:
            result, is_subscribed = self._compute_variable(type_, key, id_), False
        except KeyError:
            if type_ == SIMULATION:
                if key is None:
                    result, is_subscribed = self[SIMULATION], True
                else:
                    result, is_subscribed = self.get_subscription_results(type_, id_)[key], True
            else:
                id_ = key
                result, is_subscribed = self.get_subscription_results(type_, id_), True

        return result, is_subscribed

    @lru_cache(maxsize=None)
    def _get_subscription_results(self, type_, id_):
        return self._subscription_result_functions[type_](id_)

    @lru_cache(maxsize=None)
    def _compute_variable(self, type_, key, id_):
        return self._data_computation_functions[type_][key](id_)

    def clear_subscriptions(self):
        self._compute_variable.cache_clear()
        self._get_subscription_results.cache_clear()

    def setup(self):

        self._variables_to_subscribe = {
            PERSON: set(),
            EDGE: set(),
            LANE: set(),
            VEHICLE: set(),
            SIMULATION: set(),
            TRAFFIC_LIGHT: set(),
            BUS_SPECIFIC: set(),
            EMV_SPECIFIC: set()
        }

        consume = collections.deque(maxlen=0).extend

        self._subscription_functions = {
            PERSON: lambda ids, variables:
                consume(map(lambda id_: self._traci.person.subscribe(id_, variables), ids)),
            EDGE: lambda ids, variables:
                consume(map(lambda id_: self._traci.edge.subscribe(id_, variables), ids)),
            LANE: lambda ids, variables:
                consume(map(lambda id_: self._traci.lane.subscribe(id_, variables), ids)),
            VEHICLE: lambda ids, variables:
                consume(map(lambda id_: self._traci.vehicle.subscribe(id_, variables), ids)),
            SIMULATION: lambda ids, variables:
                self._traci.simulation.subscribe(variables),
            TRAFFIC_LIGHT: lambda ids, variables:
                consume(map(lambda id_: self._traci.trafficlight.subscribe(id_, variables), ids))
        }
        self._unsubscription_functions = {
            PERSON: lambda ids:
                consume(map(lambda id_: self._traci.person.unsubscribe(id_), ids)),
            EDGE: lambda ids:
                consume(map(lambda id_: self._traci.edge.unsubscribe(id_), ids)),
            LANE: lambda ids:
                consume(map(lambda id_: self._traci.lane.unsubscribe(id_), ids)),
            VEHICLE: lambda ids:
                consume(map(lambda id_: self._traci.vehicle.unsubscribe(id_), ids)),
            SIMULATION: lambda ids:
                self._traci.simulation.unsubscribe(),
            TRAFFIC_LIGHT: lambda ids:
                consume(map(lambda id_: self._traci.trafficlight.unsubscribe(id_), ids))
        }
        self._subscription_functions.update({
            BUS_SPECIFIC: self._subscription_functions[VEHICLE],
            EMV_SPECIFIC: self._subscription_functions[VEHICLE]
        })

        self._subscription_result_functions = {
            PERSON: lambda id_: self._traci.person.getSubscriptionResults(id_),
            EDGE: lambda id_: self._traci.edge.getSubscriptionResults(id_),
            LANE: lambda id_: self._traci.lane.getSubscriptionResults(id_),
            VEHICLE: lambda id_: self._traci.vehicle.getSubscriptionResults(id_),
            SIMULATION: lambda id_: self._traci.simulation.getSubscriptionResults(),
            TRAFFIC_LIGHT: lambda id_: self._traci.trafficlight.getSubscriptionResults(id_),
        }
        self._subscription_result_functions.update({
            BUS_SPECIFIC: self._subscription_result_functions[VEHICLE],
            EMV_SPECIFIC: self._subscription_result_functions[VEHICLE]
        })

        self._data_computation_functions = {
            SIMULATION: {
                tc.VAR_TIME: lambda _: self._traci.simulation.getTime(),
                tc.LAST_STEP_VEHICLE_ID_LIST: lambda _: self._traci.vehicle.getIDList(),
                tc.LAST_STEP_PERSON_ID_LIST: lambda _: self._traci.person.getIDList()
            },
            VEHICLE: {
                tc.VAR_VEHICLECLASS: lambda vehicle_id: self._traci.vehicle.getVehicleClass(vehicle_id),
                tc.VAR_NEXT_STOPS: lambda vehicle_id: self._traci.vehicle.getStops(vehicle_id)
            },
            TRAFFIC_LIGHT: {
                tc.TL_CURRENT_PHASE: lambda traffic_light_id: self._traci.trafficlight.getPhase(traffic_light_id)
            },
            BUS_STOP: {
                tc.LAST_STEP_VEHICLE_ID_LIST: lambda bus_stop_id: self._traci.busstop.getVehicleIDs(bus_stop_id)
            }
        }
        self._data_computation_functions.update({
            **{
                TYPE_: {
                    **self._data_computation_functions.get(TYPE_, {}),
                    **{
                        ACTOR_CLASS:
                            (lambda ACTOR_CLASS, TYPE_:
                             lambda id_: [
                                 vehicle_id
                                 for vehicle_id in self[TYPE_][id_][tc.LAST_STEP_VEHICLE_ID_LIST]
                                 if self[VEHICLE][vehicle_id][tc.VAR_VEHICLECLASS] == ACTOR_CLASS
                             ]
                             )(ACTOR_CLASS, TYPE_)
                        for ACTOR_CLASS in config.CONST.AVAILABLE_VEHICLE_ACTORS
                    }
                }
                for TYPE_ in [EDGE, LANE, SIMULATION]
            }
        })
