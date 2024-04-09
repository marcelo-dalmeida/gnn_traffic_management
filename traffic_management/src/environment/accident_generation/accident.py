import os
import sys

from libsumo import TraCIException

import config
from const import CONST
from environment.simulation_data_subscriber import VEHICLE, LANE, EDGE
from utils import xml_util
from utils.sumo import sumo_traci_util, sumo_net_util

import traci.constants as tc


class Accident:

    def __init__(self, accident_id, vehicle_id, edge_id, road_type, duration, lanes_blocked,
                 randomizer, data_subscription, execution_name):

        net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_SCENARIO, config.SCENARIO.NET_FILE)
        self._net_xml = xml_util.parse_xml(net_file)

        self.accident_id = accident_id
        self.vehicle_id = vehicle_id
        self.edge_id = edge_id
        self.road_type = road_type
        self.duration = duration
        self.lanes_blocked = lanes_blocked

        self._randomizer = randomizer
        self._data_subscription = data_subscription
        self._traci = sumo_traci_util.get_traci_connection(execution_name)

        self._start()

    def _start(self):
        self._crashed_vehicles = []
        self._current_timer = 0

        vehicle_data = self._data_subscription[VEHICLE][self.vehicle_id]

        vehicle_class = vehicle_data[tc.VAR_VEHICLECLASS]
        x, y = vehicle_data[tc.VAR_POSITION]
        route_id = vehicle_data[tc.VAR_ROUTE_ID]
        vehicle_route_index = vehicle_data[tc.VAR_ROUTE_INDEX]
        vehicle_route = vehicle_data[tc.VAR_EDGES]
        lane_number = len(list(sumo_net_util.get_edge(self._net_xml, self.edge_id)))
        lane_position = vehicle_data[tc.VAR_LANEPOSITION]
        lane_index = vehicle_data[tc.VAR_LANE_INDEX]

        crashed_lane_index = lane_index
        self._traci.vehicle.remove(self.vehicle_id)

        # Buses' routes are crashing the accident generation
        if vehicle_class == CONST.BUS:
            route_id = f'{route_id}_alt'
            self._traci.route.add(route_id, vehicle_route[vehicle_route_index:])

        # Add the crashed vehicles from scratch to avoid bad positioning
        self._crashed_vehicles.append(self.vehicle_id)
        if self.lanes_blocked > 1 and lane_number > 1:

            self._crashed_vehicles.append(f'crashed_{self.vehicle_id}')

            # Pick the lane for the additional crashed vehicle
            crashed_lane_index = lane_index
            if lane_index == lane_number - 1:
                crashed_lane_index -= 1
            elif lane_index == 0:
                crashed_lane_index += 1
            else:
                crashed_lane_index += 1 if self._randomizer.random() < 0.5 else -1

        # Add crashed vehicle behavior
        lane_id = f"{self.edge_id}_{lane_index}"
        for index, vehicle_id in enumerate(self._crashed_vehicles):

            if index > 0:
                lane_id = f"{self.edge_id}_{crashed_lane_index}"

            try:
                self._traci.vehicle.add(vehicle_id, route_id)
            except TraCIException as e:
                # bus paths which are not reached again
                print(str(e))
                try:
                    self._traci.vehicle.add(vehicle_id, "")
                except TraCIException as e:
                    # fallback in case it doesn't allow to insert a vehicle #TODO test it better
                    print(str(e))
                    vehicle_id = self._find_closest_vehicle(lane_id, lane_position)

            try:
                self._traci.vehicle.moveTo(vehicle_id, lane_id, lane_position)
            except TraCIException as e:
                # bus paths which are not reached again
                print(str(e))
                try:
                    self._traci.vehicle.moveToXY(vehicle_id, self.edge_id, lane_index, x, y, keepRoute=0)
                except TraCIException as e:
                    print(str(e))

            self._traci.vehicle.setSpeed(vehicle_id, 0.0)

            # Have the vehicle commit to the lane that it is currently in
            # This is needed to prevent the vehicle from weaving back and forth across the edge
            self._traci.vehicle.changeLaneRelative(vehicle_id, 0, self.duration)

            if index > 0:
                self._data_subscription.subscribe(VEHICLE, vehicle_id)

    def step(self):
        self._current_timer += 1

        if self._current_timer >= self.duration:
            self.clear()
            return False

        return True

    def clear(self):
        for vehicle_id in self._crashed_vehicles:
            self._traci.vehicle.remove(vehicle_id)
            self._data_subscription.unsubscribe(VEHICLE, vehicle_id)

    def _find_closest_vehicle(self, lane_id, position):
        vehicle_ids = self._data_subscription[LANE][lane_id][tc.LAST_STEP_VEHICLE_ID_LIST]

        closest_vehicle_id = None
        closest_distance = sys.maxsize

        for vehicle_id in vehicle_ids:
            lane_position = self._traci.vehicle.getLanePosition(vehicle_id)
            closest_distance = min(abs(position - lane_position), closest_distance)
            if closest_distance == lane_position:
                closest_vehicle_id = vehicle_id

        return closest_vehicle_id
