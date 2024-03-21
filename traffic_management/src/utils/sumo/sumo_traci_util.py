import math
from random import Random

import config

import traci
import traci.constants as tc
from shapely.geometry import Polygon, CAP_STYLE

from utils import math_util
from utils.sumo import sumo_util

VAR_CUMULATIVE_LENGTH = 'VAR_CUMULATIVE_LENGTH'
VAR_LANE_START_POSITION = 'VAR_LANE_START_POSITION'
VAR_LANE_END_POSITION = 'VAR_LANE_END_POSITION'
VAR_IS_PARTIAL_DETECTOR = 'VAR_IS_PARTIAL_DETECTOR'
VAR_EDGE_START_POSITION = 'VAR_EDGE_START_POSITION'
VAR_EDGE_END_POSITION = 'VAR_EDGE_END_POSITION'


def get_new_randomizer():
    SUMO_SEED = 23423

    randomizer = Random()
    randomizer.seed(SUMO_SEED)

    return randomizer


def get_traci_connection(traci_label=None):

    if traci_label is None or traci.isLibsumo():
        traci_connection = traci
    else:
        traci_connection = traci.getConnection(traci_label)

    return traci_connection


def close_connection(traci_label):
    config.PRINTING_LOCK.acquire()
    print("\n\n")
    if traci_label is None or traci.isLibsumo():
        traci.close()
    else:
        traci.main.getConnection(traci_label).close()
    config.PRINTING_LOCK.release()


def get_current_time(traci_label=None, traci_connection=None):

    if traci_connection is None:
        traci_connection = get_traci_connection(traci_label)

    return traci_connection.simulation.getTime()


def set_traffic_light_state(intersection, state, traci_label=None):
    
    traci_connection = get_traci_connection(traci_label)

    traci_connection.trafficlight.setRedYellowGreenState(intersection, state)


def get_traffic_light_state(traffic_light_id, traci_label=None):

    traci_connection = get_traci_connection(traci_label)

    return traci_connection.trafficlight.getRedYellowGreenState(traffic_light_id)


def get_average_speed(vehicle_subscription_data):

    if vehicle_subscription_data:
        speeds = [data[tc.VAR_SPEED] for data in vehicle_subscription_data]
        mean_speed = sum(speeds) / len(speeds)
    else:
        mean_speed = 0

    return mean_speed


def get_network_time_loss(vehicle_subscription_data, traci_label=None):
    traci_connection = get_traci_connection(traci_label)

    if vehicle_subscription_data:

        relative_speeds = [data[tc.VAR_SPEED] / data[tc.VAR_ALLOWED_SPEED]
                           for data in vehicle_subscription_data.values()]

        running = len(relative_speeds)
        step_length = traci_connection.simulation.getDeltaT()
        mean_relative_speed = sum(relative_speeds) / running

        time_loss = (1 - mean_relative_speed) * running * step_length
    else:
        time_loss = 0

    return time_loss


def get_movements_first_stopped_vehicle_greatest_waiting_time(
        movement_to_entering_lane, lane_vehicle_data, vehicle_subscription_data):

    result = {movement: 0 for movement in movement_to_entering_lane.keys()}

    for movement, entering_lanes in movement_to_entering_lane.items():

        greatest_waiting_time = 0
        for lane_id in entering_lanes:

            vehicle_ids = lane_vehicle_data[lane_id]
            if len(vehicle_ids) > 0:
                vehicle_id = vehicle_ids[-1]
                vehicle_waiting_time = vehicle_subscription_data[vehicle_id][tc.VAR_WAITING_TIME]
            else:
                vehicle_waiting_time = 0

            greatest_waiting_time = max(greatest_waiting_time, vehicle_waiting_time)

        result[movement] = greatest_waiting_time

    return result


def get_time_loss(vehicle_subscription_data, traci_label=None):
    traci_connection = get_traci_connection(traci_label)

    if vehicle_subscription_data:
        step_length = traci_connection.simulation.getDeltaT()
        time_loss = [
            (1 - (data[tc.VAR_SPEED] / data[tc.VAR_ALLOWED_SPEED])) * step_length
            for data in vehicle_subscription_data.values()
        ]
    else:
        time_loss = []

    return time_loss


def get_relative_occupancy(vehicle_subscription_data, detector_cumulative_length, detector_additional_info,
                           traci_label=None):

    traci_connection = get_traci_connection(traci_label)

    total_occupied_length = 0

    # It accounts for lane next entering vehicle secure gap spacing
    if vehicle_subscription_data:
        vehicle_id, vehicle = next(iter(vehicle_subscription_data.items()))
        min_gap = vehicle[tc.VAR_MINGAP]
        last_vehicle_secure_gap_margin = vehicle[tc.VAR_TAU] * vehicle[tc.VAR_SPEED] + min_gap
        vehicle_lane_id = vehicle[tc.VAR_LANE_ID]
        lane_start_position = detector_additional_info[vehicle_lane_id][VAR_LANE_START_POSITION]
        actual_distance = vehicle[tc.VAR_LANEPOSITION] - lane_start_position
        total_occupied_length += min(last_vehicle_secure_gap_margin, actual_distance)

    for vehicle_id, vehicle in vehicle_subscription_data.items():
        min_gap = vehicle[tc.VAR_MINGAP]
        leader_vehicle_result = traci_connection.vehicle.getLeader(vehicle_id)

        if leader_vehicle_result:
            leader_id, leader_vehicle_distance = leader_vehicle_result
        else:
            leader_id, leader_vehicle_distance = None, None

        if leader_id in vehicle_subscription_data:
            leader_vehicle = vehicle_subscription_data[leader_id]

            actual_distance = max(0 + min_gap, leader_vehicle_distance + min_gap)
            secure_gap = traci_connection.vehicle.getSecureGap(
                vehicle_id,
                vehicle[tc.VAR_SPEED],
                leader_vehicle[tc.VAR_SPEED],
                leader_vehicle[tc.VAR_DECEL],
                leader_id)
            secure_gap += min_gap

        else:
            vehicle_lane_id = vehicle[tc.VAR_LANE_ID]
            lane_end_position = detector_additional_info[vehicle_lane_id][VAR_LANE_END_POSITION]
            actual_distance = lane_end_position - vehicle[tc.VAR_LANEPOSITION]
            secure_gap = vehicle[tc.VAR_TAU] * vehicle[tc.VAR_SPEED] + min_gap

        occupied_length = vehicle[tc.VAR_LENGTH] + min(secure_gap, actual_distance)

        total_occupied_length += occupied_length

    relative_occupancy = total_occupied_length / detector_cumulative_length

    return relative_occupancy


def get_bounding_box(vehicle_id, vehicle_subscription_data):

    front_center = vehicle_subscription_data[vehicle_id][tc.VAR_POSITION]
    length = vehicle_subscription_data[vehicle_id][tc.VAR_LENGTH]
    width = vehicle_subscription_data[vehicle_id][tc.VAR_WIDTH]
    sumo_angle = vehicle_subscription_data[vehicle_id][tc.VAR_ANGLE]

    angle = math.radians(sumo_util.convert_sumo_angle_to_canonical_angle(sumo_angle))

    _, back_center = math_util.line(front_center, angle - math.pi, length)
    
    _, front_1 = math_util.line(front_center, angle + math.pi / 2, width / 2)
    _, front_2 = math_util.line(front_center, angle - math.pi / 2, width / 2)

    _, back_1 = math_util.line(back_center, angle + math.pi / 2, width / 2)
    _, back_2 = math_util.line(back_center, angle - math.pi / 2, width / 2)

    vehicle_bounding_box = Polygon([front_1, front_2, back_2, back_1])

    return vehicle_bounding_box


def get_blocking_vehicles(vehicle_id, polyline_path, possible_blocking_vehicles, bounding_boxes,
                          vehicle_subscription_data):

    vehicle_position = vehicle_subscription_data[vehicle_id][tc.VAR_POSITION]
    remaining_path = math_util.retrieve_remaining_path(vehicle_position, polyline_path)

    vehicle_width = vehicle_subscription_data[vehicle_id][tc.VAR_WIDTH]
    path_area = remaining_path.buffer(vehicle_width/2, resolution=0, cap_style=CAP_STYLE.flat)

    blocking_vehicles = []
    for possible_blocking_vehicle in possible_blocking_vehicles:

        bounding_box = bounding_boxes[possible_blocking_vehicle]
        is_blocking = bounding_box.intersection(path_area)

        if is_blocking:
            blocking_vehicles.append(possible_blocking_vehicle)

    return blocking_vehicles


def get_next_switching_table_program(wuat_table, traci_label=None):

    current_time = get_current_time(traci_label)
    program = wuat_table.get('startProg')
    for switch in wuat_table:
        switch_time = int(switch.get('time'))

        if current_time >= switch_time:
            program = switch.get('to')
        else:
            break

    return program


def get_traffic_light_next_switch(traffic_light_id, program):

    original_phases = [
        definition.phases
        for definition in traci.trafficlight.getCompleteRedYellowGreenDefinition(traffic_light_id)
        if definition.programID == program
    ][0]

    next_switch = traci.trafficlight.getNextSwitch(traffic_light_id)

    if len(original_phases) == 1:
        return next_switch

    current_phase_index = traci.trafficlight.getPhase(traffic_light_id)
    current_phase = original_phases[current_phase_index]

    next_phase_index = (current_phase_index + 1) % len(original_phases)
    next_phase = original_phases[next_phase_index]
    while current_phase.state == next_phase.state:
        next_switch += next_phase.duration

        next_phase_index = (next_phase_index + 1) % len(original_phases)
        next_phase = original_phases[next_phase_index]

    return next_switch


import socket
import time
import subprocess
import warnings

import sumolib  # noqa
from traci.connection import Connection
from traci.exceptions import FatalTraCIError, TraCIException

# Overridden
def start(cmd, port=None, numRetries=tc.DEFAULT_NUM_RETRIES, waitBetweenRetries=1, label="default", verbose=False,
          traceFile=None, traceGetters=True, stdout=None, doSwitch=True):
    """
    Start a sumo server using cmd, establish a connection to it and
    store it under the given label. This method is not thread-safe.

    - cmd (list): uses the Popen syntax. i.e. ['sumo', '-c', 'run.sumocfg']. The remote
      port option will be added automatically
    - numRetries (int): retries on failing to connect to sumo (more retries are needed
      if a big .net.xml file must be loaded)
    - label (string) : distinguish multiple traci connections used in the same script
    - verbose (bool): print complete cmd
    - traceFile (string): write all traci commands to FILE for debugging
    - traceGetters (bool): whether to include get-commands in traceFile
    - stdout (iostream): where to pipe sumo process stdout
    """
    if label in traci.main._connections:
        raise TraCIException("Connection '%s' is already active." % label)

    if traceFile is not None:
        traci.main._startTracing(traceFile, cmd, port, label, traceGetters)

    while numRetries >= 0 and label not in traci.main._connections:
        sumoPort = sumolib.miscutils.getFreeSocketPort() if port is None else port
        cmd2 = cmd + ["--remote-port", str(sumoPort)]
        if verbose:
            print("Calling " + ' '.join(cmd2))
        sumoProcess = subprocess.Popen(cmd2, stdout=stdout)
        try:
            return init(sumoPort, numRetries, "localhost", label, sumoProcess, waitBetweenRetries, doSwitch)
        except TraCIException as e:
            if port is not None:
                break
            warnings.warn(("Could not connect to TraCI server using port %s (%s)." +
                           " Retrying with different port.") % (sumoPort, e))
            numRetries -= 1
    raise FatalTraCIError("Could not connect.")


# Overridden
def init(port=8813, numRetries=tc.DEFAULT_NUM_RETRIES, host="localhost", label="default", proc=None,
         waitBetweenRetries=1, doSwitch=True):
    """
    Establish a connection to a TraCI-Server and store it under the given
    label. This method is not thread-safe. It accesses the connection
    pool concurrently.
    """
    traci.main._connections[label] = connect(port, numRetries, host, proc, waitBetweenRetries)
    if doSwitch:
        traci.switch(label)
    return traci.main._connections[label].getVersion()


# Overridden
def connect(port=8813, numRetries=100, host="localhost", proc=None, waitBetweenRetries=1):
    """
    Establish a connection to a TraCI-Server and return the
    connection object. The connection is not saved in the pool and not
    accessible via traci.switch. It should be safe to use different
    connections established by this method in different threads.
    """
    for retry in range(1, numRetries + 2):
        try:
            conn = Connection(host, port, proc)
            if traci.main._connectHook is not None:
                traci.main._connectHook(conn)
            return conn
        except socket.error as e:
            if proc is not None and proc.poll() is not None:
                raise TraCIException("TraCI server already finished")
            if retry > 100:
                print("Could not connect to TraCI server at %s:%s" % (host, port), e)
            if retry < numRetries + 1:
                print(" Retrying in %s seconds" % waitBetweenRetries)
                time.sleep(waitBetweenRetries)
    raise FatalTraCIError("Could not connect in %s tries" % (numRetries + 1))


def merge_detector_additional_info(detector_ids, detector_additional_info):

    additional_info = {}
    for detector_id in detector_ids:
        for key, info in detector_additional_info[detector_id].items():
            if key in additional_info:
                additional_info[key][VAR_LANE_START_POSITION] = (
                    min(
                        additional_info[key][VAR_LANE_START_POSITION],
                        info[VAR_LANE_START_POSITION]
                    ))
                additional_info[key][VAR_LANE_END_POSITION] = (
                    max(
                        additional_info[key][VAR_LANE_END_POSITION],
                        info[VAR_LANE_END_POSITION]
                    ))

                if additional_info[key][VAR_IS_PARTIAL_DETECTOR] and not info[VAR_IS_PARTIAL_DETECTOR]:
                    additional_info[key][VAR_IS_PARTIAL_DETECTOR] = info[VAR_IS_PARTIAL_DETECTOR]
            else:
                additional_info[key] = info

    return additional_info

