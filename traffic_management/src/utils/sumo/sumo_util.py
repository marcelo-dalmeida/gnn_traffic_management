import itertools
import os
import copy
import collections
import re
import warnings
import json
from functools import cmp_to_key

import pandas as pd
from lxml import etree
from sumolib import checkBinary

import numpy as np

import config
from utils.collections_util import bidict
from utils.comparator import *
from utils import math_util, xml_util, datetime_util, collections_util
from utils.sumo import sumo_net_util


def get_sumo_binary(gui=False):

    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    return sumo_binary


def detect_movements(net_xml, intersection_id, is_right_on_red=False, multi_intersection_tl_config=None):

    def get_final_direction(all_directions):

        direction = max(all_directions, key=cmp_to_key(direction_precedence_comparator))

        return direction

    if multi_intersection_tl_config is None:
        multi_intersection_tl_config = collections_util.HashableDict()

    movement_set = set()
    movements = []
    connection_id_to_movement = collections.defaultdict(list)
    movement_to_connection_id = collections.defaultdict(list)

    connection_chains_by_edge_id = (
        sumo_net_util.filter_intersection_incoming_edges(net_xml, intersection_id, multi_intersection_tl_config))

    incoming_edge_ids = list(connection_chains_by_edge_id.keys())

    for edge_index, edge_id in enumerate(incoming_edge_ids):

        connection_chains = connection_chains_by_edge_id[edge_id]

        # new sort order: from left to right
        sorted_connection_chains = list(reversed(connection_chains))

        to_edge_direction_map = bidict()
        to_edge_direction_label_map = bidict()
        for connection_chain in sorted_connection_chains:

            if all(connection.get('linkIndex') is None for connection in connection_chain):
                continue

            to_edge = ','.join(connection.get('to') for connection in connection_chain)

            all_directions = [connection.get('dir').lower() for connection in connection_chain]
            direction = get_final_direction(all_directions)

            if to_edge in to_edge_direction_map:
                assert direction == to_edge_direction_map[to_edge]
            else:
                to_edge_direction_map[to_edge] = direction

        for direction, to_edges in to_edge_direction_map.inverse.items():

            if direction == 'l':
                to_edges = list(reversed(to_edges))

            for to_edge in to_edges:

                if len(to_edges) == 1 or to_edges.index(to_edge) == 0:
                    direction_label = direction.upper()
                else:
                    direction_label = direction.upper() + str(to_edges.index(to_edge))

                to_edge_direction_label_map[to_edge] = direction_label

        for connection_chain in sorted_connection_chains:

            if all(connection.get('linkIndex') is None for connection in connection_chain):
                continue

            to_edge = ','.join(inner_connection.get('to') for inner_connection in connection_chain)

            direction_label = to_edge_direction_label_map[to_edge]

            movement = str(edge_index) + direction_label

            if movement not in movements:
                if not (is_right_on_red and direction_label[0] == 'R'):
                    movements.append(movement)
                    movement_set.add(movement)

            for connection in connection_chain:
                connection_id = sumo_net_util.get_connection_id(connection)
                connection_id_to_movement[connection_id].append(movement)
                movement_to_connection_id[movement].append(connection_id)

    return movements, connection_id_to_movement, movement_to_connection_id


def detect_movement_to_traffic_light_link_index(net_xml, traffic_light_xml, traffic_light_id,
                                                movement_to_connection_id):

    movement_to_traffic_light_link_index = {}

    for movement, connection_ids in movement_to_connection_id.items():

        link_indices = []
        for connection_id in connection_ids:
            connection = sumo_net_util.get_connection(net_xml, connection_id)
            link_index = connection.get('linkIndex')
            if link_index is not None:
                link_indices.append(int(link_index))

        movement_to_traffic_light_link_index[movement] = link_indices

    movement_to_traffic_light_link_index = (
        filter_unused_traffic_light_links(traffic_light_xml, traffic_light_id, movement_to_traffic_light_link_index))

    return movement_to_traffic_light_link_index


def detect_same_lane_origin_movements(net_xml, movement_to_connection_id):

    same_lane_origin_movements = {}

    for index_1, (movement_1, connection_ids_1) in enumerate(movement_to_connection_id.items()):
        for index_2, (movement_2, connection_ids_2) in enumerate(list(movement_to_connection_id.items())[index_1 + 1:]):
            index_2 += index_1 + 1

            if movement_1 == movement_2:
                continue

            movements = [movement_1, movement_2]
            for movement in movements:
                if movement not in same_lane_origin_movements:
                    same_lane_origin_movements[movement] = []

            for connection_id_1 in connection_ids_1:
                connection_1 = sumo_net_util.get_connection(net_xml, connection_id_1)
                for connection_id_2 in connection_ids_2:
                    connection_2 = sumo_net_util.get_connection(net_xml, connection_id_2)

                    connection_1_from_lane = connection_1.get('from') + '_' + connection_1.get('fromLane')
                    connection_2_from_lane = connection_2.get('from') + '_' + connection_2.get('fromLane')

                    if connection_1_from_lane == connection_2_from_lane:
                        if movement_2 not in same_lane_origin_movements[movement_1]:
                            same_lane_origin_movements[movement_1].append(movement_2)
                        if movement_1 not in same_lane_origin_movements[movement_2]:
                            same_lane_origin_movements[movement_2].append(movement_1)

    original_same_lane_origin_movements = copy.deepcopy(same_lane_origin_movements)
    for key, values in original_same_lane_origin_movements.items():
        movement_same_lane_origin_movements = set(values)

        unchecked_same_origin_movements = copy.deepcopy(values)
        while unchecked_same_origin_movements:
            movement = unchecked_same_origin_movements.pop(0)
            inherited_same_lane_origin_movements = set(original_same_lane_origin_movements[movement])

            difference = inherited_same_lane_origin_movements.difference(movement_same_lane_origin_movements)
            difference.discard(key)

            movement_same_lane_origin_movements.update(difference)
            unchecked_same_origin_movements.extend(difference)

        same_lane_origin_movements[key] = list(movement_same_lane_origin_movements)

    return same_lane_origin_movements


def detect_junction_link_index_to_movement(net_xml, intersection_id, connection_id_to_movement,
                                           multi_intersection_tl_config=None):

    if multi_intersection_tl_config is None:
        multi_intersection_tl_config = collections_util.HashableDict()

    connection_id_to_junction_link_index = {}
    junction_link_index_to_movement = {}

    if intersection_id in multi_intersection_tl_config:
        intersection_id = (multi_intersection_tl_config[intersection_id]['intersections'] +
                           multi_intersection_tl_config[intersection_id]['non_coordinated'])
    else:
        intersection_id = [intersection_id]

    for inner_intersection_id in intersection_id:

        junction_link_index_to_movement[inner_intersection_id] = {}

        junction = sumo_net_util.get_intersection(net_xml, inner_intersection_id)
        lane_ids = junction.get('incLanes').split(' ')

        link_index = 0
        for lane_id in lane_ids:

            connections = sumo_net_util.get_from_lane_all_connections_map(net_xml)[lane_id]

            for connection in connections:
                if connection is not None:
                    connection_id = sumo_net_util.get_connection_id(connection)
                    if connection_id in connection_id_to_movement:
                        connection_id_to_junction_link_index[connection_id] = link_index
                        movements = connection_id_to_movement[connection_id]
                        junction_link_index_to_movement[inner_intersection_id][link_index] = movements
                    link_index += 1

    return connection_id_to_junction_link_index, junction_link_index_to_movement


def detect_movements_link_states(net_xml, intersection_id, movement_to_connection_id,
                                 multi_intersection_tl_config=None):

    if multi_intersection_tl_config is None:
        multi_intersection_tl_config = collections_util.HashableDict()

    link_states = {}

    if intersection_id in multi_intersection_tl_config:
        intersection_id = (multi_intersection_tl_config[intersection_id]['intersections'] +
                           multi_intersection_tl_config[intersection_id]['non_coordinated'])
    else:
        intersection_id = [intersection_id]

    internal_lane_connections = {
        key: value
        for inner_intersection_id in intersection_id
        for key, value in sumo_net_util.get_internal_lane_connections(net_xml, inner_intersection_id).items()
    }

    for movement, connection_ids in movement_to_connection_id.items():

        connection_link_states = []
        for connection_id in connection_ids:

            connection = sumo_net_util.get_connection(net_xml, connection_id)

            connection_internal_lane = connection.get('via')
            connection_link_state = internal_lane_connections[connection_internal_lane].get('state')
            connection_link_states.append(connection_link_state)

        assert all(connection_link_state == 'M' or connection_link_state == 'm'
                   for connection_link_state in connection_link_states)

        if 'M' in connection_link_states:
            link_states[movement] = 'M'
        else:
            link_states[movement] = 'm'

    return link_states


def detect_movement_conflicts(net_xml, intersection_id, connection_id_to_movement, same_lane_origin_movements,
                              connection_id_to_junction_link_index, junction_link_index_to_movement, link_states,
                              movement_to_give_preference_to, unregulated_intersection_ids,
                              multi_intersection_tl_config=None):

    if multi_intersection_tl_config is None:
        multi_intersection_tl_config = collections_util.HashableDict()

    conflicts = {}
    minor_conflicts = {}

    if intersection_id in multi_intersection_tl_config:
        intersection_ids = (multi_intersection_tl_config[intersection_id]['intersections'] +
                            multi_intersection_tl_config[intersection_id]['non_coordinated'])
    else:
        intersection_ids = [intersection_id]

    connection_requests = {inner_intersection_id:
                               sumo_net_util.get_intersection_connection_requests(net_xml, inner_intersection_id)
                           for inner_intersection_id in intersection_ids}

    for movement, _ in movement_to_give_preference_to.items():
        conflicts[movement] = set([])
        minor_conflicts[movement] = set([])

    for connection_id, movements in connection_id_to_movement.items():

        connection = sumo_net_util.get_connection(net_xml, connection_id)
        connection_from_lane = connection.get('from') + '_' + connection.get('fromLane')
        junctions_list = net_xml.xpath(".//junction"
                                       "[@type!='internal']"
                                       "[contains(concat(' ', @incLanes, ' '), ' " + connection_from_lane + " ')]")

        assert len(junctions_list) == 1
        intersection_id = junctions_list[0].get('id')

        if intersection_id in unregulated_intersection_ids:
            continue

        connection_junction_link_index = connection_id_to_junction_link_index[connection_id]
        connection_request = connection_requests[intersection_id][connection_junction_link_index]
        conflict_indicators = connection_request.get('foes')[::-1]

        for junction_link_index, conflict_indicator in enumerate(conflict_indicators):

            if conflict_indicator == '1':

                other_movements = junction_link_index_to_movement[intersection_id][junction_link_index]

                for movement in movements:
                    for other_movement in other_movements:

                        if other_movement not in conflicts[movement]:
                            conflicts[movement].add(other_movement)

                            movement_link_state = link_states[movement]
                            other_movement_link_state = link_states[other_movement]

                            if movement_link_state != other_movement_link_state:

                                if (other_movement in movement_to_give_preference_to[movement]
                                        and movement in movement_to_give_preference_to[other_movement]):
                                    pass
                                elif (movement_link_state == 'm'
                                      and other_movement in movement_to_give_preference_to[movement]):
                                    # minor conflict
                                    minor_conflicts[movement].add(other_movement)
                                elif (movement_link_state == 'M'
                                      and movement in movement_to_give_preference_to[other_movement]):
                                    # minor conflict
                                    minor_conflicts[movement].add(other_movement)

    original_minor_conflicts = copy.deepcopy(minor_conflicts)
    for key, values in same_lane_origin_movements.items():

        selected_movements = [key] + values

        movements_minor_conflicts = set([])
        for item in selected_movements:
            movements_minor_conflicts.update(original_minor_conflicts[item])

        movements_conflicts = set([])
        for item in selected_movements:
            movements_conflicts.update(conflicts[item])

        movements_major_conflicts = movements_conflicts.difference(movements_minor_conflicts)

        for movements_minor_conflict in movements_minor_conflicts:
            movements_minor_conflict_same_origin = (
                    [movements_minor_conflict] + same_lane_origin_movements[movements_minor_conflict])
            if movements_major_conflicts.intersection(movements_minor_conflict_same_origin):
                for item in selected_movements:
                    minor_conflicts[item].difference_update(movements_minor_conflict_same_origin)

                    for same_origin in movements_minor_conflict_same_origin:
                        minor_conflicts[same_origin].discard(item)

            else:
                for item in selected_movements:
                    minor_conflicts[item].update(movements_minor_conflict_same_origin)

                    for same_origin in movements_minor_conflict_same_origin:
                        minor_conflicts[same_origin].add(item)

    original_conflicts = copy.deepcopy(conflicts)
    for key, values in same_lane_origin_movements.items():

        new_conflicts = set([])
        for value in values:

            inherited_conflicts = original_conflicts[value]
            new_conflicts.update(inherited_conflicts)

            for inherited_conflict in inherited_conflicts:
                new_conflicts.update(same_lane_origin_movements[inherited_conflict])

        for new_conflict in new_conflicts:
            conflicts[new_conflict].add(key)

        conflicts[key].update(new_conflicts)

    for key, values in conflicts.items():
        conflicts[key] = sorted(values, key=cmp_to_key(movement_comparator))

    for key, values in minor_conflicts.items():
        minor_conflicts[key] = sorted(values, key=cmp_to_key(movement_comparator))

    return conflicts, minor_conflicts


def detect_movements_preferences(net_xml, intersection_id, connection_id_to_movement,
                                 connection_id_to_junction_link_index, junction_link_index_to_movement,
                                 unregulated_intersection_ids, multi_intersection_tl_config=None):

    if multi_intersection_tl_config is None:
        multi_intersection_tl_config = collections_util.HashableDict()

    movement_to_give_preference_to = {}

    if intersection_id in multi_intersection_tl_config:
        intersection_ids = (multi_intersection_tl_config[intersection_id]['intersections'] +
                            multi_intersection_tl_config[intersection_id]['non_coordinated'])
    else:
        intersection_ids = [intersection_id]

    connection_requests = {inner_intersection_id:
                               sumo_net_util.get_intersection_connection_requests(net_xml, inner_intersection_id)
                           for inner_intersection_id in intersection_ids}

    for connection_id, movements in connection_id_to_movement.items():

        connection = sumo_net_util.get_connection(net_xml, connection_id)
        connection_from_lane = connection.get('from') + '_' + connection.get('fromLane')
        junctions_list = net_xml.xpath(".//junction"
                                       "[@type!='internal']"
                                       "[contains(concat(' ', @incLanes, ' '), ' " + connection_from_lane + " ')]")

        assert len(junctions_list) == 1
        intersection_id = junctions_list[0].get('id')

        if intersection_id in unregulated_intersection_ids:
            continue

        connection_junction_link_index = connection_id_to_junction_link_index[connection_id]
        connection_request = connection_requests[intersection_id][connection_junction_link_index]
        give_preference_to_indicators = connection_request.get('response')[::-1]

        for movement in movements:
            movement_to_give_preference_to[movement] = []

        for junction_link_index, give_preference_to_indicator in enumerate(give_preference_to_indicators):

            if give_preference_to_indicator == '1':
                other_movements = junction_link_index_to_movement[intersection_id][junction_link_index]

                for movement in movements:
                    for other_movement in other_movements:

                        if other_movement not in movement_to_give_preference_to[movement]:
                            movement_to_give_preference_to[movement].append(other_movement)

    return movement_to_give_preference_to


def detect_existing_phases(traffic_light_xml, traffic_light_id, movement_to_traffic_light_link_index):

    program_logics = sumo_net_util.get_traffic_light_program_logics(traffic_light_xml, traffic_light_id)

    traffic_light_link_indices = set([])
    for indices in movement_to_traffic_light_link_index.values():
        traffic_light_link_indices.update(indices)

    sorted_traffic_light_indices = sorted(traffic_light_link_indices)

    # ordered set
    phases = {}
    phase_original_indices = collections.defaultdict(dict)
    phase_traffic_lights = collections.defaultdict(dict)

    for program_logic in program_logics:

        program = program_logic.get('programID')
        phases[program] = {}

        for original_phase_index, original_phase in enumerate(program_logic):
            original_phase_state = original_phase.get('state')

            phase_state = np.array(list(original_phase_state))[sorted_traffic_light_indices]

            if 'y' not in phase_state:
                green_indices = [i for i, l in zip(sorted_traffic_light_indices, phase_state) if l.lower() == 'g']

                phase_movements = []

                for movement, indices in movement_to_traffic_light_link_index.items():

                    if all(i in green_indices for i in indices):
                        phase_movements.append(movement)

                if len(phase_movements) > 0:
                    phase_movements = sorted(phase_movements, key=cmp_to_key(movement_comparator))

                    phase = '_'.join(phase_movements)
                    phases_length = len(phases)
                    phases[program][phase] = None
                    new_phases_length = len(phases)

                    # Assumes that the phase is only split when at the beginning/end
                    if new_phases_length == phases_length:
                        phases[program].pop(phase)
                        phases[program][phase] = None

                    phase_original_indices[program][phase] = original_phase_index
                    phase_traffic_lights[program][phase] = original_phase.get('state')

        phases[program] = list(phases[program].keys())

    return phases, phase_original_indices, phase_traffic_lights


def detect_yellow_and_all_red_times(traffic_light_xml, traffic_light_id, phases, phase_original_indices,
                                    movement_to_traffic_light_link_index):

    phase_to_yellow_time = collections.defaultdict(lambda: collections.defaultdict(list))
    phase_to_all_red_time = collections.defaultdict(lambda: collections.defaultdict(list))

    program_logics = sumo_net_util.get_traffic_light_program_logics(traffic_light_xml, traffic_light_id)

    traffic_light_link_indices = set([])
    for _, indices in movement_to_traffic_light_link_index.items():
        traffic_light_link_indices.update(indices)
    sorted_traffic_light_indices = sorted(traffic_light_link_indices)

    for program_logic in program_logics:

        program = program_logic.get('programID')

        for index, phase in enumerate(phases[program]):
            next_index = index + 1 if index + 1 < len(phases[program]) else 0
            next_phase = phases[program][next_index]

            if len(phases[program]) > 1:

                phase_original_index = phase_original_indices[program][phase]
                next_phase_original_index = phase_original_indices[program][next_phase]

                if phase_original_index < next_phase_original_index:
                    traffic_light_change_window = (
                        list(zip(
                            range(phase_original_index, next_phase_original_index),
                            program_logic[phase_original_index:next_phase_original_index]
                        )))
                elif phase_original_index > next_phase_original_index:
                    traffic_light_change_window = (
                            list(zip(
                            range(phase_original_index, len(program_logic)),
                            program_logic[phase_original_index:]
                        )) +
                            list(zip(
                            range(0, next_phase_original_index),
                            program_logic[:next_phase_original_index]
                        )))
                else:
                    raise ValueError("Indices shouldn't be equal")

                # discard the green part
                green_state = traffic_light_change_window[0][1].get('state')
                traffic_light_change_window = [
                    (index, light) for index, light in traffic_light_change_window if light.get('state') != green_state
                ]

            else:
                traffic_light_change_window = []

            for original_phase_index, original_phase in traffic_light_change_window:

                original_phase_state = original_phase.get('state')

                phase_state = np.array(list(original_phase_state))[sorted_traffic_light_indices]

                if 'y' in phase_state:
                    yellow_time = float(original_phase.get('duration'))
                    phase_to_yellow_time[program][phase].append(yellow_time)

                else:
                    all_red_time = float(original_phase.get('duration'))
                    phase_to_all_red_time[program][phase].append(all_red_time)

            phase_to_yellow_time[program][phase] = sum(phase_to_yellow_time[program][phase])
            phase_to_all_red_time[program][phase] = sum(phase_to_all_red_time[program][phase])

    return phase_to_yellow_time, phase_to_all_red_time


def detect_original_phase_times(traffic_light_xml, traffic_light_id, phase_traffic_lights):

    phase_to_min_action_time = {}
    phase_to_original_action_time = {}
    phase_to_max_action_time = {}

    program_logics = sumo_net_util.get_traffic_light_program_logics(traffic_light_xml, traffic_light_id)

    for program_logic in program_logics:

        program = program_logic.get('programID')
        phase_to_min_action_time[program] = collections.defaultdict(int)
        phase_to_original_action_time[program] = collections.defaultdict(int)
        phase_to_max_action_time[program] = collections.defaultdict(int)

        for original_phase_index, original_phase in enumerate(program_logic):
            original_phase_state = original_phase.get('state')

            if original_phase_state in phase_traffic_lights[program].values():
                index = list(phase_traffic_lights[program].values()).index(original_phase_state)
                phase = list(phase_traffic_lights[program].keys())[index]
            else:
                continue

            duration = float(original_phase.get('duration'))
            min_duration = float(original_phase.get('minDur', duration))
            max_duration = float(original_phase.get('maxDur', duration))

            phase_to_min_action_time[program][phase] += min_duration
            phase_to_original_action_time[program][phase] += duration
            phase_to_max_action_time[program][phase] += max_duration

    return phase_to_min_action_time, phase_to_original_action_time, phase_to_max_action_time


def detect_phases(movements, conflicts, link_states, minor_conflicts, same_lane_origin_movements,
                  major_conflicts_only=False, dedicated_minor_links_phases=True):

    phases_final_set = set()

    original_conflicts = conflicts

    if major_conflicts_only:
        conflicts = {}
        for movement, conflicting_movements in original_conflicts.items():
            minor_conflicting_movements = minor_conflicts[movement]
            conflicts[movement] = list(set(conflicting_movements).difference(minor_conflicting_movements))
    else:
        conflicts = original_conflicts

    phases = []

    depth_first_search_tracking = [copy.deepcopy(movements)]
    movements_left_list = [movements]
    elements_tracking = []

    while len(depth_first_search_tracking) != 0:

        while len(depth_first_search_tracking[0]) != 0:

            elements = []
            element = depth_first_search_tracking[0].pop(0)
            elements.append(element)

            same_origin_movements = same_lane_origin_movements[element]

            for same_origin_movement in same_origin_movements:
                assert same_origin_movement in movements_left_list[-1]
                elements.append(same_origin_movement)
                depth_first_search_tracking[0].remove(same_origin_movement)

            elements_tracking.append(elements)

            movements_left = [movement
                              for movement in movements_left_list[-1]
                              if movement not in conflicts[element] and
                              movement not in elements]

            movements_left_list.append(movements_left)

            if movements_left:
                depth_first_search_tracking.insert(0, movements_left)
            else:
                phase_elements = sorted(
                    [element for element_group in elements_tracking for element in element_group],
                    key=cmp_to_key(movement_comparator))
                phases.append('_'.join(phase_elements))
                elements_tracking.pop()
                movements_left_list.pop()

        depth_first_search_tracking.pop(0)
        if elements_tracking:
            elements_tracking.pop()
        movements_left_list.pop()

    phase_sets = [set(phase.split('_')) for phase in phases]

    indices_to_remove = set()
    for i in range(0, len(phase_sets)):
        for j in range(i+1, len(phase_sets)):

            phase_i = phase_sets[i]
            phase_j = phase_sets[j]

            if phase_i.issubset(phase_j):
                indices_to_remove.add(i)
            elif phase_j.issubset(phase_i):
                indices_to_remove.add(j)

    indices_to_remove = sorted(indices_to_remove, reverse=True)
    for index_to_remove in indices_to_remove:
        phases.pop(index_to_remove)
        phase_sets.pop(index_to_remove)

    if major_conflicts_only and dedicated_minor_links_phases:
        minor_link_phases = set([])
        for phase in phases:
            phase_movements = phase.split('_')

            minor_phase_movements = []
            for movement_1 in phase_movements:

                movement_1_link = link_states[movement_1]

                if movement_1_link == 'M':
                    continue

                movement_1_conflicts = original_conflicts[movement_1]

                for movement_2 in phase_movements:
                    if movement_2 in movement_1_conflicts:
                        minor_phase_movements.append(movement_1)
                        break

            minor_link_phase = '_'.join(minor_phase_movements)
            if minor_link_phase:
                minor_link_phases.add(minor_link_phase)

        phases.extend(minor_link_phases)

    phases = sorted(phases, key=cmp_to_key(phase_comparator))

    for phase in phases:
        phases_final_set.add(phase)

    return phases


def simplify_existing_phases(phases_list, unique_phases, movement_list, unique_movements):

    phase_mapping_list = []
    for i, program_phases in enumerate(phases_list):

        phase_mapping = collections.defaultdict(bidict)

        for program, phases in program_phases.items():

            old_phases = copy.deepcopy(phases)

            movements = movement_list[i]
            missing_movements = set(unique_movements).difference(set(movements))

            import itertools

            missing_movements_set_generator = itertools.chain.from_iterable(itertools.combinations(missing_movements, r) for r in range(len(missing_movements), -1, -1))

            element = next(missing_movements_set_generator, None)

            while element is not None and old_phases:

                for p in range(len(old_phases) - 1, -1, -1):

                    old_phase = old_phases[p]

                    phase_movements_set = set(old_phase.split('_'))

                    if element != ():
                        phase_movements_set.update(element)

                    new_phase = "_".join(sorted(phase_movements_set, key=cmp_to_key(movement_comparator)))
                    if new_phase in unique_phases:
                        phase_mapping[program][old_phase] = new_phase
                        old_phases.remove(old_phase)

                element = next(missing_movements_set_generator, None)

        phase_mapping_list.append(phase_mapping)

    unique_phases = sorted(list(set(
        [
            phase
            for phase_mapping in phase_mapping_list
            for program, mapping in phase_mapping.items()
            for phase in mapping.values()
        ])), key=cmp_to_key(phase_comparator))

    return phase_mapping_list, unique_phases


def generate_bus_lanes_candidates(net_xml, bus_lane_candidate_path):

    def _is_intersection(net_xml, from_edge, to_edge):

        is_intersection_ = True

        from_connections = sumo_net_util.get_connection_map(net_xml, sumo_net_util.FROM_EDGE_MAP)[from_edge]
        to_connections = sumo_net_util.get_connection_map(net_xml, sumo_net_util.TO_EDGE_MAP)[to_edge]

        if len(from_connections) == len(to_connections):
            if (all([connection.get['to'] == to_edge for connection in from_connections])
                    and all([connection.get['from'] == from_edge for connection in to_connections])):
                is_intersection_ = False

        return is_intersection_

    exclusive_lane_candidates = []
    candidate = []
    for from_, middle, to in bus_lane_candidate_path:

        # Perform only extensions to the current exclusive lane section
        if candidate:
            if middle in candidate:
                candidate.append(to)

                if to is not None and not _is_intersection(net_xml, middle, to):
                    continue

                exclusive_lane_candidates.append(candidate)
                candidate = []
                continue
            else:
                exclusive_lane_candidates.append(candidate)
                candidate = []

        # Start new exclusive lane section
        candidate.extend([from_, middle, to])

        if to is not None and not _is_intersection(net_xml, middle, to):
            continue

        exclusive_lane_candidates.append(candidate)
        candidate = []

    return exclusive_lane_candidates


# Get exclusive lanes for bus and emergency lanes.
def get_exclusive_lanes(net_xml, candidates, exclusive_lane_side):

    def get_preferred_connections(lane_connections, lane_direction, exclusive_lane_side):

        # Get the correct connection links that will be part of the exclusive lanes
        if not lane_connections:
            return []

        if exclusive_lane_side == 'right':
            preferred_link = 0
        elif exclusive_lane_side == 'left':
            preferred_link = max([int(connection.get(lane_direction)) for connection in lane_connections])
        else:
            raise ValueError(f'Wrong exclusive lane side parameter {exclusive_lane_side}')

        preferred_connections = [connection for connection in lane_connections
                                 if int(connection.get(lane_direction)) == preferred_link]

        return preferred_connections

    selected_exclusive_lanes = {}
    for exclusive_lane in candidates:

        lane_sections = [exclusive_lane[0], *exclusive_lane[1], exclusive_lane[2]]

        connection_set = set()
        for i in range(1, len(lane_sections)-1):

            pre_section = lane_sections[i]
            from_all_to_lane_connections = sumo_net_util.get_connection_map(net_xml, sumo_net_util.TO_EDGE_MAP)[pre_section]
            connection_count = collections.Counter(
                [connection.get('from') for connection in from_all_to_lane_connections])
            connection_candidates = list(filter(lambda x: True if connection_count[x] > 1 else False, connection_count))
            lane_connections_from = [connection for connection in from_all_to_lane_connections
                                     if connection.get('from') in connection_candidates]
            preferred_connection_from = get_preferred_connections(lane_connections_from, 'toLane', exclusive_lane_side)

            post_section = lane_sections[i]
            from_lane_to_all_connections = sumo_net_util.get_connection_map(net_xml, sumo_net_util.FROM_EDGE_MAP)[post_section]
            connection_count = collections.Counter(
                [connection.get('to') for connection in from_lane_to_all_connections])
            connection_candidates = list(filter(lambda x: True if connection_count[x] > 1 else False, connection_count))
            lane_connections_to = [connection for connection in from_lane_to_all_connections
                                   if connection.get('to') in connection_candidates]
            preferred_connection_to = get_preferred_connections(lane_connections_to, 'fromLane', exclusive_lane_side)

            if ((lane_sections[i-1] is None or preferred_connection_from)
                    and (lane_sections[i+1] is None or preferred_connection_to)):

                if lane_connections_from:
                    connection_set.update(preferred_connection_from)
                if lane_connections_to:
                    connection_set.update(preferred_connection_to)

        if connection_set:
            selected_exclusive_lanes[tuple(exclusive_lane[1])] = list(connection_set)

    return selected_exclusive_lanes


def get_bus_lanes(net_xml, candidates):

    selected_bus_lanes = {}
    for bus_lane in candidates:
        from_all_to_bus_lane_connections = sumo_net_util.get_connection_map(net_xml, sumo_net_util.TO_EDGE_MAP)[bus_lane[1]]
        connection_count = collections.Counter(
            [connection.get('from') for connection in from_all_to_bus_lane_connections])
        connection_candidates = list(filter(lambda x: True if connection_count[x] > 1 else False, connection_count))
        bus_lane_connections_from = [connection for connection in from_all_to_bus_lane_connections
                                     if connection.get('from') in connection_candidates
                                     and int(connection.get('toLane')) == 0]

        from_bus_lane_to_all_connections = sumo_net_util.get_connection_map(net_xml, sumo_net_util.FROM_EDGE_MAP)[bus_lane[-2]]
        connection_count = collections.Counter(
            [connection.get('to') for connection in from_bus_lane_to_all_connections])
        connection_candidates = list(filter(lambda x: True if connection_count[x] > 1 else False, connection_count))
        bus_lane_connections_to = [connection for connection in from_bus_lane_to_all_connections
                                   if connection.get('to') in connection_candidates
                                   and int(connection.get('fromLane')) == 0]

        if ((bus_lane[0] is None or len(bus_lane_connections_from) > 0)
                and (bus_lane[-1] is None or len(bus_lane_connections_to) > 0)):

            bus_lane_connections_middle = []
            for i in range(0, len(bus_lane[1:-1])-1):
                edge_connection_id = f"{bus_lane[1:-1][i]}__{bus_lane[1:-1][i + 1]}"
                connections = sumo_net_util.get_connection_map(net_xml, sumo_net_util.EDGE_BASED_MAP)[edge_connection_id]
                bus_lane_connections_middle.extend(connections)

            selected_bus_lanes[tuple(bus_lane)] = (
                bus_lane_connections_from, bus_lane_connections_middle, bus_lane_connections_to)

    return selected_bus_lanes


def add_traffic_actors(sumocfg_file):

    sumocfg_xml = xml_util.parse_xml(sumocfg_file)

    root = sumocfg_xml.getroot()
    route_files_node = root.xpath('//configuration/input/route-files')[0]

    route_files = [config.SCENARIO.CAR_TRIPS_FILE]

    if config.SCENARIO.HAS_BUSES:
        route_files.append(config.SCENARIO.BUS_TRIPS_FILE)

    if config.SCENARIO.HAS_PASSENGERS:
        route_files.append(config.SCENARIO.PASSENGER_TRIPS_FILE)

    route_files_node.set('value', ','.join(route_files))
    xml_util.write_xml(root, sumocfg_file)


def get_sumo_cmd(sumocfg_file=None, sumocfg_parameters=None, with_gui=False, write_mode=True, step_length=1):

    if sumocfg_parameters is None and sumocfg_file is None:
        raise ValueError("Either sumocfg_parameters or work_directory must be specified")

    if sumocfg_parameters is None:
        sumocfg_parameters = {}

    sumocfg_parameters.update(
        {
            '-c': r'{0}'.format(sumocfg_file)
        }
    )

    add_traffic_actors(sumocfg_file)

    sumocfg_parameters['--step-length'] = str(step_length)

    if not write_mode:
        sumocfg_parameters.pop('--log', None)

    sumocfg_parameters_list = [str(item)
                               for key_value_pair in sumocfg_parameters.items()
                               for item in key_value_pair]

    sumo_binary = get_sumo_binary(with_gui)
    sumo_cmd = [sumo_binary, *sumocfg_parameters_list]

    return sumo_cmd


def adjusts_intersection_position(junctions, edges, x_spacing=0, y_spacing=0):
    for junction in junctions:
        junction.set('x', str(float(junction.get('x')) + x_spacing))
        junction.set('y', str(float(junction.get('y')) + y_spacing))

        if 'shape' in junction.attrib:
            junction.set('shape', math_util.translate_polyline(junction.get('shape'), x=x_spacing, y=y_spacing))

    for edge in edges:
        for lane in edge:
            lane.set('shape', math_util.translate_polyline(lane.get('shape'), x=x_spacing, y=y_spacing))


def map_connection_direction(connection):
    direction = connection.get('dir').lower()

    if direction == 'l':
        direction = 'left_turn'
    elif direction == 's':
        direction = 'straight'
    elif direction == 'r':
        direction = 'right_turn'

    return direction


def get_phase_traffic_lights(movements, movement_to_connection, is_right_on_red, phases, link_states,
                             movement_to_traffic_light_link_index):

    uncontrolled_movements = [
        movement for movement, _ in movement_to_connection.items() if movement not in movements]

    phase_traffic_lights = {}

    for phase in phases:

        links_length = len({i for v in movement_to_traffic_light_link_index.values() for i in v})
        phase_signal_string = ['r'] * links_length

        phase_movements = phase.split('_')

        for uncontrolled_movement in uncontrolled_movements:
            uncontrolled_movement_traffic_light_indices = (
                movement_to_traffic_light_link_index)[uncontrolled_movement]

            for uncontrolled_movement_traffic_light_index in uncontrolled_movement_traffic_light_indices:
                if is_right_on_red and 'R' in uncontrolled_movement:

                    movement_link_state_list = []
                    for phase_movement in phase_movements:
                        if phase_movement[0] == uncontrolled_movement[0]:
                            movement_link_state = link_states[phase_movement]
                            movement_link_state_list.append(movement_link_state)

                    if 'M' in movement_link_state_list:
                        phase_signal_string[uncontrolled_movement_traffic_light_index] = 'G'
                    else:
                        phase_signal_string[uncontrolled_movement_traffic_light_index] = 's'

        for phase_movement_index, movement in enumerate(phase_movements):

            traffic_light_indices = movement_to_traffic_light_link_index[movement]

            for traffic_light_index in traffic_light_indices:
                movement_link_state = link_states[movement]
                if movement_link_state == 'm':
                    phase_signal_string[traffic_light_index] = 'g'
                else:
                    phase_signal_string[traffic_light_index] = 'G'

        phase_traffic_lights[phase] = "".join(phase_signal_string)

    return phase_traffic_lights


def convert_sumo_angle_to_canonical_angle(sumo_angle):

    if sumo_angle <= 90:
        canonical_angle = 90 - sumo_angle
    elif sumo_angle > 90:
        canonical_angle = 90 - (sumo_angle - 360)
    else:
        raise ValueError(f"Impossible sumo angle {sumo_angle}")

    return canonical_angle


def fix_passenger_triggered_departure(passenger_trips_xml, bus_trips_xml, time_):

    bus_trips = bus_trips_xml.findall("//trip")

    past_buses = []
    for bus_trip in bus_trips:

        depart_time_in_seconds = datetime_util.convert_human_time_to_seconds(bus_trip.get('depart'))

        if depart_time_in_seconds > time_:
            break

        bus_id = bus_trip.get('id')
        past_buses.append(bus_id)

    passenger_trips = passenger_trips_xml.findall("//personFlow[@begin='triggered']")
    for p in range(len(passenger_trips)-1, -1, -1):

        if passenger_trips[p][0].get('lines') in past_buses:
            passenger_trips[p].getparent().remove(passenger_trips[p])

    return passenger_trips_xml


def fix_state(filepath):

    save_state_xml = xml_util.parse_xml(filepath)

    xml_str = etree.tostring(save_state_xml).decode()

    pattern = re.compile(r'triggered=""', re.IGNORECASE)
    replacement = 'triggered="0"'
    xml_str = pattern.sub(replacement, xml_str)

    pattern = re.compile(r'"inf"', re.IGNORECASE)
    replacement = '"1000"'

    xml_str = pattern.sub(replacement, xml_str)

    save_state_xml = etree.fromstring(xml_str).getroottree()

    save_state_xml.write(filepath, pretty_print=True)


def fix_person_plan_state(filepath):

    save_state_xml = xml_util.parse_xml(filepath)

    persons_plans = save_state_xml.findall('//person[walk]')

    for person_plan in persons_plans:
        first_step = person_plan[0]
        if first_step.tag == 'ride':
            person_plan.remove(first_step)
            person_plan.append(first_step)

    save_state_xml.write(filepath, pretty_print=True)


def get_yellow_signal_movements(traffic_light, phase, movement_to_traffic_light_link_index, traffic_light_link_indices):

    yellow_signal_indices = []
    for index in range(0, len(traffic_light)):
        if index in traffic_light_link_indices and traffic_light[index] == 'y':
            yellow_signal_indices.append(index)

    current_phase_movements = phase.split('_')
    yellow_signal_movements = []
    for movement in current_phase_movements:
        traffic_light_indices = set(movement_to_traffic_light_link_index[movement])

        if traffic_light_indices.intersection(yellow_signal_indices):
            yellow_signal_movements.append(movement)

    return yellow_signal_movements


def get_lane_based_subscription_extension(
        intersection_id, net_xml,
        entering_edges, exiting_edges, internal_lanes, intermediary_lanes,
        maximum_detection_length=None, maximum_detection_time=None, multi_intersection_tl=None):

    def get_time_based_detector_length(remaining_detection_time, current_detection_length, lane):

        lane_length = float(lane.get('length'))
        lane_speed = float(lane.get('speed'))

        maximum_detection_length = current_detection_length + (lane_speed * remaining_detection_time)
        remaining_detection_time -= max(0.0, (lane_length / lane_speed))

        return maximum_detection_length, remaining_detection_time

    def get_front_or_back_adjacent_lanes(net_xml, lane, lane_type):

        if lane_type == 'entering':

            internal_lane_chains = (
                sumo_net_util.get_previous_lanes(net_xml, lane, internal=True, over_intersection=False))
            external_lanes = (
                sumo_net_util.get_previous_lanes(net_xml, lane, over_intersection=False))
    
            previous_lanes = list(zip(internal_lane_chains, external_lanes))
            
            lanes = previous_lanes
        
        elif lane_type == 'exiting':
            
            internal_lane_chains = (
                sumo_net_util.get_next_lanes(net_xml, lane, internal=True, over_intersections=False))
            external_lanes = (
                sumo_net_util.get_next_lanes(net_xml, lane, over_intersections=False))

            next_lanes = list(zip(internal_lane_chains, external_lanes))
            
            lanes = next_lanes
            
        else:
            raise ValueError("Wrong lane type")

        return lanes

    def fill_subscription_extension_adjacent_edges(edges, lane_type):

        for edge in edges:
    
            lanes = list(edge)
    
            for lane in lanes:
    
                lane_length_list = []
                lane_stack = []
                stack_info = []
    
                detection_length = maximum_detection_length
                remaining_detection_time = maximum_detection_time
                if using_time:
                    detection_length, remaining_detection_time = get_time_based_detector_length(
                        remaining_detection_time, 0, lane)
    
                lane_id = lane.get('id')
                lane_length = float(lane.get('length'))
                lane_length_list.append(lane_length)
                accumulated_length = sum(lane_length_list)
    
                stack_info.append((accumulated_length, detection_length, remaining_detection_time))
    
                subscription_extension[lane_id] = (
                    [(lane_id, lane_length, accumulated_length, detection_length, lane_type)])
    
                adjacent_lanes = get_front_or_back_adjacent_lanes(net_xml, lane, lane_type)
                lane_stack.append(adjacent_lanes)
                while len(lane_stack) != 0:
    
                    while len(lane_stack[-1]) != 0:
    
                        internal_lanes, external_lane = lane_stack[-1].pop(0)
    
                        for internal_lane in internal_lanes[::-1]:
                            if sum(lane_length_list) > detection_length:
                                break
    
                            internal_lane_id = internal_lane.get('id')
                            internal_lane_length = float(internal_lane.get('length'))
    
                            keep_clear = internal_lane.get('keepClear')
                            if keep_clear is not None and not keep_clear:
                                accumulated_length, detection_length, remaining_detection_time = (
                                    stack_info[-1])

                                if using_time:
                                    detection_length, remaining_detection_time = get_time_based_detector_length(
                                        remaining_detection_time, accumulated_length, internal_lane)
    
                                lane_length_list.append(internal_lane_length)
                                accumulated_length = sum(lane_length_list)
    
                                stack_info.append(
                                    (accumulated_length, detection_length, remaining_detection_time))
    
                                subscription_extension[lane_id].append(
                                    (internal_lane_id, internal_lane_length,
                                     accumulated_length, detection_length, lane_type)
                                )
                            else:
                                subscription_extension[lane_id].append(
                                    (internal_lane_id, internal_lane_length,
                                     0, detection_length, lane_type)
                                )
    
                        if sum(lane_length_list) > detection_length:
                            continue
    
                        accumulated_length, detection_length, remaining_detection_time = (
                            stack_info[-1])

                        if using_time:
                            detection_length, remaining_detection_time = get_time_based_detector_length(
                                remaining_detection_time, accumulated_length, external_lane)
    
                        external_lane_id = external_lane.get('id')
                        external_lane_length = float(external_lane.get('length'))
                        lane_length_list.append(external_lane_length)
                        accumulated_length = sum(lane_length_list)
    
                        stack_info.append((accumulated_length, detection_length, remaining_detection_time))
    
                        subscription_extension[lane_id].append(
                            (external_lane_id, external_lane_length,
                             accumulated_length, detection_length, lane_type)
                        )
    
                        adjacent_lanes = get_front_or_back_adjacent_lanes(net_xml, external_lane, lane_type)
                        lane_stack.append(adjacent_lanes)
    
                    lane_stack.pop()
                    lane_length_list.pop()
                    stack_info.pop()

    if maximum_detection_length is None and maximum_detection_time is None:
        raise ValueError("Both 'maximum_detector_length' and 'maximum_detector_time' cannot be None")

    using_time = True if maximum_detection_length is None else False

    subscription_extension = {}

    lane_type = 'entering'
    fill_subscription_extension_adjacent_edges(entering_edges, lane_type)

    lane_type = 'exiting'
    fill_subscription_extension_adjacent_edges(exiting_edges, lane_type)

    lane_type = 'intersection'
    for lane in internal_lanes + intermediary_lanes:
        intersection_lane_id = lane.get('id')
        lane_length = float(lane.get('length'))

        subscription_extension[intersection_lane_id] = [
            (intersection_lane_id, lane_length, 0, 0, lane_type)
        ]

    connections = sumo_net_util.get_intersection_connections(net_xml, intersection_id, multi_intersection_tl)

    from_lane_to_intermediary_lane = {}
    for connection in connections:

        if isinstance(connection, list):
            inner_connections = connection
        else:
            inner_connections = [connection]

        from_lane = inner_connections[0].get('from') + '_' + inner_connections[0].get('fromLane')

        if from_lane not in from_lane_to_intermediary_lane:
            from_lane_to_intermediary_lane[from_lane] = set([])

        for inner_connection in inner_connections[1:]:
            connection_from_lane_id = inner_connection.get('from') + '_' + inner_connection.get('fromLane')
            inner_from_lane = sumo_net_util.get_lane(net_xml, connection_from_lane_id)

            from_lane_to_intermediary_lane[from_lane].add(inner_from_lane)

    lane_type = 'entering'
    intermediary_lanes_by_entering_lane_dict = {}
    for entering_edge in entering_edges:

        lanes = list(entering_edge)

        for lane in lanes:

            lane_id = lane.get('id')

            intermediary_lanes_by_entering_lane_dict[lane_id] = []

            intermediary_lanes = from_lane_to_intermediary_lane.get(lane_id, [])

            for intermediary_lane in intermediary_lanes:
                intermediary_lane_id = intermediary_lane.get('id')

                lane_length = float(intermediary_lane.get('length'))

                intermediary_lanes_by_entering_lane_dict[lane_id].append(
                    (intermediary_lane_id, lane_length, 0, 0, lane_type)
                )

    for key, value in intermediary_lanes_by_entering_lane_dict.items():
        subscription_extension[key] = value + subscription_extension[key]

    return subscription_extension


def get_edge_based_subscription_extension(
        net_xml, entering_edges, exiting_edges, maximum_detection_length=None, maximum_detection_time=None):
    def get_time_based_detector_length(remaining_detection_time, current_detection_length, edge):

        edge_length = float(list(edge)[0].get('length'))
        edge_speed = float(edge.get('speed'))

        maximum_detection_length = current_detection_length + (edge_speed * remaining_detection_time)
        remaining_detection_time -= max(0.0, (edge_length / edge_speed))

        return maximum_detection_length, remaining_detection_time

    def get_front_or_back_adjacent_edges(net_xml, edge, edge_type):

        if edge_type == 'entering':

            internal_edge_chains = (
                sumo_net_util.get_previous_edges(net_xml, edge, internal=True, over_intersection=False))
            external_edges = (
                sumo_net_util.get_previous_edges(net_xml, edge, over_intersection=False))

            previous_edges = list(zip(internal_edge_chains, external_edges))

            edges = previous_edges

        elif edge_type == 'exiting':

            internal_edge_chains = (
                sumo_net_util.get_next_edges(net_xml, edge, internal=True, over_intersection=False))
            external_edges = (
                sumo_net_util.get_next_edges(net_xml, edge, over_intersection=False))

            next_edges = list(zip(internal_edge_chains, external_edges))

            edges = next_edges

        else:
            raise ValueError("Wrong lane type")

        return edges

    def fill_subscription_extension_adjacent_edges(edges, edge_type):

        for edge in edges:

            edge_length_list = []
            edge_stack = []
            stack_info = []

            detection_length = maximum_detection_length
            remaining_detection_time = maximum_detection_time
            if using_time:
                detection_length, remaining_detection_time = get_time_based_detector_length(
                    remaining_detection_time, 0, edge)

            edge_id = edge.get('id')
            edge_length = float(list(edge)[0].get('length'))
            edge_length_list.append(edge_length)
            accumulated_length = sum(edge_length_list)

            stack_info.append((accumulated_length, detection_length, remaining_detection_time))

            subscription_extension[edge_id] = (
                [(edge_id, edge_length, accumulated_length, detection_length, edge_type)])

            adjacent_edges = get_front_or_back_adjacent_edges(net_xml, edge, edge_type)
            edge_stack.append(adjacent_edges)
            while len(edge_stack) != 0:

                while len(edge_stack[-1]) != 0:

                    internal_edge, external_edge = edge_stack[-1].pop(0)

                    if sum(edge_length_list) > detection_length:
                        break

                    internal_edge_id = internal_edge.get('id')
                    internal_edge_length = float(list(internal_edge)[0].get('length'))

                    keep_clear = internal_edge.get('keepClear')
                    if keep_clear is not None and not keep_clear:
                        accumulated_length, detection_length, remaining_detection_time = stack_info[-1]

                        if using_time:
                            detection_length, remaining_detection_time = get_time_based_detector_length(
                                remaining_detection_time, accumulated_length, internal_edge)

                        edge_length_list.append(internal_edge_length)
                        accumulated_length = sum(edge_length_list)

                        stack_info.append((accumulated_length, detection_length, remaining_detection_time))

                        subscription_extension[edge_id].append((internal_edge_id, internal_edge_length,
                             accumulated_length, detection_length, edge_type)
                        )
                    else:
                        subscription_extension[edge_id].append((internal_edge_id, internal_edge_length,
                             0, detection_length, edge_type)
                        )

                    if sum(edge_length_list) > detection_length:
                        continue

                    accumulated_length, detection_length, remaining_detection_time = stack_info[-1]

                    if using_time:
                        detection_length, remaining_detection_time = get_time_based_detector_length(
                            remaining_detection_time, accumulated_length, external_edge)

                    external_lane_id = external_edge.get('id')
                    external_lane_length = float(list(external_edge)[0].get('length'))
                    edge_length_list.append(external_lane_length)
                    accumulated_length = sum(edge_length_list)

                    stack_info.append((accumulated_length, detection_length, remaining_detection_time))

                    subscription_extension[edge_id].append((external_lane_id, external_lane_length,
                         accumulated_length, detection_length, edge_type)
                    )

                    adjacent_edges = get_front_or_back_adjacent_edges(net_xml, external_edge, edge_type)
                    edge_stack.append(adjacent_edges)

                edge_stack.pop()
                edge_length_list.pop()
                stack_info.pop()

    if maximum_detection_length is None and maximum_detection_time is None:
        raise ValueError("Both 'maximum_detector_length' and 'maximum_detector_time' cannot be None")

    using_time = True if maximum_detection_length is None else False

    subscription_extension = {}

    edge_type = 'entering'
    fill_subscription_extension_adjacent_edges(entering_edges, edge_type)

    edge_type = 'exiting'
    fill_subscription_extension_adjacent_edges(exiting_edges, edge_type)

    return subscription_extension


def filter_unused_traffic_light_links(traffic_light_xml, traffic_light_id, movement_to_traffic_light_link_index):

    traffic_light_logics = sumo_net_util.get_traffic_light_program_logics(traffic_light_xml, traffic_light_id)

    unused_traffic_light_links = dict(enumerate(traffic_light_logics[0][0].get('state')))
    # Looking up to the first logic should be reliable enough
    for traffic_light_logic in list(traffic_light_logics[0])[1:]:

        traffic_light_links = dict(enumerate(traffic_light_logic.get('state')))

        for k in list(unused_traffic_light_links.keys()):
            if unused_traffic_light_links[k] != traffic_light_links[k]:
                unused_traffic_light_links.pop(k)

    if len(traffic_light_logics[0]) == 1:
        unused_traffic_light_links = {}

    movement_to_traffic_light_link_index_copy = copy.deepcopy(movement_to_traffic_light_link_index)
    for movement, traffic_light_link_indices in movement_to_traffic_light_link_index_copy.items():
        for traffic_light_link in traffic_light_link_indices:
            if traffic_light_link in unused_traffic_light_links:
                movement_to_traffic_light_link_index[movement].remove(traffic_light_link)

    return movement_to_traffic_light_link_index


def remap_movements(movements_mapping, movements, connection_id_to_movement, movement_to_connection_id):

    movements = [movements_mapping[movement] for movement in movements if movements_mapping[movement]]
    connection_id_to_movement = {
        connection_id: [movements_mapping[movement] for movement in movements if movements_mapping[movement]]
        for connection_id, movements in connection_id_to_movement.items()
    }
    movement_to_connection_id = {
        movements_mapping[movement]: connection_ids
        for movement, connection_ids in movement_to_connection_id.items()
        if movements_mapping[movement]
    }

    return movements, connection_id_to_movement, movement_to_connection_id


def read_fixed_movement_labels(traffic_light_id, movements, connection_id_to_movement, movement_to_connection_id):
    try:
        with open(os.path.join(config.ROOT_DIR, config.PATH_TO_SCENARIO, 'simulation',
                               'fixed_movement_labels.json'), 'r') as file:
            movements_mapping = json.load(file)[traffic_light_id]

        movements, connection_id_to_movement, movement_to_connection_id = (
            remap_movements(movements_mapping, movements, connection_id_to_movement, movement_to_connection_id))
    except Exception as e:
        warnings.warn(f"Unable to read fixed movements, {e}")

    return movements, connection_id_to_movement, movement_to_connection_id


def get_subscription_extension_split_info(subscription_extension):

    detector_id_lane_id_tuples = [
            (detector_id, lane_id)
            for detector_id, lane_tuples in subscription_extension.items()
            for lane_id, _, _, _, _ in lane_tuples
    ]

    subscription_extension_split_info = collections.defaultdict(list)

    for (detector_id, lane_id) in detector_id_lane_id_tuples:
        subscription_extension_split_info[lane_id].append(detector_id)

    return subscription_extension_split_info


def setup_bus_lanes(net_xml):

    buslines_file = os.path.join(config.ROOT_DIR, config.PATH_TO_SCENARIO, config.SCENARIO.BUS_LINES_FILE)
    buslines = pd.read_xml(buslines_file)

    buslines_routes_file = os.path.join(config.ROOT_DIR, config.PATH_TO_SCENARIO, config.SCENARIO.BUS_LINES_FILE)
    buslines_routes = pd.read_xml(buslines_routes_file, xpath='//route')
    buslines.loc[:, 'route'] = buslines_routes['edges']
    buslines.loc[:, 'direction'] = buslines['id'].str.split('_').str[-1]

    net_edges = net_xml.xpath("//edge[not(@function)]")
    net_edges = {
        edge.get('id'): edge
        for edge in net_edges
    }

    edges_of_interest_list = [
        [net_edges[edge] for edge in edge_list]
        for edge_list in buslines['route'].str.split(' ').values
    ]

    exclusive_lane_candidates_list = []
    for edges_of_interest in edges_of_interest_list:
        exclusive_lane_candidates = set()
        edges_of_interest = [None] + edges_of_interest + [None]
        current_edges = []
        for i in range(1, len(edges_of_interest) - 1):

            edge = edges_of_interest[i]
            if edge in current_edges:
                continue

            current_edges = sumo_net_util.get_block_edges(net_xml, edge)
            current_edge_ids = tuple(edge.get('id') for edge in current_edges)

            previous_edge = edges_of_interest[i-1]
            previous_edge_id = previous_edge.get('id') if previous_edge is not None else None
            next_edge = edges_of_interest[min(i + len(current_edges), len(edges_of_interest) - 1)]
            next_edge_id = next_edge.get('id') if next_edge is not None else None

            local_candidates_paths = [tuple([previous_edge_id, current_edge_ids, next_edge_id])]

            exclusive_lane_candidates.update(local_candidates_paths)
        exclusive_lane_candidates_list.append(list(exclusive_lane_candidates))

    lines_bus_lanes = []
    for exclusive_lane_candidates in exclusive_lane_candidates_list:
        bus_lanes = get_exclusive_lanes(net_xml, exclusive_lane_candidates, 'right')
        lines_bus_lanes.append(bus_lanes)

    bus_lane_to_line_map = collections.defaultdict(list)
    for i, bus_line_bus_lanes in enumerate(lines_bus_lanes):
        for bus_lane in bus_line_bus_lanes:
            bus_lane_to_line_map[bus_lane].append((buslines.iloc[i]['line'], buslines.iloc[i]['direction']))

    bus_lanes = {
        bus_lane: connections
        for bus_line_bus_lanes in lines_bus_lanes
        for bus_lane, connections in bus_line_bus_lanes.items()
    }

    return bus_lanes, bus_lane_to_line_map


def setup_emergency_lanes(net_xml):

    net_edges = net_xml.xpath("//edge[not(@function)]")
    filtered_edges = [
        edge
        for edge in net_edges
        if len(edge) > 1
    ]

    exclusive_lane_candidates = set()
    for edge in filtered_edges:

        current_edges = sumo_net_util.get_block_edges(net_xml, edge)
        current_edge_ids = tuple(edge.get('id') for edge in current_edges)

        previous_edges = sumo_net_util.get_previous_edges(net_xml, current_edges[0])
        previous_edges = [e.get('id') for e in previous_edges] if previous_edges else [None]
        next_edges = sumo_net_util.get_next_edges(net_xml, current_edges[-1])
        next_edges = [e.get('id') for e in next_edges] if next_edges else [None]

        local_candidates_paths = list(itertools.product(previous_edges, tuple([current_edge_ids]), next_edges))

        exclusive_lane_candidates.update(local_candidates_paths)
    exclusive_lane_candidates = list(exclusive_lane_candidates)

    emergency_lanes = get_exclusive_lanes(net_xml, exclusive_lane_candidates, 'left')

    return emergency_lanes


def get_exclusive_lanes__lanes(net_xml, edge_ids, exclusive_lane_side):

    exclusive_lanes__lanes = []

    for edge_id in edge_ids:
        edge = sumo_net_util.get_edge(net_xml, edge_id)

        if exclusive_lane_side == 'right':
            exclusive_lane_index = 0
        elif exclusive_lane_side == 'left':
            exclusive_lane_index = max([int(lane.get('index')) for lane in list(edge)])
        else:
            raise ValueError(f'Wrong exclusive lane side parameter {exclusive_lane_side}')

        exclusive_lanes__lanes.append(f"{edge_id}_{exclusive_lane_index}")

    return exclusive_lanes__lanes
