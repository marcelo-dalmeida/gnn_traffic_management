import collections
import copy
import math
from functools import cmp_to_key, lru_cache

import numpy as np
from shapely.geometry import Point

from utils import collections_util
from utils.collections_util import bidict
from utils.comparator import *

REGULAR_MAP = 'connection__from_lane_to_lane_map'
EDGE_BASED_MAP = 'connection_from_edge_to_edge_map'
FROM_EDGE_MAP = 'connection_from_edge_map'
FROM_LANE_MAP = 'connection_from_lane_map'
TO_EDGE_MAP = 'connection_to_edge_map'
TO_LANE_MAP = 'connection_to_lane_map'


@lru_cache(maxsize=None)
def _get_connection_map(net_xml):

    map_elements = net_xml.findall('.//connection')

    connection_map = {}
    connection_from_edge_to_edge_map = collections.defaultdict(list)
    connection_from_edge_map = collections.defaultdict(list)
    connection_from_lane_map = collections.defaultdict(list)
    connection_to_edge_map = collections.defaultdict(list)
    connection_to_lane_map = collections.defaultdict(list)

    for element in map_elements:
        element_id = get_connection_id(element)
        connection_map[element_id] = element

        if ':' not in element_id:
            connection_from_edge_to_edge_map[get_connection_from_edge_to_edge_id(element)].append(element)
            connection_from_edge_map[get_connection_from_edge_id(element)].append(element)
            connection_from_lane_map[get_connection_from_lane_id(element)].append(element)
            connection_to_edge_map[get_connection_to_edge_id(element)].append(element)
            connection_to_lane_map[get_connection_to_lane_id(element)].append(element)

    return_type_dict = {
        REGULAR_MAP: connection_map,
        EDGE_BASED_MAP: connection_from_edge_to_edge_map,
        FROM_EDGE_MAP: connection_from_edge_map,
        FROM_LANE_MAP: connection_from_lane_map,
        TO_EDGE_MAP: connection_to_edge_map,
        TO_LANE_MAP: connection_to_lane_map,
    }

    return return_type_dict


@lru_cache(maxsize=None)
def get_intersection_map(net_xml):
    map_elements = net_xml.findall(".//junction[@type]")
    return {element.get('id'): element for element in map_elements if element.get('type') != 'internal'}


@lru_cache(maxsize=None)
def get_border_intersection_map(net_xml):
    map_elements = net_xml.findall(".//junction[@type]")
    return {element.get('id'): element for element in map_elements if element.get('type') == 'dead_end'}


@lru_cache(maxsize=None)
def get_internal_edge_map(net_xml):

    map_elements = net_xml.findall('.//edge[@function="internal"]')
    return {element.get('id'): element for element in map_elements}


@lru_cache(maxsize=None)
def get_edge_map(net_xml):

    map_elements = net_xml.findall('.//edge[@priority]')
    return {element.get('id'): element for element in map_elements}


@lru_cache(maxsize=None)
def get_internal_lane_map(net_xml):

    map_elements = net_xml.findall('.//edge[@function="internal"]/lane')
    return {element.get('id'): element for element in map_elements}


@lru_cache(maxsize=None)
def get_lane_map(net_xml):

    map_elements = net_xml.findall('.//edge[@priority]/lane')
    return {element.get('id'): element for element in map_elements}


@lru_cache(maxsize=None)
def get_from_lane_all_connections_map(net_xml):

    def get_id(connection):
        return f"{connection.get('from')}_{connection.get('fromLane')}"

    from_lane__via_connection_map = collections.defaultdict(list)

    connections = get_connection_map(net_xml).values()
    for connection in connections:
        from_lane__via_connection_map[get_id(connection)].append(connection)

    return from_lane__via_connection_map


@lru_cache(maxsize=None)
def is_intersection(net_xml, intersection_id, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    edge_to_edge_mapping = collections.defaultdict(set)

    connections = get_intersection_connections(net_xml, intersection_id, multi_intersection_config)

    for connection in connections:
        from_edge = connection.get('from')
        to_edge = connection.get('to')

        edge_to_edge_mapping[from_edge].add(to_edge)
        edge_to_edge_mapping[to_edge].add(from_edge)

        if (len(edge_to_edge_mapping[from_edge]) > 1 or
                len(edge_to_edge_mapping[to_edge]) > 1):
            return True

    return False


@lru_cache(maxsize=None)
def is_network_border(net_xml, intersection_id, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    edge_to_edge_mapping = collections.defaultdict(set)

    connections = get_intersection_connections(net_xml, intersection_id, multi_intersection_config)

    for connection in connections:
        from_edge = connection.get('from')
        to_edge = connection.get('to')

        edge_to_edge_mapping[from_edge].add(to_edge)
        edge_to_edge_mapping[to_edge].add(from_edge)

        if (len(edge_to_edge_mapping[from_edge]) > 1 or
                len(edge_to_edge_mapping[to_edge]) > 1):
            return True

    return False


@lru_cache(maxsize=None)
def get_adjacent_intersections(net_xml, intersection_id, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    edge_to_previous_intersection_mapping, edge_to_next_intersection_mapping = (
        get_edge_adjacent_intersections_mapping(net_xml))

    connections = get_intersection_connections(net_xml, intersection_id, multi_intersection_config)

    incoming_edges = set()
    outgoing_edges = set()
    for connection in connections:
        incoming_edges.add(connection.get('from'))
        outgoing_edges.add(connection.get('to'))

    adjacent_intersections = set()
    previous_intersections = []
    next_intersections = []

    for edge_id in incoming_edges:
        edge = get_edge(net_xml, edge_id)
        block_edges = get_block_edges(net_xml, edge)
        initial_edge = block_edges[0]
        initial_edge_id = initial_edge.get('id')
        previous_intersection = edge_to_previous_intersection_mapping[initial_edge_id]

        previous_intersections.append(previous_intersection)

    for edge_id in outgoing_edges:
        edge = get_edge(net_xml, edge_id)
        block_edges = get_block_edges(net_xml, edge)
        final_edge = block_edges[-1]
        final_edge_id = final_edge.get('id')
        next_intersection = edge_to_next_intersection_mapping[final_edge_id]

        next_intersections.append(next_intersection)

    adjacent_intersections.update(previous_intersections)
    adjacent_intersections.update(next_intersections)

    return list(adjacent_intersections), previous_intersections, next_intersections


@lru_cache(maxsize=None)
def get_previous_block_edges(net_xml, edge):

    edges = []

    intersection_id = edge.get('from')
    while not is_intersection(net_xml, intersection_id):

        previous_edges = get_previous_edges(net_xml, edge)
        if not previous_edges:
            break

        edge = previous_edges[0]
        edges.insert(0, edge)
        intersection_id = edge.get('from')

    return edges


@lru_cache(maxsize=None)
def get_next_block_edges(net_xml, edge):

    edges = []

    intersection_id = edge.get('to')
    while not is_intersection(net_xml, intersection_id):

        next_edges = get_next_edges(net_xml, edge)
        if not next_edges:
            break

        edge = next_edges[0]
        edges.append(edge)
        intersection_id = edge.get('to')

    return edges


@lru_cache(maxsize=None)
def get_previous_edges(net_xml, edge, internal=False, over_traffic_lights=True, over_intersection=True):

    edge_id = edge.get('id')
    intersection_id = edge.get('from')

    if not over_intersection:
        if is_intersection(net_xml, intersection_id):
            return []

    if not over_traffic_lights:

        intersection = get_intersection(net_xml, intersection_id)
        intersection_type = intersection.get('type')

        if intersection_type in ['traffic_light', 'traffic_light_right_on_red', 'traffic_light_unregulated']:
            return []

    connections = get_intersection_connections(net_xml, intersection_id)

    if internal:
        internal_connection_chains = get_intersection_internal_connection_chains(net_xml, intersection_id)

        previous_edge_ids = [
            [connection.get('from')
             for connection in internal_connection_chain]
            for internal_connection_chain in internal_connection_chains
            if internal_connection_chain[-1].get('to') == edge_id
        ]

        previous_edges = list(set([
            get_edge(net_xml, previous_edge_id)
            for edge_id_chain in previous_edge_ids
            for previous_edge_id in edge_id_chain
        ]))
    else:
        previous_edge_ids = np.unique([
            connection.get('from')
            for connection in connections
            if connection.get('to') == edge_id
        ])

        previous_edges = [get_edge(net_xml, previous_edge_id) for previous_edge_id in previous_edge_ids]

    return previous_edges


@lru_cache(maxsize=None)
def get_next_edges(net_xml, edge, internal=False, over_traffic_lights=True, over_intersection=True):

    edge_id = edge.get('id')
    intersection_id = edge.get('to')

    if not over_intersection:
        if is_intersection(net_xml, intersection_id):
            return []

    if not over_traffic_lights:

        intersection = get_intersection(net_xml, intersection_id)
        intersection_type = intersection.get('type')

        if intersection_type in ['traffic_light', 'traffic_light_right_on_red', 'traffic_light_unregulated']:
            return []

    connections = get_intersection_connections(net_xml, intersection_id)

    if internal:
        internal_connection_chains = get_intersection_internal_connection_chains(net_xml, intersection_id)

        next_edge_ids = [
            [internal_connection.get('from')
             for internal_connection in internal_connection_chain]
            for internal_connection_chain, connection in zip(internal_connection_chains, connections)
            if connection.get('from') == edge_id
        ]

        next_edges = list(set([
            get_edge(net_xml, next_edge_id)
            for edge_id_chain in next_edge_ids
            for next_edge_id in edge_id_chain
        ]))
    else:
        next_edge_ids = np.unique([
            connection.get('to')
            for connection in connections
            if connection.get('from') == edge_id
        ])

        next_edges = [get_edge(net_xml, next_edge_id) for next_edge_id in next_edge_ids]

    return next_edges


@lru_cache(maxsize=None)
def get_previous_lanes(net_xml, lane, internal=False, over_traffic_lights=True, over_intersection=True):

    lane_id = lane.get('id')
    edge_id = lane_id.rsplit('_', 1)[0]
    edge = get_edge(net_xml, edge_id)

    intersection_id = edge.get('from')

    if not over_intersection:
        if is_intersection(net_xml, intersection_id):
            return []

    if not over_traffic_lights:

        intersection = get_intersection(net_xml, intersection_id)
        intersection_type = intersection.get('type')

        if intersection_type in ['traffic_light', 'traffic_light_right_on_red', 'traffic_light_unregulated']:
            return []

    connections = get_intersection_connections(net_xml, intersection_id)

    if internal:
        internal_connection_chains = get_intersection_internal_connection_chains(net_xml, intersection_id)

        previous_lane_ids = [
            [connection.get('from') + '_' + connection.get('fromLane')
             for connection in internal_connection_chain]
            for internal_connection_chain in internal_connection_chains
            if internal_connection_chain[-1].get('to') + '_' + internal_connection_chain[-1].get('toLane') == lane_id
        ]

        previous_lanes = [
            [get_lane(net_xml, previous_lane_id) for previous_lane_id in lane_id_chain]
            for lane_id_chain in previous_lane_ids
        ]
    else:
        previous_lane_ids = [
            connection.get('from') + '_' + connection.get('fromLane')
            for connection in connections
            if connection.get('to') + '_' + connection.get('toLane') == lane_id
        ]

        previous_lanes = [get_lane(net_xml, previous_lane_id) for previous_lane_id in previous_lane_ids]

    return previous_lanes


@lru_cache(maxsize=None)
def get_next_lanes(net_xml, lane, internal=False, over_traffic_lights=True, over_intersections=True):

    lane_id = lane.get('id')
    edge_id = lane_id.rsplit('_', 1)[0]
    edge = get_edge(net_xml, edge_id)

    intersection_id = edge.get('to')

    if not over_intersections:
        if is_intersection(net_xml, intersection_id):
            return []

    if not over_traffic_lights:

        intersection = get_intersection(net_xml, intersection_id)
        intersection_type = intersection.get('type')

        if intersection_type in ['traffic_light', 'traffic_light_right_on_red', 'traffic_light_unregulated']:
            return []

    connections = get_intersection_connections(net_xml, intersection_id)

    if internal:
        internal_connection_chains = get_intersection_internal_connection_chains(net_xml, intersection_id)

        next_lane_ids = [
            [internal_connection.get('from') + '_' + internal_connection.get('fromLane')
             for internal_connection in internal_connection_chain]
            for internal_connection_chain, connection in zip(internal_connection_chains, connections)
            if connection.get('from') + '_' + connection.get('fromLane') == lane_id
        ]

        next_lanes = [
            [get_lane(net_xml, next_lane_id) for next_lane_id in lane_id_chain]
            for lane_id_chain in next_lane_ids
        ]
    else:
        next_lane_ids = [
            connection.get('to') + '_' + connection.get('toLane')
            for connection in connections if connection.get('from') + '_' + connection.get('fromLane') == lane_id
        ]

        next_lanes = [get_lane(net_xml, next_lane_id) for next_lane_id in next_lane_ids]

    return next_lanes


@lru_cache(maxsize=None)
def get_internal_edges(net_xml, connection):

    internal_edges = set()

    via_lane = connection.get('via')

    while via_lane is not None:
        edge = net_xml.find('.//edge[@function="internal"]/lane[@id="' + via_lane + '"]..')
        internal_edges.add(edge)

        via_connection = get_from_lane_all_connections_map(net_xml)[via_lane][0]
        via_lane = via_connection.get('via')

    return list(internal_edges)


@lru_cache(maxsize=None)
def get_block_edges(net_xml, edge):

    previous_edges = get_previous_block_edges(net_xml, edge)
    next_edges = get_next_block_edges(net_xml, edge)

    return previous_edges + [edge] + next_edges


@lru_cache(maxsize=None)
def get_intersection_connections(net_xml, intersection_id, multi_intersection_config=None):

    def get_intersections_incoming_edges(net_xml, intersection_id, _sorted=True):

        incoming_edges = set()

        intersection = get_intersection(net_xml, intersection_id)
        incoming_lane_ids = intersection.get('incLanes').split()
        for lane_id in incoming_lane_ids:
            edge_id = lane_id.rsplit('_')[0]
            edge = get_edge(net_xml, edge_id)

            if edge is not None:
                incoming_edges.add(edge)

        if _sorted:
            incoming_edges = sort_edges_by_angle(incoming_edges)
        else:
            incoming_edges = list(incoming_edges)

        return incoming_edges

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    connections = []
    if intersection_id in multi_intersection_config:

        connection_by_edge_id = filter_intersection_incoming_edges(
            net_xml, intersection_id, multi_intersection_config)

        for values in connection_by_edge_id.values():
            connections += values[0]
    else:
        # filter_intersection_incoming_edges uses get_intersection_connections indirectly,
        # so another way is necessary to avoid infinite loop

        incoming_edges = get_intersections_incoming_edges(net_xml, intersection_id)

        for edge in incoming_edges:
            edge_id = edge.get('id')
            edge_connections = get_connection_map(net_xml, FROM_EDGE_MAP)[edge_id]
            connections += edge_connections

    return connections


@lru_cache(maxsize=None)
def get_traffic_light_program_logics(traffic_light_xml, traffic_light_id):
    return traffic_light_xml.findall('.//tlLogic[@id="' + traffic_light_id + '"]')


@lru_cache(maxsize=None)
def get_multi_intersection_map(multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    multi_intersection_map = {
        intersection: k
        for k, v in multi_intersection_config.items()
        for intersection in v['intersections']
    }

    return multi_intersection_map


@lru_cache(maxsize=None)
def get_edge_adjacent_intersections_mapping(net_xml, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    multi_intersection_map = get_multi_intersection_map(multi_intersection_config)

    edge_to_previous_intersection_mapping = {}
    edge_to_next_intersection_mapping = {}

    edge_map = get_edge_map(net_xml)
    for edge_id, edge in edge_map.items():

        current_edges = get_block_edges(net_xml, edge)

        previous_intersection_id = current_edges[0].get('from')
        previous_intersection_id = multi_intersection_map.get(previous_intersection_id, previous_intersection_id)

        edge_to_previous_intersection_mapping[edge_id] = previous_intersection_id

        next_intersection_id = current_edges[-1].get('to')
        next_intersection_id = multi_intersection_map.get(next_intersection_id, next_intersection_id)

        edge_to_next_intersection_mapping[edge_id] = next_intersection_id

    return edge_to_previous_intersection_mapping, edge_to_next_intersection_mapping


@lru_cache(maxsize=None)
def get_edges_by_road_type(net_xml):
    road_type_map = collections.defaultdict(list)
    for edge_id, edge in get_edge_map(net_xml).items():
        road_type_map[edge.get('type')].append(edge_id)
    return road_type_map


def get_connection_map(net_xml, type_=REGULAR_MAP):

    return_type_dict = _get_connection_map(net_xml)

    return return_type_dict[type_]


def get_intersection(net_xml, intersection_id):
    return get_intersection_map(net_xml)[intersection_id]


def get_connection_id(connection):
    return f"{connection.get('from')}_{connection.get('fromLane')}__{connection.get('to')}_{connection.get('toLane')}"


def get_connection_from_edge_to_edge_id(connection):
    return f"{connection.get('from')}__{connection.get('to')}"


def get_connection_from_edge_id(connection):
    return f"{connection.get('from')}"


def get_connection_from_lane_id(connection):
    return f"{connection.get('from')}_{connection.get('fromLane')}"


def get_connection_to_edge_id(connection):
    return f"{connection.get('to')}"


def get_connection_to_lane_id(connection):
    return f"{connection.get('to')}_{connection.get('toLane')}"


def get_connection(net_xml, connection_id):
    return get_connection_map(net_xml)[connection_id]


def get_edge(net_xml, edge_id):

    try:
        return get_edge_map(net_xml)[edge_id]
    except KeyError:
        return get_internal_edge_map(net_xml)[edge_id]


def get_lane(net_xml, lane_id):

    try:
        return get_lane_map(net_xml)[lane_id]
    except KeyError:
        return get_internal_lane_map(net_xml)[lane_id]


def get_internal_lanes(net_xml, connection):

    internal_edges = get_internal_edges(net_xml, connection)

    return get_lanes_from_edges(internal_edges)


def get_lanes_from_edges(edges):
    return [lane for edge in edges for lane in edge]


def get_bus_trips(bus_trips_xml):
    return bus_trips_xml.findall('.//trip')


def get_bus_schedules(bus_trips_xml):

    bus_trips = get_bus_trips(bus_trips_xml)

    bus_schedules = {}
    for trip in bus_trips:
        bus_id = trip.get('id')
        bus_schedule = {}
        for stop in trip:
            bus_stop = stop.get('busStop')
            schedule = stop.get('arrival')
            bus_schedule[bus_stop] = schedule
        bus_schedules[bus_id] = bus_schedule

    return bus_schedules


def get_bus_stop_ids(bus_stop_xml):
    bus_stops = bus_stop_xml.findall('.//busStop')
    bus_stop_ids = [bus_stop.get("id") for bus_stop in bus_stops]

    return bus_stop_ids


def get_edge_id_block_edge_id_map(net_xml):

    edge_id_block_edge_id_map = {}

    edge_map = get_edge_map(net_xml)
    for edge_id, edge in edge_map.items():

        if edge_id in edge_id_block_edge_id_map:
            continue

        block_edges = get_block_edges(net_xml=net_xml, edge=edge)
        block_edge_tuple = tuple(edge.get('id') for edge in block_edges)

        for id_ in block_edge_tuple:
            edge_id_block_edge_id_map[id_] = block_edge_tuple

    return edge_id_block_edge_id_map


def get_intersection_internal_edges(net_xml, intersection_id, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    connections = get_intersection_connections(net_xml, intersection_id, multi_intersection_config)

    internal_edges = set()
    for inner_connections in connections:

        if not isinstance(inner_connections, list):
            inner_connections = [inner_connections]

        for inner_connection in inner_connections:
            internal_edges.update(get_internal_edges(net_xml, inner_connection))

    return list(internal_edges)


def get_multi_intersection_intermediary_edges(net_xml, intersection_id, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    edges = set([])

    if intersection_id not in multi_intersection_config:
        return list(edges)

    connections = get_intersection_connections(net_xml, intersection_id, multi_intersection_config)

    for inner_connections in connections:

        if not isinstance(inner_connections, list):
            inner_connections = [inner_connections]

        for inner_connection in inner_connections[1:]:
            connection_from = inner_connection.get('from')
            from_edge = get_edge(net_xml, connection_from)
            edges.add(from_edge)

    return list(edges)


def get_internal_lane_paths(net_xml, intersection_id, internal_lanes):

    if ',' in intersection_id:
        raise NotImplementedError()

    lanes_by_id = {lane.get('id'): lane for lane in internal_lanes}

    connections = get_intersection_connections(net_xml, intersection_id)

    lane_path = {}
    for connection in connections:
        via_lane = connection.get('via')

        polyline_lane_ids = []
        polylines = []
        while via_lane is not None:

            polyline_lane_ids.append(via_lane)
            polylines.append([])

            shape = lanes_by_id[via_lane].get('shape')

            via_lane_polyline = [list(map(float, point.split(','))) for point in shape.split()]

            for polyline in polylines:

                if polyline:
                    polyline_extension_start_index = 1
                else:
                    polyline_extension_start_index = 0

                polyline.extend(via_lane_polyline[polyline_extension_start_index:])

            via_connection = get_from_lane_all_connections_map(net_xml)[via_lane][0]
            via_lane = via_connection.get('via')

        for index, lane_id in enumerate(polyline_lane_ids):
            lane_path[lane_id] = polylines[index]

    return lane_path


def get_internal_lane_connections(net_xml, intersection_id):

    if ',' in intersection_id:
        raise NotImplementedError()

    connections = get_intersection_connections(net_xml, intersection_id)
    internal_lanes_connections = {}
    for connection in connections:
        via_lane = connection.get('via')

        while via_lane is not None:
            via_connection = get_from_lane_all_connections_map(net_xml)[via_lane][0]
            internal_lanes_connections[via_lane] = via_connection
            via_lane = via_connection.get('via')

    return internal_lanes_connections


def get_intersection_ids(net_xml, sorted_=True):

    intersections = {intersection_id: intersection
                     for intersection_id, intersection in get_intersection_map(net_xml).items()
                     if intersection.get('type') != 'dead_end' and
                     intersection.get('type') != 'internal'}

    intersection_ids = list(intersections.keys())
    if sorted_:
        intersection_ids = []
        intersection_points = []

        for intersection_id, intersection in intersections.items():
            intersection_ids.append(intersection_id)
            intersection_point = Point([float(intersection.get('x')), float(intersection.get('y'))])
            intersection_points.append(intersection_point)

        zipped_id_and_location = zip(intersection_ids, intersection_points)
        sorted_id_and_location = sorted(zipped_id_and_location, key=lambda x: cmp_to_key(location_comparator)(x[1]))

        intersection_ids = list(zip(*sorted_id_and_location))[0]

    return intersection_ids


def get_intersection_edges(net_xml, intersection_id, multi_intersection_config=None, _sorted=True,
                           edge_type='all'):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    if edge_type == 'incoming':
        direction = 'from'
    elif edge_type == 'outgoing':
        direction = 'to'
    elif edge_type == 'all':
        # do 'from' first and then 'to'
        direction = 'from'
    else:
        raise ValueError(f"Unrecognized edge type {edge_type}")

    edges = set()

    connections = get_intersection_connections(net_xml, intersection_id, multi_intersection_config)
    for inner_connections in connections:

        if isinstance(inner_connections, list):
            index = 0 if direction == 'from' else -1
            connection_edge_id = inner_connections[index].get(direction)
        else:
            connection_edge_id = inner_connections.get(direction)

        edge = get_edge(net_xml, connection_edge_id)
        edges.add(edge)

    border_intersection_map = get_border_intersection_map(net_xml)
    if intersection_id in border_intersection_map:
        # from and to is inverted because we are not dealing with connections here
        if direction != 'from':
            edges.update(net_xml.findall(f'.//edge[@from="{intersection_id}"]'))
        if direction != 'to':
            edges.update(net_xml.findall(f'.//edge[@to="{intersection_id}"]'))

    if _sorted:
        incoming = True if direction == 'from' else False
        edges = sort_edges_by_angle(edges, incoming)
    else:
        edges = list(edges)

    if edge_type == 'all':
        edges.extend(
            get_intersection_edges(
                net_xml, intersection_id, multi_intersection_config, _sorted, edge_type='outgoing')
        )

    return edges


def filter_intersection_incoming_edges(net_xml, intersection_id, multi_intersection_config=None):

    def get_incoming_edge_sort_order_function(sorted_incoming_edge_ids):
        def incoming_edge_sort_order(x):

            if x in sorted_incoming_edge_ids:
                result = sorted_incoming_edge_ids.index(x)
            else:
                result = min(map(lambda y: sorted_incoming_edge_ids.index(y), x.split(',')))

            return result

        return incoming_edge_sort_order

    def make_connection_chains(connections):

        def get_connection_chain_ids(connection_chain):
            return [get_connection_id(connection) for connection in connection_chain]

        from_lane_to_connections = collections.defaultdict(list)
        for connection in connections:
            from_lane = connection.get('from') + '_' + connection.get('fromLane')
            from_lane_to_connections[from_lane].append(connection)

        connection_chains = []

        depth_tracking = [copy.deepcopy(list(from_lane_to_connections.values()))]
        connection_tracking = []

        while len(depth_tracking) > 0:

            last_depth = depth_tracking[-1]
            connections = last_depth[0]

            if len(connections):
                connection = connections.pop(0)
                connection_tracking.append(connection)

                to_lane = connection.get('to') + '_' + connection.get('toLane')
                if to_lane in from_lane_to_connections:
                    next_connections = [copy.deepcopy(from_lane_to_connections[to_lane])]
                    depth_tracking.append(next_connections)
                else:
                    connection_chains.append(copy.deepcopy(connection_tracking))
                    connection_tracking.pop()
            else:
                last_depth.pop(0)
                if not last_depth:
                    depth_tracking.pop()
                    if connection_tracking:
                        connection_tracking.pop()

        # remove shorter paths
        connection_chains_to_remove = set()
        for connection_chain_1 in connection_chains:
            for connection_chain_2 in connection_chains:

                if connection_chain_1 == connection_chain_2:
                    continue

                if set(get_connection_chain_ids(connection_chain_1)).issubset(
                        get_connection_chain_ids(connection_chain_2)):
                    connection_chains_to_remove.add(tuple(connection_chain_1))

        for connection_chain_to_remove in connection_chains_to_remove:
            connection_chains.remove(list(connection_chain_to_remove))

        return connection_chains

    def map_connection_chains_to_edge_id(connection_chains):

        connection_chains_by_incoming_edge_id = collections.defaultdict(list)
        for connection_chain in connection_chains:
            edge_id = connection_chain[0].get('from')
            connection_chains_by_incoming_edge_id[edge_id].append(connection_chain)

        return connection_chains_by_incoming_edge_id

    def merge_edges(merge_edges_list, connection_chains_by_edge_id):

        for merge_edges in merge_edges_list:

            # from right to left
            merge_edges = list(reversed(merge_edges))

            merged_edges_id = ','.join(merge_edges)
            for edge_id in merge_edges:
                connection_chains = connection_chains_by_edge_id.pop(edge_id)
                connection_chains_by_edge_id[merged_edges_id].extend(connection_chains)

        return connection_chains_by_edge_id

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    if intersection_id not in multi_intersection_config:

        incoming_edges = get_intersection_edges(
            net_xml, intersection_id, multi_intersection_config, edge_type='incoming')

        incoming_edge_ids = [edge.get('id') for edge in incoming_edges]

        connection_chains_by_edge_id = {
            edge_id: [[connection] for connection in get_connection_map(net_xml, FROM_EDGE_MAP)[edge_id]]
            for edge_id in incoming_edge_ids
        }

        return connection_chains_by_edge_id

    intersection_configuration = multi_intersection_config[intersection_id]

    intersection_ids = intersection_configuration['intersections'] + intersection_configuration['non_coordinated']

    incoming_edges = sort_edges_by_angle([
        incoming_edge
        for intersection_id in intersection_ids
        for incoming_edge in get_intersection_edges(net_xml, intersection_id, edge_type='incoming')
    ])

    original_incoming_edge_ids = [edge.get('id') for edge in incoming_edges]
    connections = [
        connection
        for edge_id in original_incoming_edge_ids
        for connection in get_connection_map(net_xml, FROM_EDGE_MAP)[edge_id]
    ]

    connection_chains = make_connection_chains(connections)
    connection_chains_by_edge_id = map_connection_chains_to_edge_id(connection_chains)

    merge_edges_list = intersection_configuration['merge_edges']
    connection_chains_by_edge_id = merge_edges(merge_edges_list, connection_chains_by_edge_id)

    connection_chains_by_edge_id = dict(sorted(
        connection_chains_by_edge_id.items(),
        key=lambda x: get_incoming_edge_sort_order_function(original_incoming_edge_ids)(x[0])
    ))

    return connection_chains_by_edge_id


def get_intersection_id_to_traffic_light_id_map(net_xml, multi_intersection_config):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    mapping = bidict()

    intersections = {intersection_id: intersection
                     for intersection_id, intersection in get_intersection_map(net_xml).items()
                     if intersection.get('type') == 'traffic_light' or
                     intersection.get('type') == 'traffic_light_right_on_red' or
                     intersection.get('type') == 'traffic_light_unregulated'}

    for intersection_id, intersection in intersections.items():
        traffic_light_id = get_intersection_traffic_light_id(net_xml, intersection_id, multi_intersection_config)
        mapping[intersection_id] = traffic_light_id

    return mapping


def get_traffic_light_id_intersection_id_map(net_xml, multi_intersection_config=None, sorted_=True):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    inter_tl_mapping = get_intersection_id_to_traffic_light_id_map(net_xml, multi_intersection_config)
    tl_inter_mapping = inter_tl_mapping.inverse

    if sorted_:
        intersection_ids = sort_intersections(net_xml, inter_tl_mapping.keys())

        tl_inter_mapping = {
            k: v for k, v in sorted(
                tl_inter_mapping.items(), key=lambda item: intersection_ids.index(item[1][0]))
        }

    return tl_inter_mapping


def get_intersection_traffic_light_id(net_xml, intersection_id, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    tl_set = set()

    connections = get_intersection_connections(net_xml, intersection_id, multi_intersection_config)
    for connection in connections:
        tl = connection.get('tl', None)
        if tl is not None:
            tl_set.add(tl)

    if len(tl_set) == 0:
        return None

    assert len(tl_set) == 1
    traffic_light_id = list(tl_set)[0]

    return traffic_light_id


def get_intersection_connection_requests(net_xml, intersection_id):

    requests = net_xml.findall('.//junction[@id="' + intersection_id + '"]/request')

    connection_requests = {}
    for request in requests:
        request_index = request.get('index')
        connection_requests[int(request_index)] = request

    return connection_requests


def get_intersection_internal_connection_chains(net_xml, intersection_id, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    connections = get_intersection_connections(net_xml, intersection_id, multi_intersection_config)

    internal_connection_chains = []
    for inner_connections in connections:

        if not isinstance(inner_connections, list):
            inner_connections = [inner_connections]

        for inner_connection in inner_connections:
            internal_connections = []

            via_lane = inner_connection.get('via')

            while via_lane is not None:
                via_connection = get_from_lane_all_connections_map(net_xml)[via_lane][0]
                internal_connections.append(via_connection)
                via_lane = via_connection.get('via')

            internal_connection_chains.append(internal_connections)

    return internal_connection_chains


def get_network_border_edges(net_xml):

    dead_end_intersections = net_xml.findall('.//junction[@type="dead_end"]')

    dead_end_intersection_ids = [intersection.get('id') for intersection in dead_end_intersections]

    junction_to_network_entering_edges_mapping = {}
    junction_to_network_exiting_edges_mapping = {}

    for intersection_id in dead_end_intersection_ids:

        scenario_entering_edges = get_intersection_edges(net_xml, intersection_id, edge_type='outgoing')
        for entering_edge in scenario_entering_edges:
            to_intersection = entering_edge.get('to')

            if to_intersection in junction_to_network_entering_edges_mapping:
                junction_to_network_entering_edges_mapping[to_intersection].append(entering_edge)
            else:
                junction_to_network_entering_edges_mapping[to_intersection] = [entering_edge]

        scenario_exiting_edges = get_intersection_edges(net_xml, intersection_id, edge_type='incoming')
        for exiting_edge in scenario_exiting_edges:
            to_intersection = exiting_edge.get('to')

            if to_intersection in junction_to_network_exiting_edges_mapping:
                junction_to_network_exiting_edges_mapping[to_intersection].append(exiting_edge)
            else:
                junction_to_network_exiting_edges_mapping[to_intersection] = [exiting_edge]

    return junction_to_network_entering_edges_mapping, junction_to_network_exiting_edges_mapping


def sort_intersections(net_xml, intersection_ids, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    multi_intersection_map = get_multi_intersection_map(multi_intersection_config)

    intersections = []
    for intersection_id in intersection_ids:

        intersection_id = intersection_id.split(',')[0]
        intersection = get_intersection(net_xml, intersection_id)
        intersections.append(intersection)

    intersection_points = []
    for intersection in intersections:
        intersection_point = Point([float(intersection.get('x')), float(intersection.get('y'))])
        intersection_points.append(intersection_point)

    zipped_id_and_location = zip(intersection_ids, intersection_points)
    sorted_id_and_location = sorted(zipped_id_and_location, key=lambda x: cmp_to_key(location_comparator)(x[1]))

    intersection_ids = list(list(zip(*sorted_id_and_location))[0])

    intersection_ids = [
        multi_intersection_map.get(intersection_id, intersection_id)
        for intersection_id in intersection_ids
    ]

    return intersection_ids


def sort_detector_ids(net_xml, detectors_ids):

    detector_id_tuples = [detectors_id.rsplit('__', 1) for detectors_id in detectors_ids]

    intersection_ids, edge_ids = list(zip(*detector_id_tuples))

    sorted_intersection_ids = sort_intersections(net_xml, intersection_ids)

    detector_id_tuples.sort(key=lambda x: {k: v for v, k in enumerate(sorted_intersection_ids)}[x[0]])

    detectors_ids = ['__'.join(detector_id_tuple) for detector_id_tuple in detector_id_tuples]

    return detectors_ids


def sort_edges_by_angle(edges, incoming=True, clockwise=True):

    edges_and_angles = []
    for edge in edges:
        lane = edge[0]
        polyline = lane.get('shape')
        polyline_points = polyline.split()

        first_point = Point(map(float, polyline_points[0].split(',')))
        last_point = Point(map(float, polyline_points[-1].split(',')))

        if incoming:
            first_point, last_point = last_point, first_point

        normalized_point = Point([last_point.x - first_point.x, last_point.y - first_point.y])

        angle = math.atan2(normalized_point.x, normalized_point.y)

        if angle < 0:
            angle += 2 * math.pi

        edges_and_angles.append([edge, angle])

    reverse = not clockwise

    edges_and_angles.sort(key=lambda x: x[1], reverse=reverse)
    angle_sorted_edges = [edge for edge, _ in edges_and_angles]

    return angle_sorted_edges


def generate_adjacency_graph(net_xml, detector_ids=None, multi_intersection_config=None):

    if multi_intersection_config is None:
        multi_intersection_config = collections_util.HashableDict()

    if detector_ids is None:
        # Consider the intersections as nodes
        raise NotImplementedError

    edge_to_previous_intersection_mapping, edge_to_next_intersection_mapping = (
        get_edge_adjacent_intersections_mapping(net_xml, multi_intersection_config))

    from_lane__via_connection_map = get_from_lane_all_connections_map(net_xml)

    adjacency_graph = collections.defaultdict(list)

    for detector_id in detector_ids:
        intersection_id, edge_id = detector_id.rsplit('__', 1)
        edge = get_edge(net_xml, edge_id)

        incoming_edges = get_intersection_edges(
            net_xml, intersection_id, multi_intersection_config, edge_type='incoming')

        if edge in incoming_edges:
            to_edge_ids = set()
            for lane in edge:
                lane_id = lane.get('id')
                connections = from_lane__via_connection_map[lane_id]

                to_edge_ids.update([connection.get('to') for connection in connections])

            for to_edge_id in to_edge_ids:
                other_detector_id = f"{intersection_id}__{to_edge_id}"
                if other_detector_id in detector_ids:
                    adjacency_graph[detector_id].append(other_detector_id)

            continue

        outgoing_edges = get_intersection_edges(
            net_xml, intersection_id, multi_intersection_config, edge_type='outgoing')

        if edge in outgoing_edges:
            block_edges = get_block_edges(net_xml, edge)
            to_edge = block_edges[-1]
            to_edge_id = to_edge.get('id')

            other_intersection_id = edge_to_next_intersection_mapping[to_edge_id]

            other_detector_id = f"{other_intersection_id}__{to_edge_id}"
            if other_detector_id in detector_ids:
                adjacency_graph[detector_id].append(other_detector_id)

    return adjacency_graph


def get_all_detector_ids(adjacency_graph):
    detector_ids = set()
    for k, vs in adjacency_graph.items():
        for v in vs:
            source, target = k, v
            detector_ids.update({source, target})

    return detector_ids
