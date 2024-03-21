import os

import pandas as pd

import config
from environment.environment import Environment
from utils import xml_util, datetime_util


def generate_simulation_state(time_, env=None):

    seconds = datetime_util.convert_human_time_to_seconds(time_)

    path = os.path.join(config.PATH_TO_DATA, 'environment_state')

    environment_state_path = os.path.join(config.ROOT_DIR, path)

    try:
        save_state_files = next(os.walk(environment_state_path))[2]
        save_state_files.sort(key=lambda x: x.split(f'{config.EXPERIMENT.SCENARIO_NAME}_save_state_')[1].split('.')[0])

        time_ = None
        file_name = None
        for save_state_file in save_state_files:
            save_state_time = int(save_state_file.split(f'{config.EXPERIMENT.SCENARIO_NAME}_save_state_')[1].split('.')[0])

            if save_state_time == seconds:
                return

            if save_state_time > seconds:
                break

            file_name = save_state_file
            time_ = save_state_time

        if file_name is None:
            pass

    except StopIteration:
        time_ = None
        file_name = None

    if env is None:
        env = Environment(evaluate_metrics=False)
    else:
        env.setup(evaluate_metrics=False)

    env.reset()
    env.start(with_gui=False)

    step = 0

    if file_name:
        env.load_simulation_state(time_, full_path=os.path.join(environment_state_path, file_name))
        step = time_

    while step < seconds:
        env.step()
        step += 1

    path = os.path.join(config.PATH_TO_DATA, 'environment_state')

    env.save_simulation_state(name=config.EXPERIMENT.SCENARIO_NAME, path=path)

    env.end()


def get_simulation_state_path(time_):

    seconds = datetime_util.convert_human_time_to_seconds(time_)

    path = os.path.join(config.PATH_TO_DATA, 'environment_state')

    filename = f'{config.EXPERIMENT.SCENARIO_NAME}_save_state_{seconds}.0.xml'

    filepath = os.path.join(path, filename)

    if os.path.isfile(os.path.join(config.ROOT_DIR, filepath)):
        return filepath
    else:
        return None


def generate_filtered_state_files(time_, car_trips_file, bus_trips_file, passenger_trips_file):

    seconds = datetime_util.convert_human_time_to_seconds(time_)

    initial_period = seconds - 1*60*60

    car_trips_xml = xml_util.parse_xml(car_trips_file)
    bus_trips_xml = xml_util.parse_xml(bus_trips_file)
    passenger_trips_xml = xml_util.parse_xml(passenger_trips_file)

    car_flows = car_trips_xml.findall('//flow')
    for car_flow in car_flows:

        begin = float(car_flow.get('begin'))
        end = float(car_flow.get('end'))

        if end < initial_period:
            car_flow.getparent().remove(car_flow)
        elif begin < initial_period:
            car_flow.set('begin', str(initial_period))
            number = int(car_flow.get('number'))
            new_number = round(number/(end - begin)*(end - initial_period))
            car_flow.set('number', str(new_number))
        else:
            break

    removed_buses = []
    bus_trips = bus_trips_xml.findall('//trip')
    for bus_trip in bus_trips:

        depart = datetime_util.convert_human_time_to_seconds(bus_trip.get('depart'))

        if depart < initial_period:
            bus_trip.getparent().remove(bus_trip)
            removed_buses.append(bus_trip.get('id'))
        else:
            break

    passenger_trips = passenger_trips_xml.findall('//personFlow')
    for passenger_trip in passenger_trips:

        begin = passenger_trip.get('begin')

        if begin == 'triggered':
            line = passenger_trip[0].get('lines')
            if line in removed_buses:
                passenger_trip.getparent().remove(passenger_trip)
        else:
            begin = datetime_util.convert_human_time_to_seconds(time_)

            if begin < initial_period:
                passenger_trip.getparent().remove(passenger_trip)
            else:
                break

    car_trips_xml.write(car_trips_file, pretty_print=True)
    bus_trips_xml.write(bus_trips_file, pretty_print=True)
    passenger_trips_xml.write(passenger_trips_file, pretty_print=True)


def prepare_warmup():
    simulation_warmup = False
    if config.EXPERIMENT.TIME != "00:00:00":
        car_flows_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.CAR_TRIPS_FILE)
        bus_trips_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.BUS_TRIPS_FILE)
        passenger_trips_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.PASSENGER_TRIPS_FILE)

        generate_filtered_state_files(
            config.EXPERIMENT.TIME, car_flows_file, bus_trips_file, passenger_trips_file)

        simulation_warmup = True

    return simulation_warmup


def find_best_round():

    path_to_summary = os.path.join(config.ROOT_DIR, config.PATH_TO_SUMMARY)
    name_base = f"{config.EXPERIMENT.NAME.rsplit('/', 1)[1]}-test"

    file = f"{path_to_summary}/{name_base}-mean_time_loss.csv"

    if not os.path.isfile(file):
        raise ValueError("Couldn't find the best route, summary file is missing")

    metric_df = pd.read_csv(file, header=None)

    best_round = metric_df.idxmin()[0]
    best_value = metric_df.loc[best_round][0]

    file = f"{path_to_summary}/{name_base}-mean_time_loss-original.csv"
    if os.path.isfile(file):
        metric_df = pd.read_csv(file, header=None)

        local_metric_df = metric_df.loc[best_round - 4:best_round + 5]
        closest_round = (local_metric_df - best_value).abs().sort_values(by=metric_df.columns[0], kind='mergesort')\
            .index[0]
        best_round = closest_round

    return best_round


def find_last_round():

    test_dir = os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, "test_round")

    try:
        round_folders = next(os.walk(test_dir))[1]
        round_folders.sort(key=lambda x: int(x.split('_')[1]))
        last_round = round_folders[-1]
        round_ = int(last_round.split('_')[1])
    except StopIteration:
        round_ = 0

    return round_
