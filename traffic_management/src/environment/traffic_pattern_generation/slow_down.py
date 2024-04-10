import os

import config
from utils import xml_util
from utils.sumo import sumo_traci_util


class SlowDown:

    def __init__(self, slow_down_id, vehicle_id, duration,
                 randomizer, data_subscription, execution_name):

        net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_SCENARIO, config.SCENARIO.NET_FILE)
        self._net_xml = xml_util.parse_xml(net_file)

        self.slow_down_id = slow_down_id
        self.vehicle_id = vehicle_id
        self.duration = duration

        self._randomizer = randomizer
        self._data_subscription = data_subscription
        self._traci = sumo_traci_util.get_traci_connection(execution_name)

        self._start()

    def _start(self):
        self._current_timer = 0

        self._traci.vehicle.slowDown(self.vehicle_id, 0.0, self.duration)

    def step(self):
        self._current_timer += 1

        if self._current_timer >= self.duration:
            self.clear()
            return False

        return True

    def clear(self):
        pass
