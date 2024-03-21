
from datetime import datetime


def convert_human_time_to_seconds(time_):

    timedelta = datetime.strptime(time_, "%H:%M:%S") - datetime(1900, 1, 1)
    seconds = timedelta.total_seconds()

    return seconds
