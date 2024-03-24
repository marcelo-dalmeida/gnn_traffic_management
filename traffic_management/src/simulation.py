import argparse
import os
import sys
from pathlib import Path


os.environ['SUMO_HOME'] = '/usr/share/sumo'

os.environ['SUMO_HOME'] = f'{Path.home()}/code/sumo'

os.environ['LIBSUMO_AS_TRACI'] = '1'
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


from pipeline.pipeline import Pipeline


def run():

    pipeline = Pipeline()
    pipeline.generate_dataset()
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='simulation',
        description='Run the simulation')

    parser.add_argument('--run', default=True)
    args = parser.parse_args()

    if args.run:
        run()
