import os
import sys
import optparse
from random import randint

# VERIFY THAT SUMO IS INSTALLED AND PATH IS SET UP
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
else:
    sys.exit("ERROR: Please declare SUMO_HOME in your environment variables.")

from sumolib import checkBinary
import traci

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of SUMO")
    options, args = opt_parser.parse_args()
    return options

# TraCI Control Loop
def run():
    # Sets all vehicles to have a random reaction to bluelight vehicles between 1 and 10 steps
    for vehID in traci.vehicle.getIDList():
        traci.vehicle.setParameter(vehID, "device.bluelight.reactiondist", str(randint(1, 10)))

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # Cause an "accident" at time step 100 on E11
        if step == 100:
            veh_id_list = traci.edge.getLastStepVehicleIDs("E11")
            accident_ID = veh_id_list[0]
            traci.vehicle.setSpeed(str(accident_ID), 0.0)
            print("STOP", accident_ID)

        if step == 150:
            traci.vehicle.remove(str(accident_ID), 3)
            print("POOF", accident_ID)

        # Log all accidents that happen in the console
        if traci.simulation.getCollidingVehiclesNumber() > 0:
            for v in traci.simulation.getCollidingVehiclesIDList():
                print("Step:", step, "- VehicleID:", v)

    traci.close()
    sys.stdout.flush()
    pass

# Main
if __name__ == "__main__":
    options = get_options()
    FILENAME = "sim.sumocfg"

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else: sumoBinary = checkBinary('sumo-gui')

    cmd =   [   sumoBinary,
                "-c",
                FILENAME,
                "--tripinfo-output",
                "data/tripinfo.xml"
            ]

    # TraCI starts sumo as a subprocess and then this script connects and runs
    traci.start(cmd)
    print(traci.getVersion())
    run()