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

    # Count the number of lanes
    #print(traci.edge.getLaneNumber("-E9"))
    #print(traci.edge.getLaneNumber("E9"))

    # Main simulation loop
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # Stop the vehicle, enlarge it to take up multiple lanes, and move it to the middle of the road
        if step == 5:
            ### Stop the vehicle in place and enlarge it ###
            veh_id_list = traci.edge.getLastStepVehicleIDs("-E9")
            accident_ID = str(veh_id_list[0])
            traci.vehicle.setWidth(accident_ID, 7.2)
            traci.vehicle.setSpeed(accident_ID, 0.0)

            ### Move the vehicle to the middle of the road (WIP) ###
            # Calculate the how far the car needs to move to get to the center of the road
            x, y = traci.vehicle.getPosition(accident_ID)
            print(x, y)
            lane_0_shape = traci.lane.getShape("-E9_0")
            print(lane_0_shape)
            dY = (lane_0_shape[0][0] - lane_0_shape[1][0]) / 2.0
            dX = (lane_0_shape[0][1] - lane_0_shape[1][1]) / 2.0
            print(dX)
            print(dY)
            # Move the vehicle to the middle of the edge by going to the midpoint of the edge by width
            traci.vehicle.moveToXY(vehID=accident_ID, edgeID="-E9", lane=0, x=x+dX, y=y+dY, angle=0, keepRoute=0)

            # Have the vehicle commit to the lane that it is currently in
            # This is needed to prevent the vehicle from weaving back and forth across the edge
            traci.vehicle.changeLaneRelative(accident_ID, 0, 100)
        if step == 15:
            # Clear the accident 10 steps later
            print(traci.vehicle.getPosition(accident_ID))
            traci.vehicle.remove(accident_ID, 3)
            print("Accident cleared", accident_ID)

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