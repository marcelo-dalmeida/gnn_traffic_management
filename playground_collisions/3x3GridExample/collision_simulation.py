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

    # Parameters for non-accident related anomalous driving
    #   Scenario 1: Drivers slowing down below the speed limit once randomly for a random amount of time
    #   Scenario 2: Every 10 seconds, choose x% of the cars to slow down for 5 seconds
    '''# Scenario 1 parameters
    percentage_slowed = 0.4     # percentage of the cars present at slowdown time that will slow down
    slowdown_min = 15           # minimum amount of time slowed down
    slowdown_max = 20           # maximum amount of time slowed down
    slowdown_start = 200        # First step that people could slow down
    slowdown_stop = 300         # Last step people can slow down
    slowdown_steps = []         # list of timesteps to select a random vehicle to slow down'''
    # Scenario 2 parameters
    percentage_slowed = 0.1     # percentage of the cars present at slowdown time that will slow down
    slowdown_min = 5            # minimum amount of time slowed down
    slowdown_max = 5            # maximum amount of time slowed down

    # Main simulation loop
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # Stop the vehicle, enlarge it to take up multiple lanes, and move it to the middle of the road
        '''if step == 5:
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
            print("Accident cleared", accident_ID)'''

        # Non-accident related anomalous driving scenarios
        '''# Scenario 1
        if step == slowdown_start:
            for i in range(round(percentage_slowed * len(traci.vehicle.getIDList()))):
                new_step = randint(slowdown_start, slowdown_stop)
                slowdown_steps.append(new_step)
        if step in slowdown_steps:
            for i in range(slowdown_steps.count(step)):  # Repeat if this step was counted multiple times
                veh_id_list = traci.vehicle.getIDList()
                slow_id = str(veh_id_list[randint(0, len(veh_id_list)-1)])
                current_lane = traci.vehicle.getLaneID(slow_id)
                speed_limit = traci.lane.getMaxSpeed(current_lane)
                traci.vehicle.slowDown(slow_id, 0.01 * speed_limit, randint(slowdown_min, slowdown_max))'''

        # Scenario 2 (Work in progress, need to find a car stopped at a light efficiently)
        if step % 10 == 0:
            veh_id_list = traci.vehicle.getIDList()
            for i in range(round(percentage_slowed * len(veh_id_list))):  # Repeat if this step was counted multiple times
                slow_id = str(veh_id_list[randint(0, len(veh_id_list)-1)])
                current_lane = traci.vehicle.getLaneID(slow_id)
                speed_limit = traci.lane.getMaxSpeed(current_lane)
                traci.vehicle.slowDown(slow_id, 0.01 * speed_limit, randint(slowdown_min, slowdown_max))

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