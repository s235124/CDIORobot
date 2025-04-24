import sys
from ev3dev2.motor import MediumMotor, MoveTank, MoveSteering, OUTPUT_A, OUTPUT_C, SpeedPercent, OUTPUT_B, OUTPUT_D
from ev3dev2.sensor.lego import GyroSensor

def move_to_target(x, y):
    angle = 90 # <-- This should be part of arg
    secondsToTurn = 1 # Need to figure out seconds

    print("Moving to target: {}, {}".format(x, y))
    tank = MoveTank(OUTPUT_A, OUTPUT_C)
    tank.gyro = GyroSensor()
    tank.gyro.calibrate()

    port = MediumMotor(OUTPUT_B)
    shooter = MediumMotor(OUTPUT_D)

    port.on_for_seconds(speed=10, seconds=1)
    tank.on_for_seconds(left_speed=30, right_speed=30, seconds=secondsToTurn)
    port.on_for_seconds(speed=-10, seconds=1)

    tank.turn_degrees(speed=SpeedPercent(20), target_angle=angle)
    port.on_for_seconds(speed=10, seconds=1)
    shooter.on_for_seconds(speed=10, seconds=1)
    shooter.on_for_seconds(speed=-10, seconds=1)
    port.on_for_seconds(speed=-10, seconds=1)

    exit()

# Safely parse command-line args
if len(sys.argv) < 3:
    print("Usage: python3 update_robot_position.py <x> <y>")
    sys.exit(1)

try:
    target_x = int(sys.argv[1])
    target_y = int(sys.argv[2])
    move_to_target(target_x, target_y)
except ValueError:
    print("Invalid coordinates. Must be integers.")
