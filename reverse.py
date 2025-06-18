#!/usr/bin/env python3

from ev3dev2.motor import *
from time import sleep

tank = MoveTank(OUTPUT_A, OUTPUT_C)
def reverse():
    print("Reversing the robot")
    tank.on_for_seconds(left_speed=-30, right_speed=-30, seconds=1)
    sleep(0.5)
    tank.off()
if __name__ == "__main__":
    reverse()
    print("Robot has reversed")