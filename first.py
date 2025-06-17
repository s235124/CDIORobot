#!/usr/bin/env python3

from ev3dev2.motor import *
from time import sleep

shooter = MediumMotor(OUTPUT_B)

shooter.on_for_seconds(speed=-30, seconds=0.5)
sleep(0.5)
shooter.on_for_seconds(speed=30, seconds=0.5)