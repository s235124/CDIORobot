#!/usr/bin/env python3

from ev3dev2.motor import *
from ev3dev2.sound import *
from ev3dev2.button import *
from ev3dev2.sensor.lego import InfraredSensor
from ev3dev2.sensor.lego import GyroSensor
import os
from time import sleep
# import paramiko

os.system('setfont Lat15-TerminusBold14')

button = Button()

spkr = Sound()

print("STARTED")
angle = 90 # <-- This should be part of arg
secondsToTurn = 1 # Need to figure out seconds

tank_drive = MoveTank(OUTPUT_A, OUTPUT_C)
tank_drive.gyro = GyroSensor()
tank_drive.gyro.calibrate()

port = MediumMotor(OUTPUT_B)
shooter = MediumMotor(OUTPUT_D)

ir = InfraredSensor()
ir.mode = 'IR-REMOTE'

def top_left_channel_1_action(state):

    if state: # state is True (pressed) or False
        tank_drive.on(left_speed=40, right_speed=40)
    else:
        tank_drive.off()

def bottom_left_channel_1_action(state):

    if state:
        tank_drive.on(left_speed=-40, right_speed=-40)
    else:
        tank_drive.off()

def top_right_channel_1_action(state):

    if state:
        port.on(speed=30)
    else:
        port.off()

def bottom_right_channel_1_action(state):

    if state:
        port.on(speed=-30)
    else:
        port.off()

# Associate the event handlers with the functions defined above

ir.on_channel1_top_left = top_left_channel_1_action
ir.on_channel1_bottom_left = bottom_left_channel_1_action
ir.on_channel1_top_right = top_right_channel_1_action
ir.on_channel1_bottom_right = bottom_right_channel_1_action

print("Press the buttons on the remote to control the motors.")

while True:
    buttonsPressed = ir.buttons_pressed(channel=1)
    print("Buttons pressed:", buttonsPressed)
    sleep(0.5)

# while True:
#     ir.process()
#     sleep(0.01)

# For testing using buttons on the EV3 brick
# while True:

#     if button.up:
#         port.on_for_seconds(speed=100, seconds=.18)
#         time.sleep(1)
#         tank.on_for_seconds(left_speed=30, right_speed=30, seconds=secondsToTurn)
#         port.on_for_seconds(speed=-100, seconds=.18)
#     elif button.down:
#         shooter.on_for_seconds(speed=10, seconds=1)
#         time.sleep(1)
#         shooter.on_for_seconds(speed=-10, seconds=1)
#     elif button.right:
#         port.on_for_seconds(speed=20, seconds=.5)
#         time.sleep(.25)
#         shooter.on_for_seconds(speed=20, seconds=.5)
#         shooter.on_for_seconds(speed=-20, seconds=.5)
#         port.on_for_seconds(speed=-20, seconds=.5)
#     elif button.left:
#         tank.on_for_seconds(left_speed=30, right_speed=30, seconds=secondsToTurn)
#         time.sleep(1)
#         tank.on_for_seconds(left_speed=-30, right_speed=-30, seconds=secondsToTurn)
