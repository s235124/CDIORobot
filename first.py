#!/usr/bin/env python3

from ev3dev2.motor import *
from ev3dev2.sound import *
from ev3dev2.button import *
import os
import time
import paramiko

os.system('setfont Lat15-TerminusBold14')

button = Button()

spkr = Sound()
shovelMotor = MediumMotor(OUTPUT_D); shovelMotor.stop_action = 'hold'

steer_pair = MoveTank(OUTPUT_B, OUTPUT_C)

print('PROGRAM STARTED')

def pickBallUp():
    print("\npicking up ball")
    shovelMotor.on_for_seconds(speed=-100, seconds=1)
    # spkr.play_file(wav_file="/home/robot/Gruppe3CDIOPython/ballPickupSound.wav", volume=50, play_type=Sound.PLAY_NO_WAIT_FOR_COMPLETE) # goofing off, unnecessary
    steer_pair.on_for_rotations(left_speed=30, right_speed=30, rotations=1)
    shovelMotor.on_for_seconds(speed=100, seconds=1)

def move_to_target(target_x, target_y):
    # Implement logic to move towards the target
    pass

# Setup SSH client
def listen_for_commands():
    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect('<EV3-IP-address>', username='robot', password='ev3dev')

    # Continuously listen for new target coordinates
    while True:
        stdin, stdout, stderr = ssh_client.exec_command('python3 get_target_coords.py')
        target_x, target_y = stdout.read().decode().split(',')
        move_to_target(int(target_x), int(target_y))
        time.sleep(0.5)  # Update interval for new target coordinates
    
listen_for_commands()

while True:
    if button.up:
        shovelMotor.on_for_seconds(speed=80, seconds=1)
    elif button.down:
        shovelMotor.on_for_seconds(speed=-80, seconds=1)
    elif button.right:
        pickBallUp()
    elif button.left:
        exit()
