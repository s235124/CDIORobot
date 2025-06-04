import socket
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.motor import *
from time import sleep

HOST = ''  # Listen on all interfaces
PORT = 9999

motor = MoveTank(OUTPUT_A, OUTPUT_C)
port = MediumMotor(OUTPUT_B)
shooter = MediumMotor(OUTPUT_D)
motor.gyro = GyroSensor()
motor.gyro.calibrate()

def handle_command(cmd):
    print("Received command: {}".format(cmd))
    if cmd == "forward":
        motor.on(30, 30)
    elif cmd == "reverse":
        motor.on(-30, -30)
    elif cmd == "left":
        motor.on(-20, 20)
    elif cmd == "right":
        motor.on(20, -20)
    elif cmd == "stop":
        motor.off()
        port.off()
        shooter.off()
    elif cmd == "portopen":
        port.on_for_seconds(speed=10, seconds=1)
    elif cmd == "portclose":
        port.on_for_seconds(speed=-10, seconds=1)
    elif cmd == "kick":
        shooter.on_for_seconds(speed=30, seconds=1)
        sleep(0.5)
        shooter.on_for_seconds(speed=-30, seconds=1)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print("Waiting for connection...")
    conn, addr = s.accept()
    print("Connected by {}".format(addr))
    with conn:
        while True:
            data = conn.recv(1024).decode().strip()
            if not data:
                break
            handle_command(data)



# import sys
# from ev3dev2.motor import MediumMotor, MoveTank, MoveSteering, OUTPUT_A, OUTPUT_C, SpeedPercent, OUTPUT_B, OUTPUT_D

# tank = MoveTank(OUTPUT_A, OUTPUT_C)
# tank.gyro = GyroSensor()
# tank.gyro.calibrate()

# def forward():
#     tank.on_for_seconds(left_speed=40, right_speed=40, seconds=1)

# def reverse():
#     tank.on_for_seconds(left_speed=-40, right_speed=-40, seconds=1)

# def left():
#     tank.on_for_seconds(left_speed=-40, right_speed=40, seconds=1)

# def right():
#     tank.on_for_seconds(left_speed=40, right_speed=-40, seconds=1)

# def stop():
#     tank.off()
#     exit()

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         command = sys.argv[1]
#         if command == 'forward':
#             forward()
#         elif command == 'reverse':
#             reverse()
#         elif command == 'left':
#             left()
#         elif command == 'right':
#             right()
#         elif command == 'stop':
#             stop()



# FOR OLD VERSION
# def move_to_target(x, y):
#     angle = 90 # <-- This should be part of arg
#     secondsToTurn = 1 # Need to figure out seconds,  also part of args

#     print("Moving to target: {}, {}".format(x, y))
#     tank = MoveTank(OUTPUT_A, OUTPUT_C)
#     tank.gyro = GyroSensor()
#     tank.gyro.calibrate()

#     port = MediumMotor(OUTPUT_B)
#     shooter = MediumMotor(OUTPUT_D)

#     port.on_for_seconds(speed=10, seconds=1)
#     tank.on_for_seconds(left_speed=30, right_speed=30, seconds=secondsToTurn)
#     port.on_for_seconds(speed=-10, seconds=1)

#     tank.turn_degrees(speed=SpeedPercent(20), target_angle=angle)
#     port.on_for_seconds(speed=10, seconds=1)
#     shooter.on_for_seconds(speed=10, seconds=1)
#     shooter.on_for_seconds(speed=-10, seconds=1)
#     port.on_for_seconds(speed=-10, seconds=1)

#     exit()

# # Safely parse command-line args
# if len(sys.argv) < 3:
#     print("Usage: python3 update_robot_position.py <x> <y>")
#     sys.exit(1)

# try:
#     target_x = int(sys.argv[1])
#     target_y = int(sys.argv[2])
#     move_to_target(target_x, target_y)
# except ValueError:
#     print("Invalid coordinates. Must be integers.")
