import socket
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.motor import *
from ev3dev2.sound import Sound
from time import sleep

# CODE FOR MANUAL TESTING

# HOST = ''  # Listen on all interfaces
# PORT = 9999

# motor = MoveTank(OUTPUT_A, OUTPUT_C)
# port = MediumMotor(OUTPUT_D)
# shooter = MediumMotor(OUTPUT_B)
# motor.gyro = GyroSensor()
# motor.gyro.calibrate()

# def handle_command(cmd):
#     print("Received command: {}".format(cmd))
#     if cmd == "forward":
#         motor.on(30, 30)
#     elif cmd == "reverse":
#         motor.on(-30, -30)
#     elif cmd == "slowreverse":
#         motor.on(-10, -10)
#     elif cmd == "slowforward":
#         motor.on(10, 10)
#         port.on_for_seconds(speed=-10, seconds=1)
#     elif cmd == "left":
#         motor.on(-20, 20)
#     elif cmd == "right":
#         motor.on(20, -20)
#     elif cmd == "stop":
#         motor.off()
#         port.off()
#         shooter.off()
#     elif cmd == "portopen":
#         port.on_for_seconds(speed=10, seconds=1)
#     elif cmd == "portclose":
#         port.on_for_seconds(speed=-10, seconds=1)
#     elif cmd == "onemeter":
#         motor.on_for_seconds(left_speed=80, right_speed=80, seconds=4)
#     elif cmd == "kick":
#         shooter.on_for_seconds(speed=-30, seconds=0.5)
#         sleep(0.5)
#         shooter.on_for_seconds(speed=30, seconds=0.5)

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen(1)
#     print("Waiting for connection...")
#     conn, addr = s.accept()
#     print("Connected by {}".format(addr))
#     with conn:
#         while True:
#             data = conn.recv(1024).decode().strip()
#             if not data:
#                 exit()
#             handle_command(data)

# ACTUAL CODE FOR MOVING TO BALLS

def move_to_target(angle, seconds):

    print("Moving to target with angle: {} degrees and {} seconds".format(angle, seconds))

    spkr = Sound()

    spkr.set_volume(100)
    spkr.speak("Moving to target with angle: {} degrees and seconds: {} seconds".format(angle, seconds))

    tank = MoveTank(OUTPUT_A, OUTPUT_C)
    tank.gyro = GyroSensor()
    tank.gyro.calibrate()

    port = MediumMotor(OUTPUT_D)
    shooter = MediumMotor(OUTPUT_B)
    
    tank.turn_degrees(speed=SpeedPercent(20), target_angle=angle)

    # tank.on_for_seconds(left_speed=30, right_speed=30, seconds=seconds)

    # port.on_for_seconds(speed=10, seconds=1)
    # tank.on_for_seconds(left_speed=30, right_speed=30, seconds=1)
    # port.on_for_seconds(speed=-10, seconds=1)

    exit()

# Safely parse command-line args
if len(sys.argv) < 3:
    print("Usage: python3 update_robot_position.py <x> <y>")
    sys.exit(1)

try:
    angle = int(sys.argv[1])
    seconds = int(sys.argv[2])
    move_to_target(angle, seconds)
except ValueError:
    print("Invalid coordinates. Must be integers.")
