import socket
from ev3dev2.motor import *
from ev3dev2.sound import Sound
from time import sleep

# CODE FOR MANUAL TESTING

HOST = ''  # Listen on all interfaces
PORT = 9999

motor = MoveTank(OUTPUT_A, OUTPUT_C)
port = MediumMotor(OUTPUT_D)
shooter = MediumMotor(OUTPUT_B)
# motor.gyro = GyroSensor()
# motor.gyro.calibrate()

def handle_command(cmd):
    print("Received command: {}".format(cmd))
    if cmd == "forward":
        motor.on(30, 30)
    elif cmd == "reverse":
        motor.on_for_seconds(-25, -25, seconds=.5)
    elif cmd == "catchball":
        port.on_for_seconds(speed=10, seconds=1)
        motor.on_for_seconds(left_speed=10, right_speed=10, seconds=3)
        port.on_for_seconds(speed=-10, seconds=1)
    elif cmd == "slowleft":
        motor.on_for_seconds(-10, 10, seconds=.1)
    elif cmd == "slowright":
        motor.on_for_seconds(10, -10, seconds=.1)
    elif cmd == "slowreverse":
        motor.on(-10, -10)
    elif cmd == "slowforward":
        motor.on(10, 10)
    elif cmd == "left":
        motor.on_for_seconds(-20, 20, seconds=0.5)
    elif cmd == "right":
        motor.on_for_seconds(20, -20, seconds=0.5)
    elif cmd == "stop":
        motor.off()
        port.off()
        shooter.off()
    elif cmd == "dropoff":
        port.on_for_seconds(speed=10, seconds=1)
        shooter.on_for_seconds(speed=-30, seconds=0.5)
        sleep(0.5)
        shooter.on_for_seconds(speed=30, seconds=0.5)
        port.on_for_seconds(speed=-10, seconds=1)
    elif cmd == "kick":
        shooter.on_for_seconds(speed=-30, seconds=0.5)
        sleep(0.5)
        shooter.on_for_seconds(speed=30, seconds=0.5)

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
                exit()
            handle_command(data)

# ACTUAL CODE FOR MOVING TO BALLS

# def move_to_target(angle, seconds):

#     print("Moving to target with angle: {} degrees".format(angle))

#     spkr = Sound()

#     # spkr.set_volume(100)
#     # spkr.speak("Moving to target")

#     tank = MoveTank(OUTPUT_A, OUTPUT_C)
#     tank.gyro = GyroSensor()
#     tank.gyro.calibrate()

#     port = MediumMotor(OUTPUT_D)
#     shooter = MediumMotor(OUTPUT_B)
    
#     tank.turn_degrees(speed=SpeedPercent(5), target_angle=angle)

#     tank.on_for_seconds(left_speed=30, right_speed=30, seconds=seconds)

#     port.on_for_seconds(speed=10, seconds=1)
#     tank.on_for_seconds(left_speed=10, right_speed=10, seconds=2)
#     port.on_for_seconds(speed=-10, seconds=1)

#     exit()

# # Safely parse command-line args
# if len(sys.argv) < 3:
#     print("Usage: python3 update_robot_position.py <x> <y>")
#     sys.exit(1)

# try:
#     angle = int(sys.argv[1])
#     seconds = float(sys.argv[2])
#     move_to_target(angle, seconds)
# except ValueError:
#     print("Invalid coordinates. Must be integers.")
