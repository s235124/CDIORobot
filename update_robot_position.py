import socket
from ev3dev2.motor import *
from ev3dev2.sound import Sound
from time import sleep


HOST = ''  # Listen on all interfaces
PORT = 9999

motor = MoveTank(OUTPUT_A, OUTPUT_C)
port = MediumMotor(OUTPUT_D)
shooter = MediumMotor(OUTPUT_B)

KICKSECS = 0.23

def handle_command(cmd):
    print("Received command: {}".format(cmd))
    if cmd == "forward":
        motor.on(30, 30)
    elif cmd == "reverse":
        motor.on_for_seconds(-25, -25, seconds=.5)
    elif cmd == "catchball":
        port.on_for_seconds(speed=15, seconds=1)
        motor.on_for_seconds(left_speed=10, right_speed=10, seconds=2)
        port.on_for_seconds(speed=-10, seconds=1)
    elif cmd == "slowleft":
        motor.on_for_seconds(-10, 10, seconds=.1)
    elif cmd == "slowright":
        motor.on_for_seconds(10, -10, seconds=.1)
    elif cmd == "left":
        motor.on_for_seconds(-15, 15, seconds=.2)
    elif cmd == "right":
        motor.on_for_seconds(15, -15, seconds=.2)
    elif cmd == "slowreverse":
        motor.on(-10, -10)
    elif cmd == "slowforward":
        motor.on(10, 10)
    elif cmd == "fastleft":
        motor.on_for_seconds(-20, 20, seconds=0.5)
    elif cmd == "fastright":
        motor.on_for_seconds(20, -20, seconds=0.5)
    elif cmd == "stop":
        motor.off()
        port.off()
        shooter.off()
    elif cmd == "dropoff":
        port.on_for_seconds(speed=10, seconds=1)
        shooter.on_for_seconds(speed=-30, seconds=0.23)
        shooter.on_for_seconds(speed=30, seconds=0.3)
        shooter.on_for_seconds(speed=-30, seconds=0.23)
        shooter.on_for_seconds(speed=30, seconds=0.3)
        shooter.on_for_seconds(speed=-30, seconds=0.23)
        shooter.on_for_seconds(speed=30, seconds=0.3)
        shooter.on_for_seconds(speed=-30, seconds=0.23)
        shooter.on_for_seconds(speed=30, seconds=0.3)
        shooter.on_for_seconds(speed=-30, seconds=0.23)
        shooter.on_for_seconds(speed=30, seconds=0.3)
        shooter.on_for_seconds(speed=-30, seconds=0.23)
        shooter.on_for_seconds(speed=30, seconds=0.3)
        shooter.on_for_seconds(speed=-30, seconds=0.23)
        shooter.on_for_seconds(speed=30, seconds=0.3)
        port.on_for_seconds(speed=-10, seconds=1)
        motor.on_for_seconds(-25, -25, seconds=.2)
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
