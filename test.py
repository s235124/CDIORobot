# import paramiko
# import time
# import cv2
# import numpy as np
# import math

# IPADDRESS = '169.254.187.202' # REMEMBER TO UPDATE THIS

# def send_coordinates_to_ev3(target_x, target_y):
#     # Connect to the EV3 via SSH
#     ssh_client = paramiko.SSHClient()
#     ssh_client.load_system_host_keys()
#     ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh_client.connect(IPADDRESS, username='robot', password='maker')

#     # Command to update robot's movement based on the target coordinates
#     command = f'python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py {target_x} {target_y}'
#     stdin, stdout, stderr = ssh_client.exec_command(command)
    
#     # Print output or errors (for debugging)
#     print("Sending coordinates:", target_x, target_y)
#     print("STDOUT:", stdout.read().decode())
#     print("STDERR:", stderr.read().decode())

    
#     ssh_client.close()

# def listen_for_coordinates_and_update():
#     while True:
#         # Capture image and detect object
#         calculate_angle_and_distance(340,240,350,500)
#         coords = capture_and_find_object()
#         if coords:
#             target_x, target_y = coords
#             send_coordinates_to_ev3(target_x, target_y)
        
#         time.sleep(1)  # Sleep before checking for the next image

# def find_object_coordinates(image):
#     # Example processing logic to find an object (you'll need to adjust this)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
#     # Find contours of the object
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     for contour in contours:
#         if cv2.contourArea(contour) > 100:  # Avoid small noise areas
#             # Find the center of the object
#             M = cv2.moments(contour)
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 return (cx, cy)
#     return None  # If no object detected

# def capture_and_find_object():
#     # Capture image from the webcam (adjust to your actual camera)
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     cap.release()

#     if ret:
#         object_coords = find_object_coordinates(frame)
#         if object_coords:
#             print("Object detected at:", object_coords)
#             return object_coords
#         else:
#             print("No object detected.")
#     return None

# def calculate_angle_and_distance(robotx, roboty, ballx, bally):
#     height = bally - roboty
#     length = ballx - robotx

#     angle_rad = math.atan2(height, length)
#     angle_deg = math.degrees(angle_rad)
    
#     print(angle_deg)

#     distance = np.sqrt((height**2) + (length**2))

#     return angle_deg, distance

# if __name__ == "__main__":
#     listen_for_coordinates_and_update()




# import paramiko
# from pynput import keyboard

# # Connect to the EV3 brick via SSH
# def connect_to_ev3():
#     ev3_ip = IPADDRESS  # Replace with your EV3's IP address
#     ev3_user = 'robot'
#     ev3_password = 'maker'
    
#     print("Attempting Connecting to EV3 at", IPADDRESS)

#     # Start SSH session
#     ssh_client = paramiko.SSHClient()
#     ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh_client.connect(IPADDRESS, username=ev3_user, password=ev3_password)
    
#     print("Connected to EV3 at", IPADDRESS)

#     return ssh_client

# # Send command to run the motor on the EV3
# def control_motor(command, ssh_client):
#     # Example for controlling the motor (e.g., output D)
#     if command == 'w':  # Forward (e.g., motor forward)
#         ssh_client.exec_command('python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py forward')
#     elif command == 's':  # Reverse (e.g., motor reverse)
#         ssh_client.exec_command('python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py reverse')
#     elif command == 'a':  # Turn Left
#         ssh_client.exec_command('python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py left')
#     elif command == 'd':  # Turn Right
#         ssh_client.exec_command('python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py right')
#     elif command == 'q':  # Stop
#         ssh_client.exec_command('python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py stop')

# # This will handle keyboard inputs
# def on_press(key, ssh_client):
#     try:
#         # Print which key was pressed
#         print(f"Key {key.char} pressed")
#         control_motor(key.char, ssh_client)
#     except AttributeError:
#         # Special keys (e.g., space bar) can be handled here if needed
#         print(f"Special key {key} pressed")

# def on_release(key):
#     # Exit the listener when ESC is pressed
#     if key == keyboard.Key.esc:
#         return False

# # Main loop for controlling the robot
# def main():
#     ssh_client = connect_to_ev3()  # Set up SSH connection
#     print("Press 'w' for forward, 's' for reverse, 'a' for left, 'd' for right, 'q' to stop.")
    
#     # Start listening for key presses
#     with keyboard.Listener(on_press=lambda key: on_press(key, ssh_client), on_release=on_release) as listener:
#         listener.join()

# if __name__ == '__main__':
#     main()


import socket
from pynput import keyboard

HOST = '169.254.182.1'  # Replace with EV3 IP
PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
print("Connected to EV3!")

def send_command(cmd):
    print(f"Sending: {cmd}")
    sock.sendall(cmd.encode())

def on_press(key):
    try:
        if key.char in ['w', 's', 'a', 'd', 'q', 'r', 'f', 'k', 'g', 'i', 't']:
            mapping = {
                'w': 'forward',
                's': 'reverse',
                'a': 'left',
                'd': 'right',
                'r': 'portopen',
                'f': 'portclose',
                'g': 'kick',
                't': 'onemeter',
                'i': 'slowforward',
                'k': 'slowreverse',
                'q': 'stop'
            }
            send_command(mapping[key.char])
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        send_command('stop')
        sock.close()
        return False

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
