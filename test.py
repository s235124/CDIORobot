import paramiko
import time
import cv2
import numpy as np

IPADRESS = '169.254.46.72' # REMEMBER TO UPADTE THIS

def send_coordinates_to_ev3(target_x, target_y):
    # Connect to the EV3 via SSH
    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(IPADRESS, username='robot', password='maker')

    # Command to update robot's movement based on the target coordinates
    command = f'python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py {target_x} {target_y}'
    stdin, stdout, stderr = ssh_client.exec_command(command)
    
    # Print output or errors (for debugging)
    print("Sending coordinates:", target_x, target_y)
    print("STDOUT:", stdout.read().decode())
    print("STDERR:", stderr.read().decode())

    
    ssh_client.close()

def listen_for_coordinates_and_update():
    while True:
        # Capture image and detect object
        coords = capture_and_find_object()
        if coords:
            target_x, target_y = coords
            send_coordinates_to_ev3(target_x, target_y)
        
        time.sleep(1)  # Sleep before checking for the next image

def find_object_coordinates(image):
    # Example processing logic to find an object (you'll need to adjust this)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the object
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Avoid small noise areas
            # Find the center of the object
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None  # If no object detected

def capture_and_find_object():
    # Capture image from the webcam (adjust to your actual camera)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        object_coords = find_object_coordinates(frame)
        if object_coords:
            print("Object detected at:", object_coords)
            return object_coords
        else:
            print("No object detected.")
    return None

def findDistanceFromRobotToBall (robotx, roboty, ballx, bally):
    height = bally - roboty
    length = ballx - robotx

    distance = np.sqrt((height^2) + (length^2))

    return distance

if __name__ == "__main__":
    listen_for_coordinates_and_update()
