import cv2
import numpy as np
import math
import paramiko
import socket
import time

IPADDRESS = '169.254.142.235' # REMEMBER TO UPDATE THIS
PORT = 9999

sock = None

def open_SSH_connection():
    # Connect to the EV3 via SSH
    # print("Connecting to EV3 at", IPADDRESS)
    # global ssh_client
    # ssh_client = paramiko.SSHClient()
    # ssh_client.load_system_host_keys()
    # ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh_client.connect(IPADDRESS, username='robot', password='maker')

    # command = f'python3 /home/robot/Gruppe3CDIOPython/first.py'
    # stdin, stdout, stderr = ssh_client.exec_command(command)
    
    # Print output or errors (for debugging)
    # print("STDOUT:", stdout.read().decode())
    # print("STDERR:", stderr.read().decode())
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((IPADDRESS, PORT))
    print("Connected to EV3!")

def send_command(cmd):
    print(f"Sending: {cmd}")
    sock.sendall(cmd.encode())
    
# def send_coordinates_to_ev3(angle, seconds):
#     # Command to update robot's movement based on the target coordinates
#     command = f'python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py {angle} {seconds}'
#     stdin, stdout, stderr = ssh_client.exec_command(command)
    
#     # Print output or errors (for debugging)
#     print("Sending angle and seconds:", angle, seconds)
#     print("STDOUT:", stdout.read().decode())
#     print("STDERR:", stderr.read().decode())

# def reverse_ev3():
#     # Command to update robot's movement based on the target coordinates
#     command = f'python3 /home/robot/Gruppe3CDIOPython/reverse.py'
#     stdin, stdout, stderr = ssh_client.exec_command(command)
    
#     # Print output or errors (for debugging)
#     print("STDOUT:", stdout.read().decode())
#     print("STDERR:", stderr.read().decode())

# def close_SSH_connection():
#     # Close the SSH connection
#     print("Closing SSH connection")
#     ssh_client.close()

def calculate_rotation_angle(front, back, ball, return_degrees=True):
    # Calculate vectors
    print(f"Front: {front}, Back: {back}, Ball: {ball}")

    vec_orientation = (front[0] - back[0], front[1] - back[1])
    vec_to_ball = (ball[0] - back[0], ball[1] - back[1])
    
    # Compute dot product and determinant
    dot = vec_orientation[0] * vec_to_ball[0] + vec_orientation[1] * vec_to_ball[1]
    det = vec_orientation[0] * vec_to_ball[1] - vec_orientation[1] * vec_to_ball[0]
    
    # Calculate angle in radians
    angle_rad = math.atan2(det, dot)
    
    # Convert to degrees if requested
    retval = math.degrees(angle_rad) if return_degrees else angle_rad
    print(f"retval: {retval}")
    return retval

# Example usage with your coordinates
# back = (655.5, 223.0)
# front = (584.0, 180.5)
# ball = (808.0, 654.0)

# angle = calculate_rotation_angle(front, back, ball)
# print(f"Rotation angle: {angle:.2f} degrees")

def calculate_distance(p1, p2):
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return np.sqrt((dx)**2 + (dy)**2)

def find_robot(green_contours, pink_contours, boundary_box):
    if not green_contours or not pink_contours:
        # print("No robot parts detected")
        return False
    
    topx, topy, topw, toph = cv2.boundingRect(pink_contours[0])

    top = (topx + (topw / 2), topy + (toph / 2))

    shortest_distance = 0
    longest_distance = 0
    bottomx = 0
    bottomy = 0


    for green_cont in green_contours:
        x, y, w, h = cv2.boundingRect(green_cont)

        if boundary_box is not None:
                bx, by, bw, bh = boundary_box
                if not (bx <= x <= bx + bw and by <= y <= by + bh):
                    continue

        dist = calculate_distance(top, (x + w / 2, y + h / 2))
        if longest_distance == 0 or dist > longest_distance:
            longest_distance = dist
            bottomx = x + w / 2
            bottomy = y + h / 2
        
    bottom = (bottomx, bottomy)

    return top, bottom

def is_in_robot(green_contours, pink_contours, circlex, circley, boundary_box):

    # try:
    #     top, bottom = find_robot(green_contours, pink_contours, boundary_box)
    # except Exception as e:
    #     # print("Error finding robot:", e)
    #     return False
        
    # bottomx, bottomy = bottom
    # topx, topy = top

    # marginOfError = 50

    # if (bottomy < topy):
    #     # print("Robot is upside down")
    #     if (rightx - marginOfError < circlex < leftx + marginOfError) and (righty - marginOfError < circley < lefty + marginOfError):
    #         # print("Circle is in the robot area")
    #         return True
    # elif (bottomy > topy):
    #     # print("Robot is right side up")
    #     if (leftx - marginOfError < circlex < rightx + marginOfError) and (lefty - marginOfError < circley < righty + marginOfError):
    #         # print("Circle is in the robot area")
    #         return True
        
    # print("Circle is NOT in the robot area")
    return False

def check_if_hit_obstacle(bottom, ball):
    rise = ball[1] - bottom[1]
    run = ball[0] - bottom[0]

    if run == 0:
        # Avoid division by zero
        return False
    
    slope = rise / run


    pass

# Camera setup
kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
kamera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
kamera.set(cv2.CAP_PROP_EXPOSURE, -6)
# kamera.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)      # Range: 0.0 - 1.0
# kamera.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000)  # Try adjusting WB if supported

# Global variables for calibration
calibration_done = False
px_measurements = []
cm_per_pixel = 0
ball_radius_px = 0 # Optimal ball radius is 16.3 px
robot_speed = 22
# ball_counter = 0

auto = True

open_SSH_connection()

def calibrate_measurement(event, x, y, flags, param):
    global px_measurements, calibration_done, cm_per_pixel, ball_radius_px
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(px_measurements) < 2:
            px_measurements.append((x, y))
            if len(px_measurements) == 2:
                px_length = np.sqrt((px_measurements[1][0] - px_measurements[0][0])**2 + 
                                    (px_measurements[1][1] - px_measurements[0][1])**2)
                reference_cm = 5.0
                cm_per_pixel = reference_cm / px_length
                ball_radius_px = 2.5 / cm_per_pixel
                calibration_done = True

# Calibration window
cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", calibrate_measurement)

print("CALIBRATION: Click two ends of a 5cm reference object")
while not calibration_done:
    ret, frame = kamera.read()
    if not ret:
        continue
    for (x, y) in px_measurements:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    if len(px_measurements) == 2:
        cv2.line(frame, px_measurements[0], px_measurements[1], (0, 255, 0), 2)
    cv2.putText(frame, "Click two ends of 5cm object", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"Calibration complete - Expected ball radius: {ball_radius_px:.1f} pixels")

min_radius = int(ball_radius_px * 0.7)
max_radius = int(ball_radius_px * 3.5)
print(f"Detection parameters - Radius range: {min_radius}-{max_radius} pixels")
print(f"cm_per_pixel: {cm_per_pixel:.4f}")

# Optimized HSV color ranges
# Wider Orange (for light + dark shades)
lower_orange = np.array([1, 60, 60])
upper_orange = np.array([30, 255, 255])

# White (unchanged, works well for white ball)
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 80, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
lower_green = np.array([40, 40, 40])        # Expanded range
upper_green = np.array([90, 255, 255])
lower_purple = np.array([130, 40, 40])      # Expanded range
upper_purple = np.array([160, 255, 255])
lower_pink = np.array([150, 100, 100])
upper_pink = np.array([170, 255, 255])
lower_yellow = np.array([25, 100, 100])
upper_yellow = np.array([35, 255, 255])

e = True
# send_command('kick')
# time.sleep(0.5)

# Main processing loop
while True:
    ret, frame = kamera.read()
    if not ret:
        continue

    # Convert to HSV once and use for all operations
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Extract V channel for circle detection (from HSV)
    v_channel = hsv[:, :, 2]
    v_channel = cv2.medianBlur(v_channel, 5)

    # Red detection for boundaries
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Boundary processing
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_wall_area = 500
    filtered_red_contours = [cnt for cnt in red_contours if cv2.contourArea(cnt) > min_wall_area]

    boundary_box = None
    box_points = None
    inward_box = None

    # Boundary box calculation
    if filtered_red_contours:
        all_points = np.vstack(filtered_red_contours)
        rect = cv2.minAreaRect(all_points)
        box_points = cv2.boxPoints(rect)
        box_points = np.array(box_points, dtype=np.int32)
        center = np.mean(box_points, axis=0)
        shrink_factor = 0.9
        inward_box = (box_points - center) * shrink_factor + center
        inward_box = np.int32(inward_box)
        # inward_box[0][1] -= 20
        # inward_box[3][1] -= 20
        # inward_box[1][1] += 20
        # inward_box[2][1] += 20
        boundary_box = cv2.boundingRect(inward_box)
        cv2.drawContours(frame, [inward_box], 0, (0, 255, 255), 2)
        x, y, w, h = boundary_box
        cv2.putText(frame, "Wall Boundary", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # mask_inside_wall = np.zeros_like(red_mask)
    # cv2.fillPoly(mask_inside_wall, [inward_box], 255)
    # red_mask_inside = cv2.bitwise_and(red_mask, mask_inside_wall)

    # # Now use the inner mask to find potential cross contours
    # red_contours, _ = cv2.findContours(red_mask_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # frame_h, frame_w = frame.shape[:2]
    # frame_center = (frame_w // 2, frame_h // 2)

    # closest_cross = None
    # closest_distance = float('inf')
    # cross_center = (0, 0)

    # for cnt in red_contours:
    #     if cnt is None or cnt.size == 0 or cnt.shape[0] < 3:
    #         continue

    #     if cv2.contourArea(cnt) < 100:
    #         continue

    #     approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

    #     # You can modify this condition depending on your shape
    #     if 10 <= len(approx) <= 14:
    #         M = cv2.moments(cnt)
    #         if M["m00"] == 0:
    #             continue
    #         cx = int(M["m10"] / M["m00"])
    #         cy = int(M["m01"] / M["m00"])

    #         # Compute distance to frame center
    #         dist_to_center = np.sqrt((cx - frame_center[0])**2 + (cy - frame_center[1])**2)

    #         if dist_to_center < closest_distance:
    #             closest_distance = dist_to_center
    #             closest_cross = cnt
    #             cross_center = (cx, cy)

    # # Draw only the most centered valid cross
    # if closest_cross is not None:
    #     cv2.drawContours(frame, [closest_cross], -1, (0, 255, 0), 3)
    #     cv2.putText(frame, "CROSS", (cross_center[0] - 40, cross_center[1] - 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #     if e:
    #         print(f"KRYDS GENKENDT i midten: {closest_cross}")
    #     e = False

    # Robot detection using HSV
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphology to robot masks
    kernel_robot = np.ones((3, 3), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_robot)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel_robot)
    mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_OPEN, kernel_robot)
    # mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_robot)

    # Find contours for both robot parts
    green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pink_contours, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine robot contours for exclusion zone
    robot_contours = green_contours + pink_contours

    robot_front = []
    robot_back = []
    # Process green robot part
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            if boundary_box is not None:
                bx, by, bw, bh = boundary_box
                if not (bx <= x <= bx + bw and by <= y <= by + bh):
                    continue
            
            robot_front.append((x + w / 2, y + h / 2))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f"Back ({x+w/2},{y+h/2})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # reverse_sent = False
    # if len(pink_contours) <= 0 and not reverse_sent:
    #     send_command('reverse')
    #     reverse_sent = True
    # elif len(pink_contours) > 0:
    #     reverse_sent = False

    for contour in pink_contours:
        area = cv2.contourArea(contour)

        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)

            if boundary_box is not None:
                    bx, by, bw, bh = boundary_box
                    if not (bx <= x <= bx + bw and by <= y <= by + bh):
                        continue
                    
            robot_back.append((x + w / 2, y + h / 2))
            

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(frame, f"Front ({x+w/2},{y+h/2})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Circle detection using V channel
    circles = cv2.HoughCircles(
        v_channel, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=int(ball_radius_px * 2),
        param1=50, 
        param2=20, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    filtered_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = int(i[0]), int(i[1]), int(i[2])

            # Exclude circles in robot areas
            
            if is_in_robot(green_contours, pink_contours, cx, cy, boundary_box):
                continue

            # Exclude circles outside boundary
            if boundary_box is not None:
                bx, by, bw, bh = boundary_box
                if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
                    continue

                square_side = 100  # You can adjust this
                mid_x = bx + bw / 2
                mid_y = by + bh / 2
                half_side = square_side / 2
                if (mid_x - half_side <= cx <= mid_x + half_side and
                    mid_y - half_side <= cy <= mid_y + half_side):
                    continue  # Skip this circle
                

            filtered_circles.append(i)

        ball_radius_px = np.mean([i[2] for i in filtered_circles]) if filtered_circles else ball_radius_px

        # Draw detected circles
        for i in filtered_circles:
            cx, cy, r = int(i[0]), int(i[1]), int(i[2])
            if r <= ball_radius_px:
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
                cv2.putText(frame, f"Ball ({cx}, {cy})", (cx - r, cy - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # else:
            #     cv2.circle(frame, (cx, cy), r, (255, 0, 0), 2)
            #     cv2.putText(frame, "Egg", (cx - r, cy - r - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display results
    # cv2.imshow("Red Color Filter", red_mask)
    # cv2.imshow("Green Color Filter", mask_green)
    cv2.imshow("Red inside Filter", v_channel)
    cv2.imshow("Robot and Ball Detection", frame)

    if ((cv2.waitKey(1) & 0xFF == ord('f')) or auto) and green_contours:

        # if (ball_counter >= 3):
        #     print("Ball counter reached 3, going to goal for drop-off")
        #     goal = None
        #     leftx =  inward_box[0][0]
        #     rightx = inward_box[1][0]

        #     if (bottom[0] - leftx) < (rightx - bottom[0]):
        #         goal = (leftx, inward_box[0][1] + inward_box[0][1] / 2)

        if not filtered_circles:
            # print("No circles detected")
            continue

        x1, y1 = 0, 0

        shortest_distance = 0
        
        try:
            top, bottom = find_robot(green_contours, pink_contours, boundary_box)
        except Exception as e:
            # print("Error finding robot:", e)
            send_command('reverse')
            time.sleep(0.2)
            continue

        for circle in filtered_circles:
            if circle[2] < (ball_radius_px * 1.2):
                x, y, r = circle[0], circle[1], circle[2]

                dist = calculate_distance(bottom, (x, y))
                if shortest_distance == 0 or dist < shortest_distance:
                    shortest_distance = dist
                    x1, y1 = x, y
                    # print(f"Circle found at ({x1}, {y1}) with distance {dist}")
                    
        x1, y1 = float(x1), float(y1)
        x2, y2 = float(bottom[0]), float(bottom[1])

        # distanceBtwCircleAndRobot = calculate_distance((x2, y2), (x1, y1))

        angleToMove = calculate_rotation_angle((top[0],top[1]), (bottom[0],bottom[1]), (x1, y1))

        # print(f"Distances: {distanceBtwCircleAndRobot}, {ball_radius_px * 4}")
        if calculate_distance((top[0], top[1]), (x1, y1)) < (ball_radius_px * 2):

            send_command('stop')
            time.sleep(0.2)
        
            if -3 < angleToMove < 3:
                # print("Robot is close enough to the ball, catching it")
                send_command('catchball')
                time.sleep(4)
                continue 

            print(f"inside loop")
            if angleToMove > 3:
                send_command('slowright')
                time.sleep(0.2)
                continue
            elif angleToMove < -3:
                send_command('slowleft')
                time.sleep(0.2)
                continue

        print(f"out of loop")
        if angleToMove > 45:
            send_command('right')
            time.sleep(0.7)
            continue
        elif angleToMove < -45:
            send_command('left')
            time.sleep(0.7)
            continue
        if angleToMove > 3:
            send_command('slowright')
            time.sleep(0.2)
            continue
        elif angleToMove < -3:
            send_command('slowleft')
            time.sleep(0.2)
            continue
        
        # time.sleep(0.5)


        # if calculate_distance((top[0], top[1]), (x1, y1)) < (ball_radius_px * 4):
        #     send_command('stop')
        #     time.sleep(0.2)
        if calculate_distance((top[0], top[1]), (x1, y1)) < (ball_radius_px * 6):
            send_command('slowforward')
            time.sleep(0.2)
        elif calculate_distance((top[0], top[1]), (x1, y1)) > (ball_radius_px * 8):
            send_command('forward')
            # time.sleep(0.2)

        ball_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('j'):
        kamera.set(cv2.CAP_PROP_EXPOSURE, -4)
    if cv2.waitKey(1) & 0xFF == ord('k'):
        kamera.set(cv2.CAP_PROP_EXPOSURE, -8)
    if cv2.waitKey(1) & 0xFF == ord('l'):
        kamera.set(cv2.CAP_PROP_EXPOSURE, -12)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# if ssh_client is not None:
#     close_SSH_connection()

send_command('stop')
sock.close()

kamera.release()
cv2.destroyAllWindows()