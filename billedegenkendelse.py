import cv2
import numpy as np
import math
import paramiko
import socket
import time

IPADDRESS = '169.254.243.129' # REMEMBER TO UPDATE THIS
PORT = 9999

CIRCLES = 'circles'
FRONT = 'front'
BACK = 'back'
ROBOT = 'robot'
WALLS = 'walls'


sock = None

def open_ev3_connection():
    # Connect to the EV3
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((IPADDRESS, PORT))
    print("Connected to EV3!")

def send_command(cmd):
    print(f"Sending: {cmd}")
    sock.sendall(cmd.encode())

def process_angle(angle):
    if abs(angle) > 80:
        if angle > 0:
            send_command('fastright')
        else:
            send_command('fastleft')
        time.sleep(0.7)
    elif abs(angle) > 45:
        if angle > 0:
            send_command('right')
        else:
            send_command('left')
        time.sleep(0.2)
    elif abs(angle) > 2:
        if angle > 0:
            send_command('slowright')
        else:
            send_command('slowleft')
        time.sleep(0.2)

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
    # print(f"retval: {retval}")
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

def get_largest_contour_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def find_robot():
    ret, frame = kamera.read()
    if not ret:
        return False
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Front = pink
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    front = get_largest_contour_center(pink_mask)

    # Body = grøn
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    back = get_largest_contour_center(green_mask)

    return front, back


def check_if_hit_obstacle(bottom, top, ball, boundary_box, cross_coords):
    # return False
    if cross_coords is None or boundary_box is None:
        return
    
    crossx, crossy, crossw, crossh = cross_coords

    mid_x = crossx + crossw / 2
    mid_y = crossy + crossh / 2

    distance_to_cross = calculate_distance(top, (mid_x, mid_y))
    longest_allowed_distance_from_cross = np.sqrt((crossw / 2)**2 + (crossh / 2)**2) * 1.1

    if distance_to_cross < longest_allowed_distance_from_cross:
        send_command('stop')
        time.sleep(0.2)

        send_command('fastright')
        time.sleep(0.2)

        send_command('forward')
        time.sleep(3)

        send_command('stop')
    

    # # square_side_x = crossw  # You can adjust this
    # half_side = crossw / 2

    # half_side_y = crossh / 2

    # print(half_side, half_side_y)

    # rise = ball[1] - bottom[1]
    # run = ball[0] - bottom[0]

    # if run == 0:
    #     # Avoid division by zero
    #     return False
    
    # slope = rise / run

    # low = 0
    # high = 0

    # # Determine the range of x values to check
    # if (bottom[0] < ball[0]):
    #     low = bottom[0]
    #     high = ball[0]
    # else:
    #     low = ball[0]
    #     high = bottom[0]

    # # Check points along the line from bottom to ball
    # for i in range(int(low), int(high)):
    #     cx = i
    #     cy = int(bottom[1] + slope * (i - bottom[0]))
    #     if (mid_x - half_side <= cx <= mid_x + half_side and
    #         mid_y - half_side_y <= cy <= mid_y + half_side_y):
    #         return True
    
    # return False

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

hitting_obstacle = False
auto = True

if auto:
    open_ev3_connection()

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

min_radius = int(ball_radius_px -4)
max_radius = int(ball_radius_px +4)
print(f"Detection parameters - Radius range: {min_radius}-{max_radius} pixels")
print(f"cm_per_pixel: {cm_per_pixel:.4f}")

# Optimized HSV color ranges
# Wider Orange (for light + dark shades)
lower_orange = np.array([1, 60, 60])
upper_orange = np.array([30, 255, 255])
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 80, 255])
lower_blue = np.array([90, 120, 220])
upper_blue = np.array([110, 255, 255])
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])
lower_green = np.array([40, 40, 40])        # Expanded range
upper_green = np.array([90, 255, 255])
lower_purple = np.array([135, 100, 150])    # Expanded range
upper_purple = np.array([155, 255, 255])
lower_pink = np.array([150, 100, 90])
upper_pink = np.array([175, 255, 255])
lower_yellow = np.array([25, 100, 100])
upper_yellow = np.array([35, 255, 255])

def get_frame():
    ret, frame = kamera.read()
    if not ret:
        return False

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

        # Sort points by x (to get left/right)
        sorted_by_x = sorted(inward_box, key=lambda p: p[0])
        left_points = sorted_by_x[:2]   # Two with smallest x
        right_points = sorted_by_x[2:]  # Two with largest x

        # Sort each pair by y to get top/bottom
        left_top, left_bottom = sorted(left_points, key=lambda p: p[1])
        right_top, right_bottom = sorted(right_points, key=lambda p: p[1])

        # Modify y-values
        left_top[1] += 60
        left_bottom[1] -= 60
        right_top[1] += 60
        right_bottom[1] -= 60

        # Reconstruct inward_box in original order if needed
        inward_box = np.array([left_top, right_top, right_bottom, left_bottom], dtype=np.int32)
        boundary_box = cv2.boundingRect(inward_box)
        cv2.drawContours(frame, [inward_box], 0, (0, 255, 255), 2)
        x, y, w, h = boundary_box
        cv2.putText(frame, "Wall Boundary", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    mask_inside_wall = np.zeros_like(red_mask)
    cv2.fillPoly(mask_inside_wall, [inward_box], 255)
    red_mask_inside = cv2.bitwise_and(red_mask, mask_inside_wall)

    # Now use the inner mask to find potential cross contours
    red_contours, _ = cv2.findContours(red_mask_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_h, frame_w = frame.shape[:2]
    frame_center = (frame_w // 2, frame_h // 2)

    closest_cross = None
    closest_distance = float('inf')
    cross_coords = None

    for cnt in red_contours:
        if cnt is None or cnt.size == 0 or cnt.shape[0] < 3:
            continue

        if cv2.contourArea(cnt) < 100:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # You can modify this condition depending on your shape
        if 10 <= len(approx) <= 14:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Compute distance to frame center
            dist_to_center = np.sqrt((cx - frame_center[0])**2 + (cy - frame_center[1])**2)

            if dist_to_center < closest_distance:
                closest_distance = dist_to_center
                closest_cross = cnt
                cross_center = (cx, cy)

    # Draw only the most centered valid cross
    if closest_cross is not None:
        closest_cross = np.int32(closest_cross)
        pts = closest_cross.reshape(-1, 2)                # shape (N, 2)

        # 2.  Get an (x, y, w, h) bounding box
        x, y, w, h = cv2.boundingRect(pts)             # axis‑aligned box

        # 3.  Expand the box by a fixed margin (pixels) or by percentage
        margin = 25
        x_pad = max(0, x - margin)
        y_pad = max(0, y - margin)
        w_pad = min(frame_w - x_pad, w + 2*margin)
        h_pad = min(frame_h - y_pad, h + 2*margin)

        # 4.  Use / draw it
        padded_box = (x_pad, y_pad, w_pad, h_pad)
        cross_coords = padded_box
        cv2.rectangle(frame,
                    (x_pad, y_pad),
                    (x_pad + w_pad, y_pad + h_pad),
                    (0, 255, 0), 2)

    # Robot detection using HSV
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphology to robot masks
    kernel_robot = np.ones((3, 3), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_robot)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel_robot)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_robot)
    mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_OPEN, kernel_robot)
    # mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel_robot)

    # Find contours for both robot parts
    green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pink_contours, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    robot_front_contours = blue_contours + pink_contours + purple_contours

    # Process green robot part
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            if boundary_box is not None:
                bx, by, bw, bh = boundary_box
                if not (bx <= x <= bx + bw and by <= y <= by + bh):
                    continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f"Back ({x+w/2},{y+h/2})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    for contour in robot_front_contours:
        area = cv2.contourArea(contour)

        if area > 200:
            x, y, w, h = cv2.boundingRect(contour)

            if boundary_box is not None:
                    bx, by, bw, bh = boundary_box
                    if not (bx <= x <= bx + bw and by <= y <= by + bh):
                        continue

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

            # Exclude circles outside boundary
            if boundary_box is not None:
                bx, by, bw, bh = boundary_box
                if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
                    continue

            if cross_coords is not None:
                x, y, w, h = cross_coords
                if (x <= cx <= x + w and y <= cy <= y + h):
                    continue

            filtered_circles.append(i)

        # ball_radius_px = np.mean([i[2] for i in filtered_circles]) if filtered_circles else ball_radius_px

        # Draw detected circles
        for i in filtered_circles:
            cx, cy, r = int(i[0]), int(i[1]), int(i[2])
            if r <= ball_radius_px:
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
                cv2.putText(frame, f"Ball ({cx}, {cy})", (cx - r, cy - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    # Display results
    front_masks = mask_purple + mask_blue + mask_pink
    cv2.imshow("Red inside Filter", front_masks)
    cv2.imshow("Robot and Ball Detection", frame)

    return cross_coords, boundary_box, green_contours, robot_front_contours, filtered_circles

if auto:
    send_command('kick')
    time.sleep(0.5)

nearest_ball_coordinates = None
going_towards_ball = False

current_ball_counter = 0
droppingBallsOff = False

stop_program = False

previous_circles = None

cross_coords = None

# Main processing loop
while True:

    print("Big iteration")

    if droppingBallsOff == False:
        for i in range(5):
            cross_coords, boundary_box, green_contours, robot_front_contours, filtered_circles = get_frame()

            if previous_circles is None or len(previous_circles) < len(filtered_circles):
                previous_circles = filtered_circles
                print("Found more circles")

        if len(previous_circles) == 0:
            droppingBallsOff = True
            stop_program = True

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('m'):
        print(boundary_box)

    try:
        top, bottom = find_robot()
    except Exception as e:
        # print("Error finding robot:", e)
        continue

    x1, y1 = 0, 0
    shortest_distance = 0
    
    if droppingBallsOff is False:
        for circle in previous_circles:
                if circle[2] < (ball_radius_px * 1.1):
                    x, y, r = circle[0], circle[1], circle[2]

                    dist = calculate_distance(bottom, (x, y))
                    if (shortest_distance == 0 or dist < shortest_distance) and (dist > ball_radius_px * 4):
                        shortest_distance = dist
                        x1, y1 = x, y
                        # print(f"Circle found at ({x1}, {y1}) with distance {dist}")

        nearest_ball_coordinates = (x1, y1)

        going_towards_ball = True

    # print("before going towards ball")

    if (droppingBallsOff):
        print("dropping Balls Off")
        cross_coords, boundary_box, green_contours, robot_front_contours, filtered_circles = get_frame()

        try:
            top, bottom = find_robot()
        except Exception as e:
            print("Error finding robot:", e)
            send_command('reverse')
            time.sleep(0.5)
            continue

        # print("Ball counter reached 4, going to goal for drop-off")
        goal = None
        bx, by, bw, bh = boundary_box

        leftx = bx
        rightx = bx + bw
        b_boxy = by + bh / 2
        x_coord_of_middle = 0

        robotx = bottom[0]
        roboty = bottom[1]

        if (robotx - leftx) < (rightx - robotx):
            goal = (leftx, b_boxy)
            x_coord_of_middle = bx + bw / 4
        elif (robotx - leftx) > (rightx - robotx):
            goal = (rightx, b_boxy)
            x_coord_of_middle = bx + bw * 0.8

        if goal is not None:
            # print("goal is existant")

            dist_to_middle = calculate_distance((robotx, roboty), (robotx, goal[1]))

            if -20 < dist_to_middle < 20:
                angle_from_middle_to_goal = calculate_rotation_angle((top[0], top[1]), (robotx, roboty), (goal[0], goal[1]))
                if -2 < angle_from_middle_to_goal < 2:
                    dist_to_goal = calculate_distance(top, goal)
                    if -20 < dist_to_goal < 20:
                        send_command('stop')
                        time.sleep(0.2)
                        send_command('dropoff')
                        print("kicking balls out")
                        time.sleep(3)
                        droppingBallsOff = False
                        current_ball_counter = 0
                        if stop_program:
                            break
                        continue
                    
                    # check_if_hit_obstacle(bottom, top, (x1,y1), boundary_box, cross_coords)
                    send_command('slowforward')
                    continue

                process_angle(angle_from_middle_to_goal)
                continue
            
            angle_from_robot_to_middle = calculate_rotation_angle((top[0], top[1]), (robotx, roboty), (x_coord_of_middle, goal[1]))

            if -2 < angle_from_robot_to_middle < 2:
                if -10 < dist_to_middle < 10:
                    send_command('stop')
                    time.sleep(0.2)
                    continue
                send_command('forward')
                continue

            process_angle(angle_from_robot_to_middle)
            continue
        continue

    while going_towards_ball == True:
        print("small iteration")
        cross_coords, boundary_box, green_contours, robot_front_contours, filtered_circles = get_frame()
        

        if auto and green_contours:

            if not nearest_ball_coordinates:
                print("No circles detected")
                continue

            # x1, y1 = 0, 0

            # shortest_distance = 0
            
            try:
                top, bottom = find_robot()
            except Exception as e:
                # print("Error finding robot:", e)
                send_command('reverse')
                time.sleep(0.5)
                continue

            # if hitting_obstacle:
            #     print("Obstacle detected, going around")
            #     bx, by, bw, bh = boundary_box
            #     if bottom[0] < x1:
            #         print("Robot is on left side of destination")
            #         if calculate_distance((top[0], top[1]), (top[0], by)) < (ball_radius_px * 4): # The robot is near the wall
            #             print("Robot is near top of wall")
            #             angle_of_robot = calculate_rotation_angle((top[0], top[1]), (bottom[0], bottom[1]), (bx + bw, bottom[1]))
            #             if -10 < angle_of_robot < 10: # The robot is facing the right direction
            #                 print("Robot is facing destination in the interval of -5 to 5 deg")
            #                 if calculate_distance((top[0], top[1]), (bx+bw, top[1])) < (ball_radius_px * 4): # The robot is at the right place
            #                     print("Robot has arrived on the correct side")
            #                     send_command('stop')
            #                     time.sleep(0.2)
            #                     hitting_obstacle = False
            #                     continue

            #                 print("Robot is NOT near right wall")
            #                 send_command('forward')
            #                 continue
            #             else:
            #                 print("L Robot is NOT facing destination in the interval of -5 to 5 deg")
            #                 process_angle(angle_of_robot)
            #                 time.sleep(0.2)
            #                 continue
            #         else:
            #             angle_of_robot = calculate_rotation_angle((top[0], top[1]), (bottom[0], bottom[1]), (bottom[0], by))
            #             if -10 < angle_of_robot < 10:
            #                 print("L Robot is NOT near bottom wall")
            #                 send_command('forward')
            #                 time.sleep(0.2)
            #                 continue

            #             print("L Robot is NOT facing destination in the interval of -5 to 5 deg")
            #             process_angle(angle_of_robot)
            #             continue
            #     elif bottom[0] > x1:
            #         print("Robot is on right side of destination")
            #         if calculate_distance((top[0], top[1]), (top[0], by + bh)) < (ball_radius_px * 4): # The robot is near the wall
            #             print("Robot is near bottom of wall")
            #             angle_of_robot = calculate_rotation_angle((top[0], top[1]), (bottom[0], bottom[1]), (bx, bottom[1]))
            #             if -10 < angle_of_robot < 10: # The robot is facing the right direction
            #                 print("Robot is facing destination in the interval of -5 to 5 deg")
            #                 if calculate_distance((top[0], top[1]), (bx, top[1])) < (ball_radius_px * 4): # The robot is at the right place
            #                     print("Robot has arrived on the correct side")
            #                     send_command('stop')
            #                     time.sleep(0.2)
            #                     hitting_obstacle = False
            #                     continue

            #                 print("Robot is NOT near left wall")
            #                 send_command('forward')
            #                 continue
            #             else:
            #                 print("R Robot is NOT facing destination in the interval of -5 to 5 deg")
            #                 process_angle(angle_of_robot)
            #                 time.sleep(0.2)
            #                 continue
            #         else:
            #             angle_of_robot = calculate_rotation_angle((top[0], top[1]), (bottom[0], bottom[1]), (bottom[0], by + bh))
            #             if -10 < angle_of_robot < 10:
            #                 print("R Robot is NOT near bottom wall")
            #                 send_command('forward')
            #                 time.sleep(0.2)
            #                 continue

            #             print("R Robot is NOT facing bottom wall in the interval of -5 to 5 deg")
            #             process_angle(angle_of_robot)
            #             continue
            #     continue
            # else:
            #     hitting_obstacle = check_if_hit_obstacle(bottom, (x1, y1), cross_coords)
            #     print(hitting_obstacle, x1, y1)

            x1, y1 = float(x1), float(y1)

            # distanceBtwCircleAndRobot = calculate_distance((x2, y2), (x1, y1))

            angleToMove = calculate_rotation_angle((top[0],top[1]), (bottom[0],bottom[1]), (x1, y1))

            if -2 < angleToMove < 2:
                distance_to_ball = calculate_distance(top, (x1,y1))
                if distance_to_ball < (ball_radius_px * 3):
                    send_command('stop')
                    time.sleep(0.2)

                    send_command('catchball')
                    time.sleep(4)
                    current_ball_counter += 1
                    going_towards_ball = False

                    if current_ball_counter >= 4:
                        droppingBallsOff = True
                    continue

                if distance_to_ball < (ball_radius_px * 6):
                    check_if_hit_obstacle(bottom, top, (x1,y1), boundary_box, cross_coords)
                    send_command('slowforward')
                    time.sleep(0.2)
                else:
                    check_if_hit_obstacle(bottom, top, (x1,y1), boundary_box, cross_coords)
                    send_command('forward')
                continue

            process_angle(angleToMove)

            # # print(f"Distances: {distanceBtwCircleAndRobot}, {ball_radius_px * 4}")
            # if calculate_distance((top[0], top[1]), (x1, y1)) < (ball_radius_px * 2):
            #     print("Robot is close")
            #     send_command('stop')
            #     time.sleep(0.2)
            
            #     if -2 < angleToMove < 2:
            #         print("Robot is close enough to the ball, catching it")
            #         send_command('catchball')
            #         time.sleep(4)
            #         current_ball_counter += 1
            #         going_towards_ball = False
            #         print("caught ball")
            #         if current_ball_counter >= 4:
            #             droppingBallsOff = True
            #             print("dropping off balls")
            #         continue 

            #     process_angle(angleToMove)
            #     continue

            # if abs(angleToMove) > 45:
            #     if angleToMove > 0:
            #         send_command('right')
            #     else:
            #         send_command('left')
            #     time.sleep(0.7)
            #     continue
            # elif abs(angleToMove) > 2:
            #     if angleToMove > 0:
            #         send_command('slowright')
            #     # time.sleep(0.2)
            #     else:
            #         send_command('slowleft')
            #     time.sleep(0.2)
            #     continue
            
            # if calculate_distance((top[0], top[1]), (x1, y1)) < (ball_radius_px * 6):
            #     send_command('slowforward')
            #     time.sleep(0.2)
            # else:
            #     send_command('forward')

    previous_circles = None
    # print("End of iteration")
        
        
if auto:
    print("Closing connection")
    send_command('stop')
    time.sleep(0.2)
    sock.close()

kamera.release()
cv2.destroyAllWindows()