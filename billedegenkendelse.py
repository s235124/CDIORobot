import cv2
import numpy as np
import math
import paramiko

IPADDRESS = '169.254.187.202' # REMEMBER TO UPDATE THIS

def send_coordinates_to_ev3(angle, seconds):
    # Connect to the EV3 via SSH
    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(IPADDRESS, username='robot', password='maker')

    # Command to update robot's movement based on the target coordinates
    command = f'python3 /home/robot/Gruppe3CDIOPython/update_robot_position.py {angle} {seconds}'
    stdin, stdout, stderr = ssh_client.exec_command(command)
    
    # Print output or errors (for debugging)
    print("Sending coordinates:", angle, seconds)
    print("STDOUT:", stdout.read().decode())
    print("STDERR:", stderr.read().decode())

    
    ssh_client.close()

def calculate_angle(p1, p2):
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    radians = math.atan2(dx, dy)
    degrees = math.degrees(radians)
    return degrees % 360

def calculate_distance(p1, p2):
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return np.sqrt(dx**2 + dy**2)

def is_in_robot(green_contours, yellow_contours, circlex, circley):
    if not green_contours or not yellow_contours:
        print("No robot parts detected")
        return False
    
    topx, topy, topw, toph = cv2.boundingRect(yellow_contours[0])

    top = ((topx + topw) / 2, (topy + toph) / 2)

    shortest_distance = 0
    longest_distance = 0
    rightx = 0
    righty = 0
    leftx = 0
    lefty = 0
    bottomx = 0
    bottomy = 0

    for green_cont in green_contours:
        x, y, w, h = cv2.boundingRect(green_cont)

        dist = calculate_distance(top, ((x + w) / 2, (y + h) / 2))
        if shortest_distance == 0 or dist < shortest_distance:
            shortest_distance = dist
            rightx = (x + w) / 2
            righty = (y + h) / 2

    for green_cont in green_contours:
        x, y, w, h = cv2.boundingRect(green_cont)

        dist = calculate_distance(top, ((x + w) / 2, (y + h) / 2))
        if longest_distance == 0 or dist > longest_distance:
            longest_distance = dist
            bottomx = (x + w) / 2
            bottomy = (y + h) / 2

    for green_cont in green_contours:
        x, y, w, h = cv2.boundingRect(green_cont)

        dist = calculate_distance(top, ((x + w) / 2, (y + h) / 2))
        if shortest_distance < dist < longest_distance:
            leftx = (x + w) / 2
            lefty = (y + h) / 2
        
    right = (rightx, righty)
    bottom = (bottomx, bottomy)
    left = (leftx, lefty)
    angle = calculate_angle(bottom, top)

    # print(f"Robot found at top: {top}, right: {right}, left: {left}, bottom: {bottom}, angle: {angle:.2f} degrees")

    if (bottomy < topy):
        print("Robot is upside down")
        if (rightx - 20 < circlex < leftx + 20) and (righty - 20 < circley < lefty + 20):
            print("Circle is in the robot area")
            return True
    elif (bottomy > topy):
        print("Robot is right side up")
        if (leftx - 20 < circlex < rightx + 20) and (lefty - 20 < circley < righty + 20):
            print("Circle is in the robot area")
            return True
        
    print("Circle is NOT in the robot area")
    return False

# Camera setup
kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
kamera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
kamera.set(cv2.CAP_PROP_EXPOSURE, -6)

# Global variables for calibration
calibration_done = False
px_measurements = []
cm_per_pixel = 0
ball_radius_px = 0

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

    # Forbedret krydsgenkendelse (kun selve krydset, ikke hele boksen)
    for cnt in filtered_red_contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-5)

        # Kriterier for kryds: kvadratisk, ikke solid, ca. rigtig størrelse
        if 0.6 < aspect_ratio < 1.4 and 0.15 < solidity < 0.8 and 7000 < area < 20000:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Marker krydset tydeligt
            cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 3)  # Gul kontur
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)       # Rød midte
            cv2.putText(frame, "KRYDS", (cx - 30, cy - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

            # Udskriv position i pixels (eller skift til cm hvis du vil)
            # print(f"Kryds-position: ({cx}, {cy}) pixels, area={area:.0f}")


    # Boundary box calculation
    if filtered_red_contours:
        all_points = np.vstack(filtered_red_contours)
        rect = cv2.minAreaRect(all_points)
        box_points = cv2.boxPoints(rect)
        box_points = np.array(box_points, dtype=np.int32)
        center = np.mean(box_points, axis=0)
        shrink_factor = 0.90
        inward_box = (box_points - center) * shrink_factor + center
        inward_box = np.int32(inward_box)
        boundary_box = cv2.boundingRect(inward_box)
        cv2.drawContours(frame, [inward_box], 0, (0, 255, 255), 2)
        x, y, w, h = boundary_box
        cv2.putText(frame, "Wall Boundary", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
            
            robot_front.append(((x + w) / 2, (y + h) / 2))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f"Back ({(x+w)/2},{(y+h)/2})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Process yellow robot part
    for contour in pink_contours:
        area = cv2.contourArea(contour)

        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)

            if boundary_box is not None:
                    bx, by, bw, bh = boundary_box
                    if not (bx <= x <= bx + bw and by <= y <= by + bh):
                        continue
                    
            robot_back.append(((x + w) / 2, (y + h) / 2))
            

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(frame, f"Front ({(x+w)/2},{(y+h)/2})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Circle detection using V channel
    circles = cv2.HoughCircles(
        v_channel, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=int(ball_radius_px * 2),
        param1=50, 
        param2=30, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    filtered_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = float(i[0]), float(i[1]), float(i[2])

            # Exclude circles in robot areas
            # in_robot = False
            
            # for contour in robot_contours:
            #     x, y, w, h = cv2.boundingRect(contour)
            #     if (x <= cx - r and cx + r <= x + w and 
            #         y <= cy - r and cy + r <= y + h):
            #         in_robot = True
            #         break
            if is_in_robot(green_contours, pink_contours, cx, cy):
                continue

            # Exclude circles outside boundary
            if boundary_box is not None:
                bx, by, bw, bh = boundary_box
                if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
                    continue

            filtered_circles.append(i)

        # Draw detected circles
        for i in filtered_circles:
            cx, cy, r = i[0], i[1], i[2]
            if r < (ball_radius_px * 1.2):
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
                cv2.putText(frame, f"Ball ({cx}, {cy})", (cx - r, cy - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.circle(frame, (cx, cy), r, (255, 0, 0), 2)
                cv2.putText(frame, "Egg", (cx - r, cy - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display results
    cv2.imshow("Red Color Filter", red_mask)
    cv2.imshow("Green Color Filter", mask_green)
    cv2.imshow("Robot and Ball Detection", frame)

    # Distance calculation
    if cv2.waitKey(1) & 0xFF == ord('f') and green_contours:
        x1, y1, r1 = filtered_circles[0][0], filtered_circles[0][1], filtered_circles[0][2]
        x2, y2, r2 = green_contours[0][0], green_contours[0][1], green_contours[0][2]
        
        x1, y1 = float(x1), float(y1)
        x2, y2 = float(x2), float(y2)

        angleBtwCircles = calculate_angle((x1, y1), (x2, y2))

        distanceBtwCircles = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        print(f"Angle and Distance between robot ({x1}, {y1}) and ball ({x2}, {y2}): {angleBtwCircles:.2f} px, {distanceBtwCircles * cm_per_pixel:.2f} cm")

        # Calculate seconds based on distance
        ROBOTSPEEDAT30PERCENT = 13.9779 # cm/s at 30% speed
        sec = distanceBtwCircles * cm_per_pixel / ROBOTSPEEDAT30PERCENT

        send_coordinates_to_ev3(angleBtwCircles, sec)

    elif cv2.waitKey(1) & 0xFF == ord('f') and len(filtered_circles) >= 2:
        x1, y1, r1 = filtered_circles[0][0], filtered_circles[0][1], filtered_circles[0][2]
        x2, y2, r2 = filtered_circles[1][0], filtered_circles[1][1], filtered_circles[1][2]
        
        x1, y1 = float(x1), float(y1)
        x2, y2 = float(x2), float(y2)

        angleBtwCircles = calculate_angle((x1, y1), (x2, y2))

        distanceBtwCircles = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        print(f"Distance between balls ({x1}, {y1}) and ({x2}, {y2}): {distanceBtwCircles:.2f} px, {distanceBtwCircles * cm_per_pixel:.2f} cm")

    if cv2.waitKey(1) & 0xFF == ord('x'):
        print("Robot coords:", robot_front, robot_back)
        
        dist = calculate_distance(robot_front[0], robot_back[0])
        angle = calculate_angle(robot_front[0], robot_back[0])

        print(f"Distance and Angle between robot front and back: {dist:.2f} px and {angle} degrees at points {robot_front[0]} and {robot_back[0]}")

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()