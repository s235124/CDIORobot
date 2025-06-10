import cv2

import numpy as np

import math


#CALIBRATION
def calibrate_image(frame):
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))
            print(f"[CLICK] Point {len(clicks)}: ({x}, {y})")

    print("Click on two points on a ball (opposite edges)")
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse)

    while True:
        preview = frame.copy()
        for (x, y) in clicks:
            cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
        if len(clicks) == 2:
            cv2.line(preview, clicks[0], clicks[1], (0, 255, 0), 2)
        cv2.imshow("Calibration", preview)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(clicks) == 2:
            break
    cv2.destroyWindow("Calibration")

    if len(clicks) < 2:
        print("Calibration failed.")
        return None

    pixel_dist = np.linalg.norm(np.array(clicks[0]) - np.array(clicks[1]))
    cm_per_pixel = 4.0 / pixel_dist  #40mm diameter table tennis ball
    print(f"Calibration complete: {cm_per_pixel:.4f} cm/pixel")
    return cm_per_pixel

#ANGLE CALULATION (0° = down, 90° = right, 180° = up, 270° = left)
def calculate_angle(p1, p2):
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    radians = math.atan2(dx, dy)  #dx and dy is swapped for new coordinate logic
    degrees = math.degrees(radians)
    return degrees % 360


#FIND CENTER OF CONTOUR
def find_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    return (x, y)

#LOAD IMaGE
frame = cv2.imread("testbillede.png")
if frame is None:
    print("Image not found.")
    exit()

cm_per_pixel = calibrate_image(frame)
if cm_per_pixel is None:
    exit()



#DETECT ROBOT (GREEN BODY + PURRPLE FRONT)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_green = np.array([45, 100, 50])
upper_green = np.array([75, 255, 255])
lower_purple = np.array([135, 50, 50])
upper_purple = np.array([160, 255, 255])

mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

center_robot = find_center(mask_green)
front_robot = find_center(mask_purple)

if center_robot and front_robot:
    cv2.rectangle(frame, (center_robot[0]-40, center_robot[1]-60), (center_robot[0]+40, center_robot[1]+60), (0, 255, 0), 2)
    cv2.putText(frame, "Robot", (center_robot[0]-20, center_robot[1]-65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.line(frame, front_robot, center_robot, (255, 255, 0), 2)

#DETECT BALLS
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=50, param2=30, minRadius=10, maxRadius=50)

balls = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        balls.append((x, y))
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#CALCULATE DISTANCE and ANGLE TO NEAREST BALL
if center_robot and front_robot and balls:
    nearest = min(balls, key=lambda p: np.linalg.norm(np.array(p) - np.array(center_robot)))
    distance_px = np.linalg.norm(np.array(nearest) - np.array(center_robot))
    distance_cm = distance_px * cm_per_pixel

    print(f"Robot at: ({center_robot[0]}, {center_robot[1]})")
    print(f"Nearest ball at: ({nearest[0]}, {nearest[1]})")
    robot_angle = calculate_angle(center_robot, front_robot)
    target_angle = calculate_angle(center_robot, nearest)
    turn_angle = (target_angle - robot_angle + 360) % 360
    if turn_angle > 180:
        turn_angle -= 360

    cv2.circle(frame, nearest, 10, (0, 0, 255), 3)
    cv2.line(frame, center_robot, nearest, (0, 255, 255), 2)

    cv2.putText(frame, f"Robot heading: {robot_angle:.1f} deg", (10, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Ball heading: {target_angle:.1f} deg", (10, 960), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Turn: {turn_angle:+.1f} deg", (10, 990), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Distance: {distance_cm:.1f} cm", (10, 1020), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    #Console info
    print(f"[INFO] Robot heading: {robot_angle:.1f}°")
    print(f"[INFO] Turn needed: {turn_angle:+.1f}°")
    print(f"[INFO] Distance to ball: {distance_cm:.1f} cm")

cv2.imshow("Result", frame)
cv2.waitKey(0)

cv2.destroyAllWindows()
