# import cv2
# import numpy as np

# # Camera setup
# kamera = cv2.VideoCapture(0)
# kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# kamera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
# kamera.set(cv2.CAP_PROP_EXPOSURE, -4)

# # Global variables for calibration
# calibration_done = False
# px_measurements = []
# cm_per_pixel = 0
# ball_radius_px = 0

# # Mouse callback for calibration
# def calibrate_measurement(event, x, y, flags, param):
#     global px_measurements, calibration_done, cm_per_pixel, ball_radius_px
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(px_measurements) < 2:
#             px_measurements.append((x, y))
#             if len(px_measurements) == 2:
#                 # Calculate real-world measurements
#                 px_length = np.sqrt((px_measurements[1][0] - px_measurements[0][0])**2 + 
#                             (px_measurements[1][1] - px_measurements[0][1])**2)
#                 reference_cm = 5.0  # Known length of calibration object
#                 cm_per_pixel = reference_cm / px_length
#                 ball_radius_px = 2.5 / cm_per_pixel  # 5cm radius to pixels
#                 calibration_done = True

# # Calibration stage
# cv2.namedWindow("Calibration")
# cv2.setMouseCallback("Calibration", calibrate_measurement)

# print("CALIBRATION: Click two ends of a 5cm reference object")
# while not calibration_done:
#     ret, frame = kamera.read()
#     if not ret: continue
    
#     # Draw existing points
#     for (x, y) in px_measurements:
#         cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
#     if len(px_measurements) == 2:
#         cv2.line(frame, px_measurements[0], px_measurements[1], (0, 255, 0), 2)
    
#     cv2.putText(frame, "Click two ends of 5cm object", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.imshow("Calibration", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
# print(f"Calibration complete - Expected ball radius: {ball_radius_px:.1f} pixels")

# # Detection parameters based on calibration
# min_radius = int(ball_radius_px * 0.7)  # 30% size tolerance
# max_radius = int(ball_radius_px * 3.5)
# print(f"Detection parameters - Radius range: {min_radius}-{max_radius} pixels")

# # Color range for orange (adjust these values if needed)
# lower_orange = np.array([5, 100, 100])
# upper_orange = np.array([15, 255, 255])

# while True:
#     ret, frame = kamera.read()
#     if not ret: continue

#     # Color-based filtering
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, lower_orange, upper_orange)
#     filtered = cv2.bitwise_and(frame, frame, mask=mask)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)
#     circles = cv2.HoughCircles(
#         gray, 
#         cv2.HOUGH_GRADIENT, 
#         dp=1, 
#         minDist=int(ball_radius_px*2),  # Minimum distance between circles
#         param1=50, 
#         param2=30, 
#         minRadius=min_radius, 
#         maxRadius=max_radius
#     )

#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             if i[2] < 25:
#                 cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
#                 cv2.putText(frame, "Ball", (i[0]-i[2], i[1]-i[2]-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             else: # EGG
#                 cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 2)
#                 cv2.putText(frame, "Egg", (i[0]-i[2], i[1]-i[2]-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#                 print(f"x: {i[0]}, y: {i[1]}")

#     # Display detection info
#     cv2.putText(frame, f"Radius: {ball_radius_px:.1f}px", (10, 30),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#     cv2.imshow("Ball Detection", frame)
#     cv2.imshow("Color Filter", filtered)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# kamera.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Camera setup
kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
kamera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
kamera.set(cv2.CAP_PROP_EXPOSURE, -6)

# Global variables for calibration
calibration_done = False
px_measurements = []
cm_per_pixel = 0
ball_radius_px = 0

# # Mouse callback for calibration
def calibrate_measurement(event, x, y, flags, param):
    global px_measurements, calibration_done, cm_per_pixel, ball_radius_px
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(px_measurements) < 2:
            px_measurements.append((x, y))
            if len(px_measurements) == 2:
                px_length = np.sqrt((px_measurements[1][0] - px_measurements[0][0])**2 + 
                                    (px_measurements[1][1] - px_measurements[0][1])**2)
                reference_cm = 5.0  # Known length
                cm_per_pixel = reference_cm / px_length
                ball_radius_px = 2.5 / cm_per_pixel  # 5 cm diameter -> 2.5 cm radius
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

# Color ranges
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([15, 255, 255])
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

while True:
    ret, frame = kamera.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Apply morphological ops to reduce flickering
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small noise
    min_wall_area = 500
    filtered_red_contours = [cnt for cnt in red_contours if cv2.contourArea(cnt) > min_wall_area]
    
    # Forsøg at identificere det røde kryds
    for cnt in filtered_red_contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Kryds har ofte 12 hjørner og næsten kvadratisk form
        if 10 <= len(approx) <= 14 and 0.7 <= aspect_ratio <= 1.3:
            cv2.drawContours(frame, [cnt], -1, (255, 255, 0), 3)
            cv2.putText(frame, "KRYDS", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


    boundary_box = None
    if filtered_red_contours:
        all_points = np.vstack(filtered_red_contours)
        rect = cv2.minAreaRect(all_points)
        box_points = cv2.boxPoints(rect)
        box_points = np.array(box_points, dtype=np.int32)

        # Shrink box inward by moving each point closer to the center
        center = np.mean(box_points, axis=0)
        shrink_factor = 0.90  # Adjust this (e.g., 0.85 for ~15% inward)

        inward_box = (box_points - center) * shrink_factor + center
        inward_box = np.int32(inward_box)

        boundary_box = cv2.boundingRect(inward_box)

        # Draw inward box
        cv2.drawContours(frame, [inward_box], 0, (0, 255, 255), 2)
        x, y, w, h = boundary_box
        cv2.putText(frame, "Wall Boundary", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)



    # Robot (blue)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_region = cv2.bitwise_and(frame, frame, mask=mask_blue)
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Circle detection (balls and eggs)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=int(ball_radius_px * 2),
        param1=50, 
        param2=30, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, "Robot", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    filtered_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = i[0], i[1], i[2]

            # Skip if in robot region
            in_robot = False
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x <= cx - r and cx + r <= x + w and y <= cy - r and cy + r <= y + h:
                    in_robot = True
                    break
            if in_robot:
                continue

            # Skip if outside red wall boundary
            if boundary_box is not None:
                bx, by, bw, bh = boundary_box
                if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
                    continue

            filtered_circles.append(i)

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

    # Display windows
    cv2.imshow("Robot and Ball Detection", frame)
    cv2.imshow("Red Color Filter", red_mask)

    # Find distance between first two circles on 'f' key press
    if cv2.waitKey(1) & 0xFF == ord('f'):
        x1 = filtered_circles[0][0]
        y1 = filtered_circles[0][1]
        x2 = filtered_circles[1][0]
        y2 = filtered_circles[1][1]
        
        x1, y1 = float(x1), float(y1)
        x2, y2 = float(x2), float(y2)

        lengthBtwCircles = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        inCm = lengthBtwCircles * cm_per_pixel

        print(f"Distance between circles at points ({x1:.2f}, {y1:.2f}) and ({x2:.2f}, {y2:.2f}): {lengthBtwCircles:.2f} pixels and {inCm:.2f} cm")

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
