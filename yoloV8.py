import cv2
from ultralytics import YOLO

model_path = "runs/detect/train/weights/best.pt" 
image_path = "testbillede.png"             
output_path = "output_result.png"                 
conf_threshold = 0.01   #Lav threshold for test

model = YOLO(model_path)
print("✅ Model loaded")

#Kør detection
results = model.predict(source=image_path, conf=conf_threshold, save=False)


if results and results[0].boxes:
    print("Objekter fundet")

    result_img = results[0].plot()

    
    cv2.imwrite(output_path, result_img)
    print(f"✅ Gemte billedet som {output_path}")

    cv2.imshow("YOLOv8 Detektion", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Ingen objekter blev detekteret.")
