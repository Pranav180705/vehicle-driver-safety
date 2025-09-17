from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")  # Pretrained COCO model

cap = cv2.VideoCapture("bike_video.mp4")  # Replace with your video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLO detection

    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls)]
            conf = float(box.conf)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            label = f"{cls} {conf:.2f}"
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    cv2.imshow("Helmet & Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
