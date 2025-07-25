import cv2
from ultralytics import YOLO
model = YOLO("runs/classify/scrap_classifier/weights/best.pt") 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Error: Could not open webcam.")
    exit()

print(" Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize for faster processing
    resized = cv2.resize(frame, (128, 128))

    # Predict using YOLOv8 classifier
    results = model(resized, imgsz=128)

    # Get top prediction
    top1 = results[0].probs.top1  # index of top class
    class_name = model.names[top1]  # class label

    # Display result
    cv2.putText(frame, f'Predicted: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(" Scrap Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(" Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
