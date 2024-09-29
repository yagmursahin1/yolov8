from ultralytics import YOLO
import cv2

model=YOLO('/Users/yagmursahin/Desktop/allpython/qrcode/best2.pt')
red=[0,0,255]
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    if not ret:
        break 

    results=model(frame)
    for result in results:
        annotated_frame=result.plot()

    cv2.imshow('Frame',annotated_frame)
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows