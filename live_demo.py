import cv2

cam = cv2.VideoCapture('rtsp://admin:admin@123@192.168.1.245:554/cam/realmonitor?channel=8&subtype=0')
while True:
    cv2.imshow('w', cam.read()[1])
    cv2.waitKey(0)