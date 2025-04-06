import cv2 

cap = cv2.VideoCapture('https://192.168.00.00:8080/video')
cv2.namedWindow('live cam', cv2.WINDOW_NORMAL)  

while(True):
    ret, frame = cap.read()
    img_resize = cv2.resize(frame, (960, 540))
    cv2.imshow('live cam', img_resize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()