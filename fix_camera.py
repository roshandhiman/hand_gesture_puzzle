import cv2
import numpy as np

cap=cv2.VideoCapture(0)

SIZE=600

while True:

    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)

    h,w,_=frame.shape

    # crop center square
    min_dim=min(h,w)
    start_x=(w-min_dim)//2
    start_y=(h-min_dim)//2

    frame=frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

    frame=cv2.resize(frame,(SIZE,SIZE))

    cv2.imshow("Camera",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
