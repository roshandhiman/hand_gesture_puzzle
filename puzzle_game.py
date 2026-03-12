import cv2
import numpy as np
import mediapipe as mp
import random
import time

GRID=3
SIZE=540
TILE=SIZE//GRID

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(
max_num_hands=1,
min_detection_confidence=0.7,
min_tracking_confidence=0.7)

mp_draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)
cap.set(3,SIZE)
cap.set(4,SIZE)

captured=False
pieces=None
correct=None
selected=None
dragging=False
start_time=None

def split_image(img):
    parts=[]
    for y in range(GRID):
        for x in range(GRID):
            parts.append(img[y*TILE:(y+1)*TILE,x*TILE:(x+1)*TILE])
    return parts

def shuffle(parts):
    shuffled=parts.copy()
    random.shuffle(shuffled)
    return shuffled

def draw_board(parts):
    rows=[]
    for i in range(GRID):
        rows.append(np.hstack(parts[i*GRID:(i+1)*GRID]))
    return np.vstack(rows)

def fist(hand):
    tips=[8,12,16,20]
    closed=0
    for t in tips:
        if hand.landmark[t].y>hand.landmark[t-2].y:
            closed+=1
    return closed>=3

def solved():
    for i in range(len(pieces)):
        if not np.array_equal(pieces[i],correct[i]):
            return False
    return True

while True:

    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(SIZE,SIZE))

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)

    if not captured:

        cv2.rectangle(frame,(10,10),(200,60),(0,255,0),-1)
        cv2.putText(frame,"CAPTURE",(40,45),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

                h,w,_=frame.shape
                x=int(hand.landmark[8].x*w)
                y=int(hand.landmark[8].y*h)

                if 10<x<200 and 10<y<60 and fist(hand):
                    captured=True
                    correct=split_image(frame.copy())
                    pieces=shuffle(correct)
                    start_time=time.time()

    else:

        board=draw_board(pieces)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

                h,w,_=frame.shape
                x=int(hand.landmark[8].x*w)
                y=int(hand.landmark[8].y*h)

                gx=x//TILE
                gy=y//TILE
                idx=gy*GRID+gx

                if fist(hand):

                    if not dragging:
                        selected=idx
                        dragging=True

                else:

                    if dragging:
                        if selected!=idx:
                            pieces[selected],pieces[idx]=pieces[idx],pieces[selected]
                        dragging=False
                        selected=None

                if selected is not None:
                    sx=(selected%GRID)*TILE
                    sy=(selected//GRID)*TILE
                    cv2.rectangle(board,(sx,sy),(sx+TILE,sy+TILE),(0,255,0),4)

                cv2.circle(board,(x,y),10,(255,0,0),-1)

        elapsed=int(time.time()-start_time)
        remaining=max(0,120-elapsed)

        cv2.putText(board,f"Time:{remaining}",(10,35),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        if solved():
            cv2.putText(board,"PUZZLE SOLVED!",
            (90,260),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4)

        elif remaining==0:
            cv2.putText(board,"GAME OVER",
            (110,260),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

        cv2.namedWindow("Puzzle", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Puzzle", SIZE, SIZE)
        cv2.imshow("Puzzle", board)

    cv2.imshow("Camera",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
