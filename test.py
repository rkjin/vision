import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D','I','V','X'
out = cv2.VideoWriter('output.avi', fourcc, 30, (w, h))
while True:
    # 카메라 프레임 처리
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(33) == 27:
        break