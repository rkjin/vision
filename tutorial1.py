import cv2
import numpy as np

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    key = cv2.waitKey(33)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
