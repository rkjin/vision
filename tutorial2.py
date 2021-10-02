"""
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('images/lenna.bmp', cv2.IMREAD_COLOR)
colors = ['b','g','r']
bgr_planes = cv2.split(src)

for (p, c,d) in zip(bgr_planes, colors,colors):
    hist = cv2.calcHist([p],[0],None,[256],[0,255])
    plt.plot(hist, color=c)

plt.show()

cv2.destroyAllWindows()    


###################스펙트럼 늘이기, 평탄화 ######################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('images/field.bmp', cv2.IMREAD_GRAYSCALE)
src = cv2.add(src, 50)
low, high = 75, 225
idx = np.arange(0,256)
idx = 255/(high - low)*(idx -low)
idx[0:int(low)] = 0
idx[int(high +1):] =255
dst = cv2.LUT(src,idx.astype('uint8'))
dst2 = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)
dst3 = cv2.equalizeHist(src)
cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.imshow('dst2',dst)
cv2.imshow('dst3',dst)


hist = cv2.calcHist([src],[0],None,[256],[0,255])
plt.plot(hist, color='r')
hist = cv2.calcHist([dst],[0],None,[256],[0,255])
plt.plot(hist, color='g')
#hist = cv2.calcHist([dst2],[0],None,[256],[0,255])
#plt.plot(hist, color='b')
hist = cv2.calcHist([dst3],[0],None,[256],[0,255])
plt.plot(hist, color='b')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

#########################색갈 분리#############################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('images/candies.png', cv2.IMREAD_COLOR)
src_hvs = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)


def on_trackbar(pos):
    hmin = cv2.getTrackbarPos('H_min','dst')
    hmax = cv2.getTrackbarPos('H_max','dst')
    dst = cv2.inRange(src_hvs, (hmin,150,0),(hmax,255,255))
    cv2.imshow('dst',dst)

cv2.imshow('src',src)
#cv2.imshow('dst',dst)
cv2.namedWindow('dst')
cv2.createTrackbar('H_min','dst',50,179,on_trackbar)
cv2.createTrackbar('H_max','dst',80,179,on_trackbar)

cv2.waitKey(0)
cv2.destroyAllWindows()
""" """
##################### 크로마키 ############################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

capw = cv2.VideoCapture('videos/woman.mp4')
capr = cv2.VideoCapture('videos/raining.mp4')
flag = True
 
while True:
    ret, framew = capw.read()
    ret, framer = capr.read()
    framew_hsv = cv2.cvtColor(framew,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(framew_hsv,(48,150,0),(70,255,255))

    
    print(flag)
    if flag:
        frame = cv2.copyTo(framer, mask, framew)
        cv2.imshow('frame',frame)
    else:
        cv2.imshow('frame',framer)
    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == ord(' '): 
        flag = not flag 
 
capw.release()
capr.release()
cv2.destroyAllWindows()    

########################## bluring ###############################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('images/lenna.bmp', cv2.IMREAD_COLOR)
mask = np.array([[1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9]])

dst = cv2.filter2D(src,-1,mask)

dst2 = cv2.blur(src,(7,7))
for s in range(1,30,5):
    dst3 = cv2.GaussianBlur(src,(7,7),s,s,None)
    cv2.putText(dst3,f'sigma {s}',(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1, cv2.LINE_AA)
    cv2.imshow(f'sigma {s}',dst3)

cv2.imshow('src',src)
cv2.imshow('filter2D',dst)
cv2.imshow('blur',dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()

######################## blur #######################################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('images/lenna.bmp', cv2.IMREAD_COLOR)
mask = np.array([[0, -1, 0],
                [-1, 5, -1],
                [0,-1, 0]])

dst = cv2.filter2D(src,-1,mask)
dst2 = cv2.blur(src,(7,7))
dst3 = np.clip(2*src-dst2,0,255).astype(np.uint8)

cv2.imshow('src',src)
cv2.imshow('filter2D',dst)
cv2.imshow('blur',dst2)
cv2.imshow('sharp after blur',dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()

########################## MediaBlur ####################################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('images/lenna_noise.bmp', cv2.IMREAD_GRAYSCALE)

dst = cv2.medianBlur(src,5 )
dst2 = cv2.blur(src, (5,5))

cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.imshow('dst2',dst2)
#cv2.imshow('dst3',dst3)
cv2.waitKey(0)
cv2.destroyAllWindows()

########################### 카툰, 스케치 ###############################
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

def cartoon_filter(img):
    h, w,_ = img.shape
    img2 = cv2.resize(img, (w//4, h//4))
    blr = cv2.bilateralFilter(img2, -1, 20, 7)
    edge = 255 - cv2.Canny(img2, 80, 120)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    dst = cv2.bitwise_and(blr, edge)
    dst = cv2.resize(dst, (w, h), interpolation=cv2.INTER_NEAREST)
    return dst

def pencil_sketch_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 3)
    dst = cv2.divide(gray, blr, scale=255)
    return dst
flag = True
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if flag:
        cv2.imshow('frame',cartoon_filter(img))
    else:
        cv2.imshow('frame',pencil_sketch_filter(img))
    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == ord(' '): 
        flag = not flag 

cv2.waitKey()
cv2.destroyAllWindows()
 
#############################################################################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt


flag = True
cap = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
while True:
    ret, img = cap.read()

    if flag:
        img = cv2.erode(img, kernel,iterations=2)
        img = cv2.dilate(img,kernel,iterations=2)
        cv2.imshow('frame',img)
    else:
        img = cv2.dilate(img,kernel,iterations=2)
        img = cv2.erode(img, kernel,iterations=2)
        cv2.imshow('frame',img)
    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == ord(' '): 
        flag = not flag 

cv2.waitKey()
cv2.destroyAllWindows()


######################## ??? ##########################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt


flag = True
cap = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
while True:
    ret, img = cap.read()

    if flag:
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame',img)
    else:
        img = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)
        cv2.imshow('frame',img)
    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == ord(' '): 
        flag = not flag 

cv2.waitKey()
cv2.destroyAllWindows()

 
################# resize ###########################
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt


flag = True
cap = cv2.VideoCapture(0)
ret, img = cap.read()
aff = np.array([[1, 0, 0],[0.1 , 1, 0]],dtype = np.float32)
dst = cv2.warpAffine(img, aff,(0,0))
dst2 = cv2.resize(img,(0,0), fx = 1.5, fy = 1.5, interpolation=cv2.INTER_NEAREST)
dst3 = cv2.resize(img, (0,0), fx = 1.5, fy = 1.5, interpolation=cv2.INTER_AREA)
dst4 = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LANCZOS4) 
dst5 = cv2.flip(img,-1)
cv2.imshow('img',img)
cv2.imshow('dat',dst)
cv2.imshow('dat2',dst2)
cv2.imshow('dat3',dst3)
cv2.imshow('dat4',dst4)
cv2.imshow('dat5',dst5)
cv2.waitKey()
cv2.destroyAllWindows()

 
################# resize ###########################
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt


flag = True
cap = cv2.VideoCapture(0)
ret, img = cap.read()
aff = np.array([[1, 0, 0],[0.1 , 1, 0]],dtype = np.float32)
dst = cv2.warpAffine(img, aff,(0,0))
dst2 = cv2.resize(img,(0,0), fx = 1.5, fy = 1.5, interpolation=cv2.INTER_NEAREST)
dst3 = cv2.resize(img, (0,0), fx = 1.5, fy = 1.5, interpolation=cv2.INTER_AREA)
dst4 = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LANCZOS4) 
dst5 = cv2.flip(img,-1)
cv2.imshow('img',img)
cv2.imshow('dat',dst)
cv2.imshow('dat2',dst2)
cv2.imshow('dat3',dst3)
cv2.imshow('dat4',dst4)
cv2.imshow('dat5',dst5)
cv2.waitKey()
cv2.destroyAllWindows()

################# pyramid down up ###########################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt
 
src = cv2.imread('images/butterfly.jpg')
rc = (280,150,200,200)
cpy = src.copy()``
cv2.rectangle(cpy, rc, (0,0,255),2)
cv2.imshow('src', cpy)
cv2.waitKey()

for i in range (1, 4):
    src = cv2.pyrUp(src)
    cpy =src.copy()
    cv2.rectangle(cpy, rc, (0,0,255),2, shift= i )
    cv2.imshow('src', cpy)
    cv2.waitKey()
    cv2.destroyAllWindow('src')

cv2.destroyAllWindows()
 
################# pyramid down up ###########################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt


flag = True
cap = cv2.VideoCapture(0)
ret, img = cap.read()
theta = 30
theta_rad = theta / 180 * np.pi
aff = np.array([[np.cos(theta_rad), np.sin(theta_rad), 0],[-np.sin(theta_rad) , np.cos(theta_rad), 0]],dtype = np.float32)
dst = cv2.warpAffine(img, aff,(0,0))
w, h = img.shape[:2]
aff2 = cv2.getRotationMatrix2D((h/2,w/2), 30,0.5) 
dst2 = cv2.warpAffine(img, aff2,(0,0))
cv2.imshow('img',img)
cv2.imshow('dat',dst)
cv2.imshow('dat2',dst2)

cv2.waitKey()
cv2.destroyAllWindows()
 
################# perspective transform ###########################
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

 
img = cv2.imread('name2.jpg')
cv2.imshow('img',img)
dst = np.array([[274,109],[565,220],[474,440],[140,260]], np.float32)
src = np.array([[100,100],[600,100],[600,500],[100,500]],np.float32)
per = cv2.getPerspectiveTransform(src, dst)
dst = cv2.warpPerspective(img, per,(700,600))

cv2.imshow('dat',dst)

cv2.waitKey()
cv2.destroyAllWindows()
 """
################# perspective transform ########################### 작업중

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

p = [] 
def mouse(event, x, y, flags, param):
    global p, img_per
    if event ==  cv2.EVENT_LBUTTONDOWN:
        if len(p) < 4 and first:
            p.append([x, y])
            cv2.circle(blk, (x, y),25,(0, 0, 255), -1)
            cv2.polylines(blk, np.array([p]) ,False, (0,0,255),2)
            cv2.imshow('Double click to transform image',img)    
            if len(p) == 4:
                cv2.polylines(blk, np.array([p]),True, (0, 0, 255)) 
                init  = False
            img2 = cv2.addWeighted(img, 0.75, blk, 0.75, 0.0) 
            cv2.imshow('Double click to transform image',img2)   
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            for i in range(4):
                blk[:,:,:] = 0    
                if ((x > (p[i][0] -25)) and (x < (p[i][0] + 25))) and ((y > (p[i][1]  -25)) and (y < (p[i][1]  + 25))):
                    p[i] = [x, y]
                    for i in range(4):
                        cv2.circle(blk, p[i],25,(0, 0, 255), -1)
                    cv2.polylines(blk, np.array([p]) ,True, (0,0,255),2)
                    img2 = cv2.addWeighted(img, 0.75, blk, 0.75, 0.0) 
                    cv2.imshow('Double click to transform image',img2)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        dst = np.array([[100,100],[600,100],[600,500],[100,500]],np.float32)
        aff = cv2.getPerspectiveTransform(np.array([p],np.float32),dst)
        img_per = cv2.warpPerspective(img, aff, (700,600), flags = cv2.INTER_LINEAR)
        cv2.imshow('Double click to save Perspective',img_per)

def on_level_change(pos ):
    global result # function 에 선언해야 함.
    w = cv2.getTrackbarPos('W', 'Double click to save Perspective') 
    h = cv2.getTrackbarPos('H', 'Double click to save Perspective')   
    result = cv2.resize(img_per, dsize = (0,0),fx = w/100, fy=h/100,interpolation=cv2.INTER_LINEAR)
    # result 가 resize 안하면  blank 임
    cv2.imshow('Double click to save Perspective',result) 
    #cv2.imwrite('images/perspective.jpg',result) 

def save_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.imshow('Double click to save Perspective',result) 
        cv2.imwrite('images/perspective.jpg',result) 
        print('hahaha')   

global w, h, blk, first 
img = cv2.imread('name.jpg')
first = True
w, h, c = img.shape
blk = np.zeros((w, h, c), np.uint8)
result = np.zeros((w, h, c), np.uint8) 
cv2.imshow('Double click to transform image',img)
cv2.namedWindow('Double click to save Perspective',cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Double click to transform image',mouse, img)
cv2.createTrackbar('W','Double click to save Perspective',100,200, on_level_change )
cv2.createTrackbar('H','Double click to save Perspective',100,200, on_level_change )
cv2.setMouseCallback('Double click to save Perspective',save_mouse)
while True:
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows() 

"""
################# perspective transform ########################### 
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt
capture = cv2.VideoCapture('videos/road.mp4')
while True:
    ret, frame = capture.read()
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(src, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180., 160, minLineLength=100, maxLineGap=5)

    dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    minLineLength=100,
    maxLineGap=5
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1]) # 시작점 좌표
            pt2 = (lines[i][0][2], lines[i][0][3]) # 끝점 좌표
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('src', frame)
    cv2.imshow('dst', cv2.add(frame, dst))
    key = cv2.waitKey(33)
    if key == 27:
        break
cv2.waitKey()
cv2.destroyAllWindows()
""" 