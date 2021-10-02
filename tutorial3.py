"""
################# 직선 검출 ########################### 
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
 
################# ROI 직선 검출 ########################### 

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt
capture = cv2.VideoCapture('videos/road.mp4')
while True:
    ret, frame = capture.read()
    h, w, _ = frame.shape
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    edges = cv2.Canny(src, 50, 150)
    dst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('edges',edges)
    vertices = np.array([[(200,h-70),(w/2+80, h/2+20),(w/2+150, h/2+20), (w-100,h-70)]], dtype=np.int32)
    mask = np.zeros_like(edges) # mask = img와 같은 크기의 빈 이미지

    if len(edges.shape) > 2: 
        color = ( 0,0,255)
    else: 
        color = 255 # 흑백
    cv2.fillPoly(mask, vertices, color) #다각형부분(ROI 설정부분) color로 채움

    ROI_image = cv2.bitwise_and(edges, mask) # 이미지와 color로 채워진 ROI를 합침
    cv2.imshow('mask',mask)
    cv2.imshow('ROI_image',ROI_image)
    lines = cv2.HoughLinesP(ROI_image, 1, np.pi / 180., 160, minLineLength=100, maxLineGap=5)

    if lines is not None:
        for i in range(0,len(lines)):
            l = lines[i][0]
            pt1 = (lines[i][0][0], lines[i][0][1]) # 시작점 좌표
            pt2 = (lines[i][0][2], lines[i][0][3]) # 끝점 좌표
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('src', frame)
    cv2.imshow('dst', cv2.add(frame, dst))
    key = cv2.waitKey(33)
    if key == 27:
        break
cv2.waitKey()
cv2.destroyAllWindows()

 
################# 원 검출 ########################### 미완성
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

cv2.namedWindow('img')
src = cv2.imread('images/coins1.jpg')
cv2.imshow('img', src)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blr = cv2.GaussianBlur(gray, (0, 0), 1.0)

def on_trackbar(pos) :
    rmin = cv2.getTrackbarPos('minRadius', 'img')
    rmax = cv2.getTrackbarPos('maxRadius', 'img')
    th = cv2.getTrackbarPos('threshold', 'img')
    circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2 = th, minRadius=rmin, maxRadius=rmax)
    dst = src.copy() 
    if circles is not None:
        sum_of_money = 0
        for i in range(circles.shape[1]): # 조심
            cx, cy, radius = circles[0][i]
            cv2.circle(dst, (int(cx), int(cy)),int(radius), (0, 0, 255), 3, cv2.LINE_AA)

            x1 = int(cx - radius) ; y1 = int(cy-radius)
            x2 = int(cx + radius) ; y2 = int(cy+radius)
            radius = int(radius)

            crop = dst[y1:y2, x1:x2, :]
            ch, cw = crop.shape[:2]
            mask = np.zeros((ch, cw),np.uint8)
            cv2.circle(mask,(cw//2,ch//2),radius,255,-1)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hue, _, _ = cv2.split(hsv)
            hue_shift = (hue + 40) % 180
            # Hue 평균이 140보다 크면 500원, 139.1보다 크면 100원
            mean_of_hue = cv2.mean(hue_shift, mask)[0]
            if mean_of_hue > 140:
                won = 500
            elif mean_of_hue > 139.1:
                won = 100
            elif mean_of_hue > 100:
                won = 50
            else:
                won = 10
            sum_of_money += won
    else:
        print('no coin')
    cv2.imshow('img', dst)

cv2.imshow('img', src)
cv2.createTrackbar('minRadius', 'img', 0, 100, on_trackbar)
cv2.createTrackbar('maxRadius', 'img', 0, 150, on_trackbar)
cv2.createTrackbar('threshold', 'img', 0, 100, on_trackbar)
cv2.setTrackbarPos('minRadius', 'img', 10)
cv2.setTrackbarPos('maxRadius', 'img', 80)
cv2.setTrackbarPos('threshold', 'img', 40)

cv2.waitKey()
cv2.destroyAllWindows()


######################## 특 징 점  #################################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

src1 = cv2.imread('images/hough2.jpg', cv2.IMREAD_GRAYSCALE)
#src2 = cv2.imread('images/hough_resize.jpg', cv2.IMREAD_GRAYSCALE)

feature1 = cv2.SIFT_create()
feature2 = cv2.KAZE_create()
feature3 = cv2.AKAZE_create()
feature4 = cv2.ORB_create()


kp1 = feature1.detect(src1)
print(kp1[0])
kp2 = feature2.detect(src1)
kp3 = feature3.detect(src1)
kp4 = feature4.detect(src1)

dst1 = cv2.drawKeypoints(src1, kp1, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst2 = cv2.drawKeypoints(src1, kp2, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst3 = cv2.drawKeypoints(src1, kp3, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst4 = cv2.drawKeypoints(src1, kp4, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift', dst1)
cv2.imshow('kaze', dst2)
cv2.imshow('akaze', dst3)
cv2.imshow('orb', dst4)

cv2.waitKey()
cv2.destroyAllWindows()
 

######################### Homography ##################################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

src1 = cv2.imread('images/book.jpg', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('images/book3.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('org',src1)
cv2.imshow('org2',src2)

feature1 = cv2.SIFT_create()
feature2 = cv2.KAZE_create()
feature3 = cv2.AKAZE_create()
feature4 = cv2.ORB_create()


kp1, desc1 = feature2.detectAndCompute(src1, None)
kp2, desc2 = feature2.detectAndCompute(src2, None)

dst1 = cv2.drawKeypoints(src1, kp1, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst2 = cv2.drawKeypoints(src1, kp2, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('sift', dst1)
cv2.imshow('kaze', dst2)

matcher = cv2.BFMatcher_create()
matches = matcher.match(desc1, desc2)

matches = sorted(matches, key = lambda x: x.distance)
good_matches = matches[:80]

print('kp1', len(kp1))
print('kp2', len(kp2))
print('matches', len(matches))
print('good matches', len(good_matches))

pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

dst = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
(h, w) = src1.shape[:2]
corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
corners2 = cv2.perspectiveTransform(corners1, H)
cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
################################# 호모그래피 ###########################################
import sys
import numpy as np
import cv2

# 기준 영상 불러오기
src = cv2.imread('images/2021-9-27.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# 카메라 장치 열기
cap1 = cv2.VideoCapture(0)

if not cap1.isOpened():
    print('Camera open failed!')
    sys.exit()

# 필요할 경우 카메라 해상도 변경
#cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# 카메라 프레임 화면에 출력할 동영상 파일 열기
cap2 = cv2.VideoCapture('videos/seoul.mp4')

w = round(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'w: {w}, h: {h}')

if not cap2.isOpened():
    print('Video load failed!')
    sys.exit()

# AKAZE 특징점 알고리즘 객체 생성
detector = cv2.AKAZE_create()

# 기준 영상에서 특징점 검출 및 기술자 생성
kp1, desc1 = detector.detectAndCompute(src, None)

# 해밍 거리를 사용하는 매칭 객체 생성
matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

while True:
    ret1, frame1 = cap1.read()

    if not ret1:
        break

    # 매 프레임마다 특징점 검출 및 기술자 생성
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = detector.detectAndCompute(gray, None)
    print(len(kp2))
    # 특징점이 100개 이상 검출될 경우 매칭 수행
    if len(kp2) > 100:
        matches = matcher.match(desc1, desc2)

        # 좋은 매칭 선별
        matches = sorted(matches, key=lambda k: k.distance)
        print('match',len(matches))
        good_matches = matches[:80]

        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

        # 호모그래피 계산
        H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)

        inlier_cnt = cv2.countNonZero(inliers)

        # RANSAC 방법에서 정상적으로 매칭된 것의 개수가 20개 이상이면
        if inlier_cnt > 20:
            ret2, frame2 = cap2.read()

            if not ret2:
                break

            h, w = frame1.shape[:2]

            # 비디오 프레임을 투시 변환
            video_warp = cv2.warpPerspective(frame2, H, (w, h))

            white = np.full(frame2.shape[:2], 255, np.uint8)
            white = cv2.warpPerspective(white, H, (w, h))

            # 비디오 프레임을 카메라 프레임에 합성
            cv2.copyTo(video_warp, white, frame1)

    cv2.imshow('frame', frame1)
    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows() 