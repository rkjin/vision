"""
############################ knn 딥 런닝 ################################# ?????

import numpy as np 
import cv2

def on_k_changed(pos):
    global k_value
    k_value = pos
    if k_value > 1:
        k_value = 1
    trainAndDisplay()

def trainAndDisplay():
    trainData = np.array(train, dtype=np.float32)
    labelData = np.array(label, dtype=np.int32)

    knn.train(trainData, cv2.ml.ROW_SAMPLE, labelData)

    h, w = img.shape[ :2]
    for y in range(h):
        for x in range(w):
            sample = np.array([[x,y]]).astype(np.float32)
            ret,_,_,_ = knn.findNearest(sample, k_value)
            ret = int(ret)
            if ret == 0:
                img[y,x]=(128,128,255)
            elif ret == 1 :
                img[y,x]=(128,255,128)
            elif ret == 2 :
                img[y,x]=(255,128,255)

    for i in range(len(train)):
        x, y = train[i]
        l = label[i][0]
        if  l == 0:
            cv2.circle(img, (x,y), 5,(0,0,128),-1,cv2.LINE_AA )
        elif  l == 1:
            cv2.circle(img, (x,y), 5,(0,0,128),-1,cv2.LINE_AA )
        elif  l == 2:
            cv2.circle(img, (x,y), 5,(0,0,128),-1,cv2.LINE_AA )
    cv2.imshow('knn',img)

k_value = 1
img = np.full((500, 500, 3), 255, np.uint8)
knn = cv2.ml.KNearest_create()


# 학습용 랜덤 데이터 생성: 90개의 임의의 좌표 데이터 
# 학습 데이터 & 레이블
train = []
label = []

NUM = 30
rn  = np.zeros((NUM, 2), np.int32)
pos = [(150,150), (350, 150), (250, 400)]  
c_  = [0, 1, 2]   # 데이터 분류 레이블
rn = np.zeros((NUM, 2), np.int32)

# 총 90개의 임의의 좌표 데이터 생성
for idx, c in enumerate(c_):
    cv2.randn(rn, 0, 50)    #평균0, 표준편차50 정규분포를 따르는 난수 발생
    for i in range(NUM):
        train.append([rn[i, 0] + pos[idx][0], rn[i, 1] + pos[idx][1]])
        label.append([c])
cv2.namedWindow('knn')
cv2.createTrackbar('k_value', 'knn', 1, 5, on_k_changed)

# KNN 결과 출력
trainAndDisplay()

cv2.waitKey()
cv2.destroyAllWindows()

############################### Hog #######################################

import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('videos/india.mp4')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    detected, _ = hog.detectMultiScale(frame)
    for (x,y, w, h) in detected:
        c = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.rectangle(frame,(x,y,w,h),c,3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(10)==27:
        break
cv2.destroyAllWindows()
"""
############################필기체 숫자 인식##################################
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt


def on_mouse(event, x, y, flags, param):
    global old_x, old_y, img
    if event == cv2.EVENT_LBUTTONDOWN:
        old_x, old_y = x, y
    if flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.line(img,(old_x, old_y),(x,y),(255),cv2.LINE_8)
        cv2.imshow('img',img)
        old_x, old_y = x, y 

digits = cv2.imread('images/digits.png', cv2.IMREAD_GRAYSCALE)
h, w = digits.shape[:2] 
hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)

cells = [np.hsplit(row, w//20) for row in np.vsplit(digits, h//20)]
cells = np.array(cells)
cells = cells.reshape(-1, 20, 20) # shape=(5000, 20, 20)

desc = []
for img in cells:
    desc.append(hog.compute(img)) 

train_desc = np.array(desc)
train_desc = train_desc.squeeze().astype(np.float32)
train_labels = np.repeat(np.arange(10), len(train_desc)/10)

svm = cv2.ml.SVM_create()

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(2.5)
svm.setGamma(0.50625)
svm.train(train_desc, cv2.ml.ROW_SAMPLE, train_labels)
svm.save('svmdigits.yml')
img = np.zeros((400, 400), np.uint8)
cv2.imshow('img', img)
cv2.setMouseCallback('img', on_mouse)
while True:
    key = cv2.waitKey()
    if key == 27:
        break
    elif key == ord(' '):
        test_image = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        test_desc = hog.compute(test_image).T
        _, res = svm.predict(test_desc)
        print(f'분류된 결과 숫자는: {int(res)}')
        img.fill(0)
        cv2.imshow('img', img)
cv2.destroyAllWindows()
"""
############################필기체 숫자 인식##################################
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt


def on_mouse(event, x, y, flags, param):
    global old_x, old_y, img
    if event == cv2.EVENT_LBUTTONDOWN:
        old_x, old_y = x, y
    if flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.line(img,(old_x, old_y),(x,y),(255),cv2.LINE_8)
        cv2.imshow('img',img)
        old_x, old_y = x, y 

digits = cv2.imread('images/digits.png', cv2.IMREAD_GRAYSCALE)
h, w = digits.shape[:2] 
hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)

cells = [np.hsplit(row, w//20) for row in np.vsplit(digits, h//20)]
cells = np.array(cells)
cells = cells.reshape(-1, 20, 20) # shape=(5000, 20, 20)

desc = []
for img in cells:
    desc.append(hog.compute(img)) 

train_desc = np.array(desc)
train_desc = train_desc.squeeze().astype(np.float32)
train_labels = np.repeat(np.arange(10), len(train_desc)/10)

#svm = cv2.ml.SVM_create()
svm = cv2.ml.SVM_load('xml/svmdigits.yml')

img = np.zeros((400, 400), np.uint8)
cv2.imshow('img', img)
cv2.setMouseCallback('img', on_mouse)
while True:
    key = cv2.waitKey()
    if key == 27:
        break
    elif key == ord(' '):
        test_image = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        test_desc = hog.compute(test_image).T
        _, res = svm.predict(test_desc)
        print(f'분류된 결과 숫자는: {int(res)}')
        img.fill(0)
        cv2.imshow('img', img)
cv2.destroyAllWindows()



############################필기체 숫자 인식##################################
import cv2, os, sys, datetime ,random, math
import numpy as np
import matplotlib.pyplot as plt

# 1.입력 영상 불러오기
#capture = cv2.VideoCapture('videos/india.mp4')
#ret, img = capture.read()
filename = 'images/shuttle.jpg'
#filename = 'images/rose.bmp'
# classification_classes_ILSVRC2012.txt 파일에 없음
img = cv2.imread(filename)
if img is None:
    print('Image load failed!')
    sys.exit()
# 2.네트워크 불러오기
# Caffe
model = 'bvlc_googlenet.caffemodel'
config = 'deploy.prototxt.txt'
# ONNX
#model = 'googlenet/inception-v1-9.onnx'
#config = ''
net = cv2.dnn.readNet(model, config)
if net.empty():
    print('Network load failed!')
    sys.exit()

classNames = None
with open('classification_classes_ILSVRC2012.txt', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# 4.추론(예측)
blob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
net.setInput(blob)
prob = net.forward()

# 5.추론 결과 확인 & 화면 출력
out = prob.flatten()
classId = np.argmax(out)
confidence = out[classId]
text = f'{classNames[classId]} ({confidence * 100:4.2f}%)'
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1,
cv2.LINE_AA)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()


#############################음성####################################################
import win32com.client
tts = win32com.client.Dispatch("SAPI.SpVoice")
tts.Speak("마이크로소프트")

"""