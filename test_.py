import cv2
import numpy as np
import imutils
import time
import random
import urllib


class GBSJ:
    def __init__(self):
        self.info = {"Confidence": 0.2,
                     "URL": "http://172.30.1.45:5000/video_feed",
                     "Model": ["MobileNetSSD_deploy.caffemodel", "pose_iter_160000.caffemodel",
                               "pose_iter_440000.caffemodel", "pose_iter_102000.caffemodel",
                               "res10_300x300_ssd_iter_140000.caffemodel"],
                     # [0] = MobileNetSSD, [1] = pose estimation (MPII)
                     # [2] = pose estimation(coco), [3] = pose estimation(hand)
                     # [4] = object_tracking
                     "Prototxt": ["MobileNetSSD_deploy.prototxt.txt", "pose_deploy_linevec_faster_4_stages.prototxt",
                                  "pose_deploy.prototxt", "deploy.prototxt"],
                     # [0] = object detection, [1] = pose estimation, [2] = hand estimation
                     # [3] = object_tracking
                     "Label": ["background", "aeroplane", "bicycle", "bird", "boat",
                                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                "sofa", "train", "tvmonitor"]}
        self.image_bytes = bytes()

    def GBSJ_seting(self, model_num, prototxt_num):
        print("[GBSJ] : 설정 내용 불러오는중...")
        self.Net = cv2.dnn.readNetFromCaffe(self.info["Prototxt"][prototxt_num], self.info["Model"][model_num])
        # 모델 학습내용 불러오기
        # self.Url = urllib.request.urlopen(self.info["URL"])
        # 웹상에서 사진 얻기 위해 설정
        print("[GBSJ] : 설정 내용 불러옴~...")

    def GBSJ_detection_JPG(self, image):
        frame = imutils.resize(image, width=300)
        # 너비가 300으로 사이즈 맞춤

        (h, w) = frame.shape[:2]
        # shape는 배열의 각차원의 크기를 반환함
        # 고로 frame[0] = h, frame[1] = w 를 의미

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                    0.007843, (300, 300), 127.5)
        # frame을 (300, 300)으로 resize 한후
        # 스케일링값 기본 1.0
        # 크기
        # 평균 빼기 값
        # BGR => RGB 선택

        self.Net.setInput(blob)
        # Net에 bolb으로 입력을 설정함
        self.Result = self.Net.forward()
        # 입력값을 줌 & 결과 얻음

        for i in np.arange(0, self.Result.shape[2]):
            # 0 ~ 감지한 객체의 갯수 만큼의 배열
            confidence = self.Result[0, 0, i, 2]
            # 비슷한 정도를 confidence에 저장

            if confidence > self.info["Confidence"]:

                idx = int(self.Result[0, 0, i, 1])
                # 몇번째 객체인지 idx에 저장
                box = self.Result[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # 객채 위치를 저장

                object = "{name}: {percentage:.2f}%".format(name=self.info["Label"][idx],
                                                            percentage=confidence * 100)
                # Label을 나타내기 위함

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                             (255,0,0), 2)
                # 상자 그리기 image, 사각형의 위치, 색깔, 선의 두께

                object_y = startY - 15 if startY - 15 > 15 else startY + 15
                # Label을 나타낼때 위에 자리가 없으면 상자 안에 나타냄

                cv2.putText(frame, object, (startX, object_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                # 텍스트를 추가함 image, text, text의 왼쪽 아래 위치,
                #                    폰트, 폰트의 배율 인수, 색깔, 두께

        return frame

    def GBSJ_pose_estimation_JPG(self, image, check):
        """
        frame = image

        H,W = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0/255, (368, 368), 127.5)
        self.Net.setInput(blob)
        self.Result = self.Net.forward()
        points = []

        for i in np.arange(self.Result.shape[1]):
            # confidence map of corresponding body's part.
            probMap = self.Result[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (368 * point[0]) / W
            y = (368 * point[1]) / H

            if prob > 0.9:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

            return frame

        """

        frame = imutils.resize(image, width=500)

        frameH = frame.shape[0]
        frameW = frame.shape[1]

        # Prepare the frame to be fed to the network
        inpBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0 / 255, (300, 300), (0, 0, 0), swapRB=False, crop=False)

        # Set the prepared object as the input blob of the network
        self.Net.setInput(inpBlob)
        self.Result = self.Net.forward()

        H = self.Result.shape[2]
        W = self.Result.shape[3]

        # Empty list to store the detected keypoints
        self.points = []

        for i in range(len(self.Result[0])):
            # confidence map of corresponding body's part.
            probMap = self.Result[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image

            x = (point[0] * frameW) / W
            y = (point[1] * frameH) / H

            if prob > 0.1:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                self.points.append((int(x), int(y)))
            else:
                self.points.append(None)

        if check == 0:
            for pair in [[0,1],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[12,13],[9,10],[9,8],[8,14],[14,1],[14,11],[11,12]]:
                partA = pair[0]
                partB = pair[1]

                if self.points[partA] and self.points[partB]:
                    cv2.line(frame, self.points[partA], self.points[partB], (255, 255, 0), 1)
        else:
            for pair in [[0, 1], [1, 2], [2, 3], [3, 4]]:
                partA = pair[0]
                partB = pair[1]

                if self.points[partA] and self.points[partB]:
                    cv2.line(frame, self.points[partA], self.points[partB], (255, 0, 0), 3)
            for pair in [[0, 5], [5, 6], [6, 7], [7, 8]]:
                partA = pair[0]
                partB = pair[1]

                if self.points[partA] and self.points[partB]:
                    cv2.line(frame, self.points[partA], self.points[partB], (236, 220, 21), 3)

            for pair in [[0, 9], [9, 10], [10, 11], [11, 12]]:
                partA = pair[0]
                partB = pair[1]

                if self.points[partA] and self.points[partB]:
                    cv2.line(frame, self.points[partA], self.points[partB], (41, 216, 58), 3)
            for pair in [[0, 13], [13, 14], [14, 15], [15, 16]]:
                partA = pair[0]
                partB = pair[1]

                if self.points[partA] and self.points[partB]:
                    cv2.line(frame, self.points[partA], self.points[partB], (23, 55, 234), 3)
            for pair in [[0, 17], [17, 18], [18, 19], [19, 20]]:
                partA = pair[0]
                partB = pair[1]

                if self.points[partA] and self.points[partB]:
                    cv2.line(frame, self.points[partA], self.points[partB], (175, 0, 175), 3)
        return frame

        """
        # Specify the input image dimensions
        inWidth = 368
        inHeight = 368

        # Prepare the frame to be fed to the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        # Set the prepared object as the input blob of the network
        net.setInput(inpBlob)

        H = out.shape[2]
        W = out.shape[3]
        # Empty list to store the detected keypoints
        points = []
        
        for i in range(len()):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        cv2.imshow("Output-Keypoints", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    def GBSJ_traking_JPG(self, image):

        frame = imutils.resize(image, width=400)

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                     (104.0, 177.0, 123.0))
        self.Net.setInput(blob)
        self.Result = self.Net.forward()
        rects = []

        for i in range(0, self.Result.shape[2]):

            if self.Result[0, 0, i, 2] > self.info["Confidence"]:
                box = self.Result[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))

                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            text = "person : {}".format(objectID + 1)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        return frame

    def GBSJ_image_load(self):
        while True:
            self.image_bytes += self.Url.read(1024)
            # 웹상에서 byte 얻어옴 (1024) : 입력버퍼보다 크면 안댐
            im_staert = self.image_bytes.find(b'\xff\xd8')
            im_end = self.image_bytes.find(b'\xff\xd9')
            # 이미지의 시작과 끝을 찾음
            if im_staert != -1 and im_end != -1:
                jpg = self.image_bytes[im_staert:im_end + 2]
                # 사진 데이터만 저장 (+ 2) : \xff\xd9를 추가 하기 위함
                self.image_bytes = self.image_bytes[im_end + 2:]
                # 남은 byte부터 다시 저장 (+ 2) : \xff\xd9를 빼기 위함
                image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return image

    def Street(self, point_1, point_2):
        D_x = self.points[point_1][0] - self.points[point_2][0]
        D_y = self.points[point_1][1] - self.points[point_2][1]

        return np.sqrt(D_x * D_x + D_y * D_y)

    def GBSJ_rock_scissors_paper_game(self):

        self.hand = []

        if self.points[0] and self.points[10]:
            if self.points[0] and self.points[12]:
                self.hand.append(self.Street(0, 10) > self.Street(0, 12))
            else:
                self.hand.append(None)
        else:
            self.hand.append(None)

        if self.points[0] and self.points[18]:
            if self.points[0] and self.points[20]:
                self.hand.append(self.Street(0, 18) > self.Street(0, 20))
            else:
                self.hand.append(None)
        else:
            self.hand.append(None)

        if self.points[0] and self.points[18]:
            if self.points[0] and self.points[20]:
                self.hand.append(self.Street(0, 18) > self.Street(0, 20))
            else:
                self.hand.append(None)
        else:
            self.hand.append(None)

        if self.points[0] and self.points[18]:
            if self.points[0] and self.points[20]:
                self.hand.append(self.Street(0, 18) > self.Street(0, 20))
            else:
                self.hand.append(None)
        else:
            self.hand.append(None)

        if self.points[0] and self.points[18]:
            if self.points[0] and self.points[20]:
                self.hand.append(self.Street(0, 18) > self.Street(0, 20))
            else:
                self.hand.append(None)
        else:
            self.hand.append(None)



        if self.hand[0] == 1:
            # 바위
            return 1
        else:
            if self.hand[1] == 1:
                # 가위
                return 0
            else:
                # 보
                return 2

    def __del__(self):
        cv2.destroyAllWindows()

"""
Case = ["가위", "바위", "보"]
AI = random.randint(0, 2)
player = 0

A = GBSJ()

A.GBSJ_seting(3, 2)
print("[GBSJ] : 탐지 시작..")
B = cv2.VideoCapture(0)
while True:
    res, frame = B.read()
    frame = imutils.resize(frame, width=500)
    # frame = A.GBSJ_image_load()
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break
    if cv2.waitKey(1) == ord("a"):
        image = A.GBSJ_pose_estimation_JPG(frame, 1)
        cv2.imshow("detection", image)
        player = A.GBSJ_rock_scissors_paper_game()
        if player == -1:
            print("[GBSJ] : player 인식 실패..")
        else:
            print("[GBSJ] : player = " + Case[player])

if player == -1:
    print("[GBSJ] : player 인식 실패..")

    B.release()
    cv2.destroyAllWindows()

    exit()

print("AI = " + Case[AI])
print("player = " + Case[player])


if AI == player:
    print("DRAW!!!")
else:
    if AI == 0:
            if player == 1:
                print("WIN!!!")
            else:
                if player == 2:
                    print("LOST!!!")
    else:
        if AI == 1:
            if player == 0:
                print("LOST!!!")
            else:
                if player == 2:
                    print("WIN!!!")
        else:
            if player == 0:
                    print("WIN!!!")
            else:
                if player == 1:
                    print("LOST!!!")

B.release()
cv2.destroyAllWindows()


"""

"""
A = GBSJ()

A.GBSJ_seting(0, 0)
print("dsa")
B = cv2.VideoCapture(0)
while True:
    res, frame = B.read()
    frame = imutils.resize(frame, width=368)
    # frame = A.GBSJ_image_load()
    image = A.GBSJ_detection_JPG(frame)
    cv2.imshow("detection", image)
    if cv2.waitKey(1) == 27:
        break
    if cv2.waitKey(1) == ord("a"):
        image = A.GBSJ_pose_estimation_JPG(frame, 1)
        cv2.imshow("detection", image)

B.release()
cv2.destroyAllWindows()
"""

"""

1


검출 & 포인트 그리기

포인트 이어주기 (선 그리기)
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]
 
    if points[partA] and points[partB]:
        cv2.line(frameCopy, points[partA], points[partB], (0, 255, 0), 3)

"""


A = GBSJ()

A.GBSJ_seting(4, 3)
print("dsa")
B = cv2.VideoCapture(0)

from pyimagesearch.centroidtracker import CentroidTracker
ct = CentroidTracker(5)

while True:
    _, frame = B.read()

    image = A.GBSJ_traking_JPG(frame)

    cv2.imshow("detection", image)
    if cv2.waitKey(1) == 27:
        break

B.release()
cv2.destroyAllWindows()