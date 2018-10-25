import cv2
import numpy as np
import imutils
import time
import random
import urllib


class GBSJ:
    def __init__(self):
        self.info = {"Confidence": 0.4,
                     "URL": "http://172.30.1.45:5000/video_feed",
                     "Model": ["MobileNetSSD_deploy.caffemodel", "pose_iter_160000.caffemodel",
                               "pose_iter_440000.caffemodel", "pose_iter_102000.caffemodel",
                               "res10_300x300_ssd_iter_140000.caffemodel", "tiny_yolo.caffemodel",
                               "yolov3.cfg.txt"],
                     # [0] = MobileNetSSD, [1] = pose estimation (MPII)
                     # [2] = pose estimation(coco), [3] = pose estimation(hand)
                     # [4] = object_tracking, [5] = tiny_yolo, [6] = yolov3
                     "Prototxt": ["MobileNetSSD_deploy.prototxt.txt", "pose_deploy_linevec_faster_4_stages.prototxt",
                                  "pose_deploy.prototxt", "deploy.prototxt", "tiny_yolo_deploy.prototxt",
                                  "yolov3.weights"],
                     # [0] = object detection, [1] = pose estimation, [2] = hand estimation
                     # [3] = object_tracking, [4] = tiny_yolo, [5] = yolov3
                     "Label": ["background", "aeroplane", "bicycle", "iny_yolo_deploybird", "boat",
                                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                "sofa", "train", "tvmonitor"],
                     "LabelYolo": [
                         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                         "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                         "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                         "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                         "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                         "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                         "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                         "scissors", "teddy bear", "hair drier", "toothbrush"],
                     }
        self.image_bytes = bytes()

    def GBSJ_seting(self, model_num, prototxt_num, ):
        print("[GBSJ] : 설정 내용 불러오는중...")
        if (model_num == 6) & (prototxt_num == 5):
            self.Net = cv2.dnn.readNetFromDarknet(self.info["Model"][model_num], self.info["Prototxt"][prototxt_num])
            self.Net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.Net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            self.Colors = np.random.uniform(0, 255, size=(len(self.info["LabelYolo"]), 3))
        else:
            self.Net = cv2.dnn.readNetFromCaffe(self.info["Prototxt"][prototxt_num], self.info["Model"][model_num])
            self.Colors = np.random.uniform(0, 255, size=(len(self.info["Label"]), 3))
            # 모델 학습내용 불러오기
        # self.Url = urllib.request.urlopen(self.info["URL"])
        # 웹상에서 사진 얻기 위해 설정
        print("[GBSJ] : 설정 내용 불러옴~...")

    def GBSJ_detection_JPG(self, image):
        frame = imutils.resize(image, width=300)
        # 너비가 300으로 사이즈 맞춤

        colors = np.random.uniform(0, 255, size=(len(self.info["Label"]), 3))

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
                             colors[idx], 2)
                # 상자 그리기 image, 사각형의 위치, 색깔, 선의 두께

                object_y = startY - 15 if startY - 15 > 15 else startY + 15
                # Label을 나타낼때 위에 자리가 없으면 상자 안에 나타냄

                cv2.putText(frame, object, (startX, object_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                # 텍스트를 추가함 image, text, text의 왼쪽 아래 위치,
                #                    폰트, 폰트의 배율 인수, 색깔, 두께

        return frame

    def GBSJ_detection_yolo_JPG(self, frame, conf, Color):
        confThreshold = conf  # 임계값
        nmsThreshold = 0.4  # 상자 결합 할 정도

        colors = Color

        inpWidth = 320  # 네트워크에 넣을 이미지 너비 (320, 416, 608)
        inpHeight = 320  # 네트워크에 넣을 이미지 높이

        # 출력 형식 바꿔주는 함수
        def getOutputsNames(net):
            layersNames = net.getLayerNames()
            return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # 상자, 라벨 그려주는 함수
        def drawPred(idx, conf, left, top, right, bottom):

            cv2.rectangle(frame, (left, top), (right, bottom), colors[idx], 3)

            label = "{name}: {percentage:.2f}%".format(name=self.info["LabelYolo"][idx],
                                               percentage=conf * 100)

            y = top - 22 if top - 22 > 22 else top + 22

            cv2.putText(frame, label, (left + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[idx], 2)

        # 출력값 전처리 & 상자 합치는 함수
        def postprocess(frame, outs):
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            # Scan through all the bounding boxes output from the network and keep only the
            # ones with high confidence scores. Assign the box's class label as the class with the highest score.
            idxs = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:

                    scores = detection[5:]
                    idx = np.argmax(scores)
                    confidence = scores[idx]

                    if confidence > confThreshold:

                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)

                        idxs.append(idx)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

            for i in indices:
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                drawPred(idxs[i], confidences[i], left, top, left + width, top + height)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        self.Net.setInput(blob)
        # Net에 bolb으로 입력을 설정함

        self.Result = self.Net.forward(getOutputsNames(self.Net))

        postprocess(frame, self.Result)

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

        blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400),
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
YOLOv3 = GBSJ()
YOLOv3.GBSJ_seting(6, 5)
print("성공!")
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    image = YOLOv3.GBSJ_detection_yolo_JPG(frame, 0.5, YOLOv3.Colors)
    cv2.imshow("YOLOv3-detection", image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
"""

"""
Traking = GBSJ()
Traking.GBSJ_seting(4, 3)
print("성공!")
cap = cv2.VideoCapture(0)

from pyimagesearch.centroidtracker import CentroidTracker
ct = CentroidTracker(5)
while True:
    _, frame = cap.read()
    image = Traking.GBSJ_traking_JPG(frame)
    cv2.imshow("Traking-", image)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()

"""