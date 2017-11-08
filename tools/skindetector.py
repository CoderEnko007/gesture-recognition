import numpy as np
import cv2
import os

lowerB = (0, 140, 100)
upperB = (255, 170, 129)
FACE_DETECTOR_PATH = "{base_path}\\cascades\\haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))


class SkinDetector:
    def __init__(self, image):
        self.image = image

    def getImage(self):
        return self.image

    def detectFace(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # print(FACE_DETECTOR_PATH)
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        rect = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        rect = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rect]
        return rect

    def detectSkin(self, frame, mask=None):
        skinCnts = []
        area = []
        skin = cv2.inRange(frame, lowerB, upperB)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin = cv2.dilate(skin, kernel, iterations=1)
        skin = cv2.erode(skin, kernel, iterations=1)
        skin = cv2.GaussianBlur(skin, (1, 1), 0)
        cv2.imshow("skin mask", skin)
        if mask is not None:
            skin = cv2.bitwise_and(skin, skin, mask=mask)

        cnts = cv2.findContours(skin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        for c in cnts:
            a = cv2.contourArea(c)
            if a > 5000:
                skinCnts.append(c)
                area.append(a)
        skinCnts = sorted(skinCnts, key=cv2.contourArea, reverse=True)
        # return list(zip(skinCnts, area))
        return skinCnts

    def getAreaRatio(self, cnt, image=None):
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if image is not None:
            cv2.drawContours(image, [box], -1, (0, 0, 255), 2)
        minArea = cv2.contourArea(box)
        result = area / minArea
        print("area=%s, minArea=%s, ratio=%s" % (area, minArea, result))
        return result

    def getPeriAreaRatio(self, cnt):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        result = peri / area
        print("peri / area=%s" % result)
        return result

    def getAspectRatio(self, cnt):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        dx1 = box[0][0] - box[1][0]
        dy1 = box[0][1] - box[1][1]
        dxy1 = np.sqrt(dx1**2 + dy1**2)
        dx2 = box[1][0] - box[2][0]
        dy2 = box[1][1] - box[2][1]
        dxy2 = np.sqrt(dx2**2 + dy2**2)
        print("dxy1, dxy2=%s" % str((dxy1, dxy2)))
        if dxy1 < dxy2:
            width = dxy1
            height = dxy2
        else:
            width = dxy2
            height = dxy1
        result = height / width
        peri = 2*height + 2*width
        print("aspect ratio=%s" % result)
        return peri, result

    def getapproximation(self, cnt):
        epsilon = 0.025 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return approx
