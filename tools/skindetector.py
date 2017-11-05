import numpy as np
import imutils
import cv2

lowerB = (0, 138, 100)
upperB = (255, 170, 120)

class SkinDetector:
    def __init__(self, image):
        self.image = image

    def detectSkin(self):
        skinCnts = []
        area = []
        mask = cv2.inRange(self.image, lowerB, upperB)
        mask = cv2.dilate(mask, None, iterations=1)
        mask = cv2.erode(mask, None, iterations=1)
        cv2.imshow("mask", mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        # print(len(cnts))
        for c in cnts:
            a = cv2.contourArea(c)
            if a > 5000:
                skinCnts.append(c)
                area.append(a)
        return zip(skinCnts, area)

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
        ratio = height / width
        print("aspect ratio=%s" % ratio)
        return ratio

    def getapproximation(self, cnt):
        epsilon = 0.025 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return approx
