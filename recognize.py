# -*- coding:utf-8 -*-
from scipy.spatial import distance as dist
from tools.skindetector import SkinDetector
from imutils import paths
import numpy as np
import imutils
import argparse
import cv2

VIDEO_MODE = 0x01
IMAGE_MODE = 0x02


def detectHand(detector, mode, mask=None):
    orig = detector.getImage()
    startA = orig.copy()
    endA = orig.copy()
    farA = orig.copy()
    hand = orig.copy()
    ycrcb = cv2.cvtColor(orig, cv2.COLOR_BGR2YCR_CB)
    skinCnts = detector.detectSkin(ycrcb, mask)
    if len(skinCnts) == 0:
        return None
    skinHand = skinCnts[0]

    approx = SD.getapproximation(skinHand)
    # cv2.drawContours(o, [approx], -1, (0, 255, 255), 2)
    hull = cv2.convexHull(skinHand)
    # cv2.drawContours(hand, [hull], -1, (0, 255, 0), 2)
    cv2.drawContours(hand, [hull], -1, (0, 255, 255), 2)
    rect = cv2.minAreaRect(approx)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    dx1 = box[0][0] - box[1][0]
    dy1 = box[0][1] - box[1][1]
    dxy1 = float(np.sqrt(dx1 ** 2 + dy1 ** 2))
    dx2 = box[1][0] - box[2][0]
    dy2 = box[1][1] - box[2][1]
    dxy2 = float(np.sqrt(dx2 ** 2 + dy2 ** 2))
    if dxy1 < dxy2:
        width = dxy1
        height = dxy2
    else:
        width = dxy2
        height = dxy1

    M = cv2.moments(approx)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    center = (cX, cY)

    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)
    fingerPoint = []
    if defects is None:
        return None
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        # print("d[%d]=%d" % (i, d))
        # cv2.circle(hand, start, 5, (255, 0, 0), -1)
        # cv2.line(hand, start, end, (255, 255, 255), 2)
        cv2.circle(hand, end, 5, (0, 255, 0), -1)
        cv2.circle(hand, far, 5, (0, 0, 255), -1)
        cv2.circle(startA, start, 5, (0, 255, 0), 3)
        cv2.circle(endA, end, 5, (255, 0, 0), 3)
        cv2.circle(farA, far, 5, (0, 0, 255), 3)
        cv2.putText(startA, "%d" % i, (start[0] - 30, start[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(endA, "%d" % i, (end[0] - 30, end[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(farA, "%d" % i, (far[0] - 30, far[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        fingerPoint.append(start)
        fingerPoint.append(end)
    fingerPoint = sorted(set(fingerPoint), key=fingerPoint.index)
    validFinger = fingerPoint[:]
    fingerIndex = 1
    for i, f in enumerate(fingerPoint):
        dx = fingerPoint[i][0]-center[0]
        dy = fingerPoint[i][1]-center[1]
        distance = float(np.sqrt(dx**2 + dy**2))
        # print("distance[%d]=%d, h/d=%f" % (i, distance, float(distance/height)))
        if float(distance/height) > 0.4 and f[1] < center[1]:
            cv2.circle(hand, f, 10, (255, 0, 0), 5)
            cv2.putText(hand, "%d" % fingerIndex, (f[0] - 30, f[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        3)
            cv2.line(hand, f, center, (255, 255, 125), 3)
            fingerIndex += 1
        else:
            validFinger.remove(f)
    fingerNum = len(validFinger)
    cv2.circle(hand, center, 5, (255, 0, 255), 3)
    cv2.putText(hand, "%d finger" % fingerNum, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("hand", imutils.resize(hand, width=min(800, image.shape[1])))
    # cv2.imshow("start", imutils.resize(startA, width=min(500, image.shape[1])))
    # cv2.imshow("end", imutils.resize(endA, width=min(500, image.shape[1])))
    # cv2.imshow("far", imutils.resize(farA, width=min(500, image.shape[1])))
    if mode is IMAGE_MODE:
        cv2.waitKey(0)
    return skinHand


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image")
ap.add_argument("-v", "--video")
args = vars(ap.parse_args())

if args.get("image", False):
    for imagePath in paths.list_images(args["image"]):
        image = cv2.imread(imagePath)
        SD = SkinDetector(image)
        face = SD.detectFace()
        if len(face) != 0:
            mask = np.ones(image.shape[:2], dtype="uint8")
            cv2.rectangle(mask, (face[0][0], face[0][1]), (face[0][2], face[0][3]), 0, -1)
        else:
            mask = None
        detectHand(SD, IMAGE_MODE, mask)
else:
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
    while True:
        ret, image = camera.read()
        if not ret:
            break
        SD = SkinDetector(image)
        face = SD.detectFace()
        if len(face) != 0:
            mask = np.ones(image.shape[:2], dtype="uint8")
            cv2.rectangle(mask, (face[0][0], face[0][1]), (face[0][2], face[0][3]), 0, -1)
        else:
            mask = None
        detectHand(SD, VIDEO_MODE, mask)
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()
