from tools.skindetector import SkinDetector
from operator import xor
from imutils import paths
import imutils
import numpy as np
import argparse
import cv2

VIDEO_MODE = 0x01
IMAGE_MODE = 0x02

def detectHand(o, f, mode):
    startA = o.copy()
    endA = o.copy()
    farA = o.copy()
    hand = o.copy()
    SD = SkinDetector(f)
    skinCnts = SD.detectSkin()
    # print(type(skinCnts))
    for (c, area) in skinCnts:
        # cv2.drawContours(o, [c], -1, (0, 255, 0), 2)
        # SD.getAreaRatio(c)
        # SD.getPeriAreaRatio(c)

        # x, y, w, h = cv2.boundingRect(c)
        # rect = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        # cv2.drawContours(hand, [rect], -1, (128, 0, 128), 2)

        # hull = cv2.convexHull(c)
        # cv2.drawContours(o, [hull], -1, (255, 0, 0), 2)
        # hull_area = cv2.contourArea(hull)
        # print("len(hull)=%s" % len(hull))
        # hullRatio = area / hull_area
        # print("hullRatio=%s" % hullRatio)

        approx = SD.getapproximation(c)
        # cv2.drawContours(o, [approx], -1, (0, 255, 255), 2)
        hull = cv2.convexHull(approx)
        cv2.drawContours(hand, [hull], -1, (0, 255, 0), 2)
        cv2.drawContours(hand, [approx], -1, (0, 255, 255), 2)

        M = cv2.moments(approx)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        center = (cX, cY)

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        fingerPoint = []
        if defects is None:
            continue
        print(defects)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            # cv2.circle(hand, start, 5, (255, 0, 0), -1)
            cv2.line(hand, start, end, (255, 255, 255), 2)
            cv2.circle(hand, end, 5, (0, 255, 0), -1)
            cv2.circle(hand, far, 5, (0, 0, 255), -1)
            cv2.circle(startA, start, 5, (0, 255, 0), 3)
            cv2.circle(endA, end, 5, (255, 0, 0), 3)
            cv2.circle(farA, far, 5, (0, 0, 255), 3)
            cv2.putText(startA, "%d" % i, (start[0]-30, start[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(endA, "%d" % i, (end[0]-30, end[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(farA, "%d" % i, (far[0]-30, far[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            fingerPoint.append(start)
            fingerPoint.append(end)
        fingerPoint = sorted(set(fingerPoint), key=fingerPoint.index)
        validFinger = fingerPoint[:]
        fingerIndex = 1
        for i, f in enumerate(fingerPoint):
            if f[1] < center[1]:
                cv2.circle(hand, f, 10, (255, 0, 0), 5)
                cv2.putText(hand, "%d" % fingerIndex, (f[0]-30, f[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.line(hand, f, center, (255, 255, 125), 3)
                fingerIndex += 1
            else:
                validFinger.remove(f)
        fingerNum = len(validFinger)
        cv2.circle(hand, center, 5, (255, 0, 255), 3)
        cv2.putText(hand, "%d finger" % fingerNum, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("hand", imutils.resize(hand, width=min(500, image.shape[1])))
        cv2.imshow("start", imutils.resize(startA, width=min(500, image.shape[1])))
        cv2.imshow("end", imutils.resize(endA, width=min(500, image.shape[1])))
        cv2.imshow("far", imutils.resize(farA, width=min(500, image.shape[1])))
        if mode is IMAGE_MODE:
            cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False)
ap.add_argument("-w", "--video", required=False)
args = vars(ap.parse_args())

if not xor(bool(args['image']), bool(args['video'])):
    ap.error("Please specify only one image source")

if args["video"]:
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
    while True:
        ret, image = camera.read()
        if not ret:
            break
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        detectHand(image, frame, VIDEO_MODE)
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()
else:
    for imagePath in paths.list_images(args["image"]):
        print(imagePath)
        image = cv2.imread(imagePath)
        orig = image.copy()
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        detectHand(orig, frame, IMAGE_MODE)
