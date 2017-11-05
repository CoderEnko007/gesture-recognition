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
    hand = o.copy()
    SD = SkinDetector(f)
    skinCnts = SD.detectSkin()
    # print(type(skinCnts))
    for (c, area) in skinCnts:
        cv2.drawContours(o, [c], -1, (0, 255, 0), 2)
        SD.getAreaRatio(c, hand)
        SD.getPeriAreaRatio(c)
        SD.getAspectRatio(c)

        # x, y, w, h = cv2.boundingRect(c)
        # rect = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        # cv2.drawContours(hand, [rect], -1, (128, 0, 128), 2)

        hull = cv2.convexHull(c)
        cv2.drawContours(o, [hull], -1, (255, 0, 0), 2)
        hull_area = cv2.contourArea(hull)
        print("len(hull)=%s" % len(hull))
        hullRatio = area / hull_area
        print("hullRatio=%s" % hullRatio)

        approx = SD.getapproximation(c)
        cv2.drawContours(o, [approx], -1, (0, 255, 255), 2)
        hull = cv2.convexHull(approx)
        cv2.drawContours(hand, [hull], -1, (0, 255, 0), 2)
        cv2.drawContours(hand, [approx], -1, (0, 255, 255), 2)

        M = cv2.moments(approx)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        center = (cX, cY)
        cv2.circle(hand, center, 5, (255, 0, 255), -1)

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        fingerPoint = []
        if defects is None:
            continue
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            # cv2.circle(hand, start, 5, (255, 0, 0), -1)
            # cv2.circle(hand, end, 5, (0, 255, 0), -1)
            cv2.circle(hand, far, 5, (0, 0, 255), -1)
            fingerPoint.append(start)
            fingerPoint.append(end)
        fingerPoint = sorted(set(fingerPoint), key=fingerPoint.index)
        validFinger = fingerPoint[:]
        for f in fingerPoint:
            if f[1] < center[1]:
                cv2.circle(hand, f, 30, (255, 0, 0), 5)
                cv2.line(hand, f, center, (128, 128, 0), 2)
            else:
                validFinger.remove(f)
        fingerNum = len(validFinger)
        cv2.putText(hand, "%d finger" % fingerNum, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("test", imutils.resize(hand, width=min(600, image.shape[1])))
        # cv2.imshow("area", imutils.resize(orig, width=min(600, image.shape[1])))
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
        cv2.imwrite("image.jpg", image)
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
