import cv2
import math

# assume the concave part is a part of new ellipsoid.
def findConcave(imgNum):
    # imgname = 'pic/' + ("%04d" % imgNum) + '.bmp'
    imgname = './pic/ROI_0{:02d}0.png'.format(imgNum - 1)
    # imgname = './pic/Mask_0{:02d}0.png'.format(imgNum - 1)
    img = cv2.imread(imgname)  # input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 0,255 cv2.THRESH_OTSU
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE,  # external
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    # cv2.drawContours(dst, contours, -1, (0, 0, 255), 3)  # Draw contour

    maxCont = 0
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 500:  # keep the largest one, which is the target.
            continue
        maxCont = cont
    #cv2.drawContours(img, maxCont, -1, (0, 255, 255), 1)  # Draw contour

    hull = cv2.convexHull(maxCont, returnPoints=False)
    convexityDefects = cv2.convexityDefects(maxCont, hull)
    # print(convexityDefects)

    mostFarthestIndex = 0
    mostDistance = 0
    finalStartIndex = 0
    finalEndIndex = 0

    for d in range(convexityDefects.shape[0]):
        start, end, farthest, distance = convexityDefects[d][0]
        if distance > mostDistance:
            mostDistance = distance
            finalStartIndex = start
            finalEndIndex = end
            mostFarthestIndex = farthest
    mostFarthest_pt = tuple(maxCont[mostFarthestIndex][0])
    finalStart_pt = tuple(maxCont[finalStartIndex][0])
    finalEnd_pt = tuple(maxCont[finalEndIndex][0])

    # mark the start point, end point and center point (farthest point)
    cv2.circle(img, mostFarthest_pt, 3, [255, 0, 0], -1)
    cv2.circle(img, finalStart_pt, 3, [255, 0, 0], -1)
    cv2.circle(img, finalEnd_pt, 3, [255, 0, 0], -1)

    startPoint = maxCont[finalStartIndex]
    endPoint = maxCont[finalEndIndex]
    centerPoint = maxCont[mostFarthestIndex]

    # print(maxCont[finalStartIndex])
    # print(maxCont[finalEndIndex])
    # print(maxCont[mostFarthestIndex])

    # compare the distance from center point to start point and to end point
    disFromStartPoint = math.sqrt((startPoint[0][0] - centerPoint[0][0]) * (startPoint[0][0] - centerPoint[0][0]) +
                                  (startPoint[0][1] - centerPoint[0][1]) * (startPoint[0][1] - centerPoint[0][1]))

    disFromEndPoint = math.sqrt((endPoint[0][0] - centerPoint[0][0]) * (endPoint[0][0] - centerPoint[0][0]) +
                                (endPoint[0][1] - centerPoint[0][1]) * (endPoint[0][1] - centerPoint[0][1]))

    # select the point with the shorter distance, and use this point to calculate the angel of concave part
    if disFromStartPoint < disFromEndPoint:
        sidePoint = startPoint
        # sideDistance = disFromStartPoint
    else:
        sidePoint = endPoint
        # sideDistance = disFromEndPoint

    # assume the width (radius of the new ellipsoid) is vertical distance.
    width = math.fabs(centerPoint[0][1] - sidePoint[0][1])
    angel = math.atan(
        math.fabs(sidePoint[0][0] - centerPoint[0][0]) / math.fabs(sidePoint[0][1] - centerPoint[0][1])) * 180 / math.pi
    angel *= 2

    print("***")
    # print(width)
    # print(angel)

    # cv2.namedWindow("original", 1)
    # cv2.imshow('original', img)
    # cv2.namedWindow("dst", 1)
    # cv2.imshow("dst", dst)
    # # cv2.namedWindow("contour", 1)
    # # cv2.imshow('contour', img2)
    # cv2.waitKey()
    return width, angel



