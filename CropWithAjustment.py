import cv2
import numpy as np
import math


# get the area of the target (seed) from the side view of image : pic
def GetArea(path, vintValue, imageNum, sourse):
    if sourse == "original":
        imgname = path + ("%04d" % imageNum) + '.bmp'
    else:
        imgname = path + 'ROI_0{:02d}0.png'.format(imageNum - 1)
    img = cv2.imread(imgname)  # input image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, vintValue, 255, 0)  # 0,255 cv2.THRESH_OTSU
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    # cv2.drawContours(dst, contours, -1, (120, 0, 0), 2)  # Draw contour
    maxCont = 0
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 2000:  # keep the largest one, which is the target.
            continue
        maxCont = cont
    maxRect = cv2.boundingRect(maxCont)

    # test another method to get the contour(min area rect)
    # rect = cv2.minAreaRect(maxCont)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 1)  # red
    # cv2.imwrite(path + "contour_0{:02d}0.png".format(imageNum - 1), img)

    X = maxRect[0]
    Y = maxRect[1]
    width = maxRect[2]
    height = maxRect[3]
    return X, Y, width, height


# To find the expected width of each image, that is, the width when the seed located in the exact turn center.
# First, find the widths of the pair of two opposite images, then calculate the average value to get
# the expected width(the center position)
# note: the largest one is the length of the target(seed), the smallest one is the width of the target(seed)
def getExpectedValues(path, vintValue, src):
    expectedWidth = []
    heights = []  # the height of all 36 images.
    imgNum = 1
    while imgNum < 19:  # the first 18 images
        X, Y, width, height = GetArea(path, vintValue, imgNum, src)
        oppoX, oppoY, oppositeWidth, oppositeHeight = GetArea(path, vintValue, imgNum + 18, src)
        expectedWidth.append(width / 2 + oppositeWidth / 2)
        # expectedWidth.append(math.ceil(width / 2 + oppositeWidth / 2)) # round the value
        heights.append(height)
        heights.append(oppositeHeight)
        imgNum += 1
    index = 19
    while index > 1:  # the last 18 images
        expectedWidth.append(expectedWidth[19 - index])
        index -= 1
    print("expectedWidth: ")
    print(expectedWidth)
    print("********** ")
    return expectedWidth, heights


# To make sure the ratio of the seed in the image to be the same.
def normalizeImage(path, vintValue, imageNum, expectedWidth, imageWidth, imageHeight, bottomY):
    imgname = path + ("%04d" % imageNum) + '.bmp'
    img = cv2.imread(imgname)  # read input image
    # img = img[0:338, 0:720]
    X, Y, width, height = GetArea(path, vintValue, imageNum, "original")
    firstCrop = img[Y: Y + height, X: X + width]  # get the target

    H = firstCrop.shape[0]
    W = firstCrop.shape[1]

    ratio = expectedWidth / width
    newW = math.ceil(W * ratio)
    newH = math.ceil(H * ratio)
    dim = (newW, newH)
    resized = cv2.resize(firstCrop, dim, interpolation=cv2.INTER_AREA)  # resize the target (increase / decrease)

    result = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)
    start_Y = math.ceil((imageHeight - newH) / 2)
    start_X = math.ceil((imageWidth - newW) / 2)
    # make the new target in the center of the result
    if bottomY == 0:
        result[start_Y: start_Y + newH, start_X: start_X + newW] = resized
    else:
        result[bottomY-newH: bottomY, start_X: start_X + newW] = resized
    # save the image
    cv2.imwrite(path + "ROI_0{:02d}0.png".format(imageNum - 1), result)

    X, Y, width, height = GetArea(path, vintValue, imageNum, "roi")
    # X, Y, width, height = GetArea(path, vintValue, imageNum, "original")
    print("image:{} x:{} y:{} width:{} height:{}".format(imageNum, X, Y, width, height))
    return start_Y + newH


# Normalizing all images
def CropWithAdjustment(path, vintValue, imageWidth, imageHeight, expectedWidth):
    # get the stand bottomY position in the image
    bottomY = normalizeImage(path, vintValue, 1,  expectedWidth[0], imageWidth, imageHeight, 0)   # 1/0.
    imgNum = 2
    while imgNum < 37:
        normalizeImage(path, vintValue, imgNum,  expectedWidth[imgNum - 1], imageWidth, imageHeight, bottomY)
        imgNum += 1

