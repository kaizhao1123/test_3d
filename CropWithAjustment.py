import cv2
import numpy as np
import math



# get the area of the target (seed) from the side view of image : pic
def GetArea(path, imageNum, sourse):
    if sourse == "original":
        imgname = path + ("%04d" % imageNum) + '.bmp'
    else:
        imgname = path + 'ROI_0{:02d}0.png'.format(imageNum - 1)
    img = cv2.imread(imgname)  # input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 0,255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    # cv2.drawContours(dst, contours, -1, (120, 0, 0), 2)  # Draw contour
    maxCont = 0
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 500:  # keep the largest one, which is the target.
            continue
        maxCont = cont
    maxRect = cv2.boundingRect(maxCont)
    X = maxRect[0]
    Y = maxRect[1]
    width = maxRect[2]
    height = maxRect[3]
    return X, Y, width, height


# get the expected width of each image, that is the width when the seed located in the exact turn center
# find the width of the pair opposite images, then calculate the average to get the expected width(the center position)
# the largest one is the length of the target(seed), the smallest one is the width of the target(seed)
def getExpectedValues(path):
    expectedWidth = []
    heights = []  # the height of all 36 images.
    imgNum = 1
    while imgNum < 19:  # the first 18 images
        X, Y, width, height = GetArea(path, imgNum, "original")
        oppoX, oppoY, oppositeWidth, oppositeHeight = GetArea(path, imgNum + 18, "original")
        expectedWidth.append(width / 2 + oppositeWidth / 2)
        # expectedWidth.append(math.ceil(width / 2 + oppositeWidth / 2))
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


def normalizeImage(path, imageNum, expectedWidth, imageWidth, imageHeight):
    imgname = path + ("%04d" % imageNum) + '.bmp'
    img = cv2.imread(imgname)  # read input image
    X, Y, width, height = GetArea(path, imageNum, "original")
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
    result[start_Y: start_Y + newH, start_X: start_X + newW] = resized
    # save the image
    cv2.imwrite("./pic/ROI_0{:02d}0.png".format(imageNum - 1), result)

    # X, Y, width, height = GetArea(imageNum, "roi")
    # print("image:{} x:{} y:{} width:{} height:{}".format(imageNum, X, Y, width, height))


def CropWithAdjustment(path, imageWidth, imageHeight, expectedWidth):
    imgNum = 1
    while imgNum < 37:
        normalizeImage(path, imgNum,  expectedWidth[imgNum - 1], imageWidth, imageHeight)
        imgNum += 1

