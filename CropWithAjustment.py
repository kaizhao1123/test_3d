import cv2
import numpy as np
import math
from PIL import Image


# get the area of the target (seed) from the side view of image : pic
def GetArea(path, vintValue, imageNum, sourse):
    if sourse == "original":
        imgname = path + ("%04d" % imageNum) + '.bmp'
    else:
        imgname = path + 'ROI_0{:02d}0.png'.format(imageNum - 1)

    img = cv2.imread(imgname)  # input image, used to get the height

    if sourse == "original":
        img_new = img[0:230, 0:720]  # input image, used to get the width and length (crop the reflection from holder).
    else:
        img_new = img

    # ####### to get the height of each image ##########
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, 100, 255, 0)  # 0,255 cv2.THRESH_OTSU
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    maxCont = 0
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 1000:  # keep the largest one, which is the target.
            continue
        maxCont = cont
    maxRect = cv2.boundingRect(maxCont)

    # img_contour = img.copy()
    # img_contour = cv2.drawContours(img_contour, [maxCont], 0, (255, 0, 0), thickness=1)
    # img_contour = cv2.rectangle(img_contour, (maxRect[0], maxRect[1]), (maxRect[0] + maxRect[2], maxRect[1] + maxRect[3]), (0,255,0), 1)
    # cv2.imwrite("./pic/contour_00{:02d}.bmp".format(imageNum), img_contour)

    height = maxRect[3]

    # ####### to get the width of each image ##########
    gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, vintValue, 255, 0)  # 0,255 cv2.THRESH_OTSU
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    maxCont = 0
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 1000:  # keep the largest one, which is the target.
            continue
        maxCont = cont
    maxRect = cv2.boundingRect(maxCont)


    X = maxRect[0]
    Y = maxRect[1]
    width = maxRect[2]

    # center = (round(X + width / 2), round(Y + height / 2))
    # axesLength = (round(width/2+5), round(height/2)+5)
    # color = (255, 0, 0)
    # thickness = 2
    # img_cir = cv2.circle(img, center, round(width / 2), color, thickness)
    # # img_cir = cv2.ellipse(img, center,axesLength,0,0,360,color,thickness)
    # cv2.imwrite("./pic/cir_00{:02d}.bmp".format(imageNum), img_cir)

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
    X, Y, width, height = GetArea(path, vintValue, imageNum, "original")

    # draw the black circle edge
    # center = (round(X + width / 2), round(Y + height / 2))
    # color = (0, 0, 0)
    # thickness = 10
    # img = cv2.circle(img, center, round(max(width / 2 + 10, height/2 + 10)), color, thickness)

    firstCrop = img[Y: Y + height, X: X + width]  # get the target

    H = firstCrop.shape[0]
    W = firstCrop.shape[1]

    # print(H, W)

    ratio = expectedWidth / width
    newW = math.ceil(W * ratio)
    newH = math.ceil(H * ratio)
    dim = (newW, newH)
    resized = cv2.resize(firstCrop, dim, interpolation=cv2.INTER_AREA)  # resize the target (increase or decrease)

    result = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)
    start_Y = math.ceil((imageHeight - newH) / 2)
    start_X = math.ceil((imageWidth - newW) / 2)
    #
    # print(ratio)
    # print(start_Y)
    # print(newH)
    # print(start_X)
    # print(newW)

    # make the new target in the center of the result
    if bottomY == 0:
        result[start_Y: start_Y + newH, start_X: start_X + newW] = resized
    else:
        result[bottomY-newH: bottomY, start_X: start_X + newW] = resized

    ####################
    # gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # change to gray image
    # # Global threshold segmentation,  to binary image. (Otsu)
    # res, dst = cv2.threshold(gray, 25, 255, 0)  # 0,255 cv2.THRESH_OTSU
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    # dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising
    #
    # contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
    #                                        cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    # maxCont = 0
    # for cont in contours:
    #     area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
    #     if area < 1000:  # keep the largest one, which is the target.
    #         continue
    #     maxCont = cont
    # cv2.drawContours(result, [maxCont], 0, (255, 255, 255), thickness=cv2.FILLED)
    # cv2.drawContours(result, [maxCont], 0, (255, 0, 255), thickness=1)
    #####################

    # save the image
    cv2.imwrite(path + "ROI_0{:02d}0.png".format(imageNum - 1), result)

    X, Y, width, height = GetArea(path, vintValue, imageNum, "original")
    print("orignial image:{} x:{} y:{} width:{} height:{}".format(imageNum, X, Y, width, height))

    # X, Y, width, height = GetArea(path, vintValue, imageNum, "roi")
    # print("roi image:{} x:{} y:{} width:{} height:{}".format(imageNum, X, Y, width, height))

    return start_Y + newH


# Normalizing all images
def CropWithAdjustment(path, vintValue, imageWidth, imageHeight, expectedWidth):
    # get the stand bottomY position in the image
    bottomY = normalizeImage(path, vintValue, 1,  expectedWidth[0], imageWidth, imageHeight, 0)   # 1/0.
    imgNum = 2
    while imgNum < 37:
        normalizeImage(path, vintValue, imgNum,  expectedWidth[imgNum - 1], imageWidth, imageHeight, bottomY)
        imgNum += 1


# def convert2Mask(path, vintValue):
#     for imgId in range(0, 36):
#         img = cv2.imread(path + "ROI_0{:02d}0.png".format(imgId))  # read input image
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
#         # Global threshold segmentation,  to binary image. (Otsu)
#         res, dst = cv2.threshold(gray, vintValue, 255, 0)  # 0,255 cv2.THRESH_OTSU
#         element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
#         dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising
#
#         contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
#                                                cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
#         maxCont = 0
#         for cont in contours:
#             area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
#             if area < 1000:  # keep the largest one, which is the target.
#                 continue
#             maxCont = cont
#
#         img_mask = img.copy()
#         img_mask = cv2.drawContours(img_mask, [maxCont], 0, (255, 255, 255), thickness=1)
#         # img_mask = cv2.drawContours(img_mask, contours, -1, (255, 255, 255), thickness=5)
#
#         new_image = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)  # array
#
#         new_image = Image.fromarray(new_image)
#         new_image.save('pic/ROI_0{:02d}0.png'.format(imgId))
#         # img_mask = cv2.rectangle(img_mask, (maxRect[0], maxRect[1]), (maxRect[0] + maxRect[2], maxRect[1] + maxRect[3]), (0,255,0), 1)
#         # cv2.imwrite("./pic/Mask_0{:02d}0.png".format(imgId), new_image)

# just for test new idea
##############################################################################################

def PreAdjustment_blue(path,vintValue, imageHeight, imageWidth):

    # test the contour ratio (area of contour / rectangle)
    contourRatioList = []
    #
    expectedWidth = []
    heights = []
    imgNum = 1
    bottomY = 0
    while imgNum < 37:

        # image = cv2.imread(path + ("%04d" % imgNum) + '.bmp')
        # img1 = image[:, 220:420]  # keep the middle part including the target.
        img1 = cv2.imread(path + ("%04d" % imgNum) + '.bmp')

        image = cv2.GaussianBlur(img1, (3, 3), 0)
        edges = cv2.Canny(image, 0, 50)  # using low vint value
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # combine the edges image and original image
        contour_image = cv2.add(edges, img1)
        # contour_image = img1
        cv2.imwrite(path + "Contour_0{:02d}0.png".format(imgNum), contour_image)

        # get binary image of target, based on its HSV format.
        hsv = cv2.cvtColor(contour_image, cv2.COLOR_BGR2HSV)
        # hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 0])
        upper_blue = np.array([140, 255, 255])
        binary_image = cv2.inRange(hsv, lower_blue, upper_blue)
        binary_image = cv2.bitwise_not(binary_image)

        # 1. get the height
        thresh = cv2.threshold(binary_image, 10, 255, cv2.THRESH_BINARY)[1]
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
        dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
        contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        big_contour = max(contours, key=cv2.contourArea)
        contourArea = cv2.contourArea(big_contour)
        maxRect = cv2.boundingRect(big_contour)
        height = maxRect[3]

        # 2. get the X, Y, width
        binary_image_new = binary_image[0:binary_image.shape[0] - 10, 0:720]
        thresh = cv2.threshold(binary_image_new, 10, 255, cv2.THRESH_BINARY)[1]
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
        dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
        contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        big_contour = max(contours, key=cv2.contourArea)
        maxRect = cv2.boundingRect(big_contour)
        X, Y, width, _ = maxRect

        contourRatioList.append(contourArea/width/height)
        expectedWidth.append(width)
        heights.append(height)

        firstCrop = img1[Y: Y + height, X: X + width]  # get the target.
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        firstCrop_binary = binary_image[Y: Y + height, X: X + width]
        H = firstCrop.shape[0]
        W = firstCrop.shape[1]

        # create the black image with special size to carry the target.
        cropResult = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)
        start_Y = math.ceil((imageHeight - H) / 2)
        start_X = math.ceil((imageWidth - W) / 2)
        mask_image = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)

        # make the new target in the center of the result
        if imgNum == 1:
            cropResult[start_Y: start_Y + H, start_X: start_X + W] = firstCrop
            mask_image[start_Y: start_Y + H, start_X: start_X + W] = firstCrop_binary
            bottomY = start_Y + H
        else:
            cropResult[bottomY - H: bottomY, start_X: start_X + W] = firstCrop
            mask_image[bottomY - H: bottomY, start_X: start_X + W] = firstCrop_binary

        cv2.imwrite(path + "ROI_0{:02d}0.png".format(imgNum - 1), cropResult)
        cv2.imwrite(path + "Mask_0{:02d}0.png".format(imgNum - 1), mask_image)

        # new_image = Image.fromarray(mask)
        # new_image.save(path + "%04d" % imgNum + '.bmp')
        dealWithMaskImage(path, imgNum-1, imageHeight, imageWidth)
        # cv2.imwrite(path + ("%04d" % imgNum) + '.bmp', new_image)

        print("orignial image:{} x:{} y:{} width:{} height:{}".format(imgNum, X, Y, width, height))

        imgNum += 1

    return expectedWidth, heights, contourRatioList


def dealWithMaskImage(path, vintValue, imgNum, imageHeight, imageWidth):
    img = cv2.imread(path + 'Mask_0{:02d}0.png'.format(imgNum))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
    res, dst = cv2.threshold(gray, vintValue, 255, cv2.THRESH_BINARY)  # 0,255 cv2.THRESH_OTSU
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function

    maxCont = max(contours, key=cv2.contourArea)
    mask = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)
    cv2.drawContours(mask, [maxCont], 0, (255, 255, 255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)     # array
    mask = Image.fromarray(mask)     # convert to image
    mask.save(path + 'Mask_0{:02d}0.png'.format(imgNum))

# #################################################################################################
# using LED background


def getArea_LED(path, vintValue, imgNum):
    img = cv2.imread(path + ("%04d" % imgNum) + '.bmp')
    img = cv2.bitwise_not(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image

    # 1. get the height
    thresh = cv2.threshold(gray, vintValue, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big_contour = max(contours, key=cv2.contourArea)
    # contourArea = cv2.contourArea(big_contour)
    maxRect = cv2.boundingRect(big_contour)
    height = maxRect[3]

    # 2. get the X, Y, width
    gray_new = gray[0:gray.shape[0] - 10, 0:720]
    thresh = cv2.threshold(gray_new, vintValue, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big_contour = max(contours, key=cv2.contourArea)
    maxRect = cv2.boundingRect(big_contour)
    X, Y, width, _ = maxRect

    return X, Y, width, height


#
def getExpectedValues_LED(path, vintValue):
    expectedWidth = []
    heights = []  # the height of all 36 images.
    imgNum = 1
    while imgNum < 19:  # the first 18 images
        X, Y, width, height = getArea_LED(path, vintValue, imgNum)
        oppoX, oppoY, oppositeWidth, oppositeHeight = getArea_LED(path, vintValue, imgNum + 18)
        # expectedWidth.append(width / 2 + oppositeWidth / 2)
        expectedWidth.append(math.ceil(width / 2 + oppositeWidth / 2)) # round the value
        heights.append(height)
        heights.append(oppositeHeight)
        imgNum += 1
    index = 18
    while index > 0:  # the last 18 images
        expectedWidth.append(expectedWidth[18 - index])
        index -= 1
    print("expectedWidth: ")
    print(expectedWidth)
    print("********** ")

    return expectedWidth, heights


# To make sure the ratio of the seed in the image to be the same.
def normalizeImage_LED(path, vintValue, imageNum, expectedWidth, imageWidth, imageHeight, bottomY):
    imgname = path + ("%04d" % imageNum) + '.bmp'
    img = cv2.imread(imgname)  # read input image
    img = cv2.bitwise_not(img)

    X, Y, width, height = getArea_LED(path, vintValue, imageNum)

    firstCrop = img[Y: Y + height, X: X + width]  # get the target

    H = firstCrop.shape[0]
    W = firstCrop.shape[1]

    # print(H, W)

    ratio = expectedWidth / width
    newW = math.ceil(W * ratio)
    newH = math.ceil(H * ratio)
    dim = (newW, newH)
    resized = cv2.resize(firstCrop, dim, interpolation=cv2.INTER_AREA)  # resize the target (increase or decrease)

    result = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)
    start_Y = math.ceil((imageHeight - newH) / 2)
    start_X = math.ceil((imageWidth - newW) / 2)
    #
    # print(ratio)
    # print(start_Y)
    # print(newH)
    # print(start_X)
    # print(newW)

    # make the new target in the center of the result
    if bottomY == 0:
        result[start_Y: start_Y + newH, start_X: start_X + newW] = resized
    else:
        result[bottomY-newH: bottomY, start_X: start_X + newW] = resized
    # save the image
    cv2.imwrite(path + "ROI_0{:02d}0.png".format(imageNum - 1), result)

    X, Y, width, height = getArea_LED(path, vintValue, imageNum)
    print("orignial image:{} x:{} y:{} width:{} height:{}".format(imageNum, X, Y, width, height))

    # X, Y, width, height = GetArea(path, vintValue, imageNum, "roi")
    # print("roi image:{} x:{} y:{} width:{} height:{}".format(imageNum, X, Y, width, height))

    return start_Y + newH


# Normalizing all images
def CropWithAdjustment_LED(path, vintValue, imageWidth, imageHeight, expectedWidth):
    # get the stand bottomY position in the image
    bottomY = normalizeImage_LED(path, vintValue, 1,  expectedWidth[0], imageWidth, imageHeight, 0)   # 1/0.
    imgNum = 2
    while imgNum < 37:
        normalizeImage_LED(path, vintValue, imgNum,  expectedWidth[imgNum - 1], imageWidth, imageHeight, bottomY)
        imgNum += 1


def PreAdjustment_LED(path, vintValue, imageHeight, imageWidth):

    # test the contour ratio (area of contour / rectangle)
    # contourRatioList = []
    #
    expectedWidth = []
    heights = []
    imgNum = 1
    bottomY = 0
    while imgNum < 37:
        img = cv2.imread(path + ("%04d" % imgNum) + '.bmp')
        img = cv2.bitwise_not(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image

        # 1. get the height
        thresh = cv2.threshold(gray, vintValue, 255, cv2.THRESH_BINARY)[1]
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
        dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
        contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        big_contour = max(contours, key=cv2.contourArea)
        # contourArea = cv2.contourArea(big_contour)
        maxRect = cv2.boundingRect(big_contour)
        height = maxRect[3]

        # convert to binary image
        h, w, _ = img.shape
        bi = np.zeros([h, w, 3], dtype=np.uint8)
        cv2.drawContours(bi, [big_contour], 0, (255, 255, 255), thickness=-1)

        # 2. get the X, Y, width
        gray_new = gray[0:gray.shape[0] - 10, 0:720]
        thresh = cv2.threshold(gray_new, vintValue, 255, cv2.THRESH_BINARY)[1]
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
        dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
        contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        big_contour = max(contours, key=cv2.contourArea)
        maxRect = cv2.boundingRect(big_contour)
        X, Y, width, _ = maxRect

        expectedWidth.append(width)
        heights.append(height)

        # crop the target and put it into the center of black template, to get the mask image.
        firstCrop_binary = bi[Y: Y + height, X: X + width]
        start_Y = math.ceil((imageHeight - height) / 2)
        start_X = math.ceil((imageWidth - width) / 2)
        mask_image = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)

        # make the new target in the center of the result
        if imgNum == 1:
            mask_image[start_Y: start_Y + height, start_X: start_X + width] = firstCrop_binary
            bottomY = start_Y + height
        else:
            mask_image[bottomY - height: bottomY, start_X: start_X + width] = firstCrop_binary

        mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)  # array
        mask = Image.fromarray(mask)  # convert to image
        mask.save(path + 'Mask_0{:02d}0.png'.format(imgNum-1))

        # cv2.imwrite(path + "Mask_0{:02d}0.png".format(imgNum - 1), mask_image)
        # dealWithMaskImage(path, vintValue, imgNum-1, imageHeight, imageWidth)

        print("orignial image:{} x:{} y:{} width:{} height:{}".format(imgNum, X, Y, width, height))
        imgNum += 1

    return expectedWidth, heights


# combine the contour from the canny algorithm and the original image, to remove the bottom noisy.
##############################################################################################
def PreAdjustment_black(path, vintValue):
    imgNum = 1
    while imgNum < 37:
        print(imgNum)
        imgName = path + ("%04d" % imgNum) + '.bmp'
        img = cv2.imread(imgName)

        image = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(image, 0, 50)  # using low vint value.
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # modify the edges image to remove the bottom noisy.
        gap = 20
        coverLineVal = modifyEdges(edges, gap)
        width = img.shape[1]
        height = img.shape[0]
        coverImage = np.zeros((coverLineVal, width, 3))
        edges[height - coverLineVal:height, :] = coverImage

        # combine the edges image and original image
        new_image = cv2.add(edges, img)
        cv2.imwrite(path + ("%04d" % imgNum) + '.bmp', new_image)

        imgNum += 1


def isValid(list, gap):
    if list[0] < int(gap * 0.75) <= list[len(list) - 1]:
        return True
    else:
        return False


# remove the bottom noisy
def modifyEdges(edges, gap):

    height, width, _ = edges.shape

    temp = edges[height-gap:height, :]

    # get all point with color
    res = []
    for row in range(0, gap):
        for col in range(0, width):
            if temp[row][col][0] == 255:
                res.append((col, row))
    res = sorted(res)

    # keep the points existed in the same col, and separate them in groups.
    # such as [28, [0, 18]], where 28 is col, 0 and 18 is row.

    newRes = []
    pre = -1
    for i in range(0, len(res) - 1):
        if res[i][0] == pre:
            newRes.append(res[i])
        else:
            if res[i][0] == res[i + 1][0]:
                newRes.append(res[i])
            pre = res[i][0]
    all_values = [list[0] for list in newRes]
    unique_values = sorted(set(all_values))

    result = []
    for value in unique_values:
        this_group = []
        for list in newRes:
            if list[0] == value:
                this_group.append(list[1])
        result.append((value, this_group))

    # select the first 10 groups in the left side and right side respectively.
    leftPart = result[:10]
    rightPart = result[len(result) - 10:]

    # get the noisy level
    minValue = gap
    for item in leftPart:
        if isValid(item[1], gap):
            for ele in item[1]:
                if 15 <= ele <= minValue:
                    minValue = ele
    for item in rightPart:
        if isValid(item[1], gap):
            for ele in item[1]:
                if 15 <= ele <= minValue:
                    minValue = ele
    return gap - minValue


#  add light to the image. (not use currently)
def preAdjustImages(path, imageNum, vintValue):

    imgname = path + ("%04d" % imageNum) + '.bmp'

    img = cv2.imread(imgname)  # input image, used to get the height
    img_new = img[0:240, 0:720]  # input image, used to get the width/length

    # ####### to get the height of each image ##########
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, vintValue, 255, 0)  # 0,255 cv2.THRESH_OTSU
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    maxCont = 0
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 1000:  # keep the largest one, which is the target.
            continue
        maxCont = cont
    maxRect = cv2.boundingRect(maxCont)
    height = maxRect[3]

    # ####### to get the width of each image ##########
    gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, vintValue, 255, 0)  # 0,255 cv2.THRESH_OTSU
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    maxCont = 0
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 1000:  # keep the largest one, which is the target.
            continue
        maxCont = cont
    maxRect = cv2.boundingRect(maxCont)

    X = maxRect[0]
    Y = maxRect[1]
    width = maxRect[2]
    center = (round(X + width / 2), round(Y + height / 2))
    axesLength = (round(width/2+5), round(height/2)+5)
    color = (255, 0, 0)
    thickness = 2
    img_cir = cv2.circle(img, center, round(width / 2), color, thickness)
    # img_cir = cv2.ellipse(img, center,axesLength,0,0,360,color,thickness)
    cv2.imwrite("./pic/cir_00{:02d}.bmp".format(imageNum), img_cir)
