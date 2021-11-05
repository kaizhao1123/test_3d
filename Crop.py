import cv2
import math


def Crop(imageNum, imageWidth, imageHeight):
    imgname = 'pic/' + ("%04d" % imageNum) + '.bmp'
    img = cv2.imread(imgname)  # input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Local threshold segmentation, to binary image
    # dst = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31, 1) #101

    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 0,255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    # cv2.drawContours(dst, contours, -1, (120, 0, 0), 2)  # Draw contour

    count = 0  # the count of shapes
    maxArea = 0
    maxCont = []
    maxWidth = 0
    maxHeight = 0
    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 300:
            continue
        # print("{}-blob:{}".format(count,ares),end="  ") #area of each shape
        rect = cv2.boundingRect(cont)  # Extract rectangle coordinates

        cv2.rectangle(img, rect, (0, 0, 255), 1)  # Draw the stand rectangle

        if (area > maxArea) and (rect[0] > 0) and (rect[0] > 0):
            # maxArea = area
            maxCont = cont
            # maxWidth = width
            # maxHeight = height

        # Write the number in the upper left corner of the rectangle
        # count+=1
        # cv2.putText(img,str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)

    # ((x, y), r) = cv2.minEnclosingCircle(maxCont)
    # print(x, y, r)
    # cv2.circle(img, (int(x), int(y)), round(r), (255, 0, 0), 1)

    maxRect = cv2.boundingRect(maxCont)  # Extract rectangle coordinates of the target shape
    # print("image:{} x:{} y:{} width:{} height:{}".format(imageNum, maxRect[0], maxRect[1], round(maxWidth, 2),
    # round(maxHeight, 2)))
    print("image:{} x:{} y:{} width:{} height:{}".format(imageNum, maxRect[0], maxRect[1], maxRect[2], maxRect[3]))

    # save the crop image
    img_1 = cv2.imread(imgname)
    # temp_x = round(maxRect[0] - (imageWidth - maxRect[2]) / 2)
    # temp_y = round(maxRect[1] - (imageHeight - maxRect[3]) / 2)
    temp_x = math.ceil(maxRect[0] - (imageWidth - maxRect[2]) / 2)
    temp_y = math.ceil(maxRect[1] - (imageHeight - maxRect[3]) / 2)

    # print("***")
    # print(temp_y)
    # print(temp_y + imageHeight)

    crop = img_1[temp_y: temp_y + imageHeight, temp_x: temp_x + imageWidth]

    # ####### draw rec in the crop image #######
    # gray_1 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # change to gray image
    # res_1, dst_1 = cv2.threshold(gray_1, 0, 255, cv2.THRESH_OTSU)  # 0,255
    # element_1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    # dst_1 = cv2.morphologyEx(dst_1, cv2.MORPH_OPEN, element_1)  # Open operation denoising
    # contours_1, hierarchy_1 = cv2.findContours(dst_1, cv2.RETR_EXTERNAL,
    #                                       cv2.CHAIN_APPROX_SIMPLE)
    # for cont in contours_1:
    #     area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
    #     if area < 500:
    #         continue
    #     # print("{}-blob:{}".format(count,ares),end="  ") #area of each shape
    #     rect_1 = cv2.boundingRect(cont)  # Extract rectangle coordinates
    #
    #     cv2.rectangle(crop, rect_1, (0, 0, 255), 1)


    cv2.imwrite("./pic/ROI_0{:02d}0.png".format(imageNum - 1), crop)

    # show original image, gray image and crop image

    # cv2.namedWindow("original", 1)
    # cv2.imshow('original', img)
    # cv2.namedWindow("dst", 1)
    # cv2.imshow("dst", dst)
    # cv2.namedWindow("crop", 1)
    # cv2.imshow('crop', crop)
    # cv2.waitKey()

    # get the X, Y , width and height of the object
    # return maxRect[0], maxRect[1], round(maxWidth, 2), round(maxHeight, 2)
    return maxRect[0], maxRect[1], maxRect[2], maxRect[3]
