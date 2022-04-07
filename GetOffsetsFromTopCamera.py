#######################
# this file used for combining the top view camera and side view camera.
# not necessary for 3d method.
#######################

import cv2


# get the min area from the top view of image : pic_t
def GetMinArea(imageNum):
    imgname = 'pic_t/' + ("%04d" % imageNum) + '.bmp'
    img = cv2.imread(imgname)  # input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
    # Global threshold segmentation,  to binary image. (Otsu)
    res, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 0,255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
    # cv2.drawContours(dst, contours, -1, (120, 0, 0), 2)  # Draw contour

    for cont in contours:
        area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
        if area < 500:  # keep the largest one, which is the target.
            continue
        minRec = cv2.minAreaRect(cont)
    return minRec


# find two images which corresponding to the side camera images which the target located in the front and back.
# based on their center point (only Y coordinate) to get the turn center.
# 1st, find the image which the target has the largest rotation angle (almost 90), which corresponding to the
# the the side camera image which the target located in the most left or right. then find the cross image with it,
# that is the image which the target located in the front or back.
# 2nd, find the second image, which is opposite to the first image.

def FindTurnCenter():
    imageNum = 1  # start image id
    maxAngle = 0
    standImgId = 1
    # there are total 36 images, we only need to check the first 18, because other 18 are the opposite images.
    while imageNum < 19:
        minRecArea = GetMinArea(imageNum)
        print(minRecArea)
        rotationAngle = minRecArea[2]
        if maxAngle < rotationAngle:
            maxAngle = rotationAngle
            standImgId = (imageNum + 9) % 18
        imageNum += 1

    # to get the first image
    print("standImgId: ")
    print(standImgId)
    print("==============")
    minRecArea = GetMinArea(standImgId)
    firstCoordinate_Y = minRecArea[0][1]

    # to get the second image, that is the opposite fo the first image.
    oppositeImgId = standImgId + 18
    minRecArea = GetMinArea(oppositeImgId)
    secondCoordinate_Y = minRecArea[0][1]

    centerCoordinate_Y = (firstCoordinate_Y / 2) + (secondCoordinate_Y / 2)
    print("center_y: ")
    print(centerCoordinate_Y)
    print("==============")
    return centerCoordinate_Y


def GetOffSets():
    center_Y = FindTurnCenter()
    imageNum = 1
    offsets = []
    while imageNum < 37:
        minRecArea = GetMinArea(imageNum)
        coordinate_Y = minRecArea[0][1]
        offset = coordinate_Y - center_Y
        offsets.append(offset)
        imageNum += 1
    return offsets
