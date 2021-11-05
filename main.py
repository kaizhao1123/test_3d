
from CalculateVolume import CalculateVolume
import xlrd
from xlwt import Workbook
from xlutils.copy import copy

def ReadFromResult(loc):
    result = xlrd.open_workbook(loc)
    sheet1 = result.sheet_by_index(0)
    rowCount = sheet1.nrows
    wb = copy(result)
    return rowCount, wb

def CreateNewResult():
    result = Workbook()
    sheet1 = result.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'ID')
    sheet1.write(0, 1, 'Length')
    sheet1.write(0, 2, 'Width')
    sheet1.write(0, 3, 'Height')
    sheet1.write(0, 4, 'Volume')
    sheet1.write(0, 5, 'Type')

loc = "./result.xls"
rowCount, wb = ReadFromResult(loc)
sheet1 = wb.get_sheet(0)
print(rowCount)
print("&&&&&&&&&&&&&&")
path = 'pic/'
vintValue = 100
pixPerMMAtZ = 95/3.945  # 145/5.74 # 94.5/3.945 #157/6.78 #94 /3.94
imageWidth = 200
imageHeight = 200

CalculateVolume(path, vintValue, pixPerMMAtZ, imageWidth, imageHeight, sheet1, rowCount)
wb.save('result.xls')





# #########################################################


# import cv2
# import numpy as np
# import math
#
#
#
#
# imgname = 'pic/' + ("%04d" % 1) + '.bmp'
#
# img = cv2.imread(imgname)  # input image
# #img = img[0:480, 0:720]
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image
# # Global threshold segmentation,  to binary image. (Otsu)
# res, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 0,255 cv2.THRESH_OTSU
# element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
# dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, element)  # Open operation denoising
#
# contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE,    #external
#                                        cv2.CHAIN_APPROX_SIMPLE)  # Contour detection function
# # cv2.drawContours(dst, contours, -1, (0, 0, 255), 3)  # Draw contour
#
#
# maxCont = 0
# for cont in contours:
#     area = cv2.contourArea(cont)  # Calculate the area of the enclosing shape
#     if area < 500:  # keep the largest one, which is the target.
#         continue
#     maxCont = cont
# maxRect = cv2.boundingRect(maxCont)
# X = maxRect[0]
# Y = maxRect[1]
# width = maxRect[2]
# height = maxRect[3]
#
# cv2.drawContours(img, maxCont, -1, (0, 255, 255), 1)  # Draw contour
#
# hull = cv2.convexHull(maxCont, returnPoints=False)
# #cv2.drawContours(img, [hull], -1, (255, 0, 0), 1)
# #print(hull)
# convexityDefects = cv2.convexityDefects(maxCont, hull)
# #print(convexityDefects)
#
# mostFarthestIndex = 0
# mostDistance = 0
# finalStartIndex = 0
# finalEndIndex = 0
#
# for d in range(convexityDefects.shape[0]):
#     start, end, farthest, distance = convexityDefects[d][0]
#    # start_pt = tuple(maxCont[start][0])
#     #end_pt = tuple(maxCont[end][0])
#     if distance > mostDistance:
#         mostDistance = distance
#         finalStartIndex = start
#         finalEndIndex = end
#         mostFarthestIndex = farthest
# mostFarthest_pt = tuple(maxCont[mostFarthestIndex][0])
# finalStart_pt = tuple(maxCont[finalStartIndex][0])
# finalEnd_pt = tuple(maxCont[finalEndIndex][0])
#
# cv2.circle(img, mostFarthest_pt, 3, [255, 0, 0], -1)
# cv2.circle(img, finalStart_pt, 3, [255, 0, 0], -1)
# cv2.circle(img, finalEnd_pt, 3, [255, 0, 0], -1)
#
# startPoint = maxCont[finalStartIndex]
# endPoint = maxCont[finalEndIndex]
# centerPoint = maxCont[mostFarthestIndex]
#
# # print(finalStartIndex)
# # print(finalEndIndex)
# # print(mostDistance)
# # print(mostFarthestIndex)
#
# print(maxCont[finalStartIndex])
# print(maxCont[finalEndIndex])
# print(maxCont[mostFarthestIndex])
#
# disFromStartPoint = math.sqrt((startPoint[0][0] - centerPoint[0][0]) * (startPoint[0][0] - centerPoint[0][0]) +
#                               (startPoint[0][1] - centerPoint[0][1]) * (startPoint[0][1] - centerPoint[0][1]))
#
# disFromEndPoint = math.sqrt((endPoint[0][0] - centerPoint[0][0]) * (endPoint[0][0] - centerPoint[0][0]) +
#                             (endPoint[0][1] - centerPoint[0][1]) * (endPoint[0][1] - centerPoint[0][1]))
#
# # if disFromStartPoint < disFromEndPoint:
# #     weight = disFromStartPoint
# # else:
# #     weight = disFromEndPoint
#
# sidePoint = 0
# sideDistance = 0
# if disFromStartPoint < disFromEndPoint:
#     sidePoint = startPoint
#     sideDistance = disFromStartPoint
# else:
#     sidePoint = endPoint
#     sideDistance = disFromEndPoint
# weight = math.fabs(centerPoint[0][1] - sidePoint[0][1])
# angel = math.atan(math.fabs(sidePoint[0][0] - centerPoint[0][0]) / math.fabs(sidePoint[0][1] - centerPoint[0][1])) * 180 / math.pi
#
# print("***")
# print(weight)
# print(weight / sideDistance)
# print(angel)
#
#
# cv2.namedWindow("original", 1)
# cv2.imshow('original', img)
# cv2.namedWindow("dst", 1)
# cv2.imshow("dst", dst)
# #cv2.namedWindow("contour", 1)
# #cv2.imshow('contour', img2)
# cv2.waitKey()
















# #################################################3
# import cv2 as cv
# cap = cv.VideoCapture(1)  #, cv.CAP_DSHOW
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()
# #####################################################

#
# import cv2
#
#
# imgNum = 1
# while imgNum < 37:
#     imgname = 'pictures/chendi-image/' + ("%04d" % imgNum) + '.bmp'
#     img = cv2.imread(imgname)  # input image
#     img = img[0:410, 0:719]
#     cv2.imwrite("./pictures/00{:02d}.bmp".format(imgNum), img)
#     imgNum += 1









# print("****** Cropping ******")
# imgNum = 1;
# maxWidth = 0;
# minWidth = 1000;
# height = 1000;
#
# Crop(1);

# while (imgNum < 2):
#
#     objectWidth, objectHeight = Crop(imgNum);
#     if(objectWidth > maxWidth):
#         maxWidth = objectWidth;
#     if(objectWidth < minWidth):
#         minWidth = objectWidth;
#     if(objectHeight < height):
#         height = objectHeight;
#     imgNum += 1;

# print("******  Result ******")
# print("length: " + str(maxWidth));
# print("width: " + str(minWidth));
# print("height: " + str(height));

# objectWidth = Crop(36);
# print("width: " + str(objectWidth));

"""
img = cv2.imread("test.bmp")  #input image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #change to gray image

#Local threshold segmentation, to binary image
#dst = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31, 1) #101

#Global threshold segmentation,  to binary image. (Otsu)
res ,dst = cv2.threshold(gray,0 ,255, cv2.THRESH_OTSU) #0,255

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))#Morphological denoising
dst=cv2.morphologyEx(dst,cv2.MORPH_OPEN,element)  #Open operation denoising

contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #Contour detection function
cv2.drawContours(dst,contours,-1,(120,0,0),2)  #Draw contour

count  = 0; # the count of shapes
maxArea = 0;
maxCont = [];
for cont in contours:

    ares = cv2.contourArea(cont)#计算包围性状的面积
    if ares<50:   #Calculate the area of the enclosing shape
        continue
    #print("{}-blob:{}".format(count,ares),end="  ") #area of each shape
    rect = cv2.boundingRect(cont) #Extract rectangle coordinates
    cv2.rectangle(img,rect,(0,0,0xff),1)#Draw rectangle
    if((ares > maxArea) and (rect[0] > 0) and (rect[0] > 0)):
        maxArea = ares;
        maxCont = cont;

    # Write the number in the upper left corner of the rectangle
    #count+=1;
    #cv2.putText(img,str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)

maxRect = cv2.boundingRect(maxCont) #Extract rectangle coordinates of the target shape

print("x:{} y:{} width:{} height:{}".format(maxRect[0], maxRect[1], maxRect[2], maxRect[3]))#打印坐标

imageWidth = 200;
imageHeight = 200;
temp_x = round(maxRect[0] - (imageWidth - maxRect[2])/2);
temp_y = round(maxRect[1] - (imageHeight - maxRect[3])/2);
crop_rect = [temp_x,temp_y,imageWidth,imageHeight];
print(crop_rect)

img_1 = cv2.imread("test.bmp")
crop = img_1[temp_y: temp_y + imageHeight, temp_x: temp_x+imageWidth]
cv2.imshow('crop',crop);
cv2.imwrite("./00{:02d}.bmp".format(0), crop)

# show original image and gray image
cv2.namedWindow("imagshow", 2)
cv2.imshow('imagshow', img)
cv2.namedWindow("dst", 2)
cv2.imshow("dst", dst)
cv2.waitKey()

"""




"""
offsets = np.array([[0,10],[0,0]]);
camH = offsets[:, 0]
print("{}{}".format("camH:", camH))
orig_image_size = np.array([720, 540])
crop_rect = np.array([272, 145, 200, 200])
d = orig_image_size / 2 + 0.5
print("{}{}".format("d:", d))
p = orig_image_size / 2 + 0.5 - crop_rect[0:2] + 1 - camH;
print("{}{}".format("p:", p))
Z = 88.29
TurntableCenter = np.array([100,100])
ImageOfOrigin = TurntableCenter - [0, 0]
print("{}{}".format("ImageOfOrigin:", ImageOfOrigin))
PixPerMMAtZ = 145/6.35
X = -(ImageOfOrigin[0] - p[0]) / PixPerMMAtZ;
Y = -(ImageOfOrigin[1] - p[1]) / PixPerMMAtZ;
print("{}{}".format("X:", X))
print("{}{}".format("Y:", Y))
f = 2016.13
K = np.matrix([[f, 0, p[0]], [0, f, p[1]], [0, 0, 1]]);
print("{}{}".format("K:", K))
alpha = range(0, -360, -10)
print("{}{}".format("alpha[1]:", alpha[1]))
c = np.cos(alpha[0] / 180.0 * np.pi);
print("{}{}".format("c:", c))
s = np.sin(alpha[0] / 180.0 * np.pi);
print("{}{}".format("s:", s))
R = np.matrix([[c, 0, -s], [0, 1, 0], [s, 0, c]]);
print("{}{}".format("R:", R))
t = [X, Y, -Z];
print("{}{}".format("t:", t))
tT = np.matrix(t).transpose();
print("{}{}".format("tT:", tT))
P = K * np.matrix(np.concatenate((R, -tT), axis=1));
print("{}{}".format("P:", P))
P1 = P.tobytes('f')
print("{}{}".format("P1:", P1))
#for i in range (12):
#    print("{}{}".format("P1:", P1[i]))

"""

"""
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
ax.set_title('projectedVolBox')
ax.set_aspect('equal')
x = y = z = 12.5/2;
Q1 = np.array([x, y, z, 1]).reshape((1, 4));
q1 = P * Q1.transpose();
print("{}{}".format("q1:", q1))
q1 = q1 / q1[2];
print("{}{}".format("q1:", q1))
Q2 = np.array([x, y, -z, 1]).reshape((1, 4));
q2 = P * Q2.transpose();
q2 = q2 / q2[2];
Q3 = np.array([x, -y, z, 1]).reshape((1, 4));
q3 = P * Q3.transpose();
q3 = q3 / q3[2];

Q4 = np.array([x, -y, -z, 1]).reshape((1, 4));
q4 = P * Q4.transpose();
q4 = q4 / q4[2];

Q5 = np.array([-x, y, z, 1]).reshape((1, 4));
q5 = P * Q5.transpose();
q5 = q5 / q5[2];

Q6 = np.array([-x, y, -z, 1]).reshape((1, 4));
q6 = P * Q6.transpose();
q6 = q6 / q6[2];

Q7 = np.array([-x, -y, z, 1]).reshape((1, 4));
q7 = P * Q7.transpose();
q7 = q7 / q7[2];

Q8 = np.array([-x, -y, -z, 1]).reshape((1, 4));
q8 = P * Q8.transpose();
q8 = q8 / q8[2];
plt.plot([float(q1[0]), float(q2[0])], [float(q1[1]), float(q2[1])]);
plt.plot([float(q1[0]), float(q3[0])], [float(q1[1]), float(q3[1])]);
plt.plot([float(q1[0]), float(q5[0])], [float(q1[1]), float(q5[1])]);
plt.plot([float(q2[0]), float(q4[0])], [float(q2[1]), float(q4[1])]);
plt.plot([float(q2[0]), float(q6[0])], [float(q2[1]), float(q6[1])]);
plt.plot([float(q3[0]), float(q4[0])], [float(q3[1]), float(q4[1])]);
plt.plot([float(q3[0]), float(q7[0])], [float(q3[1]), float(q7[1])]);
plt.plot([float(q4[0]), float(q8[0])], [float(q4[1]), float(q8[1])]);
plt.plot([float(q5[0]), float(q6[0])], [float(q5[1]), float(q6[1])]);
plt.plot([float(q5[0]), float(q7[0])], [float(q5[1]), float(q7[1])]);
plt.plot([float(q6[0]), float(q8[0])], [float(q6[1]), float(q8[1])]);
plt.plot([float(q7[0]), float(q8[0])], [float(q7[1]), float(q8[1])]);
plt.show()
"""

"""
x = y = z = 12.5/2;
Q1 = np.array([x, y, z, 1]);
print(len(Q1))
print(Q1[1])
print("*************")

Q2 = Q1.reshape((1, 4));
print(len(Q2))
print(Q2[0][3])
print("*************")

Q3 = Q2.transpose();
print(len(Q3))
print(Q3)
print("*************")
"""
# q1 = P * Q1.transpose();
# q1 = q1 / q1[2];

# d = np.rot90(m, 1, (1,2))
# print(d)

"""
m = np.array([[1,2,3], [4,5,6], [7,8,9]])
a = np.array([m,m,m])
print(a)
print("*************")
#print(a[0][2][1])
b = np.rot90(a,axes=(0,1))
print(b)
print("*************")
c = np.rot90(b,axes=(0,2))
print(c)
"""
