# Compile the C carving program
import subprocess
import os.path
from sys import platform

import cv2
import numpy as np

from HSVSegmentSeq import HSVSegmentSeq
from TurntableCarve import TurntableCarve
from CropWithAjustment import CropWithAdjustment
from CropWithAjustment import getExpectedValues
from CropWithAjustment import PreAdjustment
import math

# only compile c programm on linux, binary for win32 included.
if platform != "win32":
    if not os.path.isfile("CarveIt.o"):
        p = subprocess.Popen("gcc -O3 CarveIt.c -lm -o CarveIt.o", shell=True)
        p.wait()


# Dummy class for storage
class Object(object):
    pass


# #################################################################


# #################################################################
def CalculateVolume(path, vintValue, pixPerMMAtZ, imageWidth, imageHeight, sheet, rowCount, auto):
    print("****** Cropping ******")

    # ############ deal with images ##################
    PreAdjustment(path, vintValue)
    allWidthData, allHeightData = getExpectedValues(path, vintValue, "original")
    CropWithAdjustment(path, vintValue, imageWidth, imageHeight, allWidthData)

    allWidthData.sort()
    result_Length = allWidthData[35]
    result_Width = allWidthData[0]
    result_Height = sum(allHeightData) / 36

    print("length: " + str(result_Length))
    print("width: " + str(result_Width))
    print("height: " + str(result_Height))

    ##################################################################
    fnroi = Object()
    fnroi.base = path + 'ROI_'
    fnroi.number = range(0, 360, 10)
    fnroi.extension = '.png'
    ##################################################################
    # initialization for 'HSVSegmentSeq'
    #
    # initial 'Mask' images
    fnmask = Object();
    fnmask.base = path + 'Mask_';
    fnmask.number = range(0, 360, 10);
    fnmask.extension = '.png';
    # color interval of foreground object in HSV space
    Hint = [0, 255]
    Sint = [0, 255]
    Vint = [vintValue, 255]
    # Vint = [30, 255]

    # segment seed using its HSV color value
    HSVSegmentSeq(fnroi, fnmask, Hint, Sint, Vint)

    ##################################################################

    # #######remove the concave part from the result of the slicing method.
    # imgNum = 1
    # allImage = []
    # while imgNum < 37:
    #     X, Y, width, height = GetArea(path, vintValue, imgNum, "roi")
    #     allImage.append((width, imgNum))
    #     imgNum += 1
    # newAllImage = sorted(allImage, key=lambda tup: tup[0])
    # img_1st_min = newAllImage[0][1]
    # img_2nd_min = allImage[(img_1st_min - 1 + 18) % 36][1]
    # concaveWidth_1, concaveAngel_1 = findConcave(path, vintValue, img_1st_min)
    # concaveWidth_2, concaveAngel_2 = findConcave(path, vintValue, img_2nd_min)
    # if concaveWidth_1 < concaveWidth_2:
    #     target = img_2nd_min
    # else:
    #     target = img_1st_min
    # concaveWidth, concaveAngel = findConcave(path, vintValue, target)
    # print("concaveId: " + str(target))
    # print("concaveWidth: " + str(concaveWidth))
    # print("concaveAngel: " + str(concaveAngel))

    ##################################################################
    # initialization for 'TurntableCarve'
    #
    # image and camera properties
    cam = Object()
    cam.alpha = range(0, -360, -10)  # rotation angle
    cam.PixPerMMAtZ = pixPerMMAtZ  # calibration value: pixel per mm at working depth: measure in image
    cam.PixPerMMSensor = 1 / 0.0069  # 4.7�m pixel size (Nikon D7000, from specs) 1/0.0062
    cam.FocalLengthInMM = 12.5  # read from lens or from calibration
    #
    # description of the reconstruction volume V as cuboid
    V = Object()
    V.VerticalOffset = 0  # Vertical offset of center of reconstruction cuboid (i.e the volume) in roi [unit: pixel]
    V.VerticalOffset_t = 10
    V.VolWidth = 10.0  # width of the volume in mm (X-direction) 10.0
    V.VolHeight = 10.0  # height of the volume in mm (Y-direction) 10.0
    V.VolDepth = 10.0  # depth of the volume in mm (Z-direction) 10.0
    V.sX = 100  # number of voxels in X-direction 100
    V.sY = 100  # number of voxels in Y-direction 100
    V.sZ = 100  # number of voxels in Z-direction 100
    #
    # perform volume carving on mask images
    volume_in_mm3, vVol = TurntableCarve(fnmask, cam, V, imageWidth, imageHeight, auto)

    pointCLoud = []
    data = vVol
    for x in range(len(data)):
        for y in range(len(data[x])):
            for z in range(len(data[x][y])):
                # shell
                if isEdgePoint(data, x, y, z):
                    subData = [x, y, z, 37, 150, 190]
                    subData = np.array(subData)
                    pointCLoud.append(subData)
                # solid
                # if data[x][y][z] == 1:
                #     subData = [x, y, z]
                #     subData = np.array(subData)
                #     pointCLoud.append(subData)
    np.savetxt("sample.txt", pointCLoud)
    ##################################################################

    ##################################################################
    # print result

    # print('length = ' + ("%0.2f" % (aveLength / pixPerMMAtZ)) + 'mm\n')
    # print('width = ' + ("%0.2f" % (aveWidth / pixPerMMAtZ)) + 'mm\n')
    # print('height = ' + ("%0.2f" % (aveHeight / pixPerMMAtZ)) + 'mm\n')
    # print('Volume = ' + ("%0.2f" % volume_in_mm3) + 'mm^3\n')

    #
    pixPerMMAtX = pixPerMMAtZ + 0.5

    result_Length = result_Length / pixPerMMAtZ
    result_Width = result_Width / pixPerMMAtX
    result_Height = result_Height / pixPerMMAtX

    # concaveWidth = concaveWidth / pixPerMMAtZ  ##
    # concaveVol = math.pi * result_Length * concaveWidth * concaveWidth / 6  ##
    # concaveVol = concaveVol * concaveAngel / 360  ##

    VolumeFormula = math.pi * result_Length * result_Width * result_Height / 6
    # volError_1 = (VolumeFormula - volume_in_mm3) / VolumeFormula
    print('VolumeOfContour = ' + ("%0.3f" % VolumeFormula) + 'mm^3\n')

    # VolumeFormula = VolumeFormula - concaveVol  ##
    # volError = (VolumeFormula - volume_in_mm3) / VolumeFormula ##
    #

    print('length = ' + ("%0.3f" % result_Length) + 'mm\n')
    print('width = ' + ("%0.3f" % result_Width) + 'mm\n')
    print('height = ' + ("%0.3f" % result_Height) + 'mm\n')
    print('Volume3D = ' + ("%0.3f" % volume_in_mm3) + 'mm^3\n')
    # print('VolumeOfConcave = ' + ("%0.3f" % concaveVol) + 'mm^3\n')       ##
    # print('VolumeOfWithoutConcave = ' + ("%0.3f" % VolumeFormula) + 'mm^3\n')
    # print('Error_1 = ' + ("%0.4f" % volError_1))
    # print('Error = ' + ("%0.4f" % volError))

    sheet.write(rowCount, 0, rowCount)
    sheet.write(rowCount, 1, result_Length)
    sheet.write(rowCount, 2, result_Width)
    sheet.write(rowCount, 3, result_Height)
    sheet.write(rowCount, 4, volume_in_mm3)
    # sheet.write(rowCount, 5, concaveVol)      ##
    ##################################################################
    return pointCLoud


def isEdgePoint(data, x, y, z):
    if data[x][y][z] == 1:
        if (data[x - 1][y][z] == 0 or data[x + 1][y][z] == 0
                or data[x][y - 1][z] == 0 or data[x][y + 1][z] == 0
                or data[x][y][z - 1] == 0 or data[x][y][z + 1] == 0):
            return True
