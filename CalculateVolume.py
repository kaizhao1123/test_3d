# Compile the C carving program
import subprocess
import os.path
from sys import platform
from HSVSegmentSeq import HSVSegmentSeq
from TurntableCarve import TurntableCarve
from CropWithAjustment import GetArea
from CropWithAjustment import CropWithAdjustment
from CropWithAjustment import getExpectedValues
from FindConcave import findConcave
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
def CalculateVolume(path, vintValue, pixPerMMAtZ, imageWidth, imageHeight, sheet, rowCount):

    print("****** Cropping ******")

    ############################################

    # imgNum = 1
    # maxWidth = 0  # that is, length
    # minWidth = 1000  # that is, width
    # height = 1000
    #
    # standX = 0
    # standY = 0
    # allLengths = []
    # allWidths = []
    # allHeights = []
    #
    # # ####################### Get offsets from the top view (top camera)
    # # offsets = GetOffSets()
    # # print(offsets)
    # # ######################
    #
    # while imgNum < 37:
    #     X, Y, objectWidth, objectHeight = Crop(imgNum, imageWidth, imageHeight)
    #     if imgNum == 1:
    #         standX = X
    #         standY = Y
    #     allLengths.append(objectWidth)
    #     allWidths.append(objectWidth)
    #     allHeights.append(objectHeight)
    #
    #     if objectWidth > maxWidth:
    #         maxWidth = objectWidth
    #
    #     if objectWidth < minWidth:
    #         minWidth = objectWidth
    #
    #     if objectHeight < height:
    #         height = objectHeight
    #
    #     imgNum += 1
    #
    # allLengths.sort(reverse=True)
    # allLengths = allLengths[:6]
    # allWidths.sort()
    # allWidths = allWidths[:4]
    # # allHeights.sort()
    # # allHeights = allHeights[:6]
    #
    # aveLength = sum(allLengths) / 6
    # # aveWidth = sum(allWidths) / 4
    # aveWidth = allWidths[0]
    # aveHeight = sum(allHeights) / 36
    # print("******  Result ******")
    # print("length: " + str(aveLength))
    # print("width: " + str(aveWidth))
    # print("height: " + str(aveHeight))
    # # print(offsets)
    # print("************")

    # ############ new method with adjusting offsets ##################

    allWidthData, allHeightData = getExpectedValues(path)
    CropWithAdjustment(path, imageWidth, imageHeight, allWidthData)

    #

    allWidthData.sort()
    result_Length = allWidthData[35]
    result_Width = allWidthData[0]
    result_Height = sum(allHeightData) / 36

    print("length: " + str(result_Length))
    print("width: " + str(result_Width))
    print("height: " + str(result_Height))

    ##################################################################
    # crop_rect = np.array([272, 145, 200, 200])
    # crop_rect = np.array([279, 206, 200, 200])
    fnroi = Object()
    fnroi.base = 'pic/ROI_'
    fnroi.number = range(0, 360, 10)
    fnroi.extension = '.png'
    ##################################################################
    # initialization for 'HSVSegmentSeq'
    #
    # initial 'Mask' images
    fnmask = Object();
    fnmask.base = 'pic/Mask_';
    fnmask.number = range(0, 360, 10);
    fnmask.extension = '.png';
    # color interval of foreground object in HSV space
    Hint = [0, 255]
    Sint = [0, 255]
    Vint = [vintValue, 255]  # 75

    # segment seed using its HSV color value
    HSVSegmentSeq(fnroi, fnmask, Hint, Sint, Vint)
    ##################################################################

    #
    imgNum = 1
    allImage = []
    while imgNum < 37:
        X, Y, width, height = GetArea(path, imgNum, "roi")
        allImage.append((width, imgNum))
        imgNum += 1
    newAllImage = sorted(allImage, key=lambda tup: tup[0])
    print(newAllImage)
    img_1st_min = newAllImage[0][1]
    img_2nd_min = allImage[(img_1st_min - 1 + 18) % 36][1]
    concaveWidth_1, concaveAngel_1 = findConcave(img_1st_min)
    concaveWidth_2, concaveAngel_2 = findConcave(img_2nd_min)
    print(img_1st_min, concaveWidth_1)
    print(img_2nd_min, concaveWidth_2)
    if concaveWidth_1 < concaveWidth_2:
        target = img_2nd_min
    else:
        target = img_1st_min
    concaveWidth, concaveAngel = findConcave(target)
    print("concaveId: " + str(target))
    print("concaveWidth: " + str(concaveWidth))
    print("concaveAngel: " + str(concaveAngel))



    ##################################################################
    # initialization for 'TurntableCarve'
    #
    # image and camera properties
    cam = Object()
    # cam.orig_image_size = np.array([720, 540])  # original size of the image, needed for principal point
    # cam.offset = offsets  # offsets of the cropping regions
    # cam.crop_rect = crop_rect  # cropping rectangle
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
    volume_in_mm3 = TurntableCarve(fnmask, cam, V, imageWidth, imageHeight)
    ##################################################################

    ##################################################################
    # print result

    # print('length = ' + ("%0.2f" % (aveLength / pixPerMMAtZ)) + 'mm\n')
    # print('width = ' + ("%0.2f" % (aveWidth / pixPerMMAtZ)) + 'mm\n')
    # print('height = ' + ("%0.2f" % (aveHeight / pixPerMMAtZ)) + 'mm\n')
    # print('Volume = ' + ("%0.2f" % volume_in_mm3) + 'mm^3\n')


    #

    result_Length = result_Length / pixPerMMAtZ
    result_Width = result_Width / pixPerMMAtZ
    result_Height = result_Height / pixPerMMAtZ

    concaveWidth = concaveWidth / pixPerMMAtZ
    concaveVol = math.pi * result_Length * concaveWidth * concaveWidth / 6
    concaveVol = concaveVol * concaveAngel / 360

    VolumeFormula = math.pi * result_Length * result_Width * result_Height / 6
    volError_1 = (VolumeFormula - volume_in_mm3) / VolumeFormula
    print('VolumeOfContour = ' + ("%0.2f" % VolumeFormula) + 'mm^3\n')

    VolumeFormula = VolumeFormula - concaveVol
    volError = (VolumeFormula - volume_in_mm3) / VolumeFormula
    #

    print('length = ' + ("%0.2f" % result_Length) + 'mm\n')
    print('width = ' + ("%0.2f" % result_Width) + 'mm\n')
    print('height = ' + ("%0.2f" % result_Height) + 'mm\n')
    print('Volume3D = ' + ("%0.2f" % volume_in_mm3) + 'mm^3\n')
    print('VolumeOfConcave = ' + ("%0.2f" % concaveVol) + 'mm^3\n')
    print('VolumeOfWithoutConcave = ' + ("%0.2f" % VolumeFormula) + 'mm^3\n')
    print('Error_1 = ' + ("%0.4f" % volError_1))
    print('Error = ' + ("%0.4f" % volError))

    # sheet.write(rowCount, 0, rowCount)
    # sheet.write(rowCount, 1, aveLength/pixPerMMAtZ)
    # sheet.write(rowCount, 2, aveWidth/pixPerMMAtZ)
    # sheet.write(rowCount, 3, aveHeight/pixPerMMAtZ)
    # sheet.write(rowCount, 4, volume_in_mm3)
    ##################################################################
