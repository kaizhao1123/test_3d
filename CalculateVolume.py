# Compile the C carving program
import subprocess
import os.path
from sys import platform

# only compile c programm on linux, binary for win32 included.
if platform != "win32":
    if not os.path.isfile("CarveIt.o"):
        p = subprocess.Popen("gcc -O3 CarveIt.c -lm -o CarveIt.o", shell=True)
        p.wait()


# Dummy class for storage
class Object(object):
    pass


from HSVSegmentSeq import HSVSegmentSeq
from TurntableCarve import TurntableCarve
from Crop import Crop
import numpy as np


##################################################################


##################################################################
def CalculateVolume(vintValue, pixPerMMAtZ, imageLength, imageWidth):
    # input
    # vintValue:  the vint value of the mask image
    # imageLength: the length of the ROI and mask image
    # imageWidth: the width of the ROI and mask image

    print("****** Cropping ******")
    imgNum = 1
    maxWidth = 0
    minWidth = 1000
    height = 1000
    aveWidth = 0
    offsets = np.zeros((2, 36), np.float)
    standX = 0
    standY = 0
    while imgNum < 37:
        X, Y, objectWidth, objectHeight = Crop(imgNum, imageLength, imageWidth)
        if imgNum == 1:
            standX = X
            standY = Y
        aveWidth += objectWidth
        if objectWidth > maxWidth:
            maxWidth = objectWidth
        if objectWidth < minWidth:
            minWidth = objectWidth
        if objectHeight < height:
            height = objectHeight
        offsets[:, (imgNum - 1)] = np.array([(X - standX), (Y - standY)])
        imgNum += 1
    aveWidth = round(aveWidth / 36)
    print("******  Result ******")
    print("length: " + str(maxWidth))
    print("width: " + str(minWidth))
    print("height: " + str(height))
    print("aveWidth: " + str(aveWidth))
    print(offsets)

    print("************")

    ##################################################################
    # crop_rect = np.array([272, 145, 200, 200])
    crop_rect = np.array([279, 206, 200, 200])
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

    ##################################################################
    # initialization for 'TurntableCarve'
    #
    # image and camera properties
    cam = Object()
    cam.orig_image_size = np.array([720, 540])  # original size of the image, needed for principal point
    cam.offset = offsets  # offsets of the cropping regions
    cam.crop_rect = crop_rect  # cropping rectangle
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
    volume_in_mm3 = TurntableCarve(fnmask, cam, V, imageLength, imageWidth)
    ##################################################################

    ##################################################################
    # print result

    print('Volume = ' + ("%0.2f" % volume_in_mm3) + 'mm^3\n')
    ##################################################################
