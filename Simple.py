import os
import sys
import xlrd
import cv2
from xlutils.copy import copy
from CalculateVolume import CalculateVolume
from time import time


def app_path():
    """Returns the base application path."""
    if hasattr(sys, 'frozen'):
        # Handles PyInstaller
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)


# Dummy class for storage
class Object(object):
    pass


# open the excel file.
def ReadFromResult(file):
    result = xlrd.open_workbook(file)
    sheet1 = result.sheet_by_index(0)
    rowCount = sheet1.nrows
    wb = copy(result)
    return rowCount, wb


# deal with the captured images and save new images into process folder
def storeImagesIntoProcess(dic_cap, dic_pro, imageCount):
    imageNum = 1
    while imageNum < (imageCount+1):
        image = cv2.imread(dic_cap + "00{:02d}.bmp".format(imageNum))
        image = image[:, 220:420]   # keep the middle part including the target.
        cv2.imwrite(dic_pro + "00{:02d}.bmp".format(imageNum), image)
        imageNum += 1


if __name__ == '__main__':

    startTime = time()

    # Create input information from user (username, seedCategory, seedId, seedWeight, volSlope, maxWt, minWt).
    inputIn = Object()
    inputIn.userName = 'kz'
    inputIn.seedCategory = 'wheat'
    inputIn.seedId = 'lg1'
    inputIn.seedWeight = 10
    inputIn.volSlope = 1.1

    # setup the some default folders.
    root_path = app_path() + '/'

    if len(sys.argv) > 1:
        capturedImageFolder = sys.argv[1]
        running_path = root_path + 'run/' + capturedImageFolder + '/'
        if not os.path.exists(running_path):
            raise ValueError("The specified path does not exist! ")
    else:
        running_path = root_path

    dic_cap = running_path + 'pic_captured/'
    dic_pro = running_path + 'pic_processing/'
    if not os.path.exists(dic_pro):
        os.makedirs(dic_pro)

    #
    pixPerMMAtZ = 76/3.94
    imageWidth = 200
    imageHeight = 200
    show3D = False
    save = False
    excel_path = root_path + 'result.xls'
    rowCount, wb = ReadFromResult(excel_path)
    sheet = wb.get_sheet(0)
    imageCount = 36

    storeImagesIntoProcess(dic_cap, dic_pro, imageCount)

    l, w, h, v1, v2, RGB = CalculateVolume(inputIn, dic_pro, pixPerMMAtZ, imageWidth, imageHeight, show3D,
                                           save, excel_path, wb, sheet, rowCount, imageCount)

    print('length = ' + ("%0.3f" % l) + 'mm')
    print('width = ' + ("%0.3f" % w) + 'mm')
    print('height = ' + ("%0.3f" % h) + 'mm')
    print('Volume3D_1 = ' + ("%0.3f" % v1) + 'mm^3')
    print('Volume3D_2 = ' + ("%0.3f" % v2) + 'mm^3')
    print(RGB)
    print('\n')
    print("Total time: --- %0.3f seconds ---" % (time() - startTime) + "\n")
    print("The calculation of '%s' " % inputIn.seedCategory + " is complete!")



