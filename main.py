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
    sheet1.write(0, 5, 'concave vol')


# open the excel file to store the results
loc = "./result.xls"
rowCount, wb = ReadFromResult(loc)
sheet1 = wb.get_sheet(0)
print(rowCount)
print("&&&&&&&&&&&&&&")

# the main parameters we need to setup
vintValue = 90  # the value of light
pixPerMMAtZ = 76 / 3.945  # 95/3.945  # 145/5.74 # 94.5/3.945 #157/6.78 #94 /3.94
imageWidth = 200
imageHeight = 200

# the source of the images
path = 'pic/'
CalculateVolume(path, vintValue, pixPerMMAtZ, imageWidth, imageHeight, sheet1, rowCount, False)

# #####duplicated tests for 20 times
# path = './pictures/11-4-3/pic'
# index = 0
# while index < 20:
#     subPath = path + str(index+1) + '/'
#     CalculateVolume(subPath, vintValue, pixPerMMAtZ, imageWidth, imageHeight, sheet1, rowCount + index, True)
#     index += 1
#     wb.save('result.xls')
