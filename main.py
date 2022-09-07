
from CalculateVolume import CalculateVolume
from CropWithAjustment import GetArea
import xlrd
from xlwt import Workbook
from xlutils.copy import copy
import os.path


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
    sheet1.write(0, 1, 'Length(mm)')
    sheet1.write(0, 2, 'Width(mm)')
    sheet1.write(0, 3, 'Height(mm)')
    sheet1.write(0, 4, 'Volume(mm^3)')
    sheet1.write(0, 5, 'concave vol')
    result.save("result.xls")


def isSame(num1, num2):
    X_1, Y_1, width_1, height_1 = GetArea(path, vintValue, num1, "original")
    print(X_1, Y_1, width_1, height_1)
    X_2, Y_2, width_2, height_2 = GetArea(path, vintValue, num2, "original")
    print(X_2, Y_2, width_2, height_2)
    if abs(X_1 - X_2 > 1) or abs(Y_1 - Y_2 > 1) or abs(width_1 - width_2 > 1) or abs(height_1 - height_2 > 1):
        print("false")
        return False
    else:
        print("true")
        return True


# open the excel file to store the results
loc = "./result.xls"
if not os.path.isfile(loc):
    CreateNewResult()
rowCount, wb = ReadFromResult(loc)
sheet1 = wb.get_sheet(0)
# print(rowCount)
# print("&&&&&&&&&&&&&&")

# the main parameters we need to setup
vintValue = 60  # 85  # the value of light
pixPerMMAtZ = 129/6.63   # 126 / 6.5  #  #19.88 #80/3.94  # 59 / 3.12  # 59/3.12   76 / 3.945  # 95/3.945  # 145/5.74 # 94.5/3.945 #157/6.78 #94 /3.94
imageWidth = 200
imageHeight = 200

# the source of the images
path = 'pic/'
points = CalculateVolume(path, vintValue, pixPerMMAtZ, imageWidth, imageHeight, sheet1, rowCount, False)


# ###### Point Cloud ######
# ##################################################
import numpy as np
import open3d as o3d

dataname ="./sample.txt"
point_cloud = np.loadtxt(dataname, skiprows=1)
# print(point_cloud)
#
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255)
pcd.estimate_normals()

# o3d.io.write_point_cloud("./sync.ply", pcd)

# make  all normals point outwards
countOfNeighbors = 10
pcd.orient_normals_consistent_tangent_plane(countOfNeighbors)

# print(np.asarray(pcd_load.normals)[:10, :])
# o3d.visualization.draw_geometries([pcd])  # , point_show_normal = True)
# o3d.io.write_point_cloud("./new.ply", pcd)


# ########################################################################################
# ******************************************
#  Alpha shape  *****  works ********
# //

# alpha = 3
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# //
# ******************************************
# #########################################################################################


# ########################################################################################
# ******************************************
# ball pivoting *******  works not well ************
# //

# estimate radius for rolling ball
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# print(avg_dist)
# radius = 1.5 * avg_dist
#
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#            pcd_load,
#            o3d.utility.DoubleVector([radius, radius * 2]))
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)

# //
# ******************************************
# ###########################################################################################


# ########################################################################################
# ******************************************
# Poisson surface reconstruction *******   ************
# //

# 1. read from the pointcloud  **** works well ****

# radius = 3
# max_nn = 10
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
#
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
# print(mesh)
# o3d.visualization.draw_geometries([mesh])

# 2. mesh method **** works well ****

print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
# mesh.compute_vertex_normals()   # add normal of the triangle, otherwise, is not 3D.
# o3d.visualization.draw_geometries([mesh])


# //
# ******************************************
# ###########################################################################################


# smooth the surface of the model (mesh filter)

mesh_out = mesh.filter_smooth_simple(number_of_iterations=5)
mesh_out.compute_vertex_normals()  # add normal of the triangle, otherwise, is not 3D.
o3d.visualization.draw_geometries([mesh_out])
print(mesh_out)
#o3d.io.write_triangle_mesh("./res.stl", mesh_out)         # save to stl file

# //
# ******************************************
# ###########################################################################################


# ##########        test camera          #############

#
# import cv2
# vid = cv2.VideoCapture(1)
#
# while (True):
#     ret, frame = vid.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
# cv2.destroyAllWindows()

#   #########################################3


#         about the carving ##############33
# def f(x, y):
#     return np.sin(np.sqrt(x ** 2 + y ** 2))
#
#
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
#
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
#
# print(Y.shape)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('3D contour')
# plt.show()

####################
#
# # Create some random data.
# volume = np.random.rand(N, N, N)
# x = np.arange(volume.shape[0])[:, None, None]
# y = np.arange(volume.shape[1])[None, :, None]
# z = np.arange(volume.shape[2])[None, None, :]
# x, y, z = np.broadcast_arrays(x, y, z)
#
# # Turn the volumetric data into an RGB array that's
# # just grayscale.  There might be better ways to make
# # ax.scatter happy.
# c = np.tile(volume.ravel()[:, None], [1, 3])
#
# # Do the plotting in a single call.
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(x.ravel(),
#            y.ravel(),
#            z.ravel(),
#            c=c)
# plt.show()

# s = datetime.now().date()
# print("time")
# print(type(s))



