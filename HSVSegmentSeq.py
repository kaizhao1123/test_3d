import numpy as np
from scipy.misc import imread, toimage
import cv2


def HSVSegmentSeq(fnin, fnout, Hint, Sint, Vint):
    # Pixelwise color segmentation in HSV space. Reads images from files and
    # writes masks to files
    #
    # Input:
    # fnin: describes names of input files
    # fnout: describes names of output files, i.e. cropped regions
    # Hint, Sint, Vint: foreground intervals in HSV channels
    #
    # #
    # # segment seed using its HSV color value
    # HSVSegmentSeq(fnroi,fnmask,Hint,Sint,Vint);
    ####################################################################

    # print info
    print('Segment images\n');

    # color segment images
    NumImgs = len(fnin.number)
    for i in range(NumImgs):
        # print a point to show progress
        print(".")

        # read image from file
        img = ReadImage(fnin, i)

        # color segmentation
        mask = HSVSegment(img, Hint, Sint, Vint)

        # write mask to file
        WriteImage(mask, fnout, i)

    # print end of line
    print('\n')


##########################################################
# read an image or mask from file
def ReadImage(fn, idx):
    imgfilename = fn.base + ("%04d" % fn.number[idx]) + fn.extension;
    img = imread(imgfilename);
    return img;


##########################################################
# write an image or mask to file
def WriteImage(img, fn, idx):
    imgfilename = fn.base + ("%04d" % fn.number[idx]) + fn.extension;
    img = toimage(img, (2 ** 16 - 1), 0, mode='I');  # workaround to create 16-bit .pngs
    img.save(imgfilename);


##########################################################
# perform HSV segmentation
def HSVSegment(rgb_image, Hint, Sint, Vint):
    # convert RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV);
    sz = hsv_image.shape;

    # define masks for each channel and set pixels with the respective
    # channel value in the given interval to foreground, i.e. 255
    hmask = np.zeros((sz[0], sz[1]), int);
    hmask[(hsv_image[:, :, 0] >= Hint[0]) & (hsv_image[:, :, 0] <= Hint[1])] = 255;
    smask = np.zeros((sz[0], sz[1]), int);
    smask[(hsv_image[:, :, 1] >= Sint[0]) & (hsv_image[:, :, 1] <= Sint[1])] = 255;
    vmask = np.zeros((sz[0], sz[1]), int);
    vmask[(hsv_image[:, :, 2] >= Vint[0]) & (hsv_image[:, :, 2] <= Vint[1])] = 255;
    # combine channel masks to a single mask
    return hmask * smask * vmask;