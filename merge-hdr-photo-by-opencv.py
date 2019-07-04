import sys
import os
import cv2
import numpy as np
import exifread
from fractions import Fraction

_str_list = []
with os.scandir('src') as it:
    for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
            _str_list.append(entry.name)


def get_photo_exif_exposuretime(input_photo):
    pass
    # Open image file for reading (binary mode)
    _f = open(input_photo, 'rb')
    # Return Exif tags
    _tags = exifread.process_file(_f)
    _exif_exposuretime = _tags['EXIF ExposureTime']
    return _exif_exposuretime


def make_filenames_times(input_filename_list):
    _filenames, _times = [], []
    for _photo in input_filename_list:
        _file_path = 'src/'+_photo
        _filenames.append(_file_path)
        _time_raw = get_photo_exif_exposuretime(_file_path)
        _time_float = float(Fraction(str(_time_raw)))
        _times.append(_time_float)
    return _filenames, np.array(_times, dtype=np.float32)


def readImagesAndTimes(input_filenames, input_times):
    # 图像文件名称列表
    filenames = input_filenames
    # 曝光时间列表
    times = input_times
    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)
    return images, times

_filenames, _times = make_filenames_times(_str_list)
images, times = readImagesAndTimes(_filenames, _times)

# 对齐输入图像
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

# 获取图像响应函数 (CRF)
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

# 将图像合并为HDR线性图像
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
# 保存图像
cv2.imwrite("hdrDebevec.hdr", hdrDebevec)

# 使用Drago色调映射算法获得24位彩色图像
tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago
cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)

# 使用Durand色调映射算法获得24位彩色图像 # openvcv-3.4.5 将其移除，之后版本再没有了
'''
tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
ldrDurand = tonemapDurand.process(hdrDebevec)
ldrDurand = 3 * ldrDurand
cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)
'''

# 使用Mantiuk色调映射算法获得24位彩色图像
tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
ldrMantiuk = 3 * ldrMantiuk
cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)

# 使用Reinhard色调映射算法获得24位彩色图像
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 1.0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)
cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
