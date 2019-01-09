# -*- coding: utf-8 -*-
# @Time: 18-11-5 下午2:38
# @Author: gtw
import cv2


def green_process(image):
    red = image[:, :, 2] - 5
    blue = image[:, :, 0]
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    result = red - blue
    cv2.imshow('result', result)
    # 从灰度图像获取二值图像
    ret, binary = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 返回指定形状和尺寸的结构元素
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1, iterations=5)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1, iterations=5)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel2, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2, iterations=2)

    cv2.imshow('open', binary)
    # 把白色区域在原图上赋值为0,弄为黑色
    image[binary == 255, :] = 0
    # cv2.imshow("rebackground1", image)

    return binary, image


# 四周有光照
def black_process_one(image):
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary, 20, 255, cv2.THRESH_BINARY_INV)# | cv2.THRESH_OTSU)
    # cv2.imshow('bw', binary)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1, iterations=3)
    # cv2.imshow('close', binary)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2, iterations=5)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel2, iterations=5)
    # cv2.imshow('reslut', binary)
    # binary = cv2.bitwise_not(binary)
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3, iterations=5)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3, iterations=5)
    # cv2.imshow('mor', binary)
    image[binary == 255, :] = 0
    # cv2.imshow('final', image)

    return binary, image


# 四周无光照
def black_process_two(image):
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1, iterations=3)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel2, iterations=3)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2, iterations=3)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2, iterations=1)
    # cv2.imshow('binary', binary)
    image[binary == 0, :] = 0

    return binary, image
