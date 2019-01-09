# -*- coding: utf-8 -*-
# @Time: 18-11-5 下午2:38
# @Author: gtw
import cv2
import numpy as np


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

    # cv2.imshow('open', binary)
    # 把白色区域在原图上赋值为0,弄为黑色
    image[binary == 255, :] = 0
    # cv2.imshow("rebackground1", image)

    return image


# 四周有光照
def black_process_one(image):
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', binary)
    ret, binary = cv2.threshold(binary, 110, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('bw', binary)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1, iterations=4)
    # cv2.imshow('close', binary)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel2, iterations=5)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2, iterations=5)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1, iterations=1)
    # cv2.imshow('reslut', binary)
    # binary = cv2.bitwise_not(binary)
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3, iterations=5)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3, iterations=5)
    # cv2.imshow('mor', binary)
    image[binary == 0, :] = 0
    # cv2.imshow('final', image)

    return binary, image


# 四周无光照
def black_process_two(image):
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary, 20, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1, iterations=3)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2, iterations=3)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel2, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel2, iterations=1)
    # cv2.imshow('binary', binary)
    image[binary == 0, :] = 0

    return binary, image


def black_process_three(image):
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary, 70, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # binary_ = cv2.Canny(binary, 50, 150)
    # circles = cv2.HoughCircles(binary_, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=250, maxRadius=300)
    # circles = np.uint16(np.around(circles))#.squeeze()
    mask = np.zeros_like(binary)
    cv2.circle(mask, (333, 240), 235, (255, 255, 255), -1)
    # for i in circles[0, :]:
    #     cv2.circle(mask, (i[0], i[1]), i[2]-20, (255, 255, 255), -1)
    #     cv2.circle(image, (i[0], i[1]), i[2]-20, (0, 0, 255), 2)
    binary[mask == 0] = 0
    # binary = cv2.Canny(binary, 50, 150)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        binary = cv2.drawContours(binary, [contour], 0, (255, 255, 255), -1)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1, iterations=5)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2, iterations=1)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1, iterations=2)
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel2, iterations=1)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 800:
            binary = cv2.drawContours(binary, [contour], 0, (0, 0, 0), -1)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
    print(hierarchy.shape[1])
    # cv2.imshow('erode', binary)
    # binary = cv2.bitwise_not(binary)
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    # binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3, iterations=5)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel3, iterations=5)
    # cv2.imshow('mor', binary)
    image[binary == 0, :] = 0
    # cv2.imshow('final', image)

    return binary, image


if __name__ == '__main__':
    import random
    import glob
    img_list = glob.glob('/home/guo/project/camera_process/combine/Images_75/Before/*/*')
    img_path = random.choice(img_list)
    # img_path = glob.glob('/home/guo/project/camera_process/combine/three_light/origin/meika/*3097.jpg')[0]
    # print(img_path)
    img = cv2.imread(img_path)
    # img = img[14:457, 56:575]
    cv2.imshow('img', img)
    binary, image = black_process_three(img)
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     image = cv2.drawContours(image, [contour], 0, (0, 255, 0), 3)
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
    # x, y, w, h = cv2.boundingRect(binary)
    # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0))
    # cv2.imshow('binary', binary)
    cv2.imshow('image', image)
    cv2.imwrite('image.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
