# -*- coding: utf-8 -*-
# @Time: 18-12-15 下午2:38 # @Author: gtw
import cv2
import numpy as np


def black_process_three(image):
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(binary, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # binary_ = cv2.Canny(binary, 50, 150)
    mask = np.zeros_like(binary)
    # (334, 240)不停的试的
    cv2.circle(mask, (333, 240), 235, (255, 255, 255), -1)
    binary[mask == 0] = 0
    cv2.circle(binary, (333, 240), 235, (255, 255, 255), 2)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        # 120000 = pi*r*r (r = 235) 整个载物台的面积
        if area > 200 and area < 120000:
            binary = cv2.drawContours(binary, [contour], 0, (255, 255, 255), -1)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1, iterations=5)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1, iterations=1)
    # binary = cv2.Canny(binary, 50, 150)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 950:# 也是试的
            binary = cv2.drawContours(binary, [contour], 0, (0, 0, 0), -1)
    cv2.imshow('binary1', binary)
    image[binary == 0, :] = 0

    return binary, image


if __name__ == '__main__':
    import random
    import glob
    img_list = glob.glob('/home/guo/project/camera_process/combine/voc_80/*.jpg')
    img_path = random.choice(img_list)
    # img_path = glob.glob('/home/guo/project/camera_process/combine/three_light/origin/meika/*3097.jpg')[0]
    # print(img_path)
    img = cv2.imread(img_path)
    cv2.imshow('img', img)
    binary, image = black_process_three(img)
    # cv2.imshow('binary', binary)
    cv2.imshow('image', image)
    cv2.imwrite('image.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
