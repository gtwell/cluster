# -*- coding: utf-8 -*-
# @Time: 18-12-7 下午2:38
# @Author: gtw
import cv2


def black_process_one(image1):
    image = image1.copy()
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary, 100, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
    cv2.imshow('binary', binary)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1, iterations=3)
    image[binary == 0, :] = 0
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[binary >= 220, :] = 0 
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('binary1', binary)
    ret, binary = cv2.threshold(binary, 10, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
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


if __name__ == '__main__':
    import random
    import glob
    random.seed(0)
    img_list = glob.glob('voc_80/*')
    img_path = random.choice(img_list)
    img = cv2.imread(img_path)
    binary, image = black_process_one(img)
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.drawContours(img, contours, 3, (0, 255, 0), -1)
    # for contour in contours:
    #     image = cv2.drawContours(img, [contour], 0, (0, 255, 0), 3)
        # x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    # x, y, w, h = cv2.boundingRect(binary)
    # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0))
    # cv2.imshow('binary', binary)
    # cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
