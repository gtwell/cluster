import cv2
import os
import os.path as osp
import numpy as np
import random
# from process import black_process_one, black_process_two
from process_new import black_process_three
import glob
import ipdb
import itertools


def mix_two_images(base, img2):
    # roi = base[55:335, 145:545]
    roi = base
    # roi = base[43:421, 139:525]
    # 将img2处理后的pre2上的物体抠出来放在img1处理后的base处理后的图上
    # img2 = img2[43:421, 139:525]
    # img2 = cv2.resize(img2, (400, 280))
    mask, pre2 = black_process_three(img2)
    # cv2.imshow('mask', mask)

    mask_inv = cv2.bitwise_not(mask)

    # mask 为掩码，与原图作用时，mask=0的地方原图变为黑色，其他地方不变
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # cv2.imshow('img1_bg', img1_bg)
    img2_bg = cv2.bitwise_and(pre2, pre2, mask=mask)

    dst = cv2.add(img1_bg, img2_bg)
    base = dst
    # base[55:335, 145:545] = dst

    return base


def mix_images(base, img2):
    # 将img2处理后的pre2上的物体抠出来放在img1处理后的base处理后的图上
    mask, pre2 = black_process_three(img2)
    # cv2.imshow('mask', mask)

    mask_inv = cv2.bitwise_not(mask)

    # mask 为掩码，与原图作用时，mask=0的地方原图变为黑色，其他地方不变
    img1_bg = cv2.bitwise_and(base, base, mask=mask_inv)
    # cv2.imshow('img1_bg', img1_bg)
    img2_bg = cv2.bitwise_and(pre2, pre2, mask=mask)

    dst = cv2.add(img1_bg, img2_bg)
    # base[55:335, 145:545] = dst

    return dst


# path = '/home/guo/project/camera_process/combine/Images_75_new/Before/'


def save_train(img_path):
    base = cv2.imread('back.jpg')
    img = cv2.imread(img_path)
    base = mix_images(base, img)
    save_folder = osp.join('train_75', '/'.join(img_path.split('/')[3:4]))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    cv2.imwrite(osp.join('train_75', '/'.join(img_path.split('/')[3:])), base)
    return base


for i in glob.glob('./Images_75_new/Before/*/*'):
    # print(i)
    save_train(img_path=i)
# n = 2
# N = 0
# for i, img_path in enumerate(glob.glob('./Images3/Before/*')):
#     for j in itertools.combinations(glob.glob(img_path+'/*'), n):
#         base = cv2.imread('back.jpg')
#         base = save_train(base, j[0])
#         base = save_train(base, j[1])
#         cv2.imwrite(osp.join('train1', img_path.split('/')[3], '/'+str(N)+'.jpg'), base)
#         N += 1
# 
