import cv2
import os
import os.path as osp
import numpy as np
import random
from process import black_process_two


def mix_two_images(base, img2):
	roi = base[28:328, 145:545]
	# 将img2处理后的pre2上的物体抠出来放在img1处理后的base处理后的图上
	img2 = cv2.resize(img2, (400, 300))
	mask, pre2 = black_process_two(img2)
	# cv2.imshow('mask', mask)

	mask_inv = cv2.bitwise_not(mask)

	# mask 为掩码，与原图作用时，mask=0的地方原图变为黑色，其他地方不变
	img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
	# cv2.imshow('img1_bg', img1_bg)
	img2_bg = cv2.bitwise_and(pre2, pre2, mask=mask)

	dst = cv2.add(img1_bg, img2_bg)
	base[28:328, 145:545] = dst

	return base 

# 物体总数
N = random.randint(3, 6) 
n = random.randint(2, 5)
path = '/home/guo/project/camera_process/combine/Images3/Before/'

# 从路径中随机选择N个文件夹（即N个物体）每个文件夹中随机选择一张图片
all_file = random.sample(os.listdir(path), N)
# image_list = [osp.join(i, random.choice(os.listdir(osp.join(path, i)))) for i in all_file]
image_list = [osp.join(i, j) for i in all_file for j in random.sample(os.listdir(osp.join(path, i)), n) ]

# img1 = cv2.imread(osp.join(path, image_list[0]))
base = cv2.imread('back.jpg')
# _, base = black_process_two(img1)

for i in random.sample(image_list, random.randint(1, 6)):
	img = osp.join(path, i)
	img = cv2.imread(img)
	base = mix_two_images(base, img)


cv2.imshow('image', base)
cv2.waitKey(0)
cv2.destroyAllWindows()
