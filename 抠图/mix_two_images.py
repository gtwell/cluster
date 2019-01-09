import cv2
import os
import numpy as np
import random
import glob
from process_new import black_process_three


def mix_two_images(base, img2):
	# roi = base[28:328, 145:545]
	roi = base[:]
	# 将img2处理后的pre2上的物体抠出来放在img1处理后的base处理后的图上
	# img2 = cv2.resize(img2, (400, 300))
	mask, pre2 = black_process_three(img2)
	# cv2.imshow('mask', mask)

	mask_inv = cv2.bitwise_not(mask)

	# mask 为掩码，与原图作用时，mask=0的地方原图变为黑色，其他地方不变
	img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
	# cv2.imshow('img1_bg', img1_bg)
	img2_bg = cv2.bitwise_and(pre2, pre2, mask=mask)

	dst = cv2.add(img1_bg, img2_bg)
	#base[28:328, 145:545] = dst
	base = dst

	return base 

path = glob.glob("/home/guo/project/camera_process/combine/Images_75/Before/*/*")

base = cv2.imread('back.jpg')
# _, base = black_process_two(img1)

# img = osp.join(path, i)
img = random.choice(path)
img = cv2.imread(img)
base = mix_two_images(base, img)


cv2.imshow('image', base)
cv2.waitKey(0)
cv2.destroyAllWindows()
