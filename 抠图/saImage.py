import serial
import time
import sys
import argparse
import cv2
import os
# from process import black_process_one, black_process_two, black_process_three
from process_new import black_process_three


parser = argparse.ArgumentParser()
parser.add_argument('--loc_before', type=str, default='./Images_75_new/Before/', help='relative path of images before location')
parser.add_argument('--loc_after', type=str, default='./Images_75_new/After/', help='relative path of images after location')
parser.add_argument('--display', type=bool, default=True, help='to display')


args = parser.parse_args()
display = args.display

if not os.path.exists(args.loc_before):
    os.makedirs(args.loc_before)
if not os.path.exists(args.loc_after):
    os.makedirs(args.loc_after)

print("enter c to close program OR enter barcode to continue...");
barcode = input("")
if barcode == 'c':
    sys.exit();
time.sleep(2)
print("starting!!!!")

if barcode:
    dir1 = os.path.join(args.loc_before, barcode)
    dir2 = os.path.join(args.loc_after, barcode)
if not os.path.exists(dir1):
    os.makedirs(dir1)
if not os.path.exists(dir2):
    os.makedirs(dir2)

cap = cv2.VideoCapture(1)

# 视频帧计数间隔频率
timeF = 6 

while True:
    count = 1
    while cap.isOpened() and count <= 601:
        ret, img = cap.read()
        if count % timeF == 0:
            now = time.time()
            cv2.imwrite(dir1 + '/' + str(now) + '.jpg', img)
            # image = cv2.imread(dir1 + '/' + str(now) + '.jpg')
            # im = cv2.imread(dir1 + '/' + str(now) + '.jpg')
            # open_image_1st(img)
            _, img = black_process_three(img)
            cv2.imwrite(dir2 + '/' + str(now) + '.jpg', img)
        count += 1
        # cv2.imshow('image', img)
        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()

print("ending!!!!")
