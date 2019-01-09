import time
import os
import cv2
# from process import black_process_one, black_process_three
from process_new import black_process_three
import time

gap = 28
def test():
    cap = cv2.VideoCapture(1)
    print("good_name:....")
    good_name = input()
    path_origin = "three_light/origin/{}/".format(good_name)
    path_after = "three_light/after/{}/".format(good_name)
    if not os.path.exists(path_origin):
        os.makedirs(path_origin)
    if not os.path.exists(path_after):
        os.makedirs(path_after)

    counts = 0
    while True:
        ret, img = cap.read()
        cv2.circle(img, (333, 240), 235, (0, 0, 255), 2)
        cv2.imshow('ori', img)
        img_name = time.time()
        #if counts % gap == 0:
        #     cv2.imwrite(path_origin+'{}.jpg'.format(img_name), img)

        _, img_process = black_process_three(img)
        #if counts % gap == 0:
        #    cv2.imwrite(path_after+'{}.jpg'.format(img_name), img_process)

        cv2.imshow('image', img_process)
        counts += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
