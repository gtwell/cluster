from annotDiago import AnnotDiago
import argparse
import os
import cv2

def generate_images(path_anno, path_gt, path_output, iou_thresh):

    txt_file = os.path.join(path_output, 'txt_file/', path_anno.split('/')[-1])
    if not os.path.exists(txt_file):
        os.makedirs(txt_file)
        print(txt_file+"文件夹已创建")

    jpg_file = os.path.join(path_output, 'jpg_file', path_anno.split('/')[-1])
    if not os.path.exists(jpg_file):
        os.makedirs(jpg_file)
        print(jpg_file+"文件夹已创建")

    # 生成txt文件
    for i in os.listdir(path_anno):
        if i.endswith('.xml'):
            xml_anno = os.path.join(path_anno, i)
            xml_gt = os.path.join(path_gt, i)
            anno_check = AnnotDiago(path_gt=xml_gt, path_anno=xml_anno, iou_thresh=iou_thresh)
            anno_check.check(path=os.path.join(txt_file, i.split('.')[0]+'.txt'))

    # 生成anno与gt不同差异的图像文件
    for i in os.listdir(txt_file):
        img = cv2.imread(os.path.join(path_anno, i.split('.')[0]+'.jpg'))
        path = os.path.join(txt_file, i)
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line:
                if line == 'iou_lower:':
                    line = f.readline().strip()
                    line = f.readline().strip().split()
                    # anno画绿色框
                    cv2.rectangle(img=img, pt1=(int(line[0]), int(line[1])), pt2=(int(line[2]), int(line[3])), color=(0, 255, 0), thickness=2)
                    line = f.readline()
                    line = f.readline().strip().split()
                    # ground truth画蓝色框
                    cv2.rectangle(img=img, pt1=(int(line[0]), int(line[1])), pt2=(int(line[2]), int(line[3])), color=(255, 0, 0), thickness=2)
                    line_iou = f.readline().strip()
                    cv2.putText(img, 'iou:{:4s}'.format(line_iou[:5]),
                                ((int(line[0])+int(line[2]))//2, (int(line[1])+int(line[3]))//2),
                                fontFace=0, fontScale=0.7, color=(0, 0, 255), thickness=2)
                    line = f.readline().strip()

                if line == 'label_not_matched:':
                    line = f.readline().strip()
                    line = f.readline().strip().split()
                    # anno画绿色框
                    cv2.rectangle(img=img, pt1=(int(line[0]), int(line[1])), pt2=(int(line[2]), int(line[3])), color=(0, 255, 0), thickness=2)
                    line = f.readline().strip()
                    line = f.readline().strip().split()
                    # ground truth画蓝色框
                    cv2.rectangle(img=img, pt1=(int(line[0]), int(line[1])), pt2=(int(line[2]), int(line[3])), color=(255, 0, 0), thickness=2)
                    cv2.putText(img, 'label_not_matched:{}'.format(line[4]), (int(line[0]), int(line[1])), \
                                fontFace=0, fontScale=0.7, color=(182, 78, 228), thickness=2)
                    line = f.readline().strip()

                if line == 'anno_no_labeled:':
                    line = f.readline().strip().split()
                    cv2.rectangle(img=img, pt1=(int(line[0]), int(line[1])), pt2=(int(line[2]), int(line[3])),
                                  color=(255, 0, 0), thickness=2)
                    cv2.putText(img, 'anno_not_exist:{}'.format(line[4]), (int(line[0]), int(line[3])), \
                                fontFace=0, fontScale=0.7, color=(0, 0, 255), thickness=2)
                    line = f.readline().strip()

                if line == 'gt_no_labeled:':
                    line = f.readline().strip().split()
                    cv2.rectangle(img=img, pt1=(int(line[0]), int(line[1])), pt2=(int(line[2]), int(line[3])),
                                  color=(0, 255, 0), thickness=2)
                    cv2.putText(img, 'gt_not_exist:{}'.format(line[4]), (int(line[0]), int(line[3])), \
                                fontFace=0, fontScale=0.5, color=(0, 0, 255), thickness=2)
                    line = f.readline().strip()

            cv2.imwrite(os.path.join(jpg_file, i.split('.')[0]+'.jpg'), img, [int( cv2.IMWRITE_JPEG_QUALITY), 100])
    # cv2.imshow('image', img)
    # cv2.waitKey(0) & 0xFF == ord('q')
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    path_anno = './1018-小关/错误案例/静态-5张'
    path_gt = './1018-小关/正确案例/静态-5张'
    path_output = './output'
    iou_thresh = 0.8
    parser = argparse.ArgumentParser("两个xml文件差异")
    parser.add_argument('--anno', type=str, default=path_anno, help="anno dir")
    parser.add_argument('--gt', type=str, default=path_gt, help="ground truth dir")
    parser.add_argument('--output', type=str, default=path_output, help="output dir")
    parser.add_argument('--thresh', type=float, default=iou_thresh, help="output dir")
    args = parser.parse_args()
    generate_images(path_anno=args.anno, path_gt=args.gt, path_output=args.output, iou_thresh=args.thresh)
