from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data.ant_test import ANT_ROOT, ANT_CLASSES as labelmap
from PIL import Image
from data.ant_test import ANTAnnotationTransform, ANTDetection, ANT_CLASSES
from data import BaseTransform
import torch.utils.data as data
from ssd import build_ssd
import cv2

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_ant_2900.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=ANT_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img_id, img = testset.pull_image(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.5:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])

                img_ = cv2.imread('ANT/Testimages/{}.jpg'.format(img_id))
                cv2.rectangle(img_, (coords[0], coords[1]),(coords[2], coords[3]), (0, 255, 0), 2)
                cv2.putText(img_, str(label_name)+':'+str(score.item()), (coords[0], coords[1]),\
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.imwrite(save_folder+'{}.jpg'.format(img_id), img_)


                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(img_id) + str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1


def test_voc():
    # load net
    num_classes = len(ANT_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = ANTDetection(args.voc_root, None, ANTAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)))

if __name__ == '__main__':
    test_voc()
