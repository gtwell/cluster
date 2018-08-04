import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import cv2
import ipdb

def xml_transform(target):
    """args:
    target: xml file
    return:
    res: [xmin, ymin, xmax, ymax, label_id]"""
    
    target = ET.parse(target).getroot()

    res = []
    for obj in target.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if difficult:
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        label = name
        bndbox.append(label)
        res += [bndbox]

    return res

def plot_rec(img, res, i):
    """args:
    img: image
    res: [xmin, ymin, xmax, ymax, label]"""

    img = cv2.imread(img)
    # (0, 255, 0) color, 2 linewidth
    for j in range(len(res)):
        cv2.rectangle(img, tuple(res[j][:2]), tuple(res[j][2:4]), (0, 255, 0), 2)
    # cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 3)
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    save_dir = './plot/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(save_dir + '{}.jpg'.format(i), img)


if __name__ == '__main__':
    # target = './蒸呢印/J01_2018.06.13 14_56_32.xml'
    # img = './J01_2018.06.27 13_39_14.jpg'
    # target = './J01_2018.06.27 13_39_14.xml'
    import os
    img = os.listdir('./')

    xml = list(filter(lambda x: x.endswith('.xml'), img))
    jpg = list(filter(lambda x: x.endswith('.jpg'), img))
    
    for i in range(len(xml)):
        target = xml[i]
        img = jpg[i]
        res = xml_transform(target)
        plot_rec(img, res, i)
        
