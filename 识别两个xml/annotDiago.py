from sklearn.utils.linear_assignment_ import linear_assignment
import xml.etree.ElementTree as ET
import numpy as np
import os
import ipdb


class AnnotDiago(object):
    
    def __init__(self, path_gt, path_anno, iou_thresh):
        
        self.data_gt = self.loadData(path_gt)
        self.data_anno = self.loadData(path_anno)
        self.iou_thresh = iou_thresh
        
    def loadData(self, path):
        # 加载xml数据
        target = ET.parse(path).getroot()
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)
            bndbox.append(name)
            # res = [xmin, ymin, xmax, ymax, label_name]
            res += [bndbox]
        return res
    
    def match(self):
        # 使用linear_assignment匹配data_gt框和data_anno框，找出1）存在但没标，2）匹配成功，3）标注但不存在
        iou_metric = []
        for box in self.data_anno:
            temp_iou = []
            for box2 in self.data_gt:
                temp_iou.append(self.iou(box[:-1], box2[:-1]))
            iou_metric.append(temp_iou)
        iou_metric = np.array(iou_metric)
        result = linear_assignment(-iou_metric)

        matched = []
        iou_lower = []
        label_not_matched = []
        # gt有anno无的标注框
        anno_no_labeled = []
        # anno有gt无的标注框
        gt_no_labeled = []
        for idx in range(len(result)):
            # 从匹配完成后的array检查iou是否大于iou阈值
            if self.iouCheck(iou_metric[result[idx][0], result[idx][1]], self.iou_thresh):
                # 满足iou阈值，检查标注label是否等于实际gt
                if self.labelMatch(self.data_anno[result[idx][0]][-1], self.data_gt[result[idx][1]][-1]):
                    # print(self.data_anno[result[idx][0]][-1], self.data_gt[result[idx][1]][-1])
                    # matched存储完全满足匹配的数组
                    matched.append(result[idx].tolist())
                else:
                    # label_not_matched存储标签不等于实际gt数组
                    label_not_matched.append(result[idx].tolist())
            else:
                # iou_lower存储小于阈值的匹配数组和iou
                iou_lower.append(result[idx].tolist()+[iou_metric[result[idx][0], result[idx][1]].item()])

        for i in range(len(self.data_gt)):
            if i not in result[:, 1]:
                # anno_no_labeled存储 gt有但anno无的标注框
                anno_no_labeled.append(i)

        for i in range(len(self.data_anno)):
            if i not in result[:, 0]:
                # gt_no_labeled存储 anno有但gt无的标注框
                gt_no_labeled.append(i)

        connected = (matched, iou_lower, label_not_matched, anno_no_labeled, gt_no_labeled)

        return connected
    
    def check(self, path):
        # 把出现错误的图片和错误的框再写入一个txt
        _, iou_lower, label_not_matched, anno_no_labeled, gt_no_labeled = self.match()
        with open(path, 'w') as f:
            if iou_lower:
                print(iou_lower)
                for i in iou_lower:
                    f.write('iou_lower:\n')
                    f.write('data_anno:\n')
                    f.write(' '.join(map(str, self.data_anno[i[0]]))+'\n')
                    f.write('data_gt:\n')
                    f.write(' '.join(map(str, self.data_gt[i[1]]))+'\n')
                    f.write(str(i[2])+'\n')
                print("iou错误写入成功")
            if label_not_matched:
                for i in label_not_matched:
                    f.write('label_not_matched:\n')
                    f.write('data_anno:\n')
                    f.write(' '.join(map(str, self.data_anno[i[0]]))+'\n')
                    f.write('data_gt:\n')
                    f.write(' '.join(map(str, self.data_gt[i[1]])) + '\n')
                print("标签错误写入成功")
            if anno_no_labeled:
                for i in anno_no_labeled:
                    f.write('anno_no_labeled:\n')
                    f.write(' '.join(map(str, self.data_gt[i])))
                    f.write('\n')
                print("anno_no_labeled写入成功")
            if gt_no_labeled:
                for i in gt_no_labeled:
                    f.write('gt_no_labeled:\n')
                    f.write(' '.join(map(str, self.data_anno[i])))
                    f.write('\n')
                print("gt_no_labeled写入成功")

        if os.path.getsize(path) == 0:
            os.remove(path)
            print(path+'无错误图像deleted')

    
    def statistics(self):
        # 统计各个label，正确的和错误的，标注框的数量
        matched, iou_lower, label_not_matched, anno_no_labeled, gt_no_labeled = self.match()
        statistics = dict()
        statistics["匹配成功"] = len(matched)
        statistics["iou低于阈值"] = len(iou_lower)
        statistics["标注不匹配"] = len(label_not_matched)
        statistics["gt有anno无"] = len(anno_no_labeled)
        statistics["anno有gt无"] = len(gt_no_labeled)

        return statistics

    @classmethod
    def iou(cls, boxA, boxB):
        # 计算两组bbox的IOU
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truthles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    @classmethod
    def iouCheck(cls, iou, thresh):
        # 判断iou是否高于thresh, 否则算错
        # 返回一个bool向量，表示哪些bbox正确标注了
        if iou >= thresh:
            return True
        else:
            return False
        
    @classmethod
    def labelMatch(cls, label_anno, label_gt):
        #  判断标注的类别是否和label_gt一致
        # 返回一个bool向量, 表示哪些label正确标注了
        if label_gt == label_anno:
            return True
        else:
            return False


if __name__ == "__main__":
    # xml_anno = './1018-小关/错误案例/静态-5张/1537351805951.xml'
    # xml_gt = './1018-小关/正确案例/静态-5张/1537351805951.xml'
    # anno_check = AnnotDiago(path_gt=xml_gt, path_anno=xml_anno, iou_thresh=0.5)
    # print("标注框个数:", len(anno_check.data_anno))
    # print("实际框个数:", len(anno_check.data_gt))
    # iou_lower, label_not_matche, anno_no_labeled, gt_no_labeled = anno_check.check()
    # print("iou小于阈值:{0}, 标注错误:{1}, gt有anno无:{2}, anno有gt无:{3}".format(iou_lower,
    #                                                 label_not_matche,
    #                                                 anno_no_labeled,
    #                                                 gt_no_labeled))

    path_anno = './1018-小关/错误案例/动态-10张/'
    path_gt = './1018-小关/正确案例/动态-10张/'
    for i in os.listdir(path_anno):
        if i.endswith('.xml'):
            xml_anno = os.path.join(path_anno, i)
            xml_gt = os.path.join(path_gt, i)
            anno_check = AnnotDiago(path_gt=xml_gt, path_anno=xml_anno, iou_thresh=0.8)
            print("错误案例标注框总个数:", len(anno_check.data_anno))
            print("正确案例实际框总个数:", len(anno_check.data_gt))
            _, iou_lower, label_not_matche, anno_no_labeled, gt_no_labeled = anno_check.match()
            print("iou小于阈值:{0}, 标注错误:{1}, gt有anno无:{2}, anno有gt无:{3}".format(iou_lower,
                                                                                label_not_matche,
                                                                                anno_no_labeled,
                                                                                gt_no_labeled))
            print(anno_check.statistics())



