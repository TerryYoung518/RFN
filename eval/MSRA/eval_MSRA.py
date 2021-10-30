import Polygon as plg
import numpy as np
import math
import cv2
import os
from MSRA import file_util
# import file_util
import glob

def get_pred(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        bbox = line.split(',')
        if len(bbox) % 2 == 1:
            print(path)
        bbox = [int(x) for x in bbox]
        bboxes.append(bbox)
    return bboxes


def get_gt(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    tags = []
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ' ')
        gt = line.split(' ')

        w_ = np.float(gt[4])
        h_ = np.float(gt[5])
        x1 = np.float(gt[2]) + w_ / 2.0
        y1 = np.float(gt[3]) + h_ / 2.0
        theta = np.float(gt[6]) / math.pi * 180

        bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
        bbox = bbox.reshape(-1)

        bboxes.append(bbox)
        tags.append(np.int(gt[1]))
    return np.array(bboxes), tags


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def get_msra_result(gt_root,pred_root):
    th = 0.5
    pred_list = file_util.read_dir(pred_root)

    count, tp, fp, tn, ta = 0, 0, 0, 0, 0
    for pred_path in pred_list:
        count = count + 1
        preds = get_pred(pred_path)
        gt_path = gt_root + 'IMG_'+pred_path.split('/')[-1].split('.')[0].split('_')[-1] + '.gt'
        gts, tags = get_gt(gt_path)

        ta = ta + len(preds)
        for gt, tag in zip(gts, tags):
            gt = np.array(gt)
            assert gt.shape[0]==8
            gt = gt.reshape(gt.shape[0] // 2, 2)
            gt_p = plg.Polygon(gt)
            difficult = tag
            flag = 0
            for pred in preds:
                pred = np.array(pred)
                assert pred.shape[0] == 8
                pred = pred.reshape(pred.shape[0] // 2, 2)
                pred_p = plg.Polygon(pred)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)
                iou = float(inter) / union
                if iou >= th:
                    flag = 1
                    tp = tp + 1
                    break

            if flag == 0:
                fp = fp + 1

    recall = float(tp) / (tp + fp)
    precision = float(tp) / ta
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))
def get_sv1k_result(gt_root,pred_root):
    th = 0.5
    print(pred_root)
    pred_list = glob.glob(pred_root+'/*.txt')
    # pred_list = glob.glob('../../'+pred_root + '/*.txt')
    # print(pred_list)
    count, tp, fp, tn, ta = 0, 0, 0, 0, 0
    for pred_path in pred_list:
        count = count + 1
        preds = get_pred(pred_path)
        gt_path = gt_root + pred_path.split('/')[-1].split('.')[0].split('_')[-1] + '.gt'
        gts, tags = get_gt(gt_path)

        ta = ta + len(preds)
        for gt, tag in zip(gts, tags):
            gt = np.array(gt)
            assert gt.shape[0]==8
            gt = gt.reshape(gt.shape[0] // 2, 2)
            gt_p = plg.Polygon(gt)
            difficult = tag
            flag = 0
            for pred in preds:
                pred = np.array(pred)
                assert pred.shape[0] == 8
                pred = pred.reshape(pred.shape[0] // 2, 2)
                pred_p = plg.Polygon(pred)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)
                iou = float(inter) / union
                if iou >= th:
                    flag = 1
                    tp = tp + 1
                    break

            if flag == 0:
                fp = fp + 1
    recall = float(tp) / (tp + fp)
    precision = float(tp) / ta
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    methodMetrics = {'precision':precision, 'recall':recall,'hmean': hmean}

    # resDict = {'calculated':True,'Message':'','method': methodMetrics}
    # print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))
    return methodMetrics,hmean
# get_msra_result("../../../../../MSRA-TD500/test/","./eval_dir/MASK_result0")
# get_sv1k_result("../../data/USTB-SV1K/SV1K_GT/","../../eval_dir/MASK_result0")
"""
if __name__ == '__main__':
    th = 0.5
    pred_list = file_util.read_dir(pred_root)

    count, tp, fp, tn, ta = 0, 0, 0, 0, 0
    for pred_path in pred_list:
        count = count + 1
        preds = get_pred(pred_path)
        gt_path = gt_root + pred_path.split('/')[-1].split('.')[0] + '.gt'
        gts, tags = get_gt(gt_path)

        ta = ta + len(preds)
        for gt, tag in zip(gts, tags):
            gt = np.array(gt)
            gt = gt.reshape(gt.shape[0] / 2, 2)
            gt_p = plg.Polygon(gt)
            difficult = tag
            flag = 0
            for pred in preds:
                pred = np.array(pred)
                pred = pred.reshape(pred.shape[0] / 2, 2)
                pred_p = plg.Polygon(pred)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)
                iou = float(inter) / union
                if iou >= th:
                    flag = 1
                    tp = tp + 1
                    break

            if flag == 0 and difficult == 0:
                fp = fp + 1

    recall = float(tp) / (tp + fp)
    precision = float(tp) / ta
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))
"""
project_root = '../../'

pred_root = project_root + 'outputs/submit_msra/'
gt_root = project_root + 'data/MSRA-TD500/test/'