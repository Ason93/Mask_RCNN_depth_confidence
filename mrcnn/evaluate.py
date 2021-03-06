#!usr/bin/env python
# -*- coding:utf-8 -*-

'''Evaluation'''

# calcualate f1_measure
def compute_f1_measure(local, ground_truth):
    overlap_area = 0
    mask_area = 0
    FP = 0
    FN = 0
    for i in range(800):
        for j in range(1280):
            if ground_truth[i][j]:
                mask_area += 1
            for k in range(local.shape[2]):
                if local[i][j][k] == ground_truth[i][j] and ground_truth[i][j]:
                    overlap_area += 1
                if local[i][j][k] and ground_truth[i][j] != local[i][j][k]:
                    FP += 1
                if local[i][j][k] != ground_truth[i][j] and ground_truth[i][j]:
                    FN += 1
    print("overlap_area", overlap_area)
    print("mask_area:", mask_area)
    TP = overlap_area
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    f1_measure = 2 * P * R / (P + R)
    return f1_measure


# calculate mAP
def compute_mAP(local, ground_truth):
    overlap_area = 0
    mask_area = 0
    FP = 0
    FN = 0
    for i in range(800):
        for j in range(1280):
            if ground_truth[i][j]:
                mask_area += 1
            for k in range(local.shape[2]):
                if local[i][j][k] == ground_truth[i][j] and ground_truth[i][j]:
                    overlap_area += 1
                if local[i][j][k] and ground_truth[i][j] != local[i][j][k]:
                    FP += 1
                if local[i][j][k] != ground_truth[i][j] and ground_truth[i][j]:
                    FN += 1
    print("overlap_area", overlap_area)
    print("mask_area:", mask_area)
    TP = overlap_area
    P = TP / (TP + FP)
    return P