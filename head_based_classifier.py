import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import csv, cPickle
import math

CLASSES = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')

def parse_args():
    parser = argparse.ArgumentParser(description='Takes detection files to generate features.')
    parser.add_argument('--train', dest='train_pkl_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--csv', dest='csv', default=None, help='Class info csv file')
    parser.add_argument('--test', dest='test_pkl_files', nargs='+', help='Training object pickle files')
    args = parser.parse_args()
    return args

def get_likely_class(img_boxes):
    global_top_score = 0
    pred_cat = ''
    for obj, boxes in img_boxes.iteritems():
        if(len(boxes) == 1):
            if(boxes[0][4] > global_top_score):
                pred_cat = obj.split('_')[0]
        elif(len(boxes) > 1):
            scores = [b[4] for b in boxes]
            top_cat_score = max(scores)
            if(top_cat_score > global_top_score):
                pred_cat = obj.split('_')[0]
        else:
            pass
    # if no boxes found
    rand_guess = False
    if(pred_cat == ''):
        pred_cat = CLASSES[randint(0,9)]
        rand_guess = True
    return pred_cat, rand_guess

def get_class_prob(objs):
    cat_score = [0.0]*10
    # get top score of all detected boxes per category.
    found = False
    for obj, boxes in objs.iteritems():
        cat = obj.split('_')[0]
        if(len(boxes) == 1):
            cat_score[CLASSES.index(cat)] = boxes[0][4]
            found = True
        elif(len(boxes) > 1):
            score = [b[4] for b in boxes]
            cat_score[CLASSES.index(cat)] = max(score)
            found = True

    if(found):
        pred_cat_idx = cat_score.index(max(cat_score))
        residue = 1.0 - max(cat_score)

        other_prob = np.random.rand(9)
        other_prob = other_prob/other_prob.sum()
        other_prob = other_prob * residue

        other = 0
        prob = [0.0]*10
        prob[pred_cat_idx] = max(cat_score)
        for c in range(10):
            if(c != pred_cat_idx):
                prob[c] = other_prob[other]
                other += 1
    else:
        prob = [0.1]*10
    return prob
    
def show_confusion_matrix(mat):
    mat = mat.astype(np.float32)
    cls_total = np.sum(mat, axis=1)
    for r in range(mat.shape[0]):
        if(cls_total[r] != 0):
            mat[r] = mat[r]/cls_total[r]

    fig, ax = plt.subplots(1, 1)
    ax.matshow(mat, aspect='equal')
    for (r, c), z in np.ndenumerate(mat):
        ax.text(c, r, '{:0.2f}'.format(z), ha='center', va='center')
    ax.set_title('Confusion matrix')

    plt.xticks(range(len(CLASSES)), CLASSES)
    plt.yticks(range(len(CLASSES)), CLASSES)
    plt.show()


def compute_accuracy(det_file, csv_label_file):
    assert(os.path.exists(csv_label_file)), 'Class info CSV file not found'
    # read the csv file containing category info and populate
    cls_info = {}
    with open(csv_label_file, 'r') as cf:
        data = csv.reader(cf)
        for row in data:
            cls_info[row[2]] = row[1]

    # create prediction dictionary
    pred_info = {}
    with open(det_file, 'r') as df:
        pred_boxes = cPickle.load(df)['boxes']

    assert (len(pred_boxes) == len(cls_info)), 'No if predictions != no of images in the set.'

    # confusion matrix 
    report = np.zeros(shape=(len(CLASSES), len(CLASSES)), dtype=np.int32)
    total_err_cnt = 0
    rand_guess_cnt = 0
    loss = 0.0
    for img, act_cls in cls_info.iteritems():
        img_boxes = pred_boxes[img]
        pred_cls, rand = get_likely_class(img_boxes)
        pred_idx = CLASSES.index(pred_cls)
        act_idx = CLASSES.index(act_cls)

        if(pred_cls != act_cls):
            total_err_cnt += 1
        if(rand):
            rand_guess_cnt += 1

        report[act_idx, pred_idx] = report[act_idx, pred_idx] + 1

        prob = get_class_prob(img_boxes)
        p_i = prob[act_idx]
        p_i = max(min(p_i, 1-1e-15), 1e-15)
        loss += math.log(p_i)

    loss = -loss/len(cls_info)
        

    print('Overall error  = {:f}%'.format(float(total_err_cnt)/len(cls_info)))
    print('Total random guesses = {:d}'.format(rand_guess_cnt))
    print('Overall loss = {:f}'.format(loss))
    show_confusion_matrix(report)

def generate_test_predictions(pkl_files):

    obj_dict = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'r') as pf:
            obj_dict.update(cPickle.load(pf)['boxes'])

    prob_dict = {}
    for img, objs in obj_dict.iteritems():
        prob_dict[img] = get_class_prob(objs)

    with open('head_based_test_predictions.pkl', 'w') as tf:
        cPickle.dump(prob_dict, tf)

if __name__=='__main__':
    args = parse_args()
    compute_accuracy(args.train_pkl_files[0], args.csv)
    generate_test_predictions(args.test_pkl_files)


