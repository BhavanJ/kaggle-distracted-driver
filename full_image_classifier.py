import sys, os
import argparse
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import math, csv

CLASSES = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')

def parse_args():
    parser = argparse.ArgumentParser(description='Full image based classifier')
    parser.add_argument('--train', dest='train_pkl_files', help='Training object pickle files')
    parser.add_argument('--csv', dest='csv', default=None, help='Class info csv file')
    parser.add_argument('--train-list', dest='train_list', default=None, help='Class info csv file')
    parser.add_argument('--test', dest='test_pkl_files', nargs='+', help='Training object pickle files')
    args = parser.parse_args()
    return args

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

def get_prob_and_cls(all_boxes):
    # run thru boxes of all 10 classes. Take one box with max score for each class
    prob = [0.1]*10
    cls = CLASSES[0]

    for idx, cls in enumerate(CLASSES):
        if(len(all_boxes[cls]) != 0):
            scores = [b[4] for b in all_boxes[cls]]
            prob[idx] = max(scores)
    cls = CLASSES[np.argmax(prob)]

    return prob, cls

def full_image_classifier(train_file, csv_label_file, train_list_file, test_files):
    assert(os.path.exists(csv_label_file)), 'Class info CSV file not found'
    # read the csv file containing category info and populate
    cls_info = {}
    with open(csv_label_file, 'r') as cf:
        data = csv.reader(cf)
        for row in data:
            cls_info[row[2]] = row[1]

    # read all detections on trainset
    with open(train_file, 'r') as tf:
        train_det = cPickle.load(tf)['boxes']

    # create the list of images in the training set. We can opt them out to find the accuracy.
    train_img_list = []
    with open(train_list_file, 'r') as f:
        for line in f:
            img_name = line.split(',')[0] + '.jpg'
            train_img_list.append(img_name)

    print len(train_img_list)
    # confusion matrix 
    train_report = np.zeros(shape=(len(CLASSES), len(CLASSES)), dtype=np.int32)
    train_err_cnt = 0
    train_loss = 0.0
    val_report = np.zeros(shape=(len(CLASSES), len(CLASSES)), dtype=np.int32)
    val_err_cnt = 0
    val_loss = 0.0
    for img, act_cls in cls_info.iteritems():
        img_boxes = train_det[img]
        prob, pred_cls = get_prob_and_cls(img_boxes)
        pred_idx = CLASSES.index(pred_cls)
        act_idx = CLASSES.index(act_cls)
        if(img in train_img_list):
            if(pred_cls != act_cls):
                train_err_cnt += 1

            train_report[act_idx, pred_idx] = train_report[act_idx, pred_idx] + 1

            p_i = prob[act_idx]
            p_i = max(min(p_i, 1-1e-15), 1e-15)
            train_loss += math.log(p_i)
        else:
            if(pred_cls != act_cls):
                val_err_cnt += 1

            val_report[act_idx, pred_idx] = val_report[act_idx, pred_idx] + 1

            p_i = prob[act_idx]
            p_i = max(min(p_i, 1-1e-15), 1e-15)
            val_loss += math.log(p_i)

    train_loss = -train_loss/len(train_img_list)
    val_loss = -val_loss/(len(cls_info)-len(train_img_list))

    print('Train error count = {:d}  Val error count = {:d}'.format(train_err_cnt, val_err_cnt))
    print('Overall train error  = {:f}%'.format(float(train_err_cnt)/len(train_img_list)))
    print('Overall train loss = {:f}'.format(train_loss))
    print('Overall val error  = {:f}%'.format(float(val_err_cnt)/(len(cls_info)-len(train_img_list))))
    print('Overall val loss = {:f}'.format(val_loss))
    show_confusion_matrix(train_report)
    show_confusion_matrix(val_report)
 
    # testing
    test_pred = {}
    for pkl_file in test_files:
        with open(pkl_file, 'r') as pf:
            test_pred.update(cPickle.load(pf)['boxes'])

    test_prob = {}
    img_no = 0
    for img, objs in test_pred.iteritems():
        prob, cls = get_prob_and_cls(objs)
        test_prob[img] = prob

        img_no += 1
        if(img_no % 1000 == 0):
            print('{:d}'.format(img_no/1000))

    with open('full_image_test_predictions.pkl', 'w') as pf:
        cPickle.dump(test_prob, pf)
        print('Stored test predictions in {:s}'.format('full_image_test_predictions.pkl'))

if __name__=='__main__':
    args = parse_args()
    full_image_classifier(args.train_pkl_files, args.csv, args.train_list, args.test_pkl_files)

