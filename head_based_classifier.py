import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import csv, cPickle
import math
from pickle_processor import naive_bayes_classifier, logistic_regression_classifier, dt_classifier, random_forest_classifier
from ensemble_prediction import is_equal_prob

CLASSES = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')
HEAD_OBJS = ('c0_head', 'c1_head', 'c2_head', 'c3_head','c4_head','c5_head','c6_head','c7_head','c8_head','c9_head')
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
    acc_wt = (0.894737, 0.653286, 0.98921, 0.714408, 0.966466, 0.729239, 0.815054, 0.795704, 0.676609, 0.883513)
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
        #cat_score = [max(0.005, cat_score[p]*acc_wt[p]) for p in range(10)]
        #cat_score = [cat_score[p]*acc_wt[p] for p in range(10)]
        #------------residue method----------------
        #pred_cat_idx = cat_score.index(max(cat_score))
        #residue = 1.0 - max(cat_score)
        #residue = max(0.05, residue)
        #other_prob = np.random.rand(9)
        #other_prob = other_prob/other_prob.sum()
        #other_prob = other_prob * residue

        #other = 0
        #prob = [0.0]*10
        #prob[pred_cat_idx] = max(cat_score)
        #for c in range(10):
        #    if(c != pred_cat_idx):
        #        prob[c] = other_prob[other]
        #        other += 1
        # ----------simple method------------
        prob = cat_score[:]
        for c in range(10):
            if (prob[c] == 0.0):
                prob[c] = 0.005
        
    else:
        prob = [0.1]*10

    total = sum(prob)
    norm_prob = [p/total for p in prob]
    return norm_prob, found
    
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

        prob, found = get_class_prob(img_boxes)
        p_i = prob[act_idx]
        p_i = max(min(p_i, 1-1e-15), 1e-15)
        loss += math.log(p_i)

    loss = -loss/len(cls_info)
        

    print('Overall error  = {:f}%'.format(float(total_err_cnt)/len(cls_info)))
    print('Total random guesses = {:d}'.format(rand_guess_cnt))
    print('Overall loss = {:f}'.format(loss))
    #show_confusion_matrix(report)

def get_feat_vector(objs):
    feat_vec = []
    for head in HEAD_OBJS:
        if(len(objs[head]) == 0):
            feat_vec.append(0.0)
        #else:
        #    feat_vec.append(1.0)
        elif(len(objs[head]) == 1):
            feat_vec.append(objs[head][0][4])
        else:
            # take the one with max score
            scores = [h[4] for h in objs[head]]
            feat_vec.append(max(scores))
    return feat_vec

def get_feat_matrix(feat_dict, train=False, cls_info=None, shuffle=False):
    assert(train == True and cls_info != None), 'Need class info for training'

    no_samples = len(feat_dict)
    ndim = 10
    X = np.zeros(shape=(no_samples, ndim), dtype=np.float32)
    if train:
        y = np.zeros(shape=(no_samples), dtype=np.float32)


    s = 0
    zero_objs = 0
    for img, objs in feat_dict.iteritems():
        feat_vec = get_feat_vector(objs)
        if(train):
            assert(cls_info.has_key(img)), 'Class info not found for the image'
            y[s] = float(CLASSES.index(cls_info[img]))
        X[s,:] = feat_vec
        s += 1
        # just a count to see how many images are with no objects.
        if(is_equal_prob(feat_vec, 0.0)):
            zero_objs += 1
    X = X[0:s, :]
    y = y[0:s]
    print('No of images with zero objects = {:d}'.format(zero_objs))
    if(train):
        if(shuffle):
            # Randomly reshuffle the data
            rand_idx = np.random.permutation(len(X))
            X = X[rand_idx]
            y = y[rand_idx]
        return X, y
    else:
        return X


def head_based_classifier(train_pkl_file, csv_label_file, test_pkl_files):
    assert(os.path.exists(csv_label_file)), 'Class info CSV file not found'
    # read the csv file containing category info and populate
    cls_info = {}
    with open(csv_label_file, 'r') as cf:
        data = csv.reader(cf)
        for row in data:
            cls_info[row[2]] = row[1]

    # create prediction dictionary for the training set
    with open(train_pkl_file, 'r') as df:
        train_pred = cPickle.load(df)['boxes']

    assert (len(train_pred) == len(cls_info)), 'No if predictions != no of images in the set.'

    X, y = get_feat_matrix(train_pred, train=True, cls_info=cls_info, shuffle=True)

    train_percent = 0.8
    no_train_samples = int(train_percent*len(X))
    train_x = X[0:no_train_samples]
    val_x = X[no_train_samples:]
    train_y = y[0:no_train_samples]
    val_y = y[no_train_samples:]
    #mean and variance normalization
    mean_x = np.mean(train_x, axis=0)
    std_x = np.maximum(np.std(train_x, axis=0), 1e-14)
    train_x = (train_x - mean_x)/std_x
    val_x = (val_x - mean_x)/std_x  
    print train_x[0:10]
    #dt_clf = dt_classifier(train_x, train_y, val_x, val_y)
    #rf_clf = random_forest_classifier(train_x, train_y, val_x, val_y)
    lr_clf = logistic_regression_classifier(train_x, train_y, val_x, val_y)
    #naive_bayes_classifier(train_x, train_y, val_x, val_y)
    #sys.exit()
    # testing
    test_pred = {}
    for pkl_file in test_pkl_files:
        with open(pkl_file, 'r') as pf:
            test_pred.update(cPickle.load(pf)['boxes'])

    test_prob = {}
    img_no = 0
    for img, objs in test_pred.iteritems():
        fvec = get_feat_vector(objs)
        x = np.array(fvec, dtype=np.float32)
        x = x.reshape(1, -1)
        x = (x - mean_x)/std_x
        prob = lr_clf.predict_proba(x)
        test_prob[img] = prob[0].tolist()

        img_no += 1
        if(img_no % 1000 == 0):
            print('{:d}'.format(img_no/1000))

    with open('head_classifer_predictions.pkl', 'w') as pf:
        cPickle.dump(test_prob, pf)
        print('Stored test predictions in {:s}'.format('head_classifer_predictions.pkl'))

def generate_test_predictions(pkl_files):

    obj_dict = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'r') as pf:
            obj_dict.update(cPickle.load(pf)['boxes'])

    prob_dict = {}
    rand_guess = 0
    for img, objs in obj_dict.iteritems():
        prob_dict[img], found = get_class_prob(objs)
        if(not found):
            rand_guess += 1

    print('There are no boxes found for {:d} test images'.format(rand_guess))
    with open('head_based_test_predictions.pkl', 'w') as tf:
        cPickle.dump(prob_dict, tf)

if __name__=='__main__':
    args = parse_args()
    print('----Majoriry vote intutive classification----')
    #compute_accuracy(args.train_pkl_files[0], args.csv)
    #generate_test_predictions(args.test_pkl_files)
    print('----Trained classifier based classification---')
    head_based_classifier(args.train_pkl_files[0], args.csv, args.test_pkl_files)

