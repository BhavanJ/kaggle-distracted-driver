import argparse
import sys, os
import numpy as np
import csv
import cPickle
import math
from feat_gen import compute_features
from kaggle_utils import plot_catwise_centroids
from copy import deepcopy

CLASSES = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')
HEAD_OBJS = ('c0_head', 'c1_head', 'c2_head', 'c3_head','c4_head','c5_head','c6_head','c7_head','c8_head','c9_head')
ISOLATED_OBJS = ('head', 'wrist', 'steering', 'radio', 'phone','cup',
    'left_hand_steering', 'right_hand_steering', 'both_hands_steering', 'right_hand_phone', 'left_hand_phone', 'drinking_near_steering')

def parse_args():
    parser = argparse.ArgumentParser(description='Takes detection files to generate features.')
    parser.add_argument('--train-obj', dest='train_obj_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--train-head', dest='train_head_file', help='Training object pickle files')
    parser.add_argument('--train-fullimg', dest='train_fullimg_file', help='Training object pickle files')
    parser.add_argument('--csv', dest='csv', default=None, help='Class info csv file')
    parser.add_argument('--test-obj', dest='test_obj_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--test-head', dest='test_head_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--test-fullimg', dest='test_fullimg_files', nargs='+', help='Training object pickle files')
    args = parser.parse_args()
    return args

def get_cls_dict(cls_csv_file):
    # create class info dictionary.
    cls_info = {}
    with open(cls_csv_file, 'r') as cf:
        data = csv.reader(cf)
        for row in data:
            cls_info[row[2]] = row[1]

    return cls_info



def create_objs_set(train_obj_files, test_obj_files, cls_info):


    # compute training objset
    train_obj_dict_list = []
    for pkl_file in train_obj_files:
        with open(pkl_file, 'r') as pf:
            train_obj_dict_list.append(cPickle.load(pf)['boxes'])

    train_objset, obj_mean_model = compute_features(train_obj_dict_list, cls_info, get_objs=True)

    # compute test feature dict
    test_obj_dict_list = []
    dummy_cls_info = {}
    test_obj_dict = {}
    for pkl_file in test_obj_files:
        with open(pkl_file, 'r') as pf:
            test_obj_dict.update(cPickle.load(pf)['boxes'])

    test_obj_dict_list.append(test_obj_dict)
    # create dummy class info requried by compute features method
    for d in test_obj_dict_list:
        for img in d:
            dummy_cls_info[img] = 'u' # unknown

    # compute features for test images
    test_obj_feat = {}
    test_objset = compute_features(test_obj_dict_list, dummy_cls_info, train=False, get_objs=True,
        head_mean_c=obj_mean_model[0],
        steering_mean_c=obj_mean_model[1],
        head_mean_box=obj_mean_model[2],
        steering_mean_box=obj_mean_model[3])

    return train_objset, test_objset

def filter_multiples(objs, labels):
    filt_objs = {}
    found = False
    max_score = 0.0
    cls = ''
    for label in labels:
        boxes = objs[label]
        if(len(boxes) == 1):
            filt_objs[label] = boxes[0]
            found = True
            if(filt_objs[label][4] > max_score):
                max_score = filt_objs[label]
                cls = label
        elif(len(boxes) > 1):
            scores = [b[4] for b in boxes]
            filt_objs[label] = boxes[scores.index(max(scores))]
            found = True
            if(filt_objs[label][4] > max_score):
                max_score = filt_objs[label]
                cls = label
        else:
            filt_objs[label] = []
    return filt_objs, found, max_score, cls

def intutive_prediction(iso_objs, heads, fullimg_boxes):
    final_prob = [0.0001]*10
    certain = False
    #def __decide_based_on_isolated_objs(iso_objs)
    # isolated objs are already filtered. just remove multiple head detections
    head_objs, found, max_score, cls = filter_multiples(heads, HEAD_OBJS)
    print head_objs
    print fullimg_boxes
    sys.exit()
    # filter  full image based objects
    # TODO:
    # found = found and full_img_found
    # check if atleast one head detections among 10 classes is found
    if(found):
        # check if atleast one object has more than 0.99 score
        if(max_score > 0.99):
            # check if more than 1 objects have score > 0.99
            if(1==1):
                # more than 1 heads have score > 0.99
                # see if this conflict can be resolved using isolated objs
                pass
            else:
                if(cls in ('c2_head', 'c7_head')):
                    # isolated objeccts are not getting detected for test images correctly. Take this path if they are not detected.
                    # final detection is the one with max score.
                    certain = True
                    final_prob[HEAD_OBJS.index(cls)] = max_score
                else:
                    # not certain. either take help of isolated objs or set certain = False
                    pass
        else:
            # need to take help from isolated object detections to decide the final prob
            pass
        
    else:
        # decision is wholey based on the isolated objects.
        pass

    
def cls_main(args):
    train_cls_info = get_cls_dict(args.csv)
    # create filtered objset for training and testing images. This set includes ISOLATED_OBJS
    print('Filtering isolated objects...')
    train_objset, test_objset = create_objs_set(args.train_obj_files, args.test_obj_files, train_cls_info)

    print len(train_objset)
    print len(test_objset)
    #plot_catwise_centroids(train_objset, 'phone')
    #plot_catwise_centroids(train_objset, 'cup')

    # create head objset dict
    with open(args.train_head_file, 'r') as df:
        train_heads = cPickle.load(df)['boxes']

    test_heads = {}
    for pkl_file in args.test_head_files:
        with open(pkl_file, 'r') as pf:
            test_heads.update(cPickle.load(pf)['boxes'])

    # create full image objset. take box with max score to create dict with small size
    train_fullimg = {}
    with open(args.train_fullimg_file, 'r') as f:
        print('Loading {:s}'.format(args.train_fullimg_file))
        temp_train_fullimg = cPickle.load(f)['boxes']
        for img, boxes in temp_train_fullimg.iteritems():
            full_img_boxes = {}
            for cls in CLASSES:
                if(len(boxes[cls]) == 1):
                    full_img_boxes[cls] = boxes[cls][0]
                elif(len(boxes[cls]) > 1):
                    scores = [b[4] for b in boxes[cls]]
                    full_img_boxes[cls] = boxes[cls][scores.index(max(scores))]
                else:
                    full_img_boxes[cls] = []
            train_fullimg[img] = deepcopy(full_img_boxes)

        del temp_train_fullimg
    test_fullimg = {}
    for pkl_file in args.test_fullimg_files:
        print('Loading {:s}'.format(pkl_file))
        with open(pkl_file, 'r') as pf:
            temp_test_fullimg = cPickle.load(pf)['boxes']
            for img, boxes in temp_test_fullimg.iteritems():
                full_img_boxes = {}
                for cls in CLASSES:
                    if(len(boxes[cls]) == 1):
                        full_img_boxes[cls] = boxes[cls][0]
                    elif(len(boxes[cls]) > 1):
                        scores = [b[4] for b in boxes[cls]]
                        full_img_boxes[cls] = boxes[cls][scores.index(max(scores))]
                    else:
                        full_img_boxes[cls] = []
                test_fullimg[img] = deepcopy(full_img_boxes)
    del temp_test_fullimg

    # make predictions on the training set
    report = np.zeros(shape=(len(CLASSES), len(CLASSES)), dtype=np.int32)
    loss = 0.0
    err_cnt = 0
    for img, act_cls in train_cls_info.iteritems():
        prob, pred_cls = intutive_prediction(iso_objs=train_objset[img], heads=train_heads[img], fullimg_boxes=train_fullimg[img])
        pred_idx = CLASSES.index(pred_cls)
        act_idx = CLASSES.index(act_cls)
        if(pred_cls != act_cls):
            err_cnt += 1

        report[act_idx, pred_idx] = report[act_idx, pred_idx] + 1

        p_i = prob[act_idx]
        p_i = max(min(p_i, 1-1e-15), 1e-15)
        loss += math.log(p_i)

    sys.exit()
    # make predictions on the testing set
    test_prob = {}
    img_no = 0
    for img, objs in test_objset.iteritems():
        prob, cls = intutive_prediction(iso_objs=objs, heads=test_heads[img])
        test_prob[img] = prob

        img_no += 1
        if(img_no % 1000 == 0):
            print('{:d}'.format(img_no/1000))

    print('\nDone predicting prob of test images')
    with open('intutive_test_predictions.pkl', 'w') as pf:
        cPickle.dump(test_predictions, pf)
        print('Stored test predictions in {:s}'.format('intutive_test_predictions.pkl'))


if __name__=='__main__':
    args = parse_args()
    cls_main(args)
