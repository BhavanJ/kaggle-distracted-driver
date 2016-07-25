import argparse
import sys, os
import numpy as np
import csv
import cPickle
import math
from feat_gen import compute_features
from kaggle_utils import plot_catwise_centroids
from copy import deepcopy
from kaggle_utils import show_confusion_matrix

CLASSES = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')
HEAD_OBJS = ('c0_head', 'c1_head', 'c2_head', 'c3_head','c4_head','c5_head','c6_head','c7_head','c8_head','c9_head')
ISOLATED_OBJS = ('head', 'wrist', 'steering', 'radio', 'phone','cup',
    'left_hand_steering', 'right_hand_steering', 'both_hands_steering', 'right_hand_phone', 'left_hand_phone', 'drinking_near_steering')

# from the histogram plot over the training data
WRIST_RADIO_LT = 150
HEAD_WRIST_LT = 200

def parse_args():
    parser = argparse.ArgumentParser(description='Takes detection files to generate features.')
    parser.add_argument('--train-obj', dest='train_obj_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--train-head', dest='train_head_file', help='Training object pickle files')
    parser.add_argument('--train-fullimg', dest='train_fullimg_file', help='Training object pickle files')
    parser.add_argument('--csv', dest='csv', default=None, help='Class info csv file')
    parser.add_argument('--test-obj', dest='test_obj_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--test-head', dest='test_head_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--test-fullimg', dest='test_fullimg_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--obj-pred', dest='obj_pred', help='Objset 1.1 and 5 based prediction file')
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
                max_score = filt_objs[label][4]
                cls = label
        elif(len(boxes) > 1):
            scores = [b[4] for b in boxes]
            filt_objs[label] = boxes[scores.index(max(scores))]
            found = True
            if(filt_objs[label][4] > max_score):
                max_score = filt_objs[label][4]
                cls = label
        else:
            filt_objs[label] = []
    return filt_objs, found, max_score, cls

def __is_present(objs, obj_cls):
    if(objs[obj_cls]):
        return objs[obj_cls][4]
    else:
        return 0.0

def _reverse_prediction(iso_objs, likely):

    conf_score = 1e-14
    if(likely == 'c0'):
        # both_hands_steering should be present
        conf_score += __is_present(iso_objs, 'both_hands_steering')
    elif(likely == 'c1'):
        conf_score += __is_present(iso_objs, 'phone')
        conf_score += __is_present(iso_objs, 'left_hand_steering')
        conf_score += __is_present(iso_objs, 'right_hand_phone')
    elif(likely == 'c2'):
        conf_score += __is_present(iso_objs, 'phone')
        conf_score += __is_present(iso_objs, 'left_hand_steering')
    elif(likely == 'c3'):
        conf_score += __is_present(iso_objs, 'phone')
        conf_score += __is_present(iso_objs, 'right_hand_steering')
        conf_score += __is_present(iso_objs, 'left_hand_phone')
    elif(likely == 'c4'):
        conf_score += __is_present(iso_objs, 'phone')
        conf_score += __is_present(iso_objs, 'right_hand_steering')
    elif(likely == 'c5'):
        conf_score += __is_present(iso_objs, 'left_hand_steering')
    elif(likely == 'c6'):
        conf_score += __is_present(iso_objs, 'left_hand_steering')
        conf_score += __is_present(iso_objs, 'cup')
        conf_score += __is_present(iso_objs, 'right_hand_steering')
        conf_score += __is_present(iso_objs, 'drinking_near_steering')
    elif(likely == 'c7'):
        conf_score += __is_present(iso_objs, 'left_hand_steering')
    elif(likely == 'c8'):
        conf_score += __is_present(iso_objs, 'right_hand_steering')
        conf_score += __is_present(iso_objs, 'left_hand_steering')
    elif(likely == 'c9'):
        conf_score += __is_present(iso_objs, 'right_hand_steering')
        conf_score += __is_present(iso_objs, 'left_hand_steering')
        conf_score += __is_present(iso_objs, 'both_hands_steering')
    else:
        raise ValueError

    return conf_score

def _forward_cls_prediction(iso_objs):
    def __relative_dist(objs, pri_obj, sec_obj):
        feat_val = 1000.0
        if((len(objs[sec_obj]) != 0) and (len(objs[pri_obj]) != 0)):
            po = objs[pri_obj]
            c1 = [(po[2]-po[0])/2. + po[0], (po[3]-po[1])/2. + po[1]]
            if(isinstance(objs[sec_obj][0], list)):
                dist = []
                for so in objs[sec_obj]:
                    c2 = [(so[2]-so[0])/2. + so[0], (so[3]-so[1])/2. + so[1]]
                    d = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                    dist.append(d)
                feat_val = min(dist)
            else:
                so = objs[sec_obj]
                c2 = [(so[2]-so[0])/2. + so[0], (so[3]-so[1])/2. + so[1]]
                feat_val = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

        return feat_val


    i_cls_idx = -100
    i_prob = 0.
    # TODO: find the proper threshold for all objects using the score distribution for all classes.
    # for now let it be 0.95
    rhp = __is_present(iso_objs, 'right_hand_phone')
    lhp = __is_present(iso_objs, 'left_hand_phone')
    cup = __is_present(iso_objs, 'cup')
    bhs = __is_present(iso_objs, 'both_hands_steering')
    phone = __is_present(iso_objs, 'phone')
    rhs = __is_present(iso_objs, 'right_hand_steering')
    lhs = __is_present(iso_objs, 'left_hand_steering')
    dns = __is_present(iso_objs, 'drinking_near_steering')

    if(bhs > 0.95):
        # c0 or c9
        i_cls_idx = 0
        i_prob = bhs
    else:
        if(rhp > 0.95):
            # c1
            i_cls_idx = 1
            i_prob += rhp
        else:
            if(lhp > 0.95):
                # c3
                i_cls_idx = 3
                i_prob += lhp
            else:
                if(dns > 0.95 or cup > 0.95):
                    # c6
                    i_cls_idx = 6
                    i_prob += dns
                    i_prob += cup
                else:
                    if(lhs > 0.95):
                        # c2, c5, c7, c8, c9
                        if(__relative_dist(iso_objs, 'radio', 'wrist') < WRIST_RADIO_LT):
                            # c5
                            i_cls_idx = 5
                            i_prob = lhs
                        else:
                            # c2, 7, 8, 9
                            if(__relative_dist(iso_objs, 'head', 'wrist') < HEAD_WRIST_LT):
                                # c2, c8
                                if(phone > 0.9):
                                    # c2
                                    i_cls_idx = 2
                                    i_prob = lhs
                                else:
                                    # c8
                                    i_cls_idx = 8
                                    i_prob = lhs
                            else:
                                # c7, 9
                                # FIXME: assigning this to c7 as c9 is covered in the first case at the top
                                i_cls_idx = 7
                                i_prob = lhs
                    else:
                        if(rhs > 0.95):
                            # c4, 8, 9
                            if(__relative_dist(iso_objs, 'head', 'wrist') < HEAD_WRIST_LT):
                                # c4, c8
                                if(phone > 0.9):
                                    # c4
                                    i_cls_idx = 4
                                    i_prob = rhs
                                else:
                                    # c8
                                    i_cls_idx = 8
                                    i_prob = lhs
                            else:
                                # c9
                                i_cls_idx = 9
                                i_prob = lhs
                        else:
                            # cannot resolve
                            i_cls_idx = 0
                            i_prob = 0.1

    return i_cls_idx, i_prob

def _isolated_obj_resolver(iso_objs, head_objs, fullimg_objs, likely):

    i_cls_idx = -100 
    i_prob = 0.1
    i_conf = False
    # if the class is not certain in the head and fullimg based detectors, this can help to
    # resolve uncertainity.
    if(likely in CLASSES):
        conf_score = _reverse_prediction(iso_objs, likely)
        if(conf_score >= 0.9):
            i_conf = True
        else:
            i_conf = False
        i_cls_idx = CLASSES.index(likely)
        h_score = head_objs[likely+'_head'][4]
        f_score = fullimg_objs[likely][4]
        # math.ceil(conf_score) gives the approx no of isolated objects that are detected to support this decision.
        i_prob = (h_score + f_score + conf_score/math.ceil(conf_score))/3
    elif(likely == 'conflict'):
        # make conf = False so that the ensemble classifier results are used to make final decision
        i_conf = False
        i_cls_idx, i_prob = _forward_cls_prediction(iso_objs)
    elif(likely == 'unsolved'):
        i_conf = False
        i_cls_idx, i_prob = _forward_cls_prediction(iso_objs)
    else:
        raise ValueError('Invalid option for variable <likely>')

    return i_cls_idx, i_prob, i_conf

def intutive_prediction(iso_objs, heads, fullimg_boxes):
    final_prob = [0.0001]*10
    certain = False
    pred_cls = 'u'

    def _get_top_classes(objs, labels, thr):
        top_cls = []
        for label in labels:
            if(objs[label] and objs[label][4] > thr):
                top_cls.append(label)
        return top_cls

    #def __decide_based_on_isolated_objs(iso_objs)
    # isolated objs are already filtered. just remove multiple head detections
    head_objs, h_found, h_max_score, h_cls = filter_multiples(heads, HEAD_OBJS)
    # filter  full image based objects
    fullimg_objs, f_found, f_max_score, f_cls = filter_multiples(fullimg_boxes, CLASSES)

    conf_thr = 0.99
    # check if atleast one head detections among 10 classes is found
    if(h_found and f_found):
        # check if head and full image based detectors giving same class predictions.
        if(h_cls.split('_')[0] == f_cls):
            print('Both detectors are confident')
            max_score = (h_max_score + f_max_score)/2
            # check if atleast one object has more than 0.99 score
            if(max_score > conf_thr):
                # check if more than 1 objects have score > 0.99
                h_top_cls = _get_top_classes(head_objs, HEAD_OBJS, conf_thr)
                f_top_cls = _get_top_classes(fullimg_objs, CLASSES, conf_thr)
                
                if(len(h_top_cls)  > 1 or len(f_top_cls) > 1):
                    # more than 1 heads have score > 0.99
                    # see if this conflict can be resolved using isolated objs
                    print('More than one class has max score')
                    i_cls, i_prob, i_conf = _isolated_obj_resolver(iso_objs, head_objs, fullimg_objs, f_cls)
                    certain = i_conf
                    final_prob[i_cls] = i_prob
                    pred_cls = CLASSES[i_cls]
                else:
                    if(h_cls in ('c2_head', 'c7_head')):
                        # isolated objeccts are not getting detected for test images correctly. Take this path if they are not detected.
                        # final detection is the one with max score.
                        certain = True
                        final_prob[HEAD_OBJS.index(h_cls)] = max_score
                        pred_cls = CLASSES[HEAD_OBJS.index(h_cls)]
                    else:
                        # not certain. either take help of isolated objs or set certain = False
                        certain = True
                        final_prob[HEAD_OBJS.index(h_cls)] = max_score
                        pred_cls = CLASSES[HEAD_OBJS.index(h_cls)]
            else:
                # need to take help from isolated object detections to decide the final prob
                print('Score is < threshold')
                i_cls, i_prob, i_conf = _isolated_obj_resolver(iso_objs, head_objs, fullimg_objs, f_cls)
                certain = i_conf
                final_prob[i_cls] = i_prob
                pred_cls = CLASSES[i_cls]
        else:
            print('Detectors are contradictiing each other.')
            # detections from head and fullimage detector are contradicting. use isolated objects
            i_cls, i_prob, i_conf = _isolated_obj_resolver(iso_objs, head_objs, fullimg_objs, 'conflict')
            certain = i_conf
            final_prob[i_cls] = i_prob
            pred_cls = CLASSES[i_cls]
        
    else:
        # decision is wholey based on the isolated objects.
        i_cls, i_prob, i_conf = _isolated_obj_resolver(iso_objs, head_objs, fullimg_objs, 'unsolved')
        certain = i_conf
        final_prob[i_cls] = i_prob
        pred_cls = CLASSES[i_cls]

    #if(certain):
    #    # normalize the prob
    #    final_prob = (np.array(final_prob)/sum(final_prob))
    #else:
    #    final_prob = np.array([0.1]*10)
    return final_prob, pred_cls, certain
    
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
                    full_img_boxes[cls] = boxes[cls]
                elif(len(boxes[cls]) > 1):
                    scores = [b[4] for b in boxes[cls]]
                    full_img_boxes[cls] = [boxes[cls][scores.index(max(scores))]]
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
                        full_img_boxes[cls] = boxes[cls]
                    elif(len(boxes[cls]) > 1):
                        scores = [b[4] for b in boxes[cls]]
                        full_img_boxes[cls] = [boxes[cls][scores.index(max(scores))]]
                    else:
                        full_img_boxes[cls] = []
                test_fullimg[img] = deepcopy(full_img_boxes)
    del temp_test_fullimg
    
    # make predictions on the training set
    report = np.zeros(shape=(len(CLASSES), len(CLASSES)), dtype=np.int32)
    loss = 0.0
    err_cnt = 0
    for img, act_cls in train_cls_info.iteritems():
        print(img)
        prob, pred_cls, conf = intutive_prediction(iso_objs=train_objset[img], heads=train_heads[img], fullimg_boxes=train_fullimg[img])
        pred_idx = CLASSES.index(pred_cls)
        act_idx = CLASSES.index(act_cls)
        if(pred_cls != act_cls):
            err_cnt += 1

        report[act_idx, pred_idx] = report[act_idx, pred_idx] + 1
        print act_idx
        print(prob[act_idx])
        p_i = prob[act_idx]
        p_i = max(min(p_i, 1-1e-15), 1e-15)
        loss += math.log(p_i)

    loss = -loss/len(train_cls_info)
    print('Error percent = {:f}%'.format((float(err_cnt)/len(train_cls_info))*100))
    print('Loss = {:f}'.format(loss))
    #show_confusion_matrix(report)
    #sys.exit()

    # use predictions based on objset 1.1 and 5 trained using dt/lr classifier
    with open(args.obj_pred, 'r') as pf:
        objset_pred = cPickle.load(pf)

    # make predictions on the testing set
    test_prob = {}
    img_no = 0
    for img, objs in test_objset.iteritems():
        prob, cls, conf = intutive_prediction(iso_objs=objs, heads=test_heads[img], fullimg_boxes=test_fullimg[img])
        # if not certain, use the probability from the objset classifier
        if(conf):
            test_prob[img] = prob
        else:
            test_prob[img] = objset_pred[img]

        img_no += 1
        if(img_no % 1000 == 0):
            print('{:d}'.format(img_no/1000))

    print('\nDone predicting prob of test images')
    with open('intutive_test_predictions.pkl', 'w') as pf:
        cPickle.dump(test_prob, pf)
        print('Stored test predictions in {:s}'.format('intutive_test_predictions.pkl'))


if __name__=='__main__':
    args = parse_args()
    cls_main(args)
