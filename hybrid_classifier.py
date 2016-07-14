import argparse
import sys, os
import numpy as np
import csv
import cPickle

from feat_gen import compute_features
from pickle_processor import naive_bayes_classifier, logistic_regression_classifier, dt_classifier, random_forest_classifier

CLASSES = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')
HEAD_OBJS = ('c0_head', 'c1_head', 'c2_head', 'c3_head','c4_head','c5_head','c6_head','c7_head','c8_head','c9_head')

def parse_args():
    parser = argparse.ArgumentParser(description='Takes detection files to generate features.')
    parser.add_argument('--train-obj', dest='train_obj_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--train-head', dest='train_head_file', help='Training object pickle files')
    parser.add_argument('--csv', dest='csv', default=None, help='Class info csv file')
    parser.add_argument('--test-obj', dest='test_obj_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--test-head', dest='test_head_files', nargs='+', help='Training object pickle files')
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

def get_head_feat_dict(objs):
    feat_vec = {}
    for head in HEAD_OBJS:
        val = -1.0
        if(len(objs[head]) == 1):
            val = objs[head][0][4]
        elif(len(objs[head]) > 1):
            # take the one with max score
            scores = [h[4] for h in objs[head]]
            val = max(scores)
        feat_vec[head] = val

    return feat_vec

def create_head_fset(train_head_file, test_head_files):
    # create prediction dictionary for the training set
    with open(train_head_file, 'r') as df:
        train_heads = cPickle.load(df)['boxes']


    # create the head features for all train images
    train_feat_dict = {}
    for img, objs in train_heads.iteritems():
        train_feat_dict[img] = get_head_feat_dict(objs)

    # clreate the head features for all test images
    test_heads = {}
    for pkl_file in test_head_files:
        with open(pkl_file, 'r') as pf:
            test_heads.update(cPickle.load(pf)['boxes'])

    test_feat_dict = {}
    for img, objs in test_heads.iteritems():
        test_feat_dict[img] = get_head_feat_dict(objs)

    return train_feat_dict, test_feat_dict


def create_objs_fset(train_obj_files, test_obj_files, cls_csv_file):

    cls_info = get_cls_dict(cls_csv_file)

    # compute training feature dict
    train_obj_dict_list = []
    for pkl_file in train_obj_files:
        with open(pkl_file, 'r') as pf:
            train_obj_dict_list.append(cPickle.load(pf)['boxes'])

    train_obj_feat, obj_mean_model = compute_features(train_obj_dict_list, cls_info)

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
    test_obj_feat = compute_features(test_obj_dict_list, dummy_cls_info, train=False,
        head_mean_c=obj_mean_model[0],
        steering_mean_c=obj_mean_model[1],
        head_mean_box=obj_mean_model[2],
        steering_mean_box=obj_mean_model[3])

    return train_obj_feat, test_obj_feat

def train_feat_dict_to_ndarray(train_obj_fset, train_head_fset, feat_labels, shuffle=False):
    no_samples = len(train_obj_fset)
    ndim = len(feat_labels)
    print('No of samples in train + val set = {:d}'.format(no_samples))
    print('Feature set dimension = {:d}'.format(ndim))

    X = np.zeros(shape=(no_samples, ndim), dtype=np.float32)
    y = np.zeros(shape=(no_samples), dtype=np.float32)

    s = 0
    for img, obj_ft in train_obj_fset.iteritems():
        # get head feature of the image and merge it with object features.
        all_ft = train_head_fset[img]
        all_ft.update(obj_ft)

        X[s,:] = [all_ft[f] for f in feat_labels]
        y[s] = float(CLASSES.index(obj_ft['cls']))
        s += 1

    if(shuffle):
        # Randomly reshuffle the data
        rand_idx = np.random.permutation(len(X))
        X = X[rand_idx]
        y = y[rand_idx]

    return X, y
 
def cls_main(args):

    # get features from isolated objects
    train_obj_fset, test_obj_fset = create_objs_fset(args.train_obj_files, args.test_obj_files, args.csv)

    # get features from head detections
    train_head_fset, test_head_fset = create_head_fset(args.train_head_file, args.test_head_files)

    # create the feature label set
    obj_feat_labels = train_obj_fset[train_obj_fset.keys()[0]].keys()
    obj_feat_labels.remove('cls')

    print obj_feat_labels
    print HEAD_OBJS

    all_feat_labels = HEAD_OBJS + tuple(obj_feat_labels)
    print all_feat_labels

    X, y = train_feat_dict_to_ndarray(train_obj_fset, train_head_fset, all_feat_labels, shuffle=True)

    # split the data into train and validation set. Do mean and variance normalization
    train_percent = 0.8
    no_train_samples = int(train_percent*len(X))
    train_x = X[0:no_train_samples]
    val_x = X[no_train_samples:]
    train_y = y[0:no_train_samples]
    val_y = y[no_train_samples:]

    #mean and variance normalization. Make sure std is not zero to avoid division by zero
    mean_x = np.mean(train_x, axis=0)
    std_x = np.maximum(np.std(train_x, axis=0), 1e-14)
    train_x = (train_x - mean_x)/std_x
    val_x = (val_x - mean_x)/std_x

    # Training
    print('Training...')
    dt_clf = dt_classifier(train_x, train_y, val_x, val_y)
    rf_clf = random_forest_classifier(train_x, train_y, val_x, val_y)
    lr_clf = logistic_regression_classifier(train_x, train_y, val_x, val_y)
    #naive_bayes_classifier(train_x, train_y, val_x, val_y)
    sys.exit()
    final_clf = lr_clf
    # compute probabilities of all classes for all images
    assert(len(test_obj_fset) == len(test_head_fset)), 'No images in object set != those in head set'
    print('Making predictions for {:d} test images'.format(len(test_obj_fset)))
    test_predictions = {}
    img_cnt = 0
    for img, obj_ft in test_obj_fset.iteritems():
        all_ft = test_head_fset[img]
        all_ft.update(obj_ft)
        fvec = [all_ft[f] for f in all_feat_labels]
        x = np.array(fvec, dtype=np.float32)
        x = x.reshape(1, -1)
        x = (x - mean_x)/std_x
        prob = final_clf.predict_proba(x)
        test_predictions[img] = prob[0].tolist()
        img_cnt += 1
        if((img_cnt % 1000) == 0):
            print(img_cnt)

    print('\nDone predicting prob of test images')
    with open('hybrid_test_predictions.pkl', 'w') as pf:
        cPickle.dump(test_predictions, pf)
        print('Stored test predictions in {:s}'.format('hybrid_test_predictions.pkl'))


if __name__=='__main__':
    args = parse_args()
    cls_main(args)
