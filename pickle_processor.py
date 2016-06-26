import sys, os
import argparse
import csv
import cPickle
import numpy as np
import sklearn
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

# local imports
from feat_gen import compute_features, CLASSES

def construct_category_info(dict_or_csv):
    # if it is dictionary, name itself includes category info
    cls_info = {}
    if(isinstance(dict_or_csv, dict)):
        for img_name in dict_or_csv:
            cls_info[img_name] = img_name.split('_')[0]
    else:
        assert(os.path.exists(dict_or_csv)), 'Class info CSV file not found'
        with open(dict_or_csv, 'r') as cf:
            data = csv.reader(cf)
            for row in data:
                cls_info[row[2]] = row[1]
    return cls_info

def parse_args():
    parser = argparse.ArgumentParser(description='Takes detection files to generate features.')
    parser.add_argument('--files', dest='pkl_files', nargs='+', help='Path to dataset directory')
    parser.add_argument('--csv', dest='csv', default=None, help='Class info csv file')
    args = parser.parse_args()
    return args

def feat_dict_to_ndarray(feat_dict):
    no_samples = len(feat_dict)
    feats = feat_dict[feat_dict.keys()[0]].keys()
    feats.remove('cls')
    ndim = len(feats)

    print no_samples
    print ndim
    print feats
    X = np.zeros(shape=(no_samples, ndim), dtype=np.float32)
    y = np.zeros(shape=(no_samples), dtype=np.float32)
    s = 0
    for img, ft in feat_dict.iteritems():
        X[s,:] = [ft[f] for f in feats]
        y[s] = float(CLASSES.index(ft['cls']))
        s += 1

    return X, y

def dt_classifier(X, y, gen_pic=False, feats=()):
    clf = tree.DecisionTreeClassifier()
    print('Training decision tree classifier...')
    trained_clf = clf.fit(X, y)

    pred = trained_clf.predict(X)
    acc = sklearn.metrics.accuracy_score(y, pred)
    print('Training Accuracy = {:f} %'.format(acc*100))
    if(gen_pic):
        assert(feats), 'Specify the feature names used for training'
        dot_data = StringIO() 
        tree.export_graphviz(trained_clf, out_file=dot_data,
            feature_names=feats,  
            class_names=CLASSES,  
            filled=True, rounded=True,  
            special_characters=True) 
        print('Generating graph of the tree...')
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        graph.write_pdf("dd_classifier.pdf") 

if __name__=='__main__':
    args = parse_args()

    if(args.csv == None):
        print('Gathering class information from the first object pickle file')
        bbox_file = args.pkl_files[0]
        with open(bbox_file, 'r') as box_file:
            box_list = cPickle.load(box_file)['boxes']
        cls_info = construct_category_info(box_list)
    else:
        print('Gathering class information from the CSV file provided.')
        cls_info = construct_category_info(args.csv)

    print('Reading all object pickle files to create a list of object dictionaries')
    obj_dict_list = []
    for pkl_file in args.pkl_files:
        with open(pkl_file, 'r') as pf:
            obj_dict_list.append(cPickle.load(pf)['boxes'])

    # compute features
    feat_dict = compute_features(obj_dict_list, cls_info)

    # convert feature dictionary to numpy array.
    X, y = feat_dict_to_ndarray(feat_dict)

    dt_classifier(X, y, gen_pic=True, feats=('f0', 'f1', 'f2', 'f5'))
