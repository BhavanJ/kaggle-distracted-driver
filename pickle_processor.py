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
def generate_kaggle_eval_metrics(train_prob, train_y, val_prob, val_y):
    """ Compute the loss metrics as per kaggle formula
    """
    def __get_loss(prob, target):
        # choose all prob corresponding to target label
        idx_row = [i for i in range(len(target))]
        idx_col = target.astype(np.int32).tolist()
        p_i = prob[idx_row, idx_col]
        # avoid log(0)
        p_i = np.maximum(np.minimum(p_i, 1-1e-15), 1e-15)
        loss = np.mean(np.log(p_i))
        return -loss
    train_loss = __get_loss(train_prob, train_y)
    val_loss = __get_loss(val_prob, val_y)
    print('Training loss = {:f}'.format(train_loss))
    print('Validation loss = {:f}'.format(val_loss))
        
     
def dt_classifier(train_x, train_y, val_x, val_y, gen_pic=False, feats=()):

    assert(len(train_x) == len(train_y)), 'No of training amples != labels'
    assert(len(val_x) == len(val_y)), 'No of validation amples != labels'

    print('No of training samples = {:d}'.format(len(train_x)))
    print('No of validation samples = {:d}'.format(len(val_x)))

    clf = tree.DecisionTreeClassifier()
    print('Training decision tree classifier...')
    trained_clf = clf.fit(train_x, train_y)

    train_pred = trained_clf.predict(train_x)
    train_pred_prob = trained_clf.predict_proba(train_x)
    acc = sklearn.metrics.accuracy_score(train_y, train_pred)
    print('Training Accuracy = {:f} %'.format(acc*100))
    val_pred = trained_clf.predict(val_x)
    val_pred_prob = trained_clf.predict_proba(val_x)
    acc = sklearn.metrics.accuracy_score(val_y, val_pred)
    print('Validation Accuracy = {:f} %'.format(acc*100))

    # compute training and validation loss
    generate_kaggle_eval_metrics(train_pred_prob, train_y, val_pred_prob, val_y)

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

    # Randomly reshuffle the data
    rand_idx = np.random.permutation(len(X))
    X = X[rand_idx]
    y = y[rand_idx]
    
    train_percent = 0.8
    no_train_samples = int(train_percent*len(X))
    train_x = X[0:no_train_samples]
    val_x = X[no_train_samples:]
    train_y = y[0:no_train_samples]
    val_y = y[no_train_samples:]

    dt_classifier(train_x, train_y, val_x, val_y, gen_pic=True, feats=('f0', 'f1', 'f2', 'f5'))
