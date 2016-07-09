import sys, os
import argparse
import csv
import cPickle
import numpy as np
import sklearn
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.calibration import CalibratedClassifierCV
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
    parser.add_argument('--train', dest='train_pkl_files', nargs='+', help='Training object pickle files')
    parser.add_argument('--csv', dest='csv', default=None, help='Class info csv file')
    parser.add_argument('--test', dest='test_pkl_files', nargs='+', help='Training object pickle files')
    args = parser.parse_args()
    return args

def feat_dict_to_ndarray(feat_dict, feats, train=True):
    no_samples = len(feat_dict)
    ndim = len(feats)

    X = np.zeros(shape=(no_samples, ndim), dtype=np.float32)
    if train:
        y = np.zeros(shape=(no_samples), dtype=np.float32)
    s = 0
    for img, ft in feat_dict.iteritems():
        X[s,:] = [ft[f] for f in feats]
        if train:
            y[s] = float(CLASSES.index(ft['cls']))
        s += 1
    if train:
        return X, y
    else:
        return X

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
    print('--------------Decision Tree Classifier-------------')
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
    print('Probability calibration.....')
    clf_isotonic = CalibratedClassifierCV(trained_clf, cv=2, method='isotonic')
    clf_isotonic.fit(train_x, train_y)
    train_prob_isotonic = clf_isotonic.predict_proba(train_x)
    val_prob_isotonic = clf_isotonic.predict_proba(val_x)
    generate_kaggle_eval_metrics(train_prob_isotonic, train_y, val_prob_isotonic, val_y)

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

    return trained_clf

def random_forest_classifier(train_x, train_y, val_x, val_y):
    print('-------Random Forest Classifier-----------')
    clf = sklearn.ensemble.forest.RandomForestClassifier()
    clf = clf.fit(train_x, train_y)
    train_pred = clf.predict(train_x)
    train_prob = clf.predict_proba(train_x)
    train_acc = clf.score(train_x, train_y)
    print('Train accuracy = {:f}'.format(train_acc))
    val_pred = clf.predict(val_x)
    val_prob = clf.predict_proba(val_x)
    val_acc = clf.score(val_x, val_y)
    print('Validation accuracy = {:f}'.format(val_acc))
    generate_kaggle_eval_metrics(train_prob, train_y, val_prob, val_y)
    print('Probability calibration.....')
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_isotonic.fit(train_x, train_y)
    train_prob_isotonic = clf_isotonic.predict_proba(train_x)
    val_prob_isotonic = clf_isotonic.predict_proba(val_x)
    generate_kaggle_eval_metrics(train_prob_isotonic, train_y, val_prob_isotonic, val_y)

def logistic_regression_classifier(train_x, train_y, val_x, val_y):
    print('-------Logistic Regression Classifier-----------')
    #clf = sklearn.linear_model.LogisticRegression(max_iter=500)
    clf = sklearn.linear_model.SGDClassifier(loss='log')
    clf = clf.fit(train_x, train_y)
    train_pred = clf.predict(train_x)
    train_prob = clf.predict_proba(train_x)
    train_acc = clf.score(train_x, train_y)
    print('Train accuracy = {:f}'.format(train_acc))
    val_pred = clf.predict(val_x)
    val_prob = clf.predict_proba(val_x)
    val_acc = clf.score(val_x, val_y)
    print('Validation accuracy = {:f}'.format(val_acc))
    generate_kaggle_eval_metrics(train_prob, train_y, val_prob, val_y)
    print('Probability calibration.....')
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_isotonic.fit(train_x, train_y)
    train_prob_isotonic = clf_isotonic.predict_proba(train_x)
    val_prob_isotonic = clf_isotonic.predict_proba(val_x)
    generate_kaggle_eval_metrics(train_prob_isotonic, train_y, val_prob_isotonic, val_y)

def naive_bayes_classifier(train_x, train_y, val_x, val_y):
    print('-------Naive Bayes Classifier-----------')
    clf = sklearn.naive_bayes.GaussianNB()
    clf = clf.fit(train_x, train_y)
    train_pred = clf.predict(train_x)
    train_prob = clf.predict_proba(train_x)
    train_acc = clf.score(train_x, train_y)
    print('Train accuracy = {:f}'.format(train_acc))
    val_pred = clf.predict(val_x)
    val_prob = clf.predict_proba(val_x)
    val_acc = clf.score(val_x, val_y)
    print('Validation accuracy = {:f}'.format(val_acc))
    generate_kaggle_eval_metrics(train_prob, train_y, val_prob, val_y)
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_isotonic.fit(train_x, train_y)
    train_prob_isotonic = clf_isotonic.predict_proba(train_x)
    val_prob_isotonic = clf_isotonic.predict_proba(val_x)
    generate_kaggle_eval_metrics(train_prob_isotonic, train_y, val_prob_isotonic, val_y)


if __name__=='__main__':
    args = parse_args()

    if(args.csv == None):
        print('Gathering class information from the first object pickle file')
        bbox_file = args.train_pkl_files[0]
        with open(bbox_file, 'r') as box_file:
            box_list = cPickle.load(box_file)['boxes']
        cls_info = construct_category_info(box_list)
    else:
        print('Gathering class information from the CSV file provided.')
        cls_info = construct_category_info(args.csv)

    print('Reading all object pickle files to create a list of object dictionaries')
    obj_dict_list = []
    for pkl_file in args.train_pkl_files:
        with open(pkl_file, 'r') as pf:
            obj_dict_list.append(cPickle.load(pf)['boxes'])

    # compute features
    feat_dict, mean_model = compute_features(obj_dict_list, cls_info)

    # order of features needs to be fixed
    train_feats = feat_dict[feat_dict.keys()[0]].keys()
    train_feats.remove('cls')
    print train_feats

    # convert feature dictionary to numpy array.
    X, y = feat_dict_to_ndarray(feat_dict, train_feats)

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

    clf = dt_classifier(train_x, train_y, val_x, val_y, gen_pic=False, feats=train_feats)

    random_forest_classifier(train_x, train_y, val_x, val_y)
    logistic_regression_classifier(train_x, train_y, val_x, val_y)
    naive_bayes_classifier(train_x, train_y, val_x, val_y)
    # Testing.
    """
    print('Training and validation done\nStarting testing...')
    print('Reading the bounding box file for testset')
    test_obj_dlist = []
    dummy_cls_info = {}
    test_obj_dict = {}
    if(args.test_pkl_files != None):
        for pkl_file in args.test_pkl_files:
            with open(pkl_file, 'r') as pf:
                test_obj_dict.update(cPickle.load(pf)['boxes'])

    test_obj_dlist.append(test_obj_dict)
    # create dummy class info requried by compute features method
    print('Creating dummy class list...')
    for d in test_obj_dlist:
        for img in d:
            dummy_cls_info[img] = 'u' # unknown

    # compute features for test images
    print('Computing featues on test set...')
    print len(mean_model)
    test_feat_dict = compute_features(test_obj_dlist, dummy_cls_info, train=False,
        head_mean_c=mean_model[0],
        steering_mean_c=mean_model[1],
        head_mean_box=mean_model[2],
        steering_mean_box=mean_model[3])

    # compute probabilities of all classes for all images
    print('Making predictions...')
    no_samples = len(test_feat_dict)
    test_predictions = {}
    for img, ft in test_feat_dict.iteritems():
        fvec = [ft[f] for f in train_feats]
        x = np.array(fvec, dtype=np.float32)
        x = x.reshape(1, -1)
        prob = clf.predict_proba(x)
        test_predictions[img] = prob[0].tolist()

    with open('test_predictions.pkl', 'w') as pf:
        cPickle.dump(test_predictions, pf)
        print('Stored test predictions in {:s}'.format('test_predictions.pkl'))
    """
    
    
