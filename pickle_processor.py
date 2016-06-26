import sys, os
import argparse
import csv
import cPickle

# local imports
from feat_gen import compute_features

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
    compute_features(obj_dict_list, cls_info)
