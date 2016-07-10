import sys
import cPickle
import argparse
import numpy as np

def generate_result_sheet(result_dict, file_name='kaggle_dd_test_result.csv'):

    with open(file_name, 'w') as rf:
        rf.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
        for img, p in result_dict.iteritems():
            rf.write('{:s},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}\n'.format(img, 
                p[0],p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]))

def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle result sheet generator')
    parser.add_argument('--list', dest='list_file', help='List of test images. One image per line')
    parser.add_argument('--pred', dest='pred_file', help='Test prediction file')

    if(len(sys.argv) < 3):
        parser.print_help()

    args = parser.parse_args()

    return args

def prob_boost(prob):
    max_idx = prob.index(max(prob))
    x = prob[max_idx]
    prob[max_idx] = 1.0
    s = 2.0 - x
    boosted = [p/s for p in prob]
    return boosted

if __name__=='__main__':
    args = parse_args()
    with open(args.pred_file, 'r') as pf:
        pred_dict = cPickle.load(pf)

    valid_imgs = pred_dict.keys()
    print('No of images in the prediction file = {:d}'.format(len(valid_imgs)))
    result = {}
    
    rand_prob_cnt = 0
    with open(args.list_file, 'r') as tf:
        for line in tf:
            img_name = line.rstrip('\n')
            if(img_name not in valid_imgs):
                prob = [0.1]*10
                rand_prob_cnt += 1
            else:
                prob = pred_dict[img_name]
            s = np.sum(prob)
            prob = prob/s
            result[img_name] = prob.tolist()
            #result[img_name] = prob
    print('Did not find predictions for {:d} images'.format(rand_prob_cnt))
    generate_result_sheet(result)
