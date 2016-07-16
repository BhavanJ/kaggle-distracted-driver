import sys
import cPickle
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble prediction generator')
    parser.add_argument('--pred', dest='pred_files', nargs='+', help='Test prediction files')
    parser.add_argument('--list', dest='img_list', help='Output prediction file')
    parser.add_argument('--out', dest='out_file', help='Output prediction file')

    if(len(sys.argv) < 3):
        parser.print_help()

    args = parser.parse_args()

    return args
def is_equal_prob(prob, val):
    equal = True
    for p in prob:
        if(p != val):
            equal = False

    return equal
def ensemble_pred(pred_files, list_file, out_file):
    pred_lists = []
    ens_prob_dict = {}
    for f in pred_files:
        with open(f, 'r') as pf:
            pred_lists.append(cPickle.load(pf))

    #weights = (0.8, 0.2)
    
    #weights = [0.894737, 0.653286, 0.98921, 0.714408, 0.966466, 0.729239, 0.815054, 0.795704, 0.676609, 0.883513]
    weights = [0.8]*10
    
    #weights = ([0.85, 0.65, 0.9, 0.7, 0.9, 0.7, 0.8, 0.75, 0.65, 0.85], [0.15, 0.35, 0.1, 0.3, 0.1, 0.3, 0.2, 0.25, 0.35, 0.15])
    with open(list_file, 'r') as tf:
        for line in tf:
            img_name = line.rstrip('\n')
            prob_list = []
            # gather prob for this image from all predictions
            for pred_dict in pred_lists:
                if(pred_dict.has_key(img_name)):
                    prob_list.append(pred_dict[img_name])
                else:
                    prob_list.append([0.1]*10)
            # weighted sum
            ens_prob = [0.0]*10
            for p, prob in enumerate(prob_list):
                # if all prob = 0.1 then skip
                if(not is_equal_prob(prob, 0.1)):
                    for i in range(10):
                        if(p == 0):
                            ens_prob[i] += weights[i] * prob[i]
                        else:
                            ens_prob[i] += ((1.-weights[i]) * prob[i])

            # check if all are 0
            if(is_equal_prob(ens_prob, 0.0)):
                print('Still all prob = 0')
                ens_prob = [0.1]*10

            ens_prob_dict[img_name] = ens_prob[:]

    with open(out_file, 'w') as of:
        cPickle.dump(ens_prob_dict, of)

if __name__=='__main__':
    args = parse_args()
    ensemble_pred(args.pred_files, args.img_list, args.out_file)
