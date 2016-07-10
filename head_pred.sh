#!/bin/bash
dir=$(dirname "$0")
echo $dir
test_bbox_dir="$dir/../test/rcnn/test-box-head"
echo $test_bbox_dir
test_bbox_files="$test_bbox_dir/*.pkl"
echo $test_bbox_files
python head_based_classifier.py --train ../test/rcnn/output/kaggle_full_trainset_bbox_objset_head.pkl \
	--csv ../driver_imgs_list.csv \
	--test $test_bbox_files
#echo "Generating submission file..."
#python -u  result_gen.py --pred head_based_test_predictions.pkl --list ../test_set.txt
