#!/bin/bash
dir=$(dirname "$0")
echo $dir

test_obj_bbox_dir="$dir/../test/rcnn/test-box"
echo $test_obj_bbox_dir
test_obj_bbox_files="$test_obj_bbox_dir/*.pkl"
echo $test_obj_bbox_files

test_head_bbox_dir="$dir/../test/rcnn/test-box-head"
echo $test_head_bbox_dir
test_head_bbox_files="$test_head_bbox_dir/*.pkl"
echo $test_head_bbox_files

python -u hybrid_classifier.py --train-obj ../test/rcnn/output/kaggle_full_trainset_bbox_objset_1.pkl \
	../test/rcnn/output/kaggle_full_trainset_bbox_objset_4.pkl \
	--train-head ../test/rcnn/output/kaggle_full_trainset_bbox_objset_head.pkl \
    --csv ../driver_imgs_list.csv \
    --test-obj $test_obj_bbox_files\
    --test-head $test_head_bbox_files

