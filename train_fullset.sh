#!/bin/bash
dir=$(dirname "$0")
echo $dir
test_bbox_dir="$dir/../test/rcnn/test-box"
echo $test_bbox_dir
test_bbox_files="$test_bbox_dir/*.pkl"
echo $test_bbox_files
python pickle_processor.py --train ../test/rcnn/output/kaggle_full_trainset_bbox_objset_1_1.pkl \
	../test/rcnn/output/kaggle_full_trainset_bbox_objset_5.pkl \
	--csv ../driver_imgs_list.csv \
	--test $test_bbox_files
#	--test ../test/rcnn/test-box/teset_set_batch_0_bbox.pkl ../test/rcnn/test-box/teset_set_batch_1_bbox.pkl \
#           ../test/rcnn/test-box/teset_set_batch_2_bbox.pkl ../test/rcnn/test-box/teset_set_batch_3_bbox.pkl \
#           ../test/rcnn/test-box/teset_set_batch_4_bbox.pkl ../test/rcnn/test-box/teset_set_batch_5_bbox.pkl \
#           ../test/rcnn/test-box/teset_set_batch_7_bbox.pkl
