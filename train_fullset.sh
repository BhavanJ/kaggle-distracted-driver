#!/bin/bash
python pickle_processor.py --train ../test/rcnn/output/kaggle_full_trainset_bbox_objset_1.pkl \
	../test/rcnn/output/kaggle_full_trainset_bbox_objset_4.pkl \
	--csv ../driver_imgs_list.csv \
	--test ../test/rcnn/test-box/teset_set_batch_0_bbox.pkl
