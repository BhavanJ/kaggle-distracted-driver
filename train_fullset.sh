#!/bin/bash
python pickle_processor.py --files ../test/rcnn/output/kaggle_full_trainset_bbox_objset_1.pkl \
	../test/rcnn/output/kaggle_full_trainset_bbox_objset_4.pkl --csv ../driver_imgs_list.csv
