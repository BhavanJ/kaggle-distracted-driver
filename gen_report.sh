#!/bin/bash
python ensemble_prediction.py --pred head_classifer_predictions.pkl objset_test_predictions.pkl \
	--list ../test_set.txt --out ens_test_predictions.pkl

python -u  result_gen.py --pred ens_test_predictions.pkl --list ../test_set.txt
