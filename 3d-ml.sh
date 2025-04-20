#!/bin/bash
python ml.py --feat flatten --model one-svm
python ml.py --feat hog --model one-svm
python ml.py --feat flatten --model IF
python ml.py --feat hog --model IF
python ml.py --feat flatten --model LOF
python ml.py --feat hog --model LOF