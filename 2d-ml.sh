#!/bin/bash
python ml.py --feat flatten --model knn
python ml.py --feat hog --model knn
python ml.py --feat flatten --model linear
python ml.py --feat hog --model linear
python ml.py --feat flatten --model logistic
python ml.py --feat hog --model logistic
python ml.py --feat flatten --model svm
python ml.py --feat hog --model svm
python ml.py --feat flatten --model rf
python ml.py --feat hog --model rf