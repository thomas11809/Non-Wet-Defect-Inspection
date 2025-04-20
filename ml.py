from pathlib import Path

import os
import numpy as np

# 2D Counting
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 3D Outlier/Novelty detection
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from torchvision import transforms
from util import *

seed = 2024
os.makedirs("results/", exist_ok=True)

# parser lib
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--feat', type=str, choices=["flatten", "hog"],
    required=True, help="feature extraction e.g., flatten, hog")
parser.add_argument('--model', type=str, choices=["knn", "linear", "logistic", "svm", "rf", "one-svm", "IF", "LOF"],
    required=True, help="classifier model e.g., \n [2d counting] knn, linear, logistic, svm, rf \n [3d outlier/novelty detection] one-svm, IF, LOF")

args = parser.parse_args()
feat = args.feat
model = args.model

if model in ["knn", "linear", "logistic", "svm", "rf"]:
    method = "2d counting"
elif model in ["one-svm", "IF", "LOF"]:
    method = "3d outlier/novelty detection"
else:
    raise
print(f"{method} - {model} ({feat})")

# *Train* at first : 2D Counting methods need to train 2D supervised ML models first.
if method == "2d counting": 
    # 2D train list
    x_train = []
    y_train = []
    for dir in make_dir_list(Path("data/1ch_dataset/2d_train_data/normal2d")):
        x_train += make_file_list(dir)
    y_train += [0] * len(x_train)
    for dir in make_dir_list(Path("data/1ch_dataset/2d_train_data/nonwet2d")):
        x_train += make_file_list(dir)
    y_train += [1] * (len(x_train) - len(y_train))

    # 2D val list
    x_val = []
    y_val = []
    for dir in make_dir_list(Path("data/1ch_dataset/2d_val_data/normal2d")):
        x_val += make_file_list(dir)
    y_val += [0] * len(x_val)
    for dir in make_dir_list(Path("data/1ch_dataset/2d_val_data/nonwet2d")):
        x_val += make_file_list(dir)
    y_val += [1] * (len(x_val) - len(y_val))


    # Shuffle 2D train data
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    shuffle_index = np.random.RandomState(seed=seed).permutation(len(x_train))
    x_train_shuffled = x_train[shuffle_index]
    y_train_shuffled = y_train[shuffle_index]
    # True indices of 2D val data
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    idx_normal = y_val==0
    idx_nonwet = y_val==1

    # 2D Feature Extraction
    if feat=="flatten":
        print("Feature Extraction Method : No, just Flatten.")
        x_train_shuffled = flatten(x_train_shuffled)
        x_val = flatten(x_val)
    elif feat=="hog":
        print("Feature Extraction Method : HOG")
        x_train_shuffled = hog_feature(x_train_shuffled)
        x_val = hog_feature(x_val)
    else:
        raise ValueError

    

    # 2D Supervised ML model
    # print(f"*** Train 2D Supervised ML model : {model} => then, validation. ***")
    if model=="knn":
        """
        for k in range(3,100,2):
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(x_train_shuffled, y_train_shuffled)
            y_pred = knn_classifier.predict(x_val)
            y_pred = np.array(y_pred)
            print_metrics(y_pred, idx_nonwet, idx_normal)
        # Make K=5 classifier for clear code after all. (temporary)
        """
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif model=="linear":
        classifier = LinearRegression()
    elif model=="logistic":
        classifier = LogisticRegression()
    elif model=="svm":
        classifier = SVC()
    elif model=="rf":
        classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError
    # 2d train
    classifier.fit(x_train_shuffled, y_train_shuffled)
    # 2d validate
    y_pred = classifier.predict(x_val)
    y_pred = np.where(y_pred > 0.5, 1 , 0)
    acc2d, rec2d, fpr2d = print_metrics(y_pred, idx_nonwet, idx_normal)
    # print(f'2d {model} model, \tAccuracy: {acc2d:.2f}%\tRecall: {rec2d:.2f}%\tFPR: {fpr2d:.2f}%')
    # print("\n"*5)


# *Test* bump folders
num_slice = 20
# Make Data list and bump labels list #
bump_slice_list = []
labels_bump = []
#1. normal
file_list = []
for folder in list_files_in_directory("data/3d_test_data/normal"):
    file_list += list_files_in_directory(folder)
bump_slice_list += file_list
num_bumps = int(len(file_list)/num_slice)
labels_bump += [0] * num_bumps
# print(f"# of normal slices, bumps\t: {len(file_list)}, {num_bumps}")
#2. nonwet
file_list = []
for folder in list_files_in_directory("data/3d_test_data/nonwet"):
    file_list += list_files_in_directory(folder)
bump_slice_list += file_list
num_bumps = int(len(file_list)/num_slice)
labels_bump += [1] * num_bumps

# True indices of bump labels
bump_slice_np = np.array(bump_slice_list)
labels_bump = np.array(labels_bump)
idx_normal_bump = labels_bump==0
idx_nonwet_bump = labels_bump==1

# 2D Feature Extraction for bump folders
if feat=="flatten":
    print("Feature Extraction Method : No, just Flatten.")
    bump_slice_np = flatten(bump_slice_np)
elif feat=="hog":
    print("Feature Extraction Method : HOG")
    bump_slice_np = hog_feature(bump_slice_np)
else:
    raise ValueError


if method == "2d counting":
    # 2D Counting (Test)
    row_rec = {"over_num_slice": f"{model} ({feat}) - Recall"}
    row_fpr = {"over_num_slice": f"{model} ({feat}) - FPR"}
    # Prediction
    cnt = 20
    for over_num_slice in range(0,cnt):
        pred_bump = classifier.predict(bump_slice_np).reshape(-1, num_slice) # (N_bump, 20)
        pred_bump = np.where(pred_bump > 0.5, 1 , 0)
        pred_bump = pred_bump.sum(axis=-1) > over_num_slice # (N_bump, )

        acc, rec, fpr = print_metrics(pred_bump, idx_nonwet_bump, idx_normal_bump)
        row_rec[over_num_slice] = round(rec,2)
        row_fpr[over_num_slice] = round(fpr,2)
    file_path = "results/2d-ml-counting.xlsx"
    excel_updater = ExcelUpdater(file_path, columns=row_rec.keys())
    excel_updater.add_row(row_rec)
    excel_updater.add_row(row_fpr)
        
elif method == "3d outlier/novelty detection":
    # 3D Bump Outlier/Novelty detection
    # Reshape 3D Bump Data : (N_bump x 20, n_feat) -> (N_bump, n_feat x 20)
    # bump_slice_np = bump_slice_np.reshape(len(labels_bump), -1) # (N_bump, n_feat x 20)
    K = 20
    bump_slice_np = bump_slice_np[::(20//K),:].reshape(len(labels_bump), -1) # (N_bump, n_feat x K)
    
    # GT outlier frcation : 132 / (754+132) = 0.149
    outliers_fraction = 0.15
    # 3D Anomaly detection ML model
    if model=="one-svm":
        classifier = OneClassSVM(nu=outliers_fraction, kernel="rbf")
    elif model=="IF":
        classifier = IsolationForest(contamination=outliers_fraction, random_state=seed)
    elif model=="LOF":
        classifier = LocalOutlierFactor(contamination=outliers_fraction)
    else:
        raise ValueError
    
    # Fit & Prediction
    pred_bump = classifier.fit_predict(bump_slice_np) # [-1:outlier, +1:inlier]
    pred_bump[pred_bump==1] = 0 # change inlier as 0
    pred_bump[pred_bump==-1] = 1 # change outlier as 1
    acc, rec, fpr = print_metrics(pred_bump, idx_nonwet_bump, idx_normal_bump)
    file_path = "results/3d-ml.xlsx"
    row_data = {"Method": f"{model} ({feat})",
                "Accuracy": round(acc,2), 
                "Recall": round(rec,2), 
                "FPR": round(fpr,2)}
    excel_updater = ExcelUpdater(file_path, columns=row_data.keys())
    excel_updater.add_row(row_data)

excel_updater.save()

print("\n"*5)
