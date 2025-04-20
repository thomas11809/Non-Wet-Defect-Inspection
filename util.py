from pathlib import Path
import os

import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from skimage.transform import resize
from skimage import io, feature #, color



# Dataset & Dataloader
def make_dir_list(path) -> list:
    return [x for x in path.iterdir() if x.is_dir()]
def make_file_list(path) -> list:
    return [x for x in path.glob("*") if x.is_file()]

class CustomDataset(Dataset):
    def __init__(self, file_list, transform=transforms.ToTensor(), need_label=True, in_channels=1):
        self.file_list = file_list
        self.transform = transform
        self.need_label = need_label
        self.in_channels = in_channels

    def __getitem__(self, i): # 모델에 들어가는 입력에 해당함
        if self.in_channels==1:
            file = self.file_list[i]
            if type(file)==str:
                file = Path(file)
            img_path = str(file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if not isinstance(img, np.ndarray):
                img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_GRAYSCALE)
            img = self.transform(img)            
        else:
            imgs = torch.empty((1,64,64))
            for file in self.file_list[i]:
                if type(file)==str:
                    file = Path(file)
                img_path = str(file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if not isinstance(img, np.ndarray):
                    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_GRAYSCALE)
                img = self.transform(img)
                imgs = torch.cat((imgs, img), dim=0)
            img = imgs[1:] # CHW (20,64,64)
        
        label = 'None'
        if self.need_label:
            if ("normal" in str(file)): # or ("good" in str(file)):
                label = 0
            elif "nonwet" in str(file):
                label = 1
            else:
                raise
        sample = {'imgs': img,
                'label': label,
                'file_name': str(file)}
        return sample
        
    def __len__(self):
        return len(self.file_list)


def list_files_in_directory(directory):
    # 지정된 디렉토리에서 파일 목록을 얻음
    files = os.listdir(directory)
    # 파일들의 전체 경로를 얻음
    file_paths = [os.path.join(directory, file) for file in files]
    return file_paths

# Accuracy, Recall, FPR
def print_metrics(y_pred, i_nonwet, i_normal):
    # TP, TN, FP, FN
    TP = (y_pred[i_nonwet]==1).sum()
    TN = (y_pred[i_normal]==0).sum()
    FP = (y_pred[i_normal]==1).sum()
    FN = (y_pred[i_nonwet]==0).sum()
    # Metrics
    accuracy = 100 * (TP+TN) / (TP+TN+FP+FN)
    recall = 100 * TP / (TP+FN)
    fpr = 100 * FP / (FP+TN)
    return accuracy, recall, fpr



# 1.1. Flatten (4096)
def flatten(input_list):
    flatten_list = []
    for img_path in input_list:
        image = io.imread(img_path)
        if image.shape!=(64,64):
            image = resize(image, (64,64))
        flattened = image.flatten()
        flatten_list.append(flattened)
    return np.array(flatten_list)
# 1.2. HOG (2916)
def hog_feature(input_list):
    hog_list = []
    for img_path in input_list:
        image = io.imread(img_path)
        hog_features = feature.hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8))
        hog_list.append(hog_features)
    return np.array(hog_list)

class ExcelUpdater:
    def __init__(self, file_path, columns):
        self.file_path = file_path
        self.columns = columns
        if os.path.exists(self.file_path):
            self.df = pd.read_excel(self.file_path)
        else:
            self.df = pd.DataFrame(columns=self.columns)
            self.df.to_excel(self.file_path, index=False)
    def add_row(self, row_data):
        new_row = pd.DataFrame([row_data], columns=self.df.columns)
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        # print(f"Add new row {row_data}")
    def save(self):
        self.df.to_excel(self.file_path, index=False)
        # print(f"Save excel file to : {self.file_path}")

