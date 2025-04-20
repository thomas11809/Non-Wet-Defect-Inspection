from pathlib import Path

import os
import numpy as np

import torch
from torch import nn
from torchmetrics import functional as FM
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from util import list_files_in_directory, print_metrics, CustomDataset, ExcelUpdater


class LitDataModule(pl.LightningDataModule):
    def __init__(self, in_channels: int = 1, batch_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(64)])
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        pass
        
    def setup(self, stage = None):
        tr_trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(64),
                                      transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        val_trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(64)])
        self.train_set = CustomDataset(x_train, transform=tr_trans, in_channels=self.in_channels)
        self.val_set = CustomDataset(x_val, transform=val_trans, in_channels=self.in_channels)
        
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        pass #return DataLoader(self.mnist_test, batch_size=32)
        
    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

def resnet_custom(in_channels, num_classes):
    net = models.resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
    num_feat = net.fc.in_features
    net.fc = nn.Linear(num_feat, num_classes)
    return net

class FocalLoss(nn.Module): # for imbalanced data
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):    
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()
        else:
            raise

class LitModel(pl.LightningModule):
    def __init__(self, in_channels=1):
        super(LitModel, self).__init__()
        self.in_channels = in_channels
        self.model = resnet_custom(in_channels=self.in_channels, num_classes=2)
        self.criterion = FocalLoss(reduction="mean")

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        imgs = batch['imgs']
        label = batch['label']
        logits = self.model(imgs)
        
        loss = self.criterion(logits, label)
        logits = torch.argmax(logits, dim=1)
        acc = FM.accuracy(logits, label, 'binary')
        prec = FM.precision(logits, label, 'binary')
        rec = FM.recall(logits, label, 'binary')

        metrics = {'train_acc': acc, 'train_loss': loss,
                  'train_precision': prec, 'train_recall': rec}
        self.log_dict(metrics, prog_bar=True)
        return loss #A dictionary. Can include any keys, but must include the key 'loss'

    def training_epoch_end(self, training_step_outputs):
        #If you need to do something with all the outputs of each training_step(), 
        #override the training_epoch_end() method.
        pass
    
    def validation_step(self, batch, batch_idx):
        imgs = batch['imgs']
        label = batch['label']
        logits = self.model(imgs)
        
        loss = self.criterion(logits, label)
        logits = torch.argmax(logits, dim=1)
        acc = FM.accuracy(logits, label, 'binary')
        prec = FM.precision(logits, label, 'binary')
        rec = FM.recall(logits, label, 'binary')
        
        metrics = {'val_acc': acc, 'val_loss': loss,
                  'val_precision': prec, 'val_recall': rec}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        #If you need to do something with all the outputs of each validation_step(), 
        #override the validation_epoch_end() method. 
        #Note that this method is called before training_epoch_end().
        pass #for pred in validation_step_outputs:
    
    def test_step(self, batch, batch_idx):
        pass
    
    def predict_step(self, batch, batch_idx):
        imgs = batch['imgs']
        logits = self.model(imgs)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class ResNetFeat(nn.Module):
    def __init__(self, model):
        super(ResNetFeat, self).__init__()
        # ResNet의 마지막 레이어 이전까지의 부분을 가져오기
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        # GAP를 적용하여 피쳐 추출
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

if __name__ == "__main__":
    os.makedirs("results/", exist_ok=True)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--over_conf', type=float, default=0.9, help='confidence manipulation')
    args = parser.parse_args()
    over_conf = args.over_conf
    print(f"\t[ confidence manipulation ~ >{over_conf} ]")

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
    # print(f"# of nonwet slices, bumps\t: {len(file_list)}, {num_bumps}")

    # True indices of bump labels
    # bump_slice_np = np.array(bump_slice_list)
    labels_bump = np.array(labels_bump)
    idx_normal_bump = labels_bump==0
    idx_nonwet_bump = labels_bump==1
    
    # Custom Dataset & DataLoader #
    in_channels = 1
    batch_size = 120
    test_trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(64)])
    test_set = CustomDataset(bump_slice_list, transform=test_trans, in_channels=in_channels)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=16)

    checkpoint_path = "./weights/resnet18_model.ckpt"
    resnet = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    resnet = resnet.model
    resnet.eval()

    # 2D Counting (Test)
    # print("*** 2D Counting (Test) starts ***")
    row_rec = {"over_num_slice": f"resnet 2d counting - Recall"}
    row_fpr = {"over_num_slice": f"resnet 2d counting - FPR"}
    # Prediction
    cnt = 20
    for over_num_slice in range(0,cnt): # over_num_slice = 4
        # CPU
        with torch.no_grad():
            logits = []
            for inputs in test_loader:
                images = inputs["imgs"]
                outputs = resnet(images) # (N, 2) ; (120,2)
                logits.append(outputs)
            logits = torch.vstack(logits) # (num_bumps x num_slice, 2) ; (num_bumps x 20, 2)
        
        # score (= confidence probability)
        scores = nn.functional.softmax(logits, dim=-1) # (num_bumps x 20, 2)
        
        # confidence manipulation
        pred_conf = []
        for i in range(len(scores)):
            if scores[i,1] > over_conf:
                pred_conf.append(1)
            else:
                pred_conf.append(0)
        pred_conf = np.array(pred_conf) # (num_bumps x 20,)
        
        pred_conf_bump = pred_conf.reshape(-1, num_slice) # (num_bumps, 20)
        pred_conf_bump = pred_conf_bump.sum(axis=-1) > over_num_slice

        acc, rec, fpr = print_metrics(pred_conf_bump, idx_nonwet_bump, idx_normal_bump)
        row_rec[over_num_slice] = round(rec,2)
        row_fpr[over_num_slice] = round(fpr,2)

    decimal_places = len(str(over_conf).split('.')[-1])
    scale = 10 ** decimal_places
    suffix = str(int(over_conf * scale)).zfill(decimal_places)
    file_path = f"results/resnet-counting_{suffix}.xlsx"
    excel_updater = ExcelUpdater(file_path, columns=row_rec.keys())
    excel_updater.add_row(row_rec)
    excel_updater.add_row(row_fpr)
    excel_updater.save()


