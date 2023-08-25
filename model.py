# モジュールの準備
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torchsummary
from torchsummary import summary
import torchmetrics
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import StepLR

# 学習済みネットワークの利用
from torchvision.models import resnet34


# 前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ネットワーク
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(*list(resnet34(pretrained=True).children())[:-1])
        self.fc_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 7)


    def forward(self, x):
        h = self.feature(x)
        h = h.view(h.size(0), -1)
        h = self.fc_bn(h)
        h = self.dropout(h)
        h = self.fc(h)
        return h


    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t), on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        x = batch
        y = self.forward(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer
