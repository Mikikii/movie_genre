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
import torchmetrics
from torchmetrics.functional import accuracy
import torchsummary
from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger

# 学習済みネットワークの利用
from torchvision.models import resnet18


import gdown
import zipfile

filepath = 'https://drive.google.com/file/d/1T6CljHiL3nVowfEyYAKPRCpcPHM27KJH/view'
filepath_converted = filepath.replace('file/d/' , 'uc?id=').replace('/view', '')
filename = 'dog_cat_data.zip'

gdown.download(filepath_converted, filename, quiet=False)

with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall()


# 前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #ResNet18 0~255 → 0~1
])

# 前処理とアノテーション（catに０、dogに１のラベル付け）
train_val = datasets.ImageFolder('dog_cat_data/train', transform)


# train と val に分割
n_train = 240
n_val = 60
train, val = torch.utils.data.random_split(train_val, [n_train, n_val])


# バッチサイズの定義
batch_size = 32

# Data Loader を定義
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)


# ResNet を特徴抽出機として使用
feature = resnet18(pretrained=True)
feature = nn.Sequential(*list(feature.children())[:-2])


# 全結合層
fc = nn.Linear(25088, 2)
# h = fc(h)


# ネットワーク
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # ResNet18 を特徴抽出機として使用するためにインスタンス化
        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True) #prog_bar=True
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
        return y


    def test_epoch_end(self, outputs):
        all_outputs = torch.cat(outputs, 0)
        return {"logits": all_outputs}



    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer


def infer(model, dataloader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch
            y = model(x)
            all_outputs.append(y)

    all_outputs = torch.cat(all_outputs, 0)
    return all_outputs


# 学習の実行
pl.seed_everything(0)
net = Net()
logger = CSVLogger(save_dir='logs', name='my_exp')
trainer = pl.Trainer(max_epochs=20, gpus=None, deterministic=False, logger=logger)
trainer.fit(net, train_loader, val_loader)


# 学習済みモデルの保存
torch.save(net.state_dict(), 'dog_cat.pth')

# ネットワークの準備
net = Net().cpu().eval()

# 重みの読み込み
net.load_state_dict(torch.load('dog_cat.pth', map_location=torch.device('cpu')))
