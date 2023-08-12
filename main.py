# 基本ライブラリ
import streamlit as st

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

from PIL import Image
import numpy as np

from model import Net


#メインパネル_アプリのタイトル
st.title('What movie posters tell　〜映画ポスターの分類〜')
st.write('アプリの自己紹介と説明書')


# モデル

# 前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #ResNet18 0~255 → 0~1
])


# ベースモデルとしてresnet18を読み込み
#base_model = resnet18(pretrained=True)
#base_model.trainable = True


# 学習済みモデルの読み込み
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "dog_cat.pth"
    model = Net(pl.LightningModule)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# 画像のアップロードと表示
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 画像の表示
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # 推論の実行
    st.write('Predicting...')

    # 画像をモデルが受け入れる形式に変換
    img_array = np.array(image.resize((240, 240))) / 255.0
    img_array = img_array[np.newaxis, ...]

    tensor_img = torch.FloatTensor(img_array).permute(0, 3, 1, 2) 
    with torch.no_grad():
        predictions = model(tensor_img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    #カテゴリに名前を付与
    number_to_name = {
    0: 'cat🐱',
    1: 'dog🐶',
    }

    # 予測結果の出力
    st.write('## Result')
    st.write('This Uploaded Image is probabily ',str(name[0]),'!')


# アプリを実行
if __name__ == '__main__':
    st.run()