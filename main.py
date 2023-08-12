# 基本ライブラリ
import streamlit as st

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
import torchsummary
from torchsummary import summary
from pytorch_lightning.loggers import CSVLogger

from PIL import Image

from model import Net


#メインパネル_アプリのタイトル
st.title('What movie posters tell　〜映画ポスターの分類〜')
st.write('アプリの自己紹介と説明書')


# モデル
# 学習済みモデルの読み込み
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "dog_cat.pth"
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
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
    predicted_class = np.argmax(predictions, axis=1)[0].item()

    # カテゴリに名前を付与
    name = {
        0: 'cat🐱',
        1: 'dog🐶',
    }

    #カテゴリに名前を付与
    #if predicted_class == 0:
     #   pred_name = 'cat 🐱',
    #elif predicted_class == 1:
     #   pred_name = 'dog 🐶'
    

    # 予測結果の出力
    st.write('## Result')
    st.write(f'This Uploaded Image is probably a {name[predicted_class]}!')
