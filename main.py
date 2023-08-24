# 基本ライブラリ
import streamlit as st

import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from model import Net


#メインパネル_アプリのタイトル
st.title('What movie posters tell　〜映画ポスターの分類〜')
st.write('Recognize the genre of a movies from the its posters. Please upload an image of poster in jpg, png or jpeg format.You would have a 50% chance of getting it right!')
st.write('映画ポスターから、その映画のジャンルを判定します。画像形式でアップロードしてください。50％くらいの確率で正解するはず!?')


# モデル
# 学習済みモデルの読み込み
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "movie_poster_R1.pt"
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 画像のアップロードと表示
uploaded_file = st.file_uploader('Upload an image of movie poster', type=['jpg', 'png', 'jpeg'])

# 前処理
def preprocess_image(image: Image.Image) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(image)
    return img_tensor.unsqueeze(0)  # バッチ次元を追加


if uploaded_file:
    # 画像の表示
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption='Uploaded Poster Image.', use_column_width=True)

    # 画像をモデルが受け入れる形式に変換
    img_array = np.array(image.resize((240, 240))) / 255.0
    img_array = img_array[np.newaxis, ...]

    #tensor_img = torch.FloatTensor(img_array).permute(0, 3, 1, 2)
    tensor_img = preprocess_image(image)
    with torch.no_grad():
        logits = model(tensor_img)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
       #predictions = model(tensor_img)
       #predicted_class = np.argmax(predictions, axis=1)[0].item()

    # カテゴリに名前を付与
    name = {
        0: 'Action アクション',
        1: 'Adventure アドベンチャー',
        2: 'Comedy コメディ',
        3: 'Suspense サスペンス・クライム',
        4: 'Drama ヒューマンドラマ',
        5: 'Family ファミリームービー',
        6: 'Fantasy ファンタジー',
        7: 'Horror ホラー',
        8: 'Mystery ミステリー',
        9: 'Musical ミュージカル映画',
        10: 'Romance ラブストーリー',
        11: 'War 戦争映画',
    }

    # 予測結果の出力
    st.write('## Result')
    st.write(f'The movie genre of this uploaded poster is probably... {name[predicted_class]}!')
