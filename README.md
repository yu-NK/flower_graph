# 正規化スペクトラルクラスタリングを用いた花弁のセグメンテーション
## 進捗状況  
- [x] CT画像から得られたグラフ構造に対して、正規化スペクトラルクラスタリング
- [x] 細線化画像を用いて、分岐点の検出と花弁の接触部分の除去
- [x] 2次元画像上で花弁の接触部分以外をクラスタ統合
- [x] 3次元方向にクラスタ統合
- [ ] 花弁の接触部分の分離

## 環境構築
* docker上で、Ubuntu 20.04.2のイメージからコンテナを作成  
`docker run -v /disk024/usrs/naka:/workspace -itd -p 8030:8030 --name naka_flower_ct ubuntu:latest /bin/bash`

* Jupyter Notebookにより作成

* ライブラリのインストール 
```
apt update

apt install python3-pip
pip3 install jupyter notebook
pip3 install pip-review tqdm joblib pandas numpy scipy matplotlib opencv-python scikit-image scikit-learn

apt install wget open3d-python networkx seaborn
apt-get install unzip git vim
apt-get install -y libgl1-mesa-dev libopencv-dev
```

## 初期設定
`adjacency_matrix/Input`にCT画像を格納

## ファイル説明
### ファイル構造
```
adjacency_matrix
├── Input
|   ├── ORA-PNG
│   ├── ORA-hand
│   └── ORA-part
|
├── Output
|   ├── Spectral_sys
|   ├── Spectral_sys_Coronal
|   ├── 3D_inte
|   ├── junk_reserch
│   ├── ...
|
├── dataset
|   ├── annotation
|   ├── anootation_NLMD
|   ├── annotation_part
│   └── test
|
└── reference
    ├── laplacian_matrix_ans.ipynb
    └── Tools.ipynb
```
### adjacency_matrix/Input
CT画像をここに格納

### adjacency_matrix/Output
出力結果をここに出力

### adjacency_matrix/reference
* `laplacian_matrix_ans.ipynb`は内海先生からいただいた第二固有値と第二固有ベクトルを計算するコード
* `Tools.ipynb`は画像処理で使うツールをまとめている