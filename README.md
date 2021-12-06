# flower_graph
**正規化スペクトラルクラスタリングを用いた花弁のセグメンテーション**

## 環境構築
docker上で、Ubuntu 20.04.2のイメージからコンテナを作成  
'docker run -v /disk024/usrs/naka:/workspace -itd -p 8030:8030 --name naka_flower_ct ubuntu:latest /bin/bash'

* ライブラリのインストール
    apt update
    apt install python3-pip
    pip3 install jupyter notebook
    pip3 install pip-review tqdm joblib pandas numpy scipy matplotlib opencv-python scikit-image scikit-learn
    apt install wget
    apt-get install unzip
    apt-get install git
    apt-get install -y libgl1-mesa-dev
    apt-get install -y libopencv-dev
    pip3 install open3d-python

    pip3 install networkx seaborn

## 初期設定
adjacency_matrix/InputにCT画像を格納

## ファイル説明