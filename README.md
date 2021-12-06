# flower_graph
**正規化スペクトラルクラスタリングを用いた花弁のセグメンテーション**  
- [x] CT画像から得られたグラフ構造に対して、正規化スペクトラルクラスタリング
- [x] 細線化画像を用いて、分岐点の検出、除去
- [x] 2次元画像上でクラスタ統合
- [ ] 3次元方向にクラスタ統合
- [ ] 花弁の接触部分の分離

## 環境構築
docker上で、Ubuntu 20.04.2のイメージからコンテナを作成  
`docker run -v /disk024/usrs/naka:/workspace -itd -p 8030:8030 --name naka_flower_ct ubuntu:latest /bin/bash`

* ライブラリのインストール 
```
apt update

apt install python3-pip
pip3 install jupyter notebook
pip3 install pip-review tqdm joblib pandas numpy scipy matplotlib opencv-python scikit-image scikit-learn

apt install wget open3d-python networkx seaborn
apt-get install unzip git
apt-get install -y libgl1-mesa-dev libopencv-dev
```

## 初期設定
adjacency_matrix/InputにCT画像を格納

## ファイル説明