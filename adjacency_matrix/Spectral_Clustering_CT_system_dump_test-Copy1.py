#!/usr/bin/env python
# coding: utf-8

# ## 1枚のCT画像をスペクトラルクラスタリング

# In[ ]:


#!pip3 install scikit-image scikit-learn


# ### ライブラリのインポート

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
#import networkx as nx
import sys
from scipy.sparse import lil_matrix
import numpy.linalg as LA
import datetime
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import inv, eigs, eigsh
from scipy.sparse import diags
#import seaborn as sns
from sklearn.cluster import KMeans
import math
from skimage.draw import line
#from scipy.spatial.distance import pdist
#from scipy.cluster.hierarchy import linkage, cophenet,fcluster,dendrogram
#from multiprocessing import Pool
#from concurrent.futures import ProcessPoolExecutor
#from collections import defaultdict
from skimage.morphology import skeletonize
import random


# ### ラプラシアン行列の作成

# In[ ]:


def laplacian_matrix(network_matrix):
    n_nodes = network_matrix.shape[0]
    
    network_matrix = network_matrix.tocsr()
    
    degree_matrix = diags( np.ravel(network_matrix.sum(axis=1)))
    
    laplacian_matrix = degree_matrix - network_matrix
    
    degree_matrix = degree_matrix.tolil()
    
    D_i = lil_matrix((degree_matrix.shape[0],degree_matrix.shape[1]),dtype='float')
    
    for i in tqdm(range(degree_matrix.shape[0])):
        D_i[i,i]= 1/degree_matrix[i,i]
        
    D_i = D_i.tocsr()
    D_i_s = D_i.sqrt()
    
    L_sym = D_i_s * laplacian_matrix * D_i_s
    
    return L_sym


# ### 固有値計算

# In[ ]:


def eigen_2nd( network_matrix ,k_c):
    
    now = datetime.datetime.now()
    print(now)
    print('ラプラシアン行列作成')
    
    L = laplacian_matrix(network_matrix)
    
    now = datetime.datetime.now()
    print(now)
    print('固有値計算作成')

    values, vectors = eigs(L,k_c,which='SR')
    
    return values, vectors


# In[ ]:


def gabor(img,k,sigma,lam,gam):
    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(0), lam, gam, 0)
    img_0 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(10), lam, gam, 0)
    img_10 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(20), lam, gam, 0)
    img_20 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(30), lam, gam, 0)
    img_30 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(40), lam, gam, 0)
    img_40 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(50), lam, gam, 0)
    img_50 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(60), lam, gam, 0)
    img_60 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(70), lam, gam, 0)
    img_70 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)
    
    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(80), lam, gam, 0)
    img_80 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(90), lam, gam, 0)
    img_90 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(100), lam, gam, 0)
    img_100 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(110), lam, gam, 0)
    img_110 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(120), lam, gam, 0)
    img_120 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)
    
    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(130), lam, gam, 0)
    img_130 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(140), lam, gam, 0)
    img_140 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(150), lam, gam, 0)
    img_150 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(160), lam, gam, 0)
    img_160 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)

    gabor = cv2.getGaborKernel((k,k), sigma, np.radians(170), lam, gam, 0)
    img_170 = cv2.filter2D(img, cv2.CV_32S, gabor,cv2.BORDER_CONSTANT)
    
    return img_0,img_10,img_20,img_30,img_40,img_50,img_60,img_70,img_80,img_90,img_100,img_110,img_120,img_130,img_140,img_150,img_160,img_170


# ### 細線化による分岐点検索

# In[ ]:


def junk(img_skel,img,img_num):
    height,width = img_skel.shape
    temp_j = np.zeros((height,width))
    
    skel_b = img.copy()
    
    for y in range(height):
        for x in range(width):
            if(img_skel[y,x]>0):
                skel_b[y,x] = (255,255,255)
            else:
                skel_b[y,x] = (0,0,0)
    
    for y in range(height):
        for x in range(width):
            if(img_skel[y,x]==255):
                check = 0
                cnt = 1
                temp_l = np.zeros((3,3))
                
                if(img_skel[y-1,x-1]==255):
                    cnt += 1
                    temp_l[0,0] = 1
                if(img_skel[y-1,x]==255):
                    cnt += 1
                    temp_l[0,1] = 1
                if(img_skel[y-1,x+1]==255):
                    cnt += 1
                    temp_l[0,2] = 1
                if(img_skel[y,x-1]==255):
                    cnt += 1
                    temp_l[1,0] = 1
                if(img_skel[y,x+1]==255):
                    cnt += 1
                    temp_l[1,2] = 1
                if(img_skel[y+1,x-1]==255):
                    cnt += 1
                    temp_l[2,0] = 1
                if(img_skel[y+1,x]==255):
                    cnt += 1
                    temp_l[2,1] = 1
                if(img_skel[y+1,x+1]==255):
                    cnt += 1
                    temp_l[2,2] = 1
                
                for y_l in range(3):
                    for x_l in range(3):
                        if(temp_l[y_l,x_l] == 1):
                            if((y_l-1)>=0):
                                if(temp_l[y_l-1,x_l] == 1):
                                    check = 1
                            if((y_l+1)<3):
                                if(temp_l[y_l+1,x_l] == 1):
                                    check = 1
                            if((x_l-1)>=0):
                                if(temp_l[y_l,x_l-1] == 1):
                                    check = 1
                            if((x_l+1)<3):
                                if(temp_l[y_l,x_l+1] == 1):
                                    check = 1

                if(cnt == 2):
                    temp_j[y,x] = 2
                elif((cnt > 3) and (check == 0)):
                    temp_j[y,x] = 1
                elif(cnt>4):
                    temp_j[y,x] = 1
                    
    for y in range(height):
        for x in range(width):
            if(temp_j[y,x]==1):
                cv2.rectangle(img, (x-3, y-3), (x+3, y+3), (0, 0, 255))
                cv2.rectangle(skel_b, (x-3, y-3), (x+3, y+3), (0, 0, 255))
                cv2.rectangle(img_skel, (x-3, y-3), (x+3, y+3), 0, thickness=-1)
            elif(temp_j[y,x]==2):
                cv2.rectangle(img, (x-3, y-3), (x+3, y+3), (0, 255, 0))
                
    cv2.imwrite('Output/Spectral_sys/03_skel-junktion/ORA-{0:03d}_skel-junction.png'.format(img_num),skel_b)
                    
    return temp_j,img,img_skel


# ### 細線化

# In[ ]:


def skel(img_num):

    #Original
    img_col = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num))

    #NLMDを用いた2値化
    img = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num),0)
    img = cv2.fastNlMeansDenoising(img,h=6)
    th_img = np.where(img>50 ,1 ,0)

    # 細線化(スケルトン化) Zha84
    skeleton = skeletonize(th_img)

    skel_b = np.zeros((img.shape[0],img.shape[1]))
    skel_b[skeleton==True] = 255
    cv2.imwrite('Output/Spectral_sys/02_skeleton/ORA-{0:03d}_skeleton.png'.format(img_num),skel_b)

    #細線化画像による分岐点探索
    temp_j,img_j,img_skel = junk(skel_b,img_col,img_num)
    
    cv2.imwrite('Output/Spectral_sys/04_img-junction/ORA-{0:03d}_img-junktion.png'.format(img_num),img_j)
    
    return temp_j,img_skel


# In[ ]:


#def spectral_clusering(img_num):
#num_list = [666,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730]
#num_list = [529,528,527,526,525,524,523,522,521,519,518,517,516,515,514,513,512,511,509,508,507,506,505,504,503,502,501,499,498,497,496,495,494,493,492,491,490]

for img_num in range(584,550,-1):
#for img_num in num_list:
    print('-----------------------------------------')
    print(img_num)

    #ノード数カウント用
    cnt = 1
    
    #クラスタ分割数
    k_c = 200

    now = datetime.datetime.now()
    print(now)
    print('ノード番号付与中')

    #入力画像を取得
    img = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num),0)
    #画像のサイズを取得
    height,width = img.shape

    #入力画像にノンローカルミーンフィルタを適用し2値化
    img_NLMD = cv2.fastNlMeansDenoising(img,h=6)
    print(img_NLMD.shape)
    th_img = np.where(img_NLMD>50 ,255,0)
    
    cv2.imwrite('Output/Spectral_sys/99_NLMD-h6/ORA-{0:03d}_NLMD-h6.png'.format(img_num),img_NLMD)

    #ノード番号を格納する配列を定義
    temp = np.zeros((height,width),dtype = 'i4')

    #閾値以上のピクセルにノード番号付与
    for y in tqdm(range(height)):
        for x in range(width):
            if(th_img[y,x]==255):
                temp[y,x] = cnt
                cnt += 1

    now = datetime.datetime.now()
    print(now)
    print('隣接行列を宣言')

    #スパース行列を使うときは以下
    A = lil_matrix((cnt-1,cnt-1),dtype='float')

    now = datetime.datetime.now()
    print(now)
    print('ガボールフィルタ')

    #パラメータ設定
    k = 30
    sigma = 1.11
    lam = 10
    gam = 0.09
    img_0,img_10,img_20,img_30,img_40,img_50,img_60,img_70,img_80,img_90,img_100,    img_110,img_120,img_130,img_140,img_150,img_160,img_170 = gabor(img_NLMD,k,sigma,lam,gam)

    now = datetime.datetime.now()
    print(now)
    print('隣接行列作成中')

    r_th = 10
    a = 0.001
    b = 0.001

    for y in tqdm(range(50,height-50)):
        for x in range(50,width-50):

            if(temp[y,x]>0):
                node1 = temp[y,x]-1
                for y_t in range(y,y+10):
                    if(y == y_t):
                        for x_t in range(x,x+10):
                            if((temp[y_t,x_t]>0) and (x != x_t)):
                                node2 = temp[y_t,x_t]-1
                                r = math.sqrt((x-x_t)**2 + (y-y_t)**2)
                                if(r < r_th):

                                    esc = 0
                                    flag = 0
                                    i = 0

                                    rr,cc = line(y,x,y_t,x_t)

                                    while(esc==0):
                                        if(th_img[rr[i],cc[i]]==0):
                                            flag = 1
                                            esc = 1
                                        elif(i < rr.shape[0]-1):
                                            i += 1
                                        else:
                                            esc = 1

                                    if(flag==0):

                                        d = math.sqrt((img_0[y,x]-img_0[y_t,x_t])**2+(img_10[y,x]-img_10[y_t,x_t])**2+(img_20[y,x]-img_20[y_t,x_t])**2+                                                      (img_30[y,x]-img_30[y_t,x_t])**2+(img_40[y,x]-img_40[y_t,x_t])**2+(img_50[y,x]-img_50[y_t,x_t])**2+                                                      (img_60[y,x]-img_60[y_t,x_t])**2+(img_70[y,x]-img_70[y_t,x_t])**2+(img_80[y,x]-img_80[y_t,x_t])**2+                                                      (img_90[y,x]-img_90[y_t,x_t])**2+(img_100[y,x]-img_100[y_t,x_t])**2+(img_110[y,x]-img_110[y_t,x_t])**2+                                                      (img_120[y,x]-img_120[y_t,x_t])**2+(img_130[y,x]-img_130[y_t,x_t])**2+(img_140[y,x]-img_140[y_t,x_t])**2+                                                      (img_150[y,x]-img_150[y_t,x_t])**2+(img_160[y,x]-img_160[y_t,x_t])**2)+(img_170[y,x]-img_170[y_t,x_t])**2

                                        W = math.exp(-a*r)+math.exp(-b*d)

                                        A[node1,node2] = W
                                        A[node2,node1] = W
                    else:
                        for x_t in range(x-10,x+10):
                            if(temp[y_t,x_t]>0):
                                node2 = temp[y_t,x_t]-1
                                r = math.sqrt((x-x_t)**2 + (y-y_t)**2)
                                if(r < r_th):

                                    esc = 0
                                    flag = 0
                                    i = 0

                                    rr,cc = line(y,x,y_t,x_t)

                                    while(esc==0):
                                        if(th_img[rr[i],cc[i]]==0):
                                            flag = 1
                                            esc = 1
                                        elif(i < rr.shape[0]-1):
                                            i += 1
                                        else:
                                            esc = 1

                                    if(flag==0):
                                        d = math.sqrt((img_0[y,x]-img_0[y_t,x_t])**2+(img_10[y,x]-img_10[y_t,x_t])**2+(img_20[y,x]-img_20[y_t,x_t])**2+                                                      (img_30[y,x]-img_30[y_t,x_t])**2+(img_40[y,x]-img_40[y_t,x_t])**2+(img_50[y,x]-img_50[y_t,x_t])**2+                                                      (img_60[y,x]-img_60[y_t,x_t])**2+(img_70[y,x]-img_70[y_t,x_t])**2+(img_80[y,x]-img_80[y_t,x_t])**2+                                                      (img_90[y,x]-img_90[y_t,x_t])**2+(img_100[y,x]-img_100[y_t,x_t])**2+(img_110[y,x]-img_110[y_t,x_t])**2+                                                      (img_120[y,x]-img_120[y_t,x_t])**2+(img_130[y,x]-img_130[y_t,x_t])**2+(img_140[y,x]-img_140[y_t,x_t])**2+                                                      (img_150[y,x]-img_150[y_t,x_t])**2+(img_160[y,x]-img_160[y_t,x_t])**2)+(img_170[y,x]-img_170[y_t,x_t])**2

                                        W = math.exp(-a*r)+math.exp(-b*d)

                                        A[node1,node2] = W
                                        A[node2,node1] = W
                                        
    A = csr_matrix(A)

    now = datetime.datetime.now()
    print(now)
    print('固有値、固有ベクトル計算中')

    values ,vectors= eigen_2nd(A,k_c)

    T = np.zeros((vectors.shape[0],vectors.shape[1]))

    vectors = vectors.astype('f8')

    for n in tqdm(range(vectors.shape[0])):
        sum_u = 0
        for k in range(vectors.shape[1]):
            sum_u += vectors[n,k] ** (2)

        sum_u = math.sqrt(sum_u)

        for k in range(vectors.shape[1]):
            T[n,k] = vectors[n,k]/sum_u
            
    now = datetime.datetime.now()
    print(now)
    print('k-meansでクラスタ分割中')
    
    pred = KMeans(n_clusters=k_c).fit_predict(T)
    
    now = datetime.datetime.now()
    print(now)
    print('k-meansのクラスタを保存')
    
    img_col = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num))
    
    colors = []
    for i in range(1, k_c+1):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    for y in range(img_col.shape[0]):
        for x in range(img_col.shape[1]):
            if(temp[y,x]>0):
                i = temp[y,x]-1
                img_col[y,x] = colors[int(pred[i])]

    cv2.imwrite('Output/Spectral_sys/01_k-means/ORA-{0:03d}_k-means-{1:03d}.png'.format(img_num,k_c),img_col)
    
    now = datetime.datetime.now()
    print(now)
    print('細線化による分岐点探索')
    
    j_p,img_skel= skel(img_num)

    now = datetime.datetime.now()
    print(now)
    print('分岐点周辺の同一クラスタを削除')
    
    cluster = np.zeros((height,width))
    del_cluster = np.zeros((height,width))

    for y in tqdm(range(height)):
        for x in range(width):
            if(temp[y,x]>0):
                i = temp[y,x]-1
                cluster[y,x] = pred[i]

    for y in tqdm(range(height)):
        for x in range(width):
            if(j_p[y,x]==1):
                if(cluster[y,x]!=0):
                    p_pred = cluster[y,x]
                    cluster[y,x] = 0
                    del_cluster[y,x] = img_NLMD[y,x]
                    for y_t in range(height):
                        for x_t in range(width):
                            if(p_pred == cluster[y_t,x_t]):
                                cluster[y_t,x_t] = 0
                                del_cluster[y_t,x_t] = img_NLMD[y_t,x_t]

    img_col = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num))

    for y in range(img_col.shape[0]):
        for x in range(img_col.shape[1]):
            if(temp[y,x]>0):
                if(cluster[y,x] != 0):
                    i = temp[y,x]-1
                    img_col[y,x] = colors[int(pred[i])]

    cv2.imwrite('Output/Spectral_sys/05_junk-del/ORA-{0:03d}_k-means-{1:03d}_junk-del.png'.format(img_num,k_c),img_col)

    cv2.imwrite('Output/Spectral_sys/06_junk-only/ORA-{0:03d}_k-means-{1:03d}_junk-only.png'.format(img_num,k_c),del_cluster)
    
    now = datetime.datetime.now()
    print(now)
    print('分岐点を除去した細線化をラベリング')
    
    img_col_skel = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num))
    skel_col = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num))

    img_skel = img_skel.astype('u1')

    nLabels, labelImages_skel, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(img_skel, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

    colors = []

    for i in range(1, n + 1):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    for y in range(height):
        for x in range(width):
            if(labelImages_skel[y,x]>0):
                img_col_skel[y,x]=colors[int(labelImages_skel[y,x])]
                skel_col[y,x]=colors[int(labelImages_skel[y,x])]
            else:
                skel_col[y,x]=(0,0,0)

    cv2.imwrite('Output/Spectral_sys/07-1_skel-label_onIMG/ORA-{0:03d}_skel-label_onIMG.png'.format(img_num),img_col_skel)
    cv2.imwrite('Output/Spectral_sys/07-2_skel-label/ORA-{0:03d}_skel-label.png'.format(img_num),skel_col)
    
    now = datetime.datetime.now()
    print(now)
    print('分岐点周辺を削除した画像を2値化')
    
    th_img_clu = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num),0)

    for y in range(height):
        for x in range(width):
            if(cluster[y,x]>0):
                th_img_clu[y,x]=255
            else:
                th_img_clu[y,x]=0

    cv2.imwrite('Output/Spectral_sys/08_junk-del-th/ORA-{0:03d}_k-means-{1:03d}_junk-del-th.png'.format(img_num,k_c),th_img_clu)

    
    now = datetime.datetime.now()
    print(now)
    print('分岐点周辺を削除した画像をラベリング')
    
    img_col = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num))

    # ラベリング処理
    nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(th_img_clu, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

    # オブジェクト情報を項目別に抽出
    n = nLabels - 1

    label_fix = np.zeros(nLabels)

    # オブジェクト情報を利用して15ピクセル以下のラベルを削除
    for i in range(1,nLabels):

        size = data[i,4]
        if(size<15):
            label_fix[i]=0
        else:
            label_fix[i]=i

    #色情報の定義
    colors = []    

    for i in range(1, nLabels+1):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    #2値化用の配列
    label_th = np.zeros((height,width))

    #ラベリングの色を付与・2値化
    for y in range(height):
        for x in range(width):
            if((labelImages[y,x]>0)and(label_fix[labelImages[y,x]]>0)):
                img_col[y,x]=colors[int(label_fix[int(labelImages[y,x])])]
                label_th[y,x]=255
            else:
                cluster[y,x]=0

    #ラベリング結果を保存
    cv2.imwrite('Output/Spectral_sys/09_junk-del-label/ORA-{0:03d}_k-means-{1:03d}_junk-del-label.png'.format(img_num,k_c),img_col)
    #ノイズ除去した画像を保存
    cv2.imwrite('Output/Spectral_sys/10_junk-del-label-th/ORA-{0:03d}_k-means-{1:03d}_junk-del-label-th.png'.format(img_num,k_c),label_th)
    
    now = datetime.datetime.now()
    print(now)
    print('ラベリングを統合（細線化を用いて）')
    
    #クラスタ統合用
    c_num = np.zeros((201))

    #細線上を辿り、1つの細線に属しているクラスタを1つのクラスタに統合
    for y in tqdm(range(height)):
        for x in range(width):
            if((img_skel[y,x]>0)and(cluster[y,x]>0)):
                c_num[int(cluster[y,x])] = labelImages_skel[y,x]

    #統合後のクラスタを格納
    cluster_fix = np.zeros((height,width))

    for y in tqdm(range(height)):
        for x in range(width):
            if(cluster[y,x]>0):
                cluster_fix[y,x]= c_num[int(cluster[y,x])]

    #統合後のクラスタを保存
    img_col = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num))

    colors = []
    for i in range(1, 600):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    for y in range(height):
        for x in range(width):
            if(cluster_fix[y,x]>0):
                img_col[y,x]=colors[int(cluster_fix[y,x])]

    cv2.imwrite('Output/Spectral_sys/11_junk-del-label-inte/ORA-{0:03d}_k-means-{1:03d}_junk-del-label-inte.png'.format(img_num,k_c),img_col)
    
    now = datetime.datetime.now()
    print(now)
    print('同一クラスタが離れている場合はクラスタを分離')
    
    cnt_cl = k_c
    chk_cl = np.zeros(600)

    for y in tqdm(range(height)):
        for x in range(width):
            if((cluster_fix[y,x]>0)and(chk_cl[int(cluster_fix[y,x])]==0)):

                temp_cl = np.zeros((height,width))

                for y_t in range(height):
                    for x_t in range(width):
                        if(cluster_fix[y,x]==cluster_fix[y_t,x_t]):
                            temp_cl[y_t,x_t] = 255

                temp_cl = temp_cl.astype('u1')

                nLabels_temp, labelImages_temp, data_temp, center_temp = cv2.connectedComponentsWithStatsWithAlgorithm(temp_cl, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

                chk_cl[int(cluster_fix[y,x])]=1
                if(nLabels_temp>2):
                    for y_t in range(height):
                        for x_t in range(width):
                            if(labelImages_temp[y_t,x_t]>0):
                                cluster_fix[y_t,x_t] = cnt_cl+labelImages_temp[y_t,x_t]
                    cnt_cl += nLabels_temp-1

    c_num_fix = np.zeros(600)

    for y in tqdm(range(height)):
        for x in range(width):
            if((img_skel[y,x]>0)and(cluster_fix[y,x]>0)):
                c_num_fix[int(cluster_fix[y,x])] = labelImages_skel[y,x]

    cluster_fix2 = np.zeros((height,width))

    for y in tqdm(range(height)):
        for x in range(width):
            if(cluster[y,x]>0):
                cluster_fix2[y,x]= c_num_fix[int(cluster_fix[y,x])]

    img_col = cv2.imread('../../flower_CT_photo/ORA/[vg-data] ORA/volume_1/ORA-{0:03d}.tif'.format(img_num))

    colors = []
    for i in range(1, 600):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    for y in range(height):
        for x in range(width):
            if(cluster_fix2[y,x]>0):
                img_col[y,x]=colors[int(cluster_fix2[y,x])]

    cv2.imwrite('Output/Spectral_sys/12_junk-del-label-inte-sp/ORA-{0:03d}_k-means-{1:03d}_junk-del-label-inte_sp.png'.format(img_num,k_c),img_col)


# In[ ]:




