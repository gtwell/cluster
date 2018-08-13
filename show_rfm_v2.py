# -*- coding: utf-8 -*-
"""
@author: gtwell
"""

import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import ipdb
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
import argparse


mpl.rcParams['font.sans-serif'] = ['FangSong'] 
mpl.rcParams['axes.unicode_minus'] = False 


def load_dataset(dataPath):

    data = pd.read_csv(dataPath,
                       names=['vip', 'L', 'firstrq', 'lastrq','R','totalbuytimes',
                              'lastyearbuytimes', 'totalamt', 'lastyearamt'])
    data = data[~data['firstrq'].str.contains('N')]
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    new_data = data.loc[:, ['R', 'lastyearbuytimes', 'totalamt']]
    new_data = new_data.sample(frac=1).reset_index(drop=True)
    # 字符转数值型
    data_copy = new_data.copy().apply(pd.to_numeric)
    new_data = preprocessing.scale(new_data, axis=0)
    new_data = pd.DataFrame(new_data, columns=['r', 'f', 'm'])
    # new_data.columns = ['r', 'f', 'm']
    print(data_copy.head(3))
    print(new_data.head(3))

    out = (new_data, data_copy)

    return out


def train_kmeans(data, K):
    """args:
    data: train set
    K: k-means the K
    return:
    cluster centers"""

    kmodel = KMeans(n_clusters = K)
    kmodel.fit(data)
    # print(kmodel.cluster_centers_)
    # print(len(kmodel.labels_))
    out = (kmodel.cluster_centers_, kmodel.labels_)

    return out


def save_file(file_name, train, centers, kmodel_labels):
    r1 = pd.Series(kmodel_labels).value_counts()
    r2 = pd.DataFrame(centers)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(train.columns) + ['类别数目']
    r.to_csv('./counts.csv', index=False)

    add_label = pd.concat([train, pd.Series(kmodel_labels, index=train.index)], axis=1)
    add_label.columns = list(train.columns) + ['聚类类别']
    add_label.to_csv(file_name, index=False)


def plot_radar(centers, K):
    """args:
    centers: return from train_kmeans"""

    # 使用ggplot的绘图风格
    plt.style.use('ggplot')

    # 构造数据
    values = []
    for i in range(K):
        values.append(centers[i])

    feature = ['距离上一次购买天数','频率','总金额']

    N = len(values[0])
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)
    # 为了使雷达图一圈封闭起来，需要下面的步骤
    for i in range(K):
        values[i] = np.concatenate((values[i],[values[i][0]]))

    angles=np.concatenate((angles,[angles[0]]))

    # 绘图
    fig=plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for i in range(K):
        # 绘制折线图
        ax.plot(angles, values[i], 'o-', linewidth=2, label = '客户{}'.format(i))
        # 填充颜色
        ax.fill(angles, values[i], alpha=0.25)

    # 添加每个特征的标签
    ax.set_thetagrids(angles * 180/np.pi, feature)
    # 设置雷达图的范围
    # ax.set_ylim(0,5)
    # 添加标题
    plt.title('不同客户聚类')

    # 添加网格线
    ax.grid(True)
    # 设置图例
    plt.legend(loc = 'best')
    # 显示图形
    plt.show()


def plot_3d(train, kmodel_label):
    """args:
    train: r f m three dimension vector
    kmodel.labels_: predicted labels
    return a 3D plot"""

    add_label = pd.concat([train, pd.Series(kmodel_label, index=train.index)], axis=1)
    r = add_label.iloc[:, 0]
    f = add_label.iloc[:, 1]
    m = add_label.iloc[:, 2]
    print(r.head(5))
    print(f.head(5))
    print(m.head(5))
    label = add_label.iloc[:, 3]
    print(label.head(5))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, f, m, c=label)
    ax.set_xlabel('距离上一次购买天数')
    ax.set_ylabel('频率')
    ax.set_zlabel('总金额')
    ax.view_init(azim=235)
    plt.show()


if __name__ == '__main__':
    input_file = './data.csv'
    K = 5
    parser = argparse.ArgumentParser("kmeans to different customs")
    parser.add_argument('--input_file', type=str, default=input_file, help="the data from database")
    parser.add_argument('--K', type=int, default=K, help='the K in kmeans algorithm')
    args = parser.parse_args()

    train_data, train_data_orig = load_dataset(args.input_file)
    print('data_length is:\n{}'.format(len(train_data)))
    centers, kmodel_labels = train_kmeans(train_data, args.K)
    np.set_printoptions(suppress=True)
    print('centers are:\n{}'.format(centers))
    # save_file('./add_label.csv', train_data_orig, centers, kmodel_labels)
    plot_radar(centers, args.K)
    # plot_3d(train_data, kmodel_labels)
    plot_3d(train_data_orig, kmodel_labels)
