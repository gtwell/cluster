#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 10:59
# @Author  : gtwell


import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
import ipdb

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


def plot_3d(data, all_labels):
    """args:
    data: r f m three dimension vector
    labels_: predicted labels
    return a 3D plot"""

    add_label = pd.concat([data, pd.Series(all_labels, index=data.index)], axis=1)
    r = add_label.iloc[:, 0]
    f = add_label.iloc[:, 1]
    m = add_label.iloc[:, 2]
    label = add_label.iloc[:, 3]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, f, m, c=label)
    ax.set_xlabel('距离上一次购买天数')
    ax.set_ylabel('频率')
    ax.set_zlabel('总金额')
    ax.view_init(azim=235)
    plt.show()


def plot_radar(centers):
    """args:
    centers: return from train_kmeans"""

    # 使用ggplot的绘图风格
    plt.style.use('ggplot')

    # 构造数据
    values = []
    for i in range(6):
        values.append(centers[i])

    feature = ['距离上一次购买天数','频率','总金额']

    N = len(values[0])
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)
    # 为了使雷达图一圈封闭起来，需要下面的步骤
    for i in range(6):
        values[i] = np.concatenate((values[i],[values[i][0]]))

    angles=np.concatenate((angles,[angles[0]]))

    # 绘图
    fig=plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for i in range(6):
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


def count_labels(centers, labels):
    r1 = pd.Series(labels).value_counts()
    r2 = pd.DataFrame(centers)
    r = pd.concat([r2, r1], axis=1, names=list(range(3)))

    return r
    # r.to_csv('./counts.csv', index=False)



def cal_users_value(dataPath, outputPath):

    data = pd.read_csv(dataPath,
                       names=['vip', 'R', 'lastyearbuytimes', 'totalamt'])
    data = data[data['lastyearbuytimes'] != -1]
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    new_data = data.loc[:, ['R', 'lastyearbuytimes', 'totalamt']]
    #new_data = new_data.sample(frac=1).reset_index(drop=True)
    scaler = preprocessing.StandardScaler().fit(new_data)
    new_data = scaler.transform(new_data)

    kmodel = KMeans(n_clusters = 6, random_state=1)
    kmodel.fit(new_data)

    np.set_printoptions(suppress=True)
    centers = kmodel.cluster_centers_
    print(centers)
    values = centers[:, 1] + centers[:, 2] - centers[:, 0]
    print('values:', values)
    print(np.argsort(values))

    label_before = kmodel.predict(centers)
    label_after = np.argsort(values)[::-1]
    print('第三行:{}'.format(scaler.transform(np.expand_dims(data.iloc[2, 1:], axis=0))))
    print('第三行:{}'.format(kmodel.predict(scaler.transform(np.expand_dims(data.iloc[2, 1:], axis=0)))))

    print('label before:', label_before)
    print('label after:', label_after)

    dicts = dict(zip(label_after, label_before))

    all_labels = pd.Series(kmodel.labels_)
    print('before:', all_labels.head())
    all_labels.replace(dicts, inplace=True)
    print('after:', all_labels.head())

    # centers = np.concatenate([kmodel.cluster_centers_, np.expand_dims(kmodel.predict(kmodel.cluster_centers_), axis=1)], axis=1)
    # plot_radar(centers[label_after, :])
    # plot_3d(pd.DataFrame(new_data, columns=['r', 'f', 'm']), all_labels)
    print(count_labels(centers[label_after, :], all_labels))

    add_label = pd.concat([data, pd.Series(all_labels, index=data.index)], axis=1)
    add_label.columns = list(data.columns) + ['label']
    add_label.to_csv(outputPath, index=False)


if __name__ == '__main__':
    input_file = './data.csv'
    output_file = './out.csv'
    cal_users_value(input_file, outputPath=output_file)

