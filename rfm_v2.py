import pandas as pd
import datetime
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

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


def load_dataset(input_file):
    """args:
    file: input file
    return:
    x_train: train set
    y_train: test set
    x_train_copy: to show the original data in the plot3D"""

    # origin_data = pd.read_csv(input_file, names=['vip', 'jdrq', 'lastdays', 'buytimes', 'totalamt'])
    origin_data = pd.read_csv(input_file, names=['vip', 'lastdays', 'totalamt', 'buytimes'], nrows=30000)
    now = datetime.datetime.now()
    origin_data['lastdays'] = now - pd.to_datetime(origin_data['lastdays'])
    origin_data['lastdays'] = [i.days for i in origin_data['lastdays']]
    origin_data = origin_data.dropna(axis=0)
    # print(origin_data[origin_data.isnull().values==True])

    # origin_data = origin_data.loc[:200000, ['lastdays', 'buytimes', 'totalamt']]

    origin_data = origin_data.loc[:20000, ['lastdays', 'totalamt', 'buytimes']]
    x_train, y_train = train_test_split(origin_data, test_size=0.2, random_state=0)
    x_train_copy = x_train.copy()
    y_train_copy = y_train.copy()

    # 均值归一化
    # x_train = preprocessing.scale(x_train)

    # 最大最小值归一化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_train = min_max_scaler.fit_transform(x_train)

    # 鲁棒归一化(离群值)
    x_train = preprocessing.robust_scale(x_train)

    x_train = pd.DataFrame(x_train)
    x_train.columns=['r', 'f', 'm']

    out = (x_train, y_train, x_train_copy)

    return out


def train_kmeans(data, K):
    """args:
    data: train set
    K: k-means the K
    return:
    cluster centers"""
    
    kmodel = MiniBatchKMeans(n_clusters = K, batch_size=1000)
    kmodel.fit(data)
    # print(kmodel.cluster_centers_)
    # print(len(kmodel.labels_))
    out = (kmodel.cluster_centers_, kmodel)

    return out


def save_file(file_name, x_train, kmodel):
    r1 = pd.Series(kmodel.labels_).value_counts()
    r2 = pd.DataFrame(kmodel.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(x_train.columns) + ['类别数目']
    r.to_csv('./counts.csv', index=False)

    add_label = pd.concat([x_train, pd.Series(kmodel.labels_, index=x_train.index)], axis=1)
    add_label.columns = list(x_train.columns) + ['聚类类别']
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


def plot_3d(x_train, kmodel):
    """args:
    x_train: r f m three dimension vector
    kmodel.labels_: predicted labels
    return a 3D plot"""

    add_label = pd.concat([x_train, pd.Series(kmodel.labels_, index=x_train.index)], axis=1)
    r = add_label.iloc[:, 0]
    f = add_label.iloc[:, 1]
    m = add_label.iloc[:, 2]
    print(r.head(10))
    print(f.head(10))
    print(m.head(10))
    label = add_label.iloc[:, 3]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, f, m, c=label)
    ax.set_xlabel('距离上一次购买天数')
    ax.set_ylabel('总金额')
    ax.set_zlabel('频率')
    ax.view_init(azim=235)
    plt.show()



if __name__ == '__main__':
    input_file = '../from2015.csv'
    K = 5 
    parser = argparse.ArgumentParser("kmeans to different customs")
    parser.add_argument('--input_file', type=str, default=input_file, help="the data from database")
    parser.add_argument('--K', type=int, default=K, help='the K in kmeans algorithm')
    args = parser.parse_args()
    train_data, test_data, train_data_orig = load_dataset(args.input_file)
    print(len(train_data))
    centers, kmodel = train_kmeans(train_data, args.K)
    np.set_printoptions(suppress=True)
    # save_file('./add_label.csv', train_data, kmodel) 
    # plot_radar(centers, args.K)
    plot_3d(train_data_orig, kmodel)
