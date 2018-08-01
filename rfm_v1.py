import pandas as pd
import datetime
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import ipdb
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def normalize(d):
    # d is a (n x dimension) np array
    d -= np.mean(d, axis=0)
    d /= np.std(d, axis=0)
    d /= np.max(d, axis=0)
    return d


def load_dataset(input_file):
    """args:
    file: input file
    return:
    x_train: train set
    y_train: test set"""

    origin_data = pd.read_csv(input_file, names=['vip', 'jdrq', 'lastrq', 'buytimes', 'totalamt'])
    now = datetime.datetime.now()
    origin_data['jdrq'] = now - pd.to_datetime(origin_data['jdrq'])
    origin_data['jdrq'] = [i.days for i in origin_data['jdrq']]

    origin_data['jdrq'] = normalize(origin_data['jdrq'])
    origin_data['buytimes'] = normalize(origin_data['buytimes'])
    origin_data['totalamt'] = normalize(origin_data['totalamt'])


    rfm_data = origin_data.loc[:100000, ['jdrq', 'buytimes', 'totalamt']]
    rfm_data.columns=['r', 'f', 'm']
    rfm_data = rfm_data.dropna(axis=0)
    # rfm_data_bn = (rfm_data - rfm_data.mean(axis=0))/(rfm_data.std(axis=0))
    x_train, y_train = train_test_split(rfm_data, test_size=0.2, random_state=0)
    # print(x_train[x_train.isnull().values==True])

    return (x_train, y_train)


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

    feature = ['Recency','Frequency','Monetary']

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
        ax.plot(angles, values[i], 'o-', linewidth=2, label = 'guest {}'.format(i))
        # 填充颜色
        ax.fill(angles, values[i], alpha=0.25)

    # 添加每个特征的标签
    ax.set_thetagrids(angles * 180/np.pi, feature)
    # 设置雷达图的范围
    # ax.set_ylim(0,5)
    # 添加标题
    plt.title('different guest')

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
    ax.set_xlabel('Recence')
    ax.set_ylabel('Frequence')
    ax.set_zlabel('Monetary')
    ax.view_init(azim=235)
    plt.show()



if __name__ == '__main__':
    input_file = './year.csv'
    K = 6 
    train_data, test_data = load_dataset(input_file)
    print(len(train_data))
    centers, kmodel = train_kmeans(train_data, K)
    np.set_printoptions(suppress=True)
    save_file('./add_label.csv', train_data, kmodel) 
    # plot_radar(centers, K)
    plot_3d(train_data, kmodel)
