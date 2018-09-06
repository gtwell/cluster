import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import time
from impala.dbapi import connect
import os

def count_labels(centers, labels):
    r1 = pd.Series(labels).value_counts()
    r2 = pd.DataFrame(centers)
    r = pd.concat([r2, r1], axis=1, names=list(range(3)))

    return r
    # r.to_csv('./counts.csv', index=False)


def compute_value(data):

    data = pd.DataFrame(data,
                       columns=['vip', 'R', 'lastyearbuytimes', 'totalamt'])
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
    values = centers[:, 1] + centers[:, 2] - centers[:, 0]

    label_before = kmodel.predict(centers)
    label_after = np.argsort(values)[::-1]

    dicts = dict(zip(label_after, label_before))

    all_labels = pd.Series(kmodel.labels_)
    all_labels.replace(dicts, inplace=True)

    # centers = np.concatenate([kmodel.cluster_centers_, np.expand_dims(kmodel.predict(kmodel.cluster_centers_), axis=1)], axis=1)
    # plot_radar(centers[label_after, :])
    # plot_3d(pd.DataFrame(new_data, columns=['r', 'f', 'm']), all_labels)
    # print(count_labels(centers[label_after, :], all_labels))

    add_label = pd.concat([data, pd.Series(all_labels, index=data.index)], axis=1)
    add_label.columns = list(data.columns) + ['label']
    
    def cal_score(label):
        if label == 0:
            return 6
        elif label == 1:
            return 5
        elif label == 2:
            return 4
        elif label == 3:
            return 3
        elif label == 4:
            return 2
        elif label == 5:
            return 1
    add_label['score'] = add_label.apply(lambda x : cal_score(x.label), axis = 1)
    
    uservalue_data = add_label[['vip','score']]
    uservalue_data.insert(1, 'labelid', 3)
    uservalue_data.to_csv('/home/lilanz/xiaohong/user_profile/user_data/uservalue_output.csv', index=False, header = False)

if __name__=='__main__':
    try:
        conn = connect(host='192.168.35.111', port=10000, database='query', auth_mechanism='PLAIN')
        cur = conn.cursor()
        print(time.ctime(), 'hive connected successfully')
    except:
        RuntimeError('hive connected error')

    try:
        select_sql =  'select vip,r,totalbuytimes,totalamt from userprofilethirdlabelzjb where lastrq is not NULL and totalbuytimes >= 0 and totalamt >= 0'
        cur.execute(select_sql)
        data = cur.fetchall()
        print(time.ctime(), 'export data successfully')
    except:
        print(time.ctime(), 'export data failed')
        
    compute_value(data)
    print(time.ctime(), 'user value information has been computed')
    
    try:
        delete_sql = "hive -database query -e \"truncate table t_sa_userprofile  partition (label_id = 3)\" "
        os.system(delete_sql)
        print(time.ctime(), 'delete old data done')

    except:
        print(time.ctime(), 'delete old data failed')

    try:
            sql = "hive -database query -e \"load data local inpath '/home/lilanz/xiaohong/user_profile/user_data/uservalue_output.csv' into table t_sa_userprofile partition (label_id=3)\""
            os.system(sql)
            print(time.ctime(), 'insert consumer value successfully')
    except:
            print(time.ctime(), 'insert consumer value failed')
        
    
        
