#### 安装依赖包
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install git cmake build-essential
```

```
make all -j8
make pycaffe
make test -j8
###### 可选
make runtest -j8

###### 重新make需要进行
make clean
```

#### 参考
https://blog.csdn.net/qq_31347869/article/details/89469001
