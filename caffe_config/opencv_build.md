安装依赖包
```
sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg.dev libtiff4.dev libswscale-dev libjasper-dev
```

#### 切换到opencv文件
```
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..

make
##### make install 需要超级权限
sudo make install
```

#### 修改环境变量
```
sudo vim /etc/ld.so.conf.d/opencv.conf
```

加入`/usr/local/lib`

```
sudo ldconfig
```


```
vim ~/.bashrc
export  PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
source ~/.bashrc
```


#### 参考：
1. https://blog.csdn.net/u013066730/article/details/79411767
2. https://blog.csdn.net/orDream/article/details/84311697
