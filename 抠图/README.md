这个项目为实际在圆盘上模拟伪造的图片，为了进行数据扩充。

saImage.py: 录入图片的文件，直接运行该文件即可，然后输入对应商品的名称即可开始采集图片，输出两个文件夹，一个是原图，另外一个是处理后的图片文件夹

process.py, process1.py: 分别是四周无灯管进行处理的源码（抠图）,四周一根灯管(抠图）
process_new.py: 四周有三根灯管对图像处理的源码(抠图）
combination.py(周围无灯管）, combination_light.py(周围一根灯管）, mix_two_images.py(三根灯管）: 将采集的图片与back.jpg图片融合，等于将商品放入back.jpg中

convert_train.py:批量处理，将所有文件转换

cap_camera.py: 这是进行测试用的，直接接上摄像头进行观察结果, 方便现场调试。

box_process.py: 抠出物体后利用opencv找到物体最小外接矩形，等于标注框。（未完善）

