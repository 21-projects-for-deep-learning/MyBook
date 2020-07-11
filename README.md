# 深度学习训练营 - 21天实战
## 书名： 深度学习训练营 - 21天实战
## 基于Tensorflow + Keras + Scikit-learn框架编写而来的21个实战项目

Github地址：https://github.com/21-projects-for-deep-learning/MyBook

![京东图书：深度学习训练营](https://img13.360buyimg.com/n1/jfs/t1/121507/30/3245/261556/5ecf50a7E51da8c88/9c429e8719022f69.jpg)

#### 本书所有的代码集合 (大约27MB)
链接: https://pan.baidu.com/s/1C6waPKRlRfQ_Dj8MDatdEg 提取码: eu3k

如果百度网盘不能下载，可以直接下载本项目，其中[Python Code.zip](https://github.com/21-projects-for-deep-learning/MyBook/blob/master/Python%20Code.zip)就是所有的源代码


### 宋体的字体下载
链接:https://pan.baidu.com/s/1TThtFKeeGX6VjquwABQWOw  密码:y6u7

<hr/>
<br/>

### 书中数据集和相关链接

#### 1.1
数据集链接：https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html 

#### 6.3.4
`plot()`方法中的marker参数表示绘制的线的样式，全部的样式可以在以下链接查看：https://matplotlib.org/api/markers_api.html

#### 8.1
平行语料库的数据集下载地址有以下两个
-	http://www.manythings.org/anki/
-	http://www.statmt.org/europarl/

#### 9.1
MNIST数据集下载：http://yann.lecun.com/exdb/mnist/

#### 10.1
狗狗数据集：http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

#### 10.4.1
下载狗狗的数据集
```python
dataset_path = tf.keras.utils.get_file("Images", 
                  "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", 
                  untar=True)
```

#### 11.1
人脸数据集：http://vis-www.cs.umass.edu/lfw/lfw.tgz
```
# -O参数表示远程文件名，这里就是一个下载地址
$curl -O http://vis-www.cs.umass.edu/lfw/lfw.tgz
```

#### 11.1.3
人脸检测模型，下载地址是：https://github.com/opencv/opencv/tree/master/data/haarcascades

#### 11.2.1  
下载代码仓库
```
git clone https://github.com/21-projects-for-deep-learning/facenet
```
预训练模型地址：https://pan.baidu.com/s/1mWyoy3AmwRaIuco6XlxWpQ

#### 11.4.1
方式2安装，手动安装，所以要下载dlib的git仓库。
```
git clone https://github.com/davisking/dlib.git
```

#### 13.1.2
下该TensorFlow的实现版本的代码库，代码如下：
```
git clone https://github.com/21-projects-for-deep-learning/tf-pose-estimation.git
```

#### 13.2.2
Keras的多人姿态实时评估
```
git clone https://github.com/21-projects-for-deep-learning/keras_Realtime_Multi-Person_Pose_Estimation.git
```

#### 14.1
病理图像数据集下载地址
```
https://challenge.kitware.com/#challenges
```

#### 14.3.1
TensorFlow迁移学习实现分类
```
git clone https://github.com/21-projects-for-deep-learning/Simple_Transfer_Learning.git
```

#### 15.3.1
1.下载tensorflow/models
```
git clone https://github.com/tensorflow/models.git
```
2.protoc下载地址：https://github.com/google/protobuf/releases。

#### 16.1.2
本章所需要的代码库都在image2text的项目里，通过git clone可以将其克隆下来，代码如下：
```
git clone https://github.com/21-projects-for-deep-learning/image2text.git
```
image2text库的代码是从Google的models项目中迁移出来的，然后进行了一些本章讲解时的相应调整。
数据集是完整的取自val2017，然后拆分它为训练集和验证集。通过curl -O命令加上文件地址，就可以下载，命令如下：
```
curl -O http://images.cocodataset.org/zips/val2017.zip
```
解压val2017.zip文件通过unzip命令，如下：
```
unzip val2017.zip
```
下载标注文件（Annotations）也通过curl -O加上文件地址，命令如下：
```
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

#### 16.2.4
下载预训练模型Inception v3
```
curl -O http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
```

#### 19.1.2
MNIST图像数据集下载
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz


#### 19.2.1
LFW（Labeled Faces in the Wild）官方网站上下载数据集
http://vis-www.cs.umass.edu/lfw/lfw.tgz


#### 20.1.1
效果预览地址：http://waifu2x.udp.jp/

LFW数据集下载页面
http://vis-www.cs.umass.edu/lfw/

SRGAN预览效果页面地址
https://bigjpg.com/
https://waifu2x.me/


#### 21.1.3
图片数据集
https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

#### 21.4.1
下载dlib包：http://dlib.net/files/dlib-19.16.tar.bz2



