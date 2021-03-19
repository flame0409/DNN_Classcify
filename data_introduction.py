import tensorflow as tf
#utf-8
import _pickle as cPickel
import os
import numpy
import matplotlib.pyplot as plt
#3.0用_pickel替换cPickle

#cifar10文件保存路径：D:\BSWJ\DataSet\cifar-10-batches-py
cifar_dir = "D:\BSWJ\DataSet\cifar-10-batches-py"
#print(os.listdir(cifar_dir))

#简单看一下cifar内部的文件格式

with open(os.path.join(cifar_dir,"data_batch_1"),'rb') as file:
    data = cPickel.load(file,encoding='bytes')
    #print(data)
    #print(type(data))#数据单元的格式：字典
    #print(data.keys())#[b'batch_label', b'labels', b'data', b'filenames']
    print(data[b'labels'])#10种图片分类
    #print(data[b'data'].shape)#32*32*3的图片格式共一万个图片


    #尝试解析一下文件的像素点对应矩阵
    image_arr = data[b'data'][1000]#取第八张图片
    image_arr = image_arr.reshape((3, 32, 32))#将1*3072转为3*32*32
    #print(image_arr)
    image_arr = image_arr.transpose((1,2,0))#交换一下通道才可正常显示
    plt.imshow(image_arr)
    plt.show()




