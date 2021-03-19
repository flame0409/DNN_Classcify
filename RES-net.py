#RES-net模型实现,1卷积1池化接多个残差连接块（可选降采样），可拓展，一般在16-19层
#正确率：5000次0.6
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import _pickle as cPickle
import numpy as np
import openpyxl

cifar_dir = "D:\BSWJ\DataSet\cifar-10-batches-py"
#print(os.listdir(cifar_dir))
#读取文件函数
def load_data(filename):
    with open(filename,'rb') as file:
        data = cPickle.load(file,encoding='bytes')
        return data[b'data'],data[b'labels']

#res-net残差函数,并对通道数进行调整,一般来说，通过一个加倍残差函数，通道数加倍
def residual_block(x, output_channel):
    input_channel = x.get_shape( ).as_list()[-1]#读取通道数
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2,2)#降采样
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1,1)
    else:
        raise Exception("输入通道与输出通道不匹配")
    conv1 = tf.layers.conv2d(x,output_channel,
                             (3,3),
                             strides = strides,
                             padding = 'same',
                             activation = tf.nn.relu,
                             name = 'conv1')
    conv2 = tf.layers.conv2d(conv1,output_channel,
                             (3,3),
                             strides = (1,1),
                             padding = 'same',
                             activation = tf.nn.relu,
                             name = 'conv2')
    #分支加法，需要对恒等变换分支进行降采样
    if increase_dim:
        pooled_x = tf.layers.average_pooling2d(
            x, (2,2), (2,2), padding = 'valid'
        )#降采样
        padded_x = tf.pad(pooled_x,
                          [[0,0],
                           [0,0],
                           [0,0],
                           [(input_channel // 2), (input_channel // 2)]])#残差翻倍，通道数补充
    else:
        padded_x = x
    output_x = padded_x + conv2 #conv2是经过卷积后的，padded是恒等变换
    return output_x

def res_net(x,
            num_residual_blocks,
            num_filter_base,
            class_num):
    """
    各变量含义：
    x: 接受输入的图像
    num_residual_blocks:每个stage中对应的残差块数量
    num_subsampling：stage数量
    num_filter_base:原始输入通道数
    class_num: 增强泛化能力
    """
    num_subsampling = len(num_residual_blocks)
    layers = []#卷积层数组
    input_size = x.get_shape( ).as_list( )[1:]
    # 1. 通过普通卷积层
    with tf.variable_scope('conv0'):#创建初始图变量
        conv0 = tf.layers.conv2d(x,
                                 num_filter_base,
                                 (3,3),
                                 padding = 'same',
                                 strides= (1, 1),
                                 activation = tf.nn.relu,
                                 name= 'conv0')
        layers.append(conv0)
        #因为数据集图片小，所以省略去一个池化层

        #循环处理每个stage
    for sample_id in range(num_subsampling):
        #每个stage中循环处理每个残差块
        for i in range(num_residual_blocks[sample_id]):
            with tf.variable_scope("conv%d_%d" % (sample_id,i)):
                conv = residual_block(
                    layers[-1],#取最后一个层
                    num_filter_base * (2 ** sample_id))#输出通道数
                layers.append(conv)
    multiplier = 2 ** (num_subsampling - 1)#预测输出神经元图大小
    assert layers[-1].get_shape( ).as_list( )[1:] == [
        input_size[0] / multiplier,
        input_size[1] / multiplier,
        num_filter_base * multiplier
    ]#确定输出与预测输出大小一致
    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1,2])#池化
        logits = tf.layers.dense(global_pool, class_num)
        layers.append(logits)
    return layers[-1]


#y=x*w+b x:内容 y：标签
x = tf.placeholder(tf.float32,[None,3072])#None数据大小（形状）
y = tf.placeholder(tf.int64,[None])
#将x的3072*1的向量转换为三通道图片
x_image = tf.reshape(x, [-1, 3, 32, 32])
#交换通道顺序 x_image:32*32*3
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

#调用resnet处理流程：
y_ = res_net(x_image, [2,3,2], 32, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
#取y_中最大值标签为预测分类
predict = tf.arg_max(y_, 1)
correct_prediction = tf.equal(predict,y)#预测值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))#准确率

#梯度下降
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


#cifar-10数据处理类
class CifarData:
    def __init__(self,filenames,need_shuffle): #need_shuffle:打乱训练集顺序，降低依赖，提升泛化能力
        # 读入数据
        all_data = []
        all_labels = []
        for filename in filenames:
            data,labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)#n行的3072*1
        #归一化缩放,提升准确率关键步骤
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)#一维向量集
        self._num_examples = self._data.shape[0]#n个数据
        self._need_shuffle = need_shuffle
        self._indicator = 0 #当前遍历所处位置
            #print(self._data.shape)
            #print(self._labels.shape)
        if self._need_shuffle:
            self._shuffle_data()
    #打乱训练数据集
    def _shuffle_data(self):
        #使用permutation对0至_num_examples进行混排
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    #
    def next_batch(self,batch_size):
        end_indicator = self._indicator+batch_size
        if end_indicator > self._num_examples:#结束一轮后，重新洗牌
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no examples")
        #取出对应batchsize中的数据及标签
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels




#读取的文件名，按实际情况改变
train_filnames = [os.path.join(cifar_dir,'data_batch_%d' % i) for i in range(1,6)]
test_filenames = [os.path.join(cifar_dir,'test_batch')]
train_data = CifarData(train_filnames,True)
test_data = CifarData(test_filenames,False)
#batch_data,batch_labels = train_data.next_batch(5)
#print(batch_data)
#print(batch_labels)

#执行计算
    #初始化
init = tf.global_variables_initializer()
batch_size = 20
#训练数据次数大于数据量，反复训练
train_steps = 5000
#测试数据步长乘以batchsize为测试集大小
test_steps = 100
    #创建session进行计算图执行
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val,_ = sess.run(
            [loss,accuracy,train_op],
            feed_dict={x:batch_data, y:batch_labels})
        if (i+1) % 500 == 0:
            print("训练中：Step：%d, 损失值:%4.5f, 准确率:%4.5f" % (i+1,loss_val,acc_val))
        #每5000测试一遍测试集准确率
        if (i+1) % 5000 == 0:
            test_data = CifarData(test_filenames,False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data,test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict={x:batch_data, y:batch_labels}
                )
                all_test_acc_val.append(test_acc_val)
                test_acc = np.mean(all_test_acc_val)
            print("测试：准确率：%4.5f" % (test_acc))


