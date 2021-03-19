import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
#使用VGGnet进行初步探索


image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


#1. 读取数据：标签和数据
labels_path = "D:\BSWJ\smali\CNN_TF-IDF_only.csv"
data_path = "D:\BSWJ\smali\CNN_TF-IDF_onlydata.csv"
labels_path_test = "D:\BSWJ\smali\\test\CNN_TF-IDF_only.csv"
data_paths_test = "D:\BSWJ\smali\\test\CNN_TF-IDF_onlydata.csv"
#取数据
data = np.loadtxt(data_path, skiprows = 1, dtype = float ,delimiter=',')
#print(data.shape)
#取标签
labels = np.loadtxt(labels_path, skiprows = 1, dtype = int, usecols= 0,delimiter = ',')
data_test = np.loadtxt(data_paths_test, skiprows = 1, dtype = float ,delimiter=',')
#取标签
labels_test = np.loadtxt(labels_path_test, skiprows = 1, dtype = int, usecols= 0,delimiter = ',')
data = data * 100 #归一化缩放
data_test = data_test * 100
#print(data)

#2.构建神经网络框架
x = tf.placeholder(tf.float32, [None,343])#数据占位符
y = tf.placeholder(tf.int64,[None])#标签占位符
x_reshape = tf.reshape(x, [-1, 7, 7, 7])#将数组转化为777的矩阵
#卷积层
conv1_1 = tf.layers.conv2d(x_reshape,
                         32,#输出通道数
                         (3,3),#卷积核大小3*3
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv1_1')
conv1_2 = tf.layers.conv2d(conv1_1,
                         32,#输出通道数
                         (3,3),#卷积核大小3*3
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv1_2')
#两层卷积后一层池化层
pooling1 = tf.layers.max_pooling2d(conv1_2,
                                   (2, 2), #卷积核大小
                                   (2, 2), #步长
                                    padding = 'same',
                                   name = 'pool1')
#卷积层2
conv2_1 = tf.layers.conv2d(pooling1,
                         32,#输出通道数
                         (3,3),#卷积核大小3*3
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv2_1')

conv2_2 = tf.layers.conv2d(conv2_1,
                         32,#输出通道数
                         (3,3),#卷积核大小3*3
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv2_2')
#池化层2
pooling2 = tf.layers.max_pooling2d(conv2_2,
                                   (2, 2), #卷积核大小
                                   (2, 2), #步长
                                   padding = 'same',
                                   name = 'pool2')
#卷积层3
conv3_1 = tf.layers.conv2d(pooling2,
                         32,#输出通道数
                         (3,3),#卷积核大小3*3
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv3_1')
conv3_2 = tf.layers.conv2d(conv3_1,
                         32,#输出通道数
                         (3,3),#卷积核大小3*3
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv3_2')
#池化层3
pooling3 = tf.layers.max_pooling2d(conv2_2,
                                   (2, 2), #卷积核大小
                                   (2, 2), #步长
                                   padding= 'same',
                                   name = 'pool3')
flatten = tf.layers.flatten(pooling3)
#全连接层：映射到两个类型上去：
y_ = tf.layers.dense(flatten,2)

with tf.name_scope('loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
    loss_summary = scalar_summary('loss', loss)

predict = tf.arg_max(y_, 1)
correct_prediction = tf.equal(predict,y)#预测值

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))#准确率
    acc_summary = scalar_summary('accuracy', accuracy)

#梯度下降
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

merged = merge_summary([loss_summary, acc_summary])

#数据处理函数
class AndroidData:
    def __init__(self, data, labels, need_shuffle):
        self._data = data
        self._labels = labels
        self._need_shuffle = need_shuffle
        self._indicator = 0  # 当前遍历所处位置
        self._num_examples = self._data.shape[0]  # n个数据
        print(self._num_examples)
        if self._need_shuffle:
            self._shuffle_data()
        self._num_examples = self._data.shape[0]#n个数据
    def _shuffle_data(self):
        # 使用permutation对0至_num_examples进行混排
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:  # 结束一轮后，重新洗牌
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no examples")
            # 取出对应batchsize中的数据及标签
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels
train_data = AndroidData(data,labels,True)


init = tf.global_variables_initializer()
batch_size = 20
#训练数据次数大于数据量，反复训练
train_steps = 10000
#测试数据步长乘以batchsize为测试集大小
test_steps = 100
    #创建session进行计算图执行
counter=0
with tf.Session() as sess:
    sess.run(init)
    writer = SummaryWriter('D:\BSWJ\logs', sess.graph)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val,summary,_ = sess.run(
            [loss,accuracy,merged,train_op],
            feed_dict={x:batch_data, y:batch_labels})
        counter=counter+1
        writer.add_summary(summary, counter)
        if (i+1) % 500 == 0:
            print("训练中：Step：%d, 损失值:%4.5f, 准确率:%4.5f" % (i+1,loss_val,acc_val))
        #每5000测试一遍测试集准确率
        if (i+1) % 5000 == 0:
            test_data = AndroidData(data_test,labels_test,True)
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

