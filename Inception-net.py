#Inception-net模型实现,分组卷积网络
#一万次训练正确率在 准确率：0.95000左右
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import _pickle as cPickle
import numpy as np

cifar_dir = "D:\BSWJ\DataSet\cifar-10-batches-py"
#print(os.listdir(cifar_dir))
#读取文件函数
def load_data(filename):
    with open(filename,'rb') as file:
        data = cPickle.load(file,encoding='bytes')
        return data[b'data'],data[b'labels']
#y=x*w+b x:内容 y：标签
x = tf.placeholder(tf.float32,[None,3072])#None数据大小（形状）
y = tf.placeholder(tf.int64,[None])
#将x的3072*1的向量转换为三通道图片
x_image = tf.reshape(x, [-1, 3, 32, 32])
#交换通道顺序 x_image:32*32*3
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

#Inception分组卷积层
def inception_block(x,
                    output_channel_for_each_path,#每一个分组卷积的输出通道数
                    name):
    with tf.variable_scope(name):
        conv1_1 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[2],
                                   (1, 1),
                                   strides= (1, 1),
                                   padding= 'same',
                                   activation= tf.nn.relu,
                                   name = 'conv1_1') #scope_name/cov1
        conv3_3 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[2],
                                   (3, 3),
                                   strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv3_3')
        conv5_5 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[2],
                                   (5, 5),
                                   strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv5_5')
        max_pooling = tf.layers.max_pooling2d(x,
                                              (2,2),
                                              (2,2),
                                              name = 'max_pooling')
    #maxpooling后大小变为二分之一，要用padding补足原大小
    max_pooling_shape = max_pooling.get_shape( ).as_list( )[1:]
    input_shape = x.get_shape( ).as_list( )[1:]
    width_padding = (input_shape[0] - max_pooling_shape[0]) // 2
    heigth_padding = (input_shape[1] - max_pooling_shape[1]) // 2
    padding_pooling = tf.pad(max_pooling,
                             [[0,0],
                              [width_padding, width_padding],
                              [heigth_padding, heigth_padding],
                              [0,0]])
    concat_layer = tf.concat([conv1_1,conv3_3,conv5_5,padding_pooling],
                             axis = 3)
    return concat_layer

conv1 = tf.layers.conv2d(x_image,
                         32,#输出通道数
                         (3,3),#卷积核大小3*3
                         padding = 'same',
                         activation = tf.nn.relu,
                         name = 'conv1')
pooling1 = tf.layers.max_pooling2d(conv1,
                                   (2, 2), #卷积核大小
                                   (2, 2), #步长
                                   name = 'pool1')
inception_2a = inception_block(pooling1,[16, 16, 16],name= 'inception_2a')
inception_2b = inception_block(inception_2a,[16, 16, 16],name= 'inception_2b')
pooling2 = tf.layers.max_pooling2d(inception_2b,
                                   [2, 2],
                                   [2, 2],
                                   name= 'pool2')
inception_3a = inception_block(pooling2,[16, 16, 16],name= 'inception_3a')
inception_3b = inception_block(inception_3a,[16, 16, 16],name= 'inception_3b')
pooling3 = tf.layers.max_pooling2d(inception_3b,
                                   [2, 2],
                                   [2, 2],
                                   name= 'pool3')
flatten = tf.layers.flatten(pooling3)

y_ = tf.layers.dense(flatten,10)

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
train_steps = 100000
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


