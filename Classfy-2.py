#单神经元的逻辑斯蒂回归模型实现二分类
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

x = tf.placeholder(tf.float32,[None,3072])#None数据大小（形状）
y = tf.placeholder(tf.int64,[None])
#创建两个变量w，b w：3072*1 b:1
w = tf.get_variable('w',[x.get_shape()[-1],1],
                    initializer=tf.random_normal_initializer(0,1))#随机初始化为一串0-1中的3072个值
b = tf.get_variable('b',[1],
                    initializer=tf.constant_initializer(0.0))#常量初始化

#获取输出,内积
y_ = tf.matmul(x,w) + b
#使用sigmoid进行激活,矩阵相乘后变为[None,1]的形式
p_y_1 = tf.nn.sigmoid(y_)
y_reshaped = tf.reshape(y, (-1,1))
y_reshaped = tf.cast(y_reshaped,tf.float32)
#print(y_reshaped.dtype,x.dtype)

#损失函数 reduce_mean:均值，差值平方求均值
loss = tf.reduce_mean(tf.square(y_reshaped - p_y_1))
predict = p_y_1>0.5
correct_prediction = tf.equal(tf.cast(predict,tf.float32),y_reshaped)#预测值
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
            #二分类，只取标签为0,1的数据
            for item ,label in list(zip(data,labels)):
                #print(list(zip(data,lables)))
                #二分累，选其中两个数据集
                if label in [0,1]:
                    all_data.append(item)
                    all_labels.append(label)
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
#训练数据步长大于数据量，反复训练
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
        if i % 500 == 0:
            print("训练中：Step：%d, 损失值:%4.5f, 准确率:%4.5f" % (i,loss_val,acc_val))
        #每5000测试一遍测试集准确率
        if i % 5000 == 0:
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


