# adapted from mnist_tutorial_cw.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
#用Tensorflow中的FLAGS可以自动接收到python脚本运行时的命令行参数，并可将其作为定义的全局变量引用
from tensorflow.python.platform import flags

import logging
import os
from cleverhans.attacks import DeepFool
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans_tutorials.tutorial_models import ModelBasicCNN

FLAGS = flags.FLAGS

#针对mnist的deepfool攻击函数
def mnist_tutorial_deepfool(train_start=0, train_end=60000, #读60000训练
                            test_start=0,test_end=10000, #读10000测试
                            viz_enabled=True, nb_epochs=6,
                            batch_size=128, nb_classes=2, source_samples=10,
                            learning_rate=0.001, attack_iterations=100,
                            model_path=os.path.join("models", "mnist")):
    """
    MNIST tutorial for Deepfool's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples激活对抗例子
    :param nb_epochs: number of epochs to train model（一个epoch指代所有的数据送入网络中完成一次前向计算及反向传播的过程。）
    :param batch_size: size of training batches
    :param nb_classes: number of output classes（输出几类）
    :param source_samples: number of test inputs to attack（测试输入用于攻击的数量）
    :param learning_rate: learning rate for training（学习率）
    :param model_path: path to the model file（文件路径）
    :param attack_iterations: 攻击迭代次数
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies精确度报告
    report = AccuracyReport()

    # MNIST-specific dimensions图像尺寸28*28*1
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = make_basic_picklable_cnn()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow(构建训练模型)
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': os.path.join(*os.path.split(model_path)[:-1]),
        'filename': os.path.split(model_path)[-1]
    }

    rng = np.random.RandomState([2018, 8, 9])
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(model_path+".meta"):
        tf_model_load(sess, model_path)
    else:
        model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                    save=os.path.exists("models"), rng=rng)
        print("save success")

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a DeepFool attack object
    deepfool = DeepFool(model, back='tf', sess=sess)


    idxs = [np.where(np.argmax(Y_test, axis=1) == i)[0][1] for i in range(10)]
    print("idxs:",idxs)

    # construct adv_inputs
    grid_shape = (nb_classes, 2, img_rows, img_cols, channels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')
    print("grid_viz_data",grid_viz_data.shape)
    adv_inputs = X_test[idxs].reshape([-1,28,28,1])

    deepfool_params = {'nb_candidate': 10,
                       'overshoot': 0.02,
                       'max_iter': attack_iterations,
                       'nb_classes': 10,
                       'clip_min': 0.,
                       'clip_max': 1.}

    adv = deepfool.generate_np(adv_inputs, **deepfool_params)

    print("adv success")

    adv_accuracy = 1-model_eval(sess, x, y, preds, adv, Y_test[idxs],
                                args={'batch_size': 10})

    for j in range(10):
        grid_viz_data[j, 0] = adv_inputs[j]
        grid_viz_data[j, 1] = adv[j]

    print(grid_viz_data.shape)

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1.-adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if viz_enabled:
        import matplotlib.pyplot as plt
        _ = grid_visual(grid_viz_data)

    return report


def main(argv=None):
    mnist_tutorial_deepfool(viz_enabled=FLAGS.viz_enabled,
                            nb_epochs=FLAGS.nb_epochs,
                            batch_size=FLAGS.batch_size,
                            nb_classes=FLAGS.nb_classes,
                            source_samples=FLAGS.source_samples,
                            learning_rate=FLAGS.learning_rate,
                            attack_iterations=FLAGS.attack_iterations,
                            model_path=FLAGS.model_path)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_boolean('attack_iterations', 50,
                         'Number of iterations to run attack')

    tf.app.run()
