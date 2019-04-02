from tensorflow.examples.tutorials.mnist import input_data
import config
import utils
from dcgan_cluster_mnist_v71 import WGAN
# from dcgan_cluster_mnist_v35 import WGAN
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from keras.datasets import cifar10
from tensorflow.contrib.learn.python.learn.datasets import base

if __name__ == '__main__':

    # opts = config.config_mnist
    # dataset = input_data.read_dat a_sets('MNIST_data')

    # opts = config.config_reuters10k
    # dataset = utils.load_data('reuters10k')

    # opts = config.config_reuters
    # dataset = utils.get_reuters()

    # y0 = y[y == 0].shape[0]
    # y1 = y[y == 1].shape[0]
    # y2 = y[y == 2].shape[0]
    # y3 = y[y == 3].shape[0]
    # y4 = y[y == 4].shape[0]
    # y5 = y[y == 5].shape[0]
    # y6 = y[y == 6].shape[0]
    # y7 = y[y == 7].shape[0]
    # y8 = y[y == 8].shape[0]
    # y9 = y[y == 9].shape[0]

    # opts = config.config_svhn
    # dataset = utils.get_svhn()
    # x, y = dataset.train.next_batch(100)
    # gan_gs = WGAN(opts)
    # gan_gs.train(dataset, True, pre_model_path='Model/pre_svhn/model.ckpt', acc1_model_path='Model/svhn/model.ckpt',acc2_model_path='kmeans_Model/svhn/model.ckpt', img_path='dcgan_imgs/svhn',log_path='log/svhn/results.csv')


    # opts = config.config_hhar
    # dataset = utils.load_data('har')
    # x, y = dataset.train.next_batch(100)
    # gan_gs = WGAN(opts)
    # gan_gs.train(dataset, True, model_path='Model/hhar/model.ckpt',img_path = 'dcgan_imgs/hhar', log_path = 'log/hhar/results.csv')

    # opts = config.config_mnist
    # dataset = utils.load_data('mnist')
    # x, y = dataset.train.next_batch(100)
    # gan_gs = WGAN(opts)
    # gan_gs.train(dataset, True, pre_model_path='Model/pre_mnist/model.ckpt', acc1_model_path='Model/mnist/model.ckpt',acc2_model_path='kmeans_Model/mnist/model.ckpt', img_path='dcgan_imgs/mnist',log_path='log/mnist/results.csv')
    # gan_gs.train(dataset, True, pre_model_path='Model/pre_mnist/model.ckpt', acc1_model_path='Model/mnist/model.ckpt',acc2_model_path='kmeans_Model/', img_path='dcgan_imgs/mnist',log_path='log/mnist/results.csv')
    # gan_gs.train(dataset, False, pre_model_path='kmeans_Model/mnist_14_cluster/model.ckpt', acc1_model_path='Model/mnist/model.ckpt',acc2_model_path='kmeans_Model/', img_path='dcgan_imgs/mnist',log_path='log/mnist/results.csv')
    # gan_gs.get_2pts(dataset,model_path='kmeans_Model/mnist0/model.ckpt')
    # gan_gs.save_pic_cluster_n(dataset, model_path='kmeans_Model/mnist_14_cluster/model.ckpt')
    # opts = config.config_reuters
    # dataset = utils.get_reuters()

    # opts = config.config_reuters10k
    # dataset = utils.load_data('reuters10k')
    # x, y = dataset.train.next_batch(100)
    # gan_gs = WGAN(opts)
    # gan_gs.train(dataset, True,model_path='Model/reuters/model.ckpt',img_path = 'dcgan_imgs/reuters', log_path = 'log/reuters10/results.csv')

    opts = config.config_fashion
    dataset = utils.load_data('fashion')
    gan_gs = WGAN(opts)
    gan_gs.train(dataset, True, pre_model_path='kmeans_Model/fashion_14_cluster/model.ckpt', acc1_model_path='Model/fashion/model.ckpt',acc2_model_path='kmeans_Model/', img_path='dcgan_imgs/fashion',log_path='log/fashion/results.csv')
    # gan_gs.calculate_acc(dataset, model_path='Model/fashion/model.ckpt', img_path='dcgan_imgs/fashion')

    # opts = config.config_usps
    # dataset = utils.load_data('usps')
    # gan_gs = WGAN(opts)
    # gan_gs.train(dataset, True,pre_model_path= 'Model/pre_usps/model.ckpt', acc1_model_path='Model/usps/model.ckpt',acc2_model_path='kmeans_Model/usps/model.ckpt',img_path = 'dcgan_imgs/usps', log_path = 'log/usps/results.csv')


