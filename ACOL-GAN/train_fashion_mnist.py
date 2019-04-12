from tensorflow.examples.tutorials.mnist import input_data
import config
import utils
from acol_gan_mode import ACOL_GAN
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from keras.datasets import cifar10
from tensorflow.contrib.learn.python.learn.datasets import base

if __name__ == '__main__':
    opts = config.config_fashion
    dataset = utils.load_data('fashion')
    gan_gs = WGAN(opts)
    gan_gs.train(dataset, True, pre_model_path='kmeans_Model/fashion_14_cluster/model.ckpt', acc1_model_path='Model/fashion/model.ckpt',acc2_model_path='kmeans_Model/', img_path='dcgan_imgs/fashion',log_path='log/fashion/results.csv')
    # gan_gs.calculate_acc(dataset, model_path='Model/fashion/model.ckpt', img_path='dcgan_imgs/fashion')


