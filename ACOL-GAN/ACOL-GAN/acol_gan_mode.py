import tensorflow as tf
import ops
import os
import numpy as np
import utils
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
import csv
from keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class ACOL_GAN():

    def __init__(self, opts):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.opts = opts
        self.data_shape = opts['data_shape']

        self.add_placeholders()

        self.encoded = self.encoder(opts, inputs=self.sample_points)
        self.re_sample_points = self.decoder(opts, self.encoded)

        self.encoded_r = self.encoder(opts, inputs=self.sample_points_r,reuse=True)
        self.re_sample_points_r = self.decoder(opts, self.encoded_r,reuse=True)

        self.sample_noise_1, self.limit_1 = self.add_sample(self.choise_list)
        self.generated = self.decoder(opts, self.sample_noise_1, True)
        self.re_sample_noise = self.encoder(opts, self.generated, reuse = True)

        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10)
        self.build_loss()

        self.add_optimizers()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def build_loss(self):
        """
        Calculate loss
        :return:
        """
        self.re_loss_real = tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(self.sample_points, [self.opts['batch_size'], -1]) - tf.reshape(self.re_sample_points,[self.opts['batch_size'], -1])),axis=1), axis=0)
        self.re_loss = self.re_loss_real
        self.re_loss += tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(self.sample_points_r, [self.opts['batch_size'], -1]) - tf.reshape(self.re_sample_points_r,[self.opts['batch_size'], -1])),axis=1), axis=0)

        self.D_loss, self.G_loss, self.pred, self.real_logits, self.fake_logits, self.l0_logits, self.Dr_loss, self.l0_logits_r,self.pred_r = self.gan_penalty(self.opts)
        self.regularization = self.get_regularization(self.opts, self.pred) + self.get_regularization(self.opts, self.pred_r)

    def get_regularization(self, opts, logits):
        U = tf.matmul(tf.transpose(logits), logits)
        v = tf.reshape(tf.diag_part(U), [1, -1])
        V = tf.matmul(tf.transpose(v), v)
        affinity = (tf.reduce_sum(U) - tf.trace(U)) / ((2 * (opts['num_cluster']*2 - 1)) * tf.trace(U) + 1e-35)
        balance = (tf.reduce_sum(V) - tf.trace(V)) / ((2 * (opts['num_cluster']*2 - 1)) * tf.trace(V) + 1e-35)
        l2_reg = tf.reduce_sum(tf.square(logits))
        regularization = 3 * affinity + 1.5 * (1 - balance) + 0.000001 * l2_reg
        return regularization

    def discriminator(self, opts, inputs, reuse=False):

        num_units = opts['d_num_filters']
        num_layers = opts['d_num_layers']

        with tf.variable_scope('discriminator', reuse=reuse):
            hi = inputs
            for i in range(num_layers):
                scale = 2 ** (i)
                hi = ops.linear(opts, hi, num_units, scope='h%d_lin' % (i + 1))
                hi = tf.nn.relu(hi)
                hi = tf.nn.dropout(hi, self.keep_prob)

            # ACOL layer
            logit = ops.linear(opts, hi, opts['num_cluster'] + opts['num_cluster'], scope='hfinal_lin')
            pred = tf.nn.softmax(logit)
            id_matrix = tf.one_hot(tf.concat((tf.zeros(opts['num_cluster'],dtype=tf.int32), tf.ones(opts['num_cluster'],dtype=tf.int32)), axis=0),2)
            self.mat2 = id_matrix

            bi_logit = tf.matmul(pred, id_matrix)

        return bi_logit, pred, logit

    def gan_penalty(self,opts):
        """
        objective function of gan
        :param opts: configuration dictionary
        :return:
        """

        D_logit, pred, D_logit_ml = self.discriminator(opts, tf.concat((self.encoded, self.sample_noise_1), axis=0))

        labels = tf.concat((tf.zeros(opts['batch_size'], dtype=tf.int32),tf.ones(opts['batch_size'], dtype=tf.int32)), axis=0)
        labels = tf.one_hot(labels, 2)
        D_loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(D_logit + 1e-35), axis=1)) * 5

        labels = tf.concat((tf.ones( opts['batch_size'], dtype=tf.int32), tf.zeros( opts['batch_size'], dtype=tf.int32)),axis=0)
        labels = tf.one_hot(labels, 2)
        G_loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(D_logit + 1e-35), axis=1)) * 1

        D_logit_fake = tf.slice(D_logit_ml, [opts['batch_size'], 0], [opts['batch_size'], -1])
        D_logit_real = tf.slice(D_logit_ml, [0, 0], [opts['batch_size'], -1])

        # Cross entropy
        D_loss += self.loss_decay * 0.5 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.one_hot(self.choise_list + opts['num_cluster'], 2 * opts['num_cluster'])))
        G_loss += self.loss_decay * 0.5 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.one_hot(self.choise_list, 2 * opts['num_cluster'])))

        self.D_logit = D_logit


        # Data Augmentation
        D_logit_r, pred_r, D_logit_ml_r = self.discriminator(opts, tf.concat((self.encoded_r, self.sample_noise_1), axis=0),reuse=True)

        labels = tf.concat((tf.zeros(opts['batch_size'], dtype=tf.int32), tf.ones(opts['batch_size'], dtype=tf.int32)),axis=0)
        labels = tf.one_hot(labels, 2)
        D_loss += tf.reduce_mean(-tf.reduce_sum(labels * tf.log(D_logit_r + 1e-35), axis=1)) * 5

        labels = tf.concat((tf.ones(opts['batch_size'], dtype=tf.int32), tf.zeros(opts['batch_size'], dtype=tf.int32)),axis=0)
        labels = tf.one_hot(labels, 2)
        G_loss += tf.reduce_mean(-tf.reduce_sum(labels * tf.log(D_logit_r + 1e-35), axis=1)) * 1

        D_logit_fake_r = tf.slice(D_logit_ml_r, [opts['batch_size'], 0], [opts['batch_size'], -1])
        D_logit_real_r = tf.slice(D_logit_ml_r, [0, 0], [opts['batch_size'], -1])

        # Cross entropy
        D_loss += self.loss_decay * 0.5 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_fake_r, labels=tf.one_hot(self.choise_list + opts['num_cluster'], 2 * opts['num_cluster'])))
        G_loss += self.loss_decay * 0.5 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_fake_r, labels=tf.one_hot(self.choise_list, 2 * opts['num_cluster'])))

        Dr_loss = self.loss_decay * 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(D_logit_real - D_logit_real_r), axis=1))
        D_loss += Dr_loss

        return D_loss , G_loss, pred, D_logit_real,D_logit_fake, tf.maximum(0., D_logit_ml), Dr_loss,  tf.maximum(0., D_logit_ml_r),pred_r

    def add_sample(self, choise_list, reuse = False):
        """
        Obtaining a Gaussian mixture distribution
        :param choise_list:
        :param reuse:
        :return:
        """
        opts = self.opts
        label_one_hot = tf.one_hot(choise_list, opts['num_cluster'])
        label_one_hot += tf.random_normal((tf.shape(choise_list)[0], opts['num_cluster']), stddev=0.1)
        num_units = 512
        num_layers = opts['e_num_layers']
        limit = 0
        with tf.variable_scope('sample', reuse=reuse):
            hi = label_one_hot
            i = 0
            for i in range(num_layers):
                hi = ops.linear(opts, hi, num_units, scope='h%d_lin' % i)
                hi = tf.nn.relu(hi)
                hi = tf.nn.dropout(hi, self.keep_prob)
            if opts['GS'] == 'GS':
                mean = ops.linear(opts, hi, opts['latent_dim'], scope='mean_lin')
                log_sigmas = ops.linear(opts, hi, opts['latent_dim'], scope='log_sigmas_lin')
                epsilon = tf.random_normal(tf.stack([tf.shape(choise_list)[0], opts['latent_dim']]))
                limit = tf.reduce_mean(tf.square(1 - tf.reduce_sum(log_sigmas, axis=1)), axis=0)
                res = mean + tf.multiply(epsilon, tf.exp(log_sigmas))
            else:
                hi = ops.linear(opts, hi, opts['latent_dim'], scope='mean_lin')
                shape = tf.shape(hi)
                res = hi + tf.truncated_normal(shape, 0.0, 0.01)
        return res , limit

    def add_placeholders(self):
        """
        添加placeholder
        :return:
        """
        self.sample_points = tf.placeholder(tf.float32, [None] + self.data_shape, name='real_pts_ph')
        self.sample_points_r = tf.placeholder(tf.float32, [None] + self.data_shape, name='real_pts_ph_r')
        self.choise_list = tf.placeholder(dtype=tf.int32, shape=[None])
        self.range = tf.placeholder(dtype=tf.int32, shape=[None])
        self.cond = tf.placeholder(dtype=tf.float32, shape=[None])
        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.loss_decay = tf.placeholder(tf.float32, name='loss_decay_ph')
        self.keep_prob = tf.placeholder(tf.float32)

    def encoder(self, opts, inputs, is_training=True, reuse=False):
        """
        Encoding Network
        :param opts:
        :param inputs:
        :param is_training:
        :param reuse:
        :return:
        """

        # Add noise if needed
        if opts['e_noise'] == 'add_noise':

            def add_noise(x):
                shape = tf.shape(x)
                return x + tf.truncated_normal(shape, 0.0, 0.01)

            def do_nothing(x):
                return x

            inputs = tf.cond(tf.constant(is_training, tf.bool), lambda: add_noise(inputs), lambda: do_nothing(inputs))

        num_units = opts['e_num_filters']
        num_layers = opts['e_num_layers']
        layer_x = inputs
        with tf.variable_scope('encoder', reuse=reuse):
            for i in range(num_layers):
                scale = 2 ** (num_layers - i - 1)
                layer_x = ops.conv2d(opts, layer_x, num_units / scale,
                                     scope='h%d_conv' % i)
                if opts['batch_norm']:
                    layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='h%d_bn' % i)
                layer_x = tf.nn.relu(layer_x)

            if opts['GS'] == 'GS':
                mean = ops.linear(opts, layer_x, opts['latent_dim'], scope='mean_lin')
                log_sigmas = ops.linear(opts, layer_x, opts['latent_dim'], scope='log_sigmas_lin')
                log_sigmas = tf.clip_by_value(log_sigmas, -50, 50)
                epsilon = tf.random_normal((opts['batch_size'], opts['latent_dim']), 0., 1., dtype=tf.float32)
                res = mean + tf.multiply(epsilon, tf.sqrt(1e-8 + tf.exp(log_sigmas)))
            else:
                res = ops.linear(opts, layer_x, opts['latent_dim'], scope='mean_lin')

        return res

    def decoder(self, opts, inputs, reuse=False, is_training=True):
        output_shape = opts['data_shape']
        num_units = opts['g_num_filters']
        num_layers = opts['g_num_layers']
        height = output_shape[0] // 2 ** (num_layers - 1)
        width = output_shape[1] // 2 ** (num_layers - 1)

        with tf.variable_scope('decoder', reuse=reuse):
            h0 = ops.linear(
                opts, inputs, num_units * height * width, scope='h0_lin')
            h0 = tf.reshape(h0, [-1, height, width, num_units])
            h0 = tf.nn.relu(h0)
            layer_x = h0
            for i in range(num_layers - 1):
                scale = 2 ** (i + 1)
                _out_shape = [opts['batch_size'], height * scale,
                              width * scale, num_units // scale]
                layer_x = ops.deconv2d(opts, layer_x, _out_shape, scope='h%d_deconv' % i)
                if opts['batch_norm']:
                    layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='h%d_bn' % i)
                layer_x = tf.nn.relu(layer_x)

            _out_shape = [opts['batch_size']] + list(output_shape)
            last_h = ops.deconv2d(opts, layer_x, _out_shape, d_w=1, d_h=1, scope='hfinal_deconv')
            net = tf.nn.sigmoid(last_h)
        return net

    def add_optimizers(self):

        opts = self.opts
        lr = opts['lr']
        lr_adv = opts['lr_adv']

        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        sample_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sample')

        global_step_1 = tf.Variable(0, trainable=False, name='global_step_1')
        opt = self.optimizer(lr, self.lr_decay)
        self.gt_opt = opt.minimize(loss= self.re_loss + self.G_loss + self.limit_1,global_step=global_step_1, var_list=encoder_vars + decoder_vars + sample_vars)

        global_step_2 = tf.Variable(0, trainable=False, name='global_step_2')
        opt = self.optimizer(lr_adv, self.lr_decay)
        self.dis_opt = opt.minimize(loss= self.D_loss + 1 * self.regularization,global_step=global_step_2, var_list=dis_vars)

        global_step_3 = tf.Variable(0, trainable=False, name='global_step_3')
        opt = self.optimizer(lr, self.lr_decay)
        self.pre_opt = opt.minimize(loss= self.re_loss_real, global_step=global_step_3, var_list=encoder_vars + decoder_vars)

    def calculate_acc(self,data, model_path = 'Model/model.ckpt',img_path='dcgan_imgs'):
        opts = self.opts
        self.saver.restore(self.sess, model_path)
        test_image = data.test.images
        test_y = data.test.labels
        perm = np.arange(test_y.shape[0])
        np.random.shuffle(perm)
        test_image = test_image[perm]
        test_y = test_y[perm]

        fake_acc_sum = 0
        bi_acc_sum = 0
        reloss_sum = 0
        pred_real_1 = np.array([])
        klayer_list = np.array([])
        for i in range(test_image.shape[0] // opts['batch_size']):
            imgs = test_image[i * opts['batch_size']:(i + 1) * opts['batch_size'], :]
            labels = test_y[i * opts['batch_size']:(i + 1) * opts['batch_size']]
            imgs = np.reshape(imgs, [-1] + opts['data_shape'])
            choise_list = [i % (opts['num_cluster'] * opts['range']) for i in range(opts['batch_size'])]

            feed_d = {
                self.sample_points: imgs,
                self.choise_list: np.array(choise_list),
                self.keep_prob: 1
            }
            labels = np.reshape(labels, [-1])
            imgs1, pred, reloss, imgs2, logit, emd_data, noise, klayer, flayer = self.sess.run(
                [self.re_sample_points, self.pred, self.re_loss, self.generated, self.D_logit, self.encoded,
                 self.sample_noise_1, self.real_logits, self.fake_logits], feed_dict=feed_d)
            real_pred_1 = np.argmax(pred[:opts['batch_size'], :opts['num_cluster']], axis=1)
            pred_fake = np.argmax(pred[opts['batch_size']:, opts['num_cluster']:], axis=1)
            acc2, _ = utils.cluster_acc(pred_fake, np.array(choise_list))
            if i == 0:
                klayer_list = klayer
                pred_real_1 = real_pred_1
                utils.draw_gan(imgs, imgs1, imgs2, klayer, real_pred_1, labels, np.array(choise_list), flayer,
                               pred_fake,
                               0, img_path, opts['name'], num_cluster=opts['batch_size'] // opts['num_cluster'])
            else:
                klayer_list = np.concatenate((klayer_list, klayer), axis=0)
                pred_real_1 = np.concatenate((pred_real_1, real_pred_1), axis=0)

            bi_pred = np.argmax(logit, axis=1)
            acc3, _ = utils.cluster_acc(bi_pred, np.array(np.concatenate(
                (np.zeros(opts['batch_size'], dtype=np.int32), np.ones(opts['batch_size'], dtype=np.int32)), axis=0)))
            fake_acc_sum += acc2
            bi_acc_sum += acc3
            reloss_sum += reloss

        test_batch_num = (test_image.shape[0] // opts['batch_size'])
        test_y = np.reshape(test_y[:test_image.shape[0] // opts['batch_size'] * opts['batch_size']], [-1])

        real_pred_2 = KMeans(n_clusters=opts['num_cluster']).fit_predict(klayer_list)
        acc_1_sum, _ = utils.cluster_acc(pred_real_1, test_y)
        acc_2_sum, _ = utils.cluster_acc(real_pred_2, test_y)
        nmi = metrics.normalized_mutual_info_score(test_y, pred_real_1)
        ari = metrics.adjusted_rand_score(test_y, pred_real_1)
        print(
            "....acc_1: %f, acc_2: %f, reloss: %f, fake_acc: %f, bi_acc: %f, nmi: %f, ari: %f ...." % (
                acc_1_sum, acc_2_sum,
                reloss_sum / test_batch_num, fake_acc_sum / test_batch_num,
                bi_acc_sum / test_batch_num,nmi,ari))

    def optimizer(self, lr, decay = 1.):
        opts = self.opts
        lr *= decay
        if opts['optimizer'] == 'ADM':
            return tf.train.AdamOptimizer(lr, beta1=opts["adam_beta1"])
        else:
            return tf.train.RMSPropOptimizer(lr,momentum=0.95)

    def pre_train(self, data, is_init = True, model_path = 'Model/pre_model.ckpt', img_path='dcgan_imgs'):
        opts = self.opts
        if is_init:
            self.sess.run(self.init)
        else:
            self.saver.restore(self.sess, model_path)
        batches_num = data.train.num_examples // opts['batch_size']
        counter = 0
        decay = 1
        reloss_min = 99999

        for epoch in range(opts["epoch_num"]):
            if epoch > 0 and epoch % 10 == 0:
                decay *= 0.9
            for it in range(batches_num):
                batch_images, batch_y = data.train.next_batch(opts['batch_size'])
                batch_images = np.reshape(batch_images, [opts['batch_size']] + opts['data_shape'])

                feed_d = {
                    self.sample_points: batch_images,
                    self.lr_decay: decay,
                    self.keep_prob: 0.75,
                }
                self.sess.run(self.pre_opt, feed_dict=feed_d)

                r_batch_images = self.datagen.flow(batch_images, shuffle=True, batch_size=opts['batch_size']).next()
                feed_d = {
                    self.sample_points: r_batch_images,
                    self.lr_decay: decay,
                    self.keep_prob: 0.75
                }
                self.sess.run(self.pre_opt, feed_dict=feed_d)

                counter += 1
                if counter % opts['print_every'] == 0:
                    test_image = data.test.images
                    test_y = data.test.labels
                    perm = np.arange(test_y.shape[0])
                    np.random.shuffle(perm)
                    test_image = test_image[perm]
                    reloss_sum = 0
                    for i in range(test_image.shape[0] // opts['batch_size']):
                        imgs = test_image[i * opts['batch_size']:(i + 1) * opts['batch_size'], :]

                        imgs = np.reshape(imgs, [-1] + opts['data_shape'])

                        feed_d = {
                            self.sample_points: imgs,
                            self.lr_decay: decay,
                            self.keep_prob: 1
                        }

                        imgs1, reloss = self.sess.run(
                            [self.re_sample_points, self.re_loss_real], feed_dict=feed_d)

                        reloss_sum += reloss
                        if i == 0:
                            utils.draw_img(imgs, imgs1, counter, img_path, opts['name'])
                    test_batch_num = test_image.shape[0] // opts['batch_size']
                    if reloss_min >= reloss_sum / test_batch_num:
                        self.saver.save(self.sess, model_path)
                        reloss_min = reloss_sum / test_batch_num
                    print(
                        "....eapoch %d ,step %d,reloss: %f ...." % (epoch, counter, reloss_sum / test_batch_num))
        print('test')

    def train(self, data, is_init = True, pre_model_path = 'Model/pre_model.ckpt', acc1_model_path ='Model/model.ckpt', acc2_model_path ='Model/model.ckpt', img_path='dcgan_imgs', log_path ='log/results.csv'):

        opts = self.opts
        if is_init:
            self.sess.run(self.init)
        else:
            self.saver.restore(self.sess, pre_model_path)

        batches_num = data.train.num_examples // opts['batch_size']
        counter = 0
        decay = 1
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        loss_decay = 1
        acc1_max = 0
        acc2_max = 0
        logfile = open(log_path, 'w',newline='',buffering=1)
        logwriter = csv.DictWriter(logfile,
                                   fieldnames=['epoch', 'step', 'softmax_acc', 'softmax_nmi', 'softmax_ari',
                                               'acc_kmeans', 'reloss', 'fake_acc', 'bi_acc'])
        logwriter.writeheader()

        for epoch in range(opts["epoch_num"]):
            if epoch > 0 and epoch % 10 == 0:
                decay *= 0.9

            for it in range(batches_num):
                batch_images, batch_y = data.train.next_batch(opts['batch_size'])
                batch_images = np.reshape(batch_images, [opts['batch_size']] + opts['data_shape'])
                choise_list = np.random.randint(opts['num_cluster'], size=opts['batch_size'])
                batch_images_r = self.datagen.flow(batch_images, shuffle=False, batch_size=opts['batch_size']).next()

                feed_d = {
                    self.sample_points: batch_images,
                    self.sample_points_r: batch_images_r,
                    self.choise_list: choise_list,
                    self.lr_decay: decay,
                    self.keep_prob: 0.75,
                    self.loss_decay: loss_decay
                }

                self.sess.run(self.gt_opt, feed_dict=feed_d)
                self.sess.run(self.dis_opt, feed_dict=feed_d)

                counter += 1
                if counter % opts['print_every'] == 0:
                    test_image = data.test.images
                    test_y = data.test.labels
                    perm = np.arange(test_y.shape[0])
                    np.random.shuffle(perm)
                    test_image = test_image[perm]
                    test_y = test_y[perm]
                    fake_acc_sum = 0
                    bi_acc_sum = 0
                    reloss_sum = 0
                    pred_real_1 = np.array([])
                    klayer_list = np.array([])
                    for i in range(test_image.shape[0] // opts['batch_size']):
                        imgs = test_image[i * opts['batch_size']:(i + 1) * opts['batch_size'], :]
                        labels = test_y[i * opts['batch_size']:(i + 1) * opts['batch_size']]
                        imgs = np.reshape(imgs, [-1] + opts['data_shape'])
                        choise_list = [i % opts['num_cluster'] for i in range(opts['batch_size'])]

                        feed_d = {
                            self.sample_points: imgs,
                            self.choise_list: np.array(choise_list),
                            self.lr_decay: decay,
                            self.keep_prob: 1
                        }
                        labels = np.reshape(labels, [-1])
                        imgs1, pred, reloss, imgs2, logit, emd_data, noise, klayer, flayer = self.sess.run(
                            [self.re_sample_points, self.pred, self.re_loss, self.generated, self.D_logit, self.encoded,
                             self.sample_noise_1, self.real_logits, self.fake_logits], feed_dict=feed_d)
                        real_pred_1 = np.argmax(pred[:opts['batch_size'], :opts['num_cluster']], axis=1)
                        pred_fake = np.argmax(pred[opts['batch_size']:, opts['num_cluster']:], axis=1)
                        acc2, _ = utils.cluster_acc(pred_fake, np.array(choise_list))
                        if i == 0:
                            klayer_list = klayer
                            pred_real_1 = real_pred_1
                            utils.draw_gan(imgs, imgs1, imgs2, klayer, real_pred_1, labels, np.array(choise_list),
                                           flayer,
                                           pred_fake,
                                           0, img_path, opts['name'],
                                           num_cluster=opts['batch_size'] // opts['num_cluster'])
                        else:
                            klayer_list = np.concatenate((klayer_list, klayer), axis=0)
                            pred_real_1 = np.concatenate((pred_real_1, real_pred_1), axis=0)

                        bi_pred = np.argmax(logit, axis=1)
                        acc3, _ = utils.cluster_acc(bi_pred, np.array(np.concatenate(
                            (np.zeros(opts['batch_size'], dtype=np.int32), np.ones(opts['batch_size'], dtype=np.int32)),
                            axis=0)))
                        fake_acc_sum += acc2
                        bi_acc_sum += acc3
                        reloss_sum += reloss

                    test_batch_num = (test_image.shape[0] // opts['batch_size'])
                    test_y = np.reshape(test_y[:test_image.shape[0] // opts['batch_size'] * opts['batch_size']], [-1])

                    pred_real_2 = KMeans(n_clusters=opts['num_cluster']).fit_predict(klayer_list)
                    acc_1_sum, _ = utils.cluster_acc(pred_real_1,test_y)
                    acc_2_sum, _ = utils.cluster_acc(pred_real_2, test_y)
                    nmi = metrics.normalized_mutual_info_score(test_y, pred_real_1)
                    ari = metrics.adjusted_rand_score(test_y, pred_real_1)

                    if acc1_max < acc_1_sum:
                        self.saver.save(self.sess, acc1_model_path)
                        acc1_max = acc_1_sum
                    if acc2_max < acc_2_sum:
                        self.saver.save(self.sess, acc2_model_path)
                        acc2_max = acc_2_sum
                    logwriter.writerow(dict(epoch=epoch, step=counter, softmax_acc=acc_1_sum, softmax_nmi=nmi, softmax_ari=ari,acc_kmeans=acc_2_sum, reloss =reloss_sum/test_batch_num,
                                            fake_acc=fake_acc_sum / test_batch_num, bi_acc=bi_acc_sum / test_batch_num))
                    print(
                        "....eapoch %d ,step %d, acc_1: %f, acc_2: %f, reloss: %f, fake_acc: %f, bi_acc: %f ...." % (
                            epoch, counter, acc_1_sum , acc_2_sum,
                            reloss_sum / test_batch_num, fake_acc_sum / test_batch_num,
                            bi_acc_sum / test_batch_num))
        logfile.close()
        print('test')





