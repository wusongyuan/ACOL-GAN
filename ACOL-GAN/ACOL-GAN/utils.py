import numpy as np
import tensorflow as tf
import ops
import matplotlib.pyplot as plt
import matplotlib
import gc
import gzip
from six.moves import cPickle
import scipy.io as scio
import sys
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base

from tensorflow.examples.tutorials.mnist import input_data
import h5py


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
from keras.datasets import cifar10
from sklearn.manifold import TSNE
plt.switch_backend('agg')
import scipy.io as sio
def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

# def draw(data0, data1, data2,name):
#   imgs0 = np.reshape(data0, [-1, 28, 28])
#   imgs1 = np.reshape(data1, [-1, 28, 28])
#   imgs2 = np.reshape(data2, [-1, 28, 28])
#
#   plt.figure(figsize=(15,10))
#   for i in range(1, 65):
#     plt.subplot(8, 32, i)
#     plt.imshow(imgs0[i - 1], clim=[0, 1], cmap='bone')
#     plt.axis('off')
#
#
#   for i in range(1,65):
#     plt.subplot(8, 32, i + 96)
#     plt.imshow(imgs1[i - 1], clim=[0, 1], cmap='bone')
#     plt.axis('off')
#
#   plt.savefig("wgan_minit/"+str(name)+".png")
#   plt.clf()  # normally I use these lines to release the memory
#   plt.close()
#   del imgs0,imgs1,imgs2
#   gc.collect()

def draw_prcess(emd_data,pred,test_y, cts,noise, choise_list, name, dir):
  plt.figure(figsize=(4 * 400 / 100, 400 / 100))
  plt.subplot(141)
  # plt.xlim(-1, 1)
  # plt.ylim(-1, 1)
  p1 = emd_data[pred == 0]
  plt.plot(p1[:, 0], p1[:, 1], '.')
  p2 = emd_data[pred == 1]
  plt.plot(p2[:, 0], p2[:, 1], 'b+')
  p3 = emd_data[pred == 2]
  plt.plot(p3[:, 0], p3[:, 1], '*')
  p4 = emd_data[pred == 3]
  plt.plot(p4[:, 0], p4[:, 1], 'v')
  p5 = emd_data[pred == 4]
  plt.plot(p5[:, 0], p5[:, 1], '<')
  p6 = emd_data[pred == 5]
  plt.plot(p6[:, 0], p6[:, 1], '>')
  p7 = emd_data[pred == 6]
  plt.plot(p7[:, 0], p7[:, 1], '^')
  p8 = emd_data[pred == 7]
  plt.plot(p8[:, 0], p8[:, 1], '8')
  p9 = emd_data[pred == 8]
  plt.plot(p9[:, 0], p9[:, 1], 'p')
  p10 = emd_data[pred == 9]
  plt.plot(p10[:, 0], p10[:, 1], 'x')

  plt.subplot(142)
  # plt.xlim(-1, 1)
  # plt.ylim(-1, 1)
  p1 = emd_data[test_y == 0]
  plt.plot(p1[:, 0], p1[:, 1], '.')
  p2 = emd_data[test_y == 1]
  plt.plot(p2[:, 0], p2[:, 1], 'b+')
  p3 = emd_data[test_y == 2]
  plt.plot(p3[:, 0], p3[:, 1], '*')
  p4 = emd_data[test_y == 3]
  plt.plot(p4[:, 0], p4[:, 1], 'v')
  p5 = emd_data[test_y == 4]
  plt.plot(p5[:, 0], p5[:, 1], '<')
  p6 = emd_data[test_y == 5]
  plt.plot(p6[:, 0], p6[:, 1], '>')
  p7 = emd_data[test_y == 6]
  plt.plot(p7[:, 0], p7[:, 1], '^')
  p8 = emd_data[test_y == 7]
  plt.plot(p8[:, 0], p8[:, 1], '8')
  p9 = emd_data[test_y == 8]
  plt.plot(p9[:, 0], p9[:, 1], 'p')
  p10 = emd_data[test_y == 9]
  plt.plot(p10[:, 0], p10[:, 1], 'x')

  plt.subplot(143)
  plt.plot(cts[:, 0], cts[:, 1], '.')

  plt.subplot(144)
  # plt.xlim(-1, 1)
  # plt.ylim(-1, 1)
  p1 = noise[choise_list == 0]
  plt.plot(p1[:, 0], p1[:, 1], '.')
  p2 = noise[choise_list == 1]
  plt.plot(p2[:, 0], p2[:, 1], 'b+')
  p3 = noise[choise_list == 2]
  plt.plot(p3[:, 0], p3[:, 1], '*')
  p4 = noise[choise_list == 3]
  plt.plot(p4[:, 0], p4[:, 1], 'v')
  p5 = noise[choise_list == 4]
  plt.plot(p5[:, 0], p5[:, 1], '<')
  p6 = noise[choise_list == 5]
  plt.plot(p6[:, 0], p6[:, 1], '>')
  p7 = noise[choise_list == 6]
  plt.plot(p7[:, 0], p7[:, 1], '^')
  p8 = noise[choise_list == 7]
  plt.plot(p8[:, 0], p8[:, 1], '8')
  p9 = noise[choise_list == 8]
  plt.plot(p9[:, 0], p9[:, 1], 'p')
  p10 = noise[choise_list == 9]
  plt.plot(p10[:, 0], p10[:, 1], 'x')

  plt.savefig(dir+"/"+str(name)+".png")
  plt.clf()
  plt.close()
  del emd_data,pred,test_y, cts
  gc.collect()

def draw(data0, data1, data2, name, dir, dataset = 'MNIST'):
  if dataset == 'MNIST':
    shape = [-1, 28, 28]
  elif dataset == 'CIFAR10':
    shape = [-1, 32, 32, 3]
  elif dataset == 'hhar':
    shape = [-1, 561]
  imgs0 = np.reshape(data0, shape)
  imgs1 = np.reshape(data1, shape)
  imgs2 = np.reshape(data2, shape)

  plt.figure(figsize=(2 * 560 / 100, 2 * 560 / 100))
  gs = matplotlib.gridspec.GridSpec(2, 2)

  image = np.concatenate(np.split(imgs0, 10), axis=2)
  image = np.concatenate(image.tolist(), axis=0)
  ax = plt.subplot(gs[0, 0])
  if dataset == 'MNIST':
    plt.imshow(image, clim=[0, 1], cmap='bone')
  elif dataset == 'CIFAR10':
    plt.imshow(image)
  plt.text(0.47, 1., "original pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
  plt.axis('off')

  image = np.concatenate(np.split(imgs1, 10), axis=2)
  image = np.concatenate(image.tolist(), axis=0)
  ax = plt.subplot(gs[0, 1])
  if dataset == 'MNIST':
    plt.imshow(image, clim=[0, 1], cmap='bone')
  elif dataset == 'CIFAR10':
    plt.imshow(image)
  plt.text(0.47, 1., "reconstructed pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
  plt.axis('off')

  image = np.concatenate(np.split(imgs2, 10), axis=2)
  image = np.concatenate(image.tolist(), axis=0)
  ax = plt.subplot(gs[1, 1])
  if dataset == 'MNIST':
    plt.imshow(image, clim=[0, 1], cmap='bone')
  elif dataset == 'CIFAR10':
    plt.imshow(image)
  plt.text(0.47, 1., "generated pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
  plt.axis('off')

  plt.savefig(dir+"/"+str(name)+".png")
  plt.clf()
  plt.close()
  del imgs0,imgs1,imgs2
  gc.collect()

def draw_noise(img, data, y,name, dir, dataset = 'MNIST'):
  if dataset == 'MNIST':
    shape = [-1, 28, 28]
  elif dataset == 'CIFAR10':
    shape = [-1, 32, 32, 3]
  elif dataset == 'hhar':
    shape = [-1, 561]
  elif dataset == 'reuters10k':
    shape = [-1, 2000]
  elif dataset == 'stl':
    shape = [-1, 96,96,3]
  img = np.reshape(img, shape)
  logo = ['.', 'b+', '*', 'v', '<', '>', '^', '8', 'p', 'x']
  tsne = TSNE(n_components=2, init='pca', random_state=0)
  emd_data = tsne.fit_transform(data)
  plt.figure(figsize=(2 * 560 / 100, 1 * 560 / 100))
  gs = matplotlib.gridspec.GridSpec(1, 2)
  image = np.concatenate(np.split(img, 10), axis=2)
  image = np.concatenate(image.tolist(), axis=0)
  ax = plt.subplot(gs[0, 0])
  if dataset == 'MNIST':
    plt.imshow(image, clim=[0, 1], cmap='bone')
  elif dataset == 'CIFAR10' or dataset == 'stl':
    plt.imshow(image)
  plt.text(0.47, 1., "original pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
  plt.axis('off')

  ax = plt.subplot(gs[0, 1])
  for i in range(10):
    p = emd_data[y == i]
    plt.plot(p[:, 0], p[:, 1], logo[i])

  plt.text(0.47, 1., "noise distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)
  # plt.axis('off')
  plt.savefig(dir + "/" + str(name) + ".png")
  plt.clf()
  plt.close()
  del data,emd_data
  gc.collect()

def draw_gan_2(data0, data1, data2, emd_data, pred, test_y, choise_list, noise, pred_noise, name, dir, dataset = 'MNIST'):
  tsne = TSNE(n_components=2, init='pca', random_state=0)
  logo = ['.','b+','*','v','<','>','^','8','p','x']
  if dataset == 'MNIST':
    shape = [-1, 28, 28]
  elif dataset == 'CIFAR10':
    shape = [-1, 32, 32, 3]
  elif dataset == 'hhar':
    shape = [-1, 561]
  elif dataset == 'reuters10k':
    shape = [-1, 2000]
  elif dataset == 'stl':
    shape = [-1,96, 96, 3]

  imgs0 = np.reshape(data0, shape)
  imgs1 = np.reshape(data1, shape)
  imgs2 = np.reshape(data2, shape)

  plt.figure(figsize=(2 * 560 / 100, 2 * 560 / 100))
  gs = matplotlib.gridspec.GridSpec(2, 2)



  ax = plt.subplot(gs[0, 0])
  emd_data = tsne.fit_transform(emd_data)
  for i in range(10):
    p = emd_data[pred == i]
    plt.plot(p[:,0], p[:,1],logo[i])
  plt.text(0.47, 1., "pred distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)

  ax = plt.subplot(gs[0, 1])
  for i in range(10):
    p = emd_data[test_y == i]
    plt.plot(p[:,0], p[:,1],logo[i])
  plt.text(0.47, 1., "real distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)

  ax = plt.subplot(gs[1, 1])
  noise = tsne.fit_transform(noise)
  choise_list = np.array(choise_list)
  for i in range(10):
    p = noise[choise_list == i]
    plt.plot(p[:,0], p[:,1],logo[i])
  plt.text(0.47, 1., "real generate distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)

  ax = plt.subplot(gs[1, 0])
  for i in range(10):
    p = noise[pred_noise == i]
    plt.plot(p[:, 0], p[:, 1], logo[i])
  plt.text(0.47, 1., "pred generate distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)

  plt.savefig(dir + "/" + str(name) + ".png")
  plt.clf()
  plt.close()
  del imgs0, imgs1, imgs2
  gc.collect()

def draw_img(data0,data1, name, dir, dataset = 'MNIST'):
  if dataset == 'MNIST':
    shape = [-1, 28, 28]
  elif dataset == 'CIFAR10':
    shape = [-1, 32, 32, 3]
  elif dataset == 'hhar':
    shape = [-1, 561]
  elif dataset == 'reuters10k':
    shape = [-1, 2000]
  elif dataset == 'stl':
    shape = [-1, 96, 96, 3]
  elif dataset == 'usps':
    shape = [-1, 16, 16]

  plt.figure(figsize=(2 * 560 / 100, 1 * 560 / 100))
  gs = matplotlib.gridspec.GridSpec(1, 2)

  ax = plt.subplot(gs[0, 0])
  imgs0 = np.reshape(data0, shape)
  image = np.concatenate(np.split(imgs0, 10), axis=2)
  image = np.concatenate(image.tolist(), axis=0)
  plt.imshow(image, clim=[0, 1], cmap='bone')
  plt.text(0.47, 1., "real pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
  plt.axis('off')

  ax = plt.subplot(gs[0, 1])
  imgs0 = np.reshape(data1, shape)
  image = np.concatenate(np.split(imgs0, 10), axis=2)
  image = np.concatenate(image.tolist(), axis=0)
  plt.imshow(image, clim=[0, 1], cmap='bone')
  plt.text(0.47, 1., "rec pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
  plt.axis('off')

  plt.savefig(dir + "/" + str(name) + ".png")
  plt.clf()
  plt.close()
  del imgs0
  gc.collect()

def draw_distribution(emd_data, pred,dataset, name):
  logo = ['.', 'b+', '*', 'v', '<', '>', '^', '8', 'p', 'x', 's', 'h', ',', 'o', '1', '2', '3', '4', '8', 'P']
  plt.figure()
  for i in range(10):
    p = emd_data[pred == i]
    plt.plot(p[:,0], p[:,1],logo[i])
  plt.savefig('features/'+dataset+'/img/'+str(name) + ".png")

def draw_gan(data0, data1, data2, emd_data, pred, test_y, choise_list, noise, pred_noise, name, dir, dataset = 'MNIST', num_cluster = 10):
  tsne = TSNE(n_components=2, init='pca', random_state=0)
  logo = ['.','b+','*','v','<','>','^','8','p','x','s','h',',','o','1','2','3','4','8','P']

  if dataset == 'MNIST':
    shape = [-1, 28, 28]
  elif dataset == 'CIFAR10':
    shape = [-1, 32, 32, 3]
  elif dataset == 'hhar':
    shape = [-1, 561]
  elif dataset == 'reuters10k':
    shape = [-1, 2000]
  elif dataset == 'stl':
    shape = [-1,96, 96, 3]
  elif dataset == 'usps':
    shape = [-1,16, 16]
  imgs0 = np.reshape(data0, shape)
  imgs1 = np.reshape(data1, shape)
  imgs2 = np.reshape(data2, shape)
  emd_data = np.reshape(emd_data, [imgs2.shape[0], -1])
  noise = np.reshape(noise, [imgs2.shape[0], -1])
  plt.figure(figsize=(4 * 560 / 100, 2 * 560 / 100))
  gs = matplotlib.gridspec.GridSpec(2, 4)

  if dataset == 'CIFAR10' or dataset == 'MNIST' or dataset == 'stl' or dataset == 'usps':
    image = np.concatenate(np.split(imgs0, 10), axis=2)
    image = np.concatenate(image.tolist(), axis=0)
    ax = plt.subplot(gs[0, 0])
    plt.imshow(image, clim=[0, 1], cmap='bone')
    plt.text(0.47, 1., "original pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
    plt.axis('off')

    image = np.concatenate(np.split(imgs1, 10), axis=2)
    image = np.concatenate(image.tolist(), axis=0)
    ax = plt.subplot(gs[0, 1])
    plt.imshow(image, clim=[0, 1], cmap='bone')
    plt.text(0.47, 1., "reconstructed pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
    plt.axis('off')

    image = np.concatenate(np.split(imgs2, 10), axis=2)
    image = np.concatenate(image.tolist(), axis=0)
    ax = plt.subplot(gs[0, 2])
    plt.imshow(image, clim=[0, 1], cmap='bone')
    plt.text(0.47, 1., "generated pic", ha="center", va="bottom", size=20, transform=ax.transAxes)
    plt.axis('off')

  ax = plt.subplot(gs[1, 0])
  emd_data = tsne.fit_transform(emd_data)
  for i in range(10):
    p = emd_data[pred == i]
    plt.plot(p[:,0], p[:,1],logo[i])
  plt.text(0.47, 1., "pred distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)



  ax = plt.subplot(gs[1, 1])
  for i in range(10):
    p = emd_data[test_y == i]
    plt.plot(p[:,0], p[:,1],logo[i])
  plt.text(0.47, 1., "real distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)


  np.savetxt('features/'+dataset+'/features/'+str(name)+'.txt',emd_data)
  np.savetxt('features/' + dataset + '/labels/' + str(name)+'.txt', test_y)


  ax = plt.subplot(gs[1, 2])
  noise = tsne.fit_transform(noise)
  choise_list = np.array(choise_list)
  for i in range(10):
    p = noise[choise_list == i]
    plt.plot(p[:,0], p[:,1],logo[i])
  plt.text(0.47, 1., "real generate distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)

  ax = plt.subplot(gs[1, 3])
  for i in range(10):
    p = noise[pred_noise == i]
    plt.plot(p[:, 0], p[:, 1], logo[i])
  plt.text(0.47, 1., "pred generate distribution", ha="center", va="bottom", size=20, transform=ax.transAxes)

  plt.savefig(dir + "/" + str(name) + ".png")
  plt.clf()
  plt.close()
  del imgs0, imgs1, imgs2
  gc.collect()


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      # if reshape:
      #   assert images.shape[3] == 1
      #   images = images.reshape(images.shape[0],
      #                           images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        # images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in range(batch_size)], [
          fake_label for _ in range(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), np.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def load_data(dataset,dtype=dtypes.float32,reshape=True,validation_size=1000,test_size = 1000, seed=None):
  path = 'dataset/' + dataset + '/'
  if dataset == 'mnist':
    path = path + 'mnist.pkl.gz'
    if path.endswith(".gz"):
      f = gzip.open(path, 'rb')
    else:
      f = open(path, 'rb')

    if sys.version_info < (3,):
      (x_train, y_train), (x_test, y_test) = cPickle.load(f)
    else:
      (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")

    f.close()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    train_images = np.concatenate((x_train, x_test))
    train_labels = np.concatenate((y_train, y_test))

  if dataset == 'reuters10k':
    # data = scio.loadmat(path + 'reuters10k.mat')reutersidf10k.npy
    # train_images = data['X']
    # train_labels = data['Y'].squeeze()

    data = np.load(path + 'reutersidf10k.npy').item()
    train_images = data['data']
    train_labels = data['label'].squeeze()

  if dataset == 'fashion':
    dataset = input_data.read_data_sets('data/fashion')
    x_train= dataset.train.images
    y_train = dataset.train.labels
    x_test= dataset.test.images
    y_test = dataset.test.labels

    train_images = np.concatenate((x_train, x_test))
    train_labels = np.concatenate((y_train, y_test))

  if dataset == 'usps':
    x_train, y_train, x_test, y_test= read_usps_dataset()
    train_images = np.concatenate((x_train, x_test))
    train_labels = np.concatenate((y_train, y_test))
    # data0 = train_labels[train_labels==0].shape[0]
    # data1 = train_labels[train_labels==1].shape[0]
    # data2 = train_labels[train_labels==2].shape[0]
    # data3 = train_labels[train_labels==3].shape[0]
    # data4 = train_labels[train_labels==4].shape[0]
    # data5 = train_labels[train_labels==5].shape[0]
    # data6 = train_labels[train_labels==6].shape[0]
    # data7 = train_labels[train_labels==7].shape[0]
    # data8 = train_labels[train_labels==8].shape[0]
    # data9 = train_labels[train_labels==9].shape[0]
    # print('test')

  if dataset == 'har':
    data = scio.loadmat(path + 'HAR.mat')
    X = data['X']
    X = X.astype('float32')
    Y = data['Y'] - 1
    train_images = X[:10200]
    train_labels = Y[:10200]

    # data0 = train_labels[train_labels == 0].shape
    # data1 = train_labels[train_labels == 1].shape
    # data2 = train_labels[train_labels == 2].shape
    # data4 = train_labels[train_labels == 3].shape
    # data5 = train_labels[train_labels == 4].shape
    # data6 = train_labels[train_labels == 5].shape
    # size = 1
  # if dataset == 'reuters':
  #   (X_train, y_train), (X_test, y_test) = reuters.load_data()
  #   train_images = np.array([])
  #   test_images = np.array([])
  #   for i in range(len(X_train)):
  #     train_images = np.concatenate((train_images,np.array(X_train[i], dtype=np.float32)), axis=0)
  #   train_images = np.reshape(train_images,[-1, len([X_train[0]])])
  #   train_labels = np.array(y_train, dtype=np.int32)
  #
  #   for i in range(len(X_test)):
  #     test_images = np.concatenate((test_images, np.array(X_test[i], dtype=np.float32)), axis=0)
  #   test_images = np.reshape(test_images,[-1, len([X_test[0]])])
  #   test_labels = np.array(y_test, dtype=np.int32)
  #   test_size = 1

  # if not 0 <= validation_size <= len(train_images):
  #   raise ValueError('Validation size should be between 0 and {}. Received: {}.'
  #                    .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]

  train_images = train_images
  train_labels = train_labels
  test_images = train_images
  test_labels = train_labels

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)

def get_cifar10(dtype=dtypes.float32,reshape=True,seed=None):
  train,test = cifar10.load_data()
  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train[0] / 255, train[1], **options)
  test = DataSet(test[0] / 255, test[1], **options)

  dataset = base.Datasets(train=train, validation=None, test=test)
  return dataset


def read_usps_dataset(filename='dataset/usps/usps.h5'):
    with h5py.File(filename, 'r') as hf:
      train = hf.get('train')
      X_tr = train.get('data')[:]
      y_tr = train.get('target')[:]
      test = hf.get('test')
      X_te = test.get('data')[:]
      y_te = test.get('target')[:]
    return X_tr,y_tr,X_te,y_te


def get_reuters(dtype=dtypes.float32,reshape=True,seed=None):

  X_train, y_train, X_test, y_test = make_reuters_data()
  # train_images = np.array([])
  # test_images = np.array([])
  # for i in range(len(X_train)):
  #   train_images = np.concatenate((train_images,np.array(X_train[i], dtype=np.float32)), axis=0)
  # train_images = np.reshape(train_images,[-1, len(X_train[0])])
  # train_labels = np.array(y_train, dtype=np.int32)

  # for i in range(len(X_test)):
  #   test_images = np.concatenate((test_images, np.array(X_test[i], dtype=np.float32)), axis=0)
  # test_images = np.reshape(test_images,[-1, len([X_test[0]])])
  # test_labels = np.array(y_test, dtype=np.int32)

  # data0 = train_labels[train_labels == 0].shape[0]
  # data1 = train_labels[train_labels == 1].shape[0]
  # data2 = train_labels[train_labels == 2].shape[0]
  # data3 = train_labels[train_labels == 3].shape[0]
  #
  # test_size = 1

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  train = DataSet(X_train, y_train, **options)
  test = DataSet(X_test, y_test, **options)

  dataset = base.Datasets(train=train, validation=None, test=test)
  return dataset


def load_svhn(order='tf', path=None):

    # input image dimensions
    img_rows, img_cols, img_channels,  = 32, 32, 3
    nb_classes = 10

    if order == 'tf':
        input_shape=(img_rows, img_cols, img_channels)
    elif order == 'th':
        input_shape=(img_channels, img_rows, img_cols)

    if path is None:
        train_data = sio.loadmat('handwritten/SVHN//train_32x32.mat')
    else:
        train_data = sio.loadmat(path + 'train_32x32.mat')

    # access to the dict
    X_train = train_data['X']
    if order == 'tf':
        X_train = X_train.reshape(img_channels*img_rows*img_cols, X_train.shape[-1]).T
        X_train = X_train.reshape(len(X_train), input_shape[0], input_shape[1], input_shape[2])
    elif order == 'th':
        X_train = X_train.T.swapaxes(2,3)
    X_train = X_train.astype('float32')
    X_train /= 255

    y_train = train_data['y']
    y_train = y_train.reshape(len(y_train))
    y_train = y_train%nb_classes

    del train_data

    if path is None:
        test_data = sio.loadmat('handwritten/SVHN/test_32x32.mat')
    else:
        test_data = sio.loadmat(path + 'test_32x32.mat')

    # access to the dict
    X_test = test_data['X']
    if order == 'tf':
        X_test = X_test.reshape(img_channels*img_rows*img_cols, X_test.shape[-1]).T
        X_test = X_test.reshape(len(X_test), input_shape[0], input_shape[1], input_shape[2])
    elif order == 'th':
        X_test = X_test.T.swapaxes(2,3)
    X_test = X_test.astype('float32')
    X_test /= 255

    y_test= test_data['y']
    y_test = y_test.reshape(y_test.shape[0])
    y_test = y_test%nb_classes

    del test_data

    # if extra:
    #     if path is None:
    #         extra_data = sio.loadmat('/home/ozsel/Jupyter/datasets/svhn/extra_32x32.mat')
    #     else:
    #         extra_data = sio.loadmat(path + 'extra_32x32.mat')
    #
    #     # access to the dict
    #     X_extra = extra_data['X']
    #     if order == 'tf':
    #         X_extra = X_extra.reshape(img_channels*img_rows*img_cols, X_extra.shape[-1]).T
    #         X_extra = X_extra.reshape(len(X_extra), input_shape[0], input_shape[1], input_shape[2])
    #     elif order == 'th':
    #         X_extra = X_extra.T.swapaxes(2,3)
    #     X_extra = X_extra.astype('float32')
    #     X_extra /= 255
    #
    #     y_extra= extra_data['y']
    #     y_extra = y_extra.reshape(y_extra.shape[0])
    #     y_extra = y_extra%nb_classes
    #
    #     del extra_data
    # else:
    #     X_extra = None
    #     y_extra = None

    return (X_train, y_train), (X_test, y_test)

def get_svhn(dtype=dtypes.float32,reshape=True,seed=None):
  train,test = load_svhn()
  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train[0], train[1], **options)
  test = DataSet(test[0], test[1], **options)

  dataset = base.Datasets(train=train, validation=None, test=test)
  return dataset

def make_reuters_data(data_dir='dataset/reuters'):
  np.random.seed(1234)
  from sklearn.feature_extraction.text import CountVectorizer
  from os.path import join
  did_to_cat = {}
  cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
  with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
    for line in fin.readlines():
      line = line.strip().split(' ')
      cat = line[0]
      did = int(line[1])
      if cat in cat_list:
        did_to_cat[did] = did_to_cat.get(did, []) + [cat]
    # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
    for did in list(did_to_cat.keys()):
      if len(did_to_cat[did]) > 1:
        del did_to_cat[did]

  dat_list = ['lyrl2004_tokens_test_pt0.dat',
              'lyrl2004_tokens_test_pt1.dat',
              'lyrl2004_tokens_test_pt2.dat',
              'lyrl2004_tokens_test_pt3.dat',
              'lyrl2004_tokens_train.dat']
  data = []
  target = []
  cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
  del did
  for dat in dat_list:
    with open(join(data_dir, dat)) as fin:
      for line in fin.readlines():
        if line.startswith('.I'):
          if 'did' in locals():
            assert doc != ''
            if did in did_to_cat:
              data.append(doc)
              target.append(cat_to_cid[did_to_cat[did][0]])
          did = int(line.strip().split(' ')[1])
          doc = ''
        elif line.startswith('.W'):
          assert doc == ''
        else:
          doc += line

  print((len(data), 'and', len(did_to_cat)))
  assert len(data) == len(did_to_cat)

  x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
  y = np.asarray(target)

  from sklearn.feature_extraction.text import TfidfTransformer
  x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
  x = x[:685000].astype(np.float32)
  print(x.dtype, x.size)
  y = y[:685000]
  x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
  print('todense succeed')

  p = np.random.permutation(x.shape[0])
  x = x[p]
  y = y[p]
  print('permutation finished')

  return x, y, x, y

def get_stl10(dtype=dtypes.float32,reshape=True,seed=None):

  train = scio.loadmat('D:\\科研\\GanGSCluster\\dataset\\stl10_matlab\\train.mat')
  test = scio.loadmat('D:\\科研\\GanGSCluster\\dataset\\stl10_matlab\\test.mat')
  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train['X'] / 255, train['y'].squeeze(), **options)
  test = DataSet(test['X'] / 255, test['y'].squeeze(), **options)

  dataset = base.Datasets(train=train, validation=None, test=test)
  return dataset
