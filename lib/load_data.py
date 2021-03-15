# functions to load different datasets (mnist, fmnist, cifar10) 

import gzip, zipfile, tarfile
import os, shutil, re, string, fnmatch
import urllib.request
import pickle as pkl
import numpy as np
import sys
import imageio
from PIL import Image
        
def _download_mnist(dataset):
    """
    download mnist dataset if not present
    """
    origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
    print('Downloading data from %s' %origin)
    urllib.request.urlretrieve(origin, dataset)


def _download_cifar10(dataset):
    """
    download cifar10 dataset if not present
    """
    origin = ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    
    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin,dataset)

def _download_fmnist(dataset,subset,labels=False):

    if subset=='test':
        subset = 't10k'
    if labels:
        origin = ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/%s-labels-idx1-ubyte.gz'%subset)
    else:
        origin = ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/%s-images-idx3-ubyte.gz'%subset)

    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin,dataset)


def _get_datafolder_path():
    """
    returns data path
    """
    #where am I? return full path
    full_path = os.path.abspath('./')
    path = full_path +'/data'
    return path


def download_mnist(data_dir):
  dataset=os.path.join(data_dir,'mnist/mnist.pkl.gz')

  if not os.path.isfile(dataset):
    datasetfolder = os.path.dirname(dataset)
    if not os.path.exists(datasetfolder):
      print('creating ', datasetfolder)
      os.makedirs(datasetfolder)
    _download_mnist(dataset)
		

def download_fmnist(data_dir):
  data = {}
  for subset in ['train','test']:
    data[subset]={}
    for labels in [True,False]:
      if labels:
        dataset=os.path.join(data_dir,'fmnist/fmnist_%s_labels.gz'%subset)
      else:
        dataset=os.path.join(data_dir,'fmnist/fmnist_%s_images.gz'%subset)
      datasetfolder = os.path.dirname(dataset)
      if not os.path.isfile(dataset):
        if not os.path.exists(datasetfolder):
          os.makedirs(datasetfolder)
        _download_fmnist(dataset,subset,labels)
      with gzip.open(dataset, 'rb') as path:
        if labels:
          data[subset]['labels'] = np.frombuffer(path.read(), dtype=np.uint8,offset=8)
        else:
          data[subset]['images'] = np.frombuffer(path.read(), dtype=np.uint8,offset=16)


def load_mnist(data_dir,flatten=False,params=None):
	
    dataset=os.path.join(data_dir,'mnist/mnist.pkl.gz')

    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            print('creating ', datasetfolder)
            os.makedirs(datasetfolder)
        _download_mnist(dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
    f.close()

    if flatten:
        x_train, targets_train = train_set[0], train_set[1]
        x_test,  targets_test  = test_set[0], test_set[1]
        x_valid, targets_valid = valid_set[0], valid_set[1]
    else:
        x_train, targets_train = train_set[0].reshape((-1,28,28,1)).astype("float32") , train_set[1]
        x_test,  targets_test  = test_set[0].reshape((-1,28,28,1)).astype("float32"), test_set[1]
        x_valid, targets_valid = valid_set[0].reshape((-1,28,28,1)).astype("float32"), valid_set[1]
    
    return x_train, targets_train, x_valid, targets_valid, x_test, targets_test


def load_fmnist(data_dir,flatten=False,params=None):
   
    data = {}
    for subset in ['train','test']:
        data[subset]={}
        for labels in [True,False]:
            if labels:
                dataset=os.path.join(data_dir,'fmnist/fmnist_%s_labels.gz'%subset)
            else:
                dataset=os.path.join(data_dir,'fmnist/fmnist_%s_images.gz'%subset)
            datasetfolder = os.path.dirname(dataset)
            if not os.path.isfile(dataset):
                if not os.path.exists(datasetfolder):
                    os.makedirs(datasetfolder)
                _download_fmnist(dataset,subset,labels)
            with gzip.open(dataset, 'rb') as path:
                if labels:
                    data[subset]['labels'] = np.frombuffer(path.read(), dtype=np.uint8,offset=8)
                else:
                    data[subset]['images'] = np.frombuffer(path.read(), dtype=np.uint8,offset=16)
                    
    x_train = data['train']['images'].reshape((-1,28*28))
    x_test  = data['test']['images'].reshape((-1,28*28))
    if not flatten:
        x_train = x_train.reshape((-1,28,28,1)).astype("float32")/255
        x_test  = x_test.reshape((-1,28,28,1)).astype("float32")/255

    y_train = data['train']['labels']
    y_test  = data['test']['labels']

    return x_train, y_train, x_test, y_test, None, None
	
	
	

def load_mnist_vs_fmnist(data_dir,flatten=False,params=None):

	#mnist
  _,_,_,_, x_test,_ = load_mnist(data_dir,flatten)
	##mix
  _,_, xf_test,_,_,_ = load_fmnist(data_dir,flatten)

  xn_test = np.concatenate((xf_test[0:5000],x_test[0:5000]))
  yn_test = np.concatenate((np.zeros(5000),np.ones(5000)))
  
  return xn_test, yn_test


