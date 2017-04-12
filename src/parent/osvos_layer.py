caffe_root = '../../caffe_osvos/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image
import scipy.io
import cv2
import random

class OSVOSDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.data_root_dir = params['data_root_dir']
        self.mean = np.array((104.00699, 116.66877, 122.67892), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.pair_list = params.get('pair_list')

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data, and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = self.pair_list
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        #if 'train' not in self.split:
        #    self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        #print(self.idx)
        #sys.stdout.flush()
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])

        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        im = Image.open('{}/{}'.format(self.data_root_dir, idx.split()[0]))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
        gt = Image.open('{}/{}'.format(self.data_root_dir, idx.split()[1])).convert('L')
        label = np.array(gt, dtype=np.float32)/255
        label = label[np.newaxis, ...]
        return label

