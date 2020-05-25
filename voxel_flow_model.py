"""Implements a voxel flow model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.loss_utils import l1_loss, l2_loss, vae_loss, hue_loss, lightness_loss
from utils.geo_layer_utils import vae_gaussian_layer
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid

class Voxel_flow_model(object):
  def __init__(self, is_train=True):
    self.is_train = is_train

  def inference(self, input_images):
    """Inference on a set of input_images.
    Args:
    """
    return self._build_model(input_images) 

  def loss(self, predictions, targets):
    """Compute the necessary loss for training.
    Args:
    Returns:
    """
    # self.reproduction_loss = l2_loss(predictions, targets) #l2_loss(predictions, targets)
    # self.reproduction_loss = hue_loss(predictions, targets)
    self.reproduction_loss = lightness_loss(predictions, targets)
    # self.prior_loss = vae_loss(self.z_mean, self.z_logvar, prior_weight = 0.0001)

    # return [self.reproduction_loss, self.prior_loss]
    return self.reproduction_loss

  def _build_model(self, input_images):
    """Build a VAE model.
    Args:
    """

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
      
      # Define network      
      batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'is_training': self.is_train,
      }
      with slim.arg_scope([slim.batch_norm], is_training = self.is_train, updates_collections=None):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
          net = slim.conv2d(input_images, 64, [5, 5], stride=1, scope='conv1')
          net = slim.max_pool2d(net, [2, 2], scope='pool1')
          net = slim.conv2d(net, 128, [5, 5], stride=1, scope='conv2')
          net = slim.max_pool2d(net, [2, 2], scope='pool2')
          net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')
          net = slim.max_pool2d(net, [2, 2], scope='pool3')
          net = tf.image.resize_bilinear(net, [64,64])
          net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
          net = tf.image.resize_bilinear(net, [128,128])
          net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv5')
          net = tf.image.resize_bilinear(net, [256,256])
          net = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv6')
    num_in=4
    num_out=4
    net = slim.conv2d(net, (num_in+1)*num_out, [5, 5], stride=1, activation_fn=tf.tanh,
    normalizer_fn=None, scope='conv7')
    print('/n/n/n/n')
    print(net.get_shape())
    print('/n/n/n/n')
    g=globals()
    l=locals()
    # mask=np.zeros(shape=(None,256,256,10,4))
    for i in range(num_out):
        exec("flow{}=net[:,:,:,{}:{}]".format(i+1,(num_in+1)*i,(num_in+1)*i+2),g,l)
        for j in range(num_in-1):
            exec("mask{}_{}=tf.expand_dims(net[:,:,:,{}],3)".format(j+1,i+1,(num_in+1)*i+2+j),g,l)
            # exec("mask{}_{}=locals()['mask{}_{}']".format(j+1,i+1,j+1,i+1))
            # print(net[:,:,:,(num_in+1)*i+2+j])
    
            # mask[:,:,:,j+1,i+1]=net[:,:,:,(num_in+1)*i+2+j]
    # print(locals())
    # print(mask[:,:,:,1,1])
            
    # flow = net[:, :, :, 0:2]
    # mask1 = tf.expand_dims(net[:, :, :, 2], 3)
    # mask2 = tf.expand_dims(net[:, :, :, 3], 3)
    # mask3 = tf.expand_dims(net[:, :, :, 4], 3)
    # mask4 = tf.expand_dims(net[:, :, :, 5], 3)
    # mask5 = tf.expand_dims(net[:, :, :, 6], 3)
    # mask6 = tf.expand_dims(net[:, :, :, 7], 3)
    # mask7 = tf.expand_dims(net[:, :, :, 8], 3)
    # mask8 = tf.expand_dims(net[:, :, :, 9], 3)
    # mask9 = tf.expand_dims(net[:, :, :, 10], 3)

    # flow2 = net[:, :, :, 11:13]
    # mask1_2 = tf.expand_dims(net[:, :, :, 13], 3)
    # mask2_2 = tf.expand_dims(net[:, :, :, 14], 3)
    # mask3_2 = tf.expand_dims(net[:, :, :, 15], 3)
    # mask4_2 = tf.expand_dims(net[:, :, :, 16], 3)
    # mask5_2 = tf.expand_dims(net[:, :, :, 17], 3)
    # mask6_2 = tf.expand_dims(net[:, :, :, 18], 3)
    # mask7_2 = tf.expand_dims(net[:, :, :, 19], 3)
    # mask8_2 = tf.expand_dims(net[:, :, :, 20], 3)
    # mask9_2 = tf.expand_dims(net[:, :, :, 21], 3)

    # flow3 = net[:, :, :, 22:24]
    # mask1_3 = tf.expand_dims(net[:, :, :, 24], 3)
    # mask2_3 = tf.expand_dims(net[:, :, :, 25], 3)
    # mask3_3 = tf.expand_dims(net[:, :, :, 26], 3)
    # mask4_3 = tf.expand_dims(net[:, :, :, 27], 3)
    # mask5_3 = tf.expand_dims(net[:, :, :, 28], 3)
    # mask6_3 = tf.expand_dims(net[:, :, :, 29], 3)
    # mask7_3 = tf.expand_dims(net[:, :, :, 30], 3)
    # mask8_3 = tf.expand_dims(net[:, :, :, 31], 3)
    # mask9_3 = tf.expand_dims(net[:, :, :, 32], 3)

    # flow4 = net[:, :, :, 33:35]
    # mask1_4 = tf.expand_dims(net[:, :, :, 35], 3)
    # mask2_4 = tf.expand_dims(net[:, :, :, 36], 3)
    # mask3_4 = tf.expand_dims(net[:, :, :, 37], 3)
    # mask4_4 = tf.expand_dims(net[:, :, :, 38], 3)
    # mask5_4 = tf.expand_dims(net[:, :, :, 39], 3)
    # mask6_4 = tf.expand_dims(net[:, :, :, 40], 3)
    # mask7_4 = tf.expand_dims(net[:, :, :, 41], 3)
    # mask8_4 = tf.expand_dims(net[:, :, :, 42], 3)
    # mask9_4 = tf.expand_dims(net[:, :, :, 43], 3)

    exec('grid_x, grid_y = meshgrid(256, 256)',g,l)
    exec('grid_x = tf.tile(grid_x, [8, 1, 1])',g,l) # batch_size = 32
    exec('grid_y = tf.tile(grid_y, [8, 1, 1])',g,l) # batch_size = 32

    for i in range(num_out):
        exec('flow{}=0.5*flow{}'.format(i+1,i+1),g,l)
    # flow = 0.5 * flow
    # flow2 = 0.5 * flow2
    # flow3 = 0.5 * flow3
    # flow4 = 0.5 * flow4

    for i in range(num_out):
        for j in range(num_in):
            if j<num_in//2:
                coff=1-2/num_in*j
            else:
                coff=1-2/num_in*(j+1)
            exec('coor_x_{}_{}=grid_x+{}*flow{}[:,:,:,0]'.format(j+1,i+1,coff,i+1),g,l)
            exec('coor_y_{}_{}=grid_y+{}*flow{}[:,:,:,1]'.format(j+1,i+1,coff,i+1),g,l)


    # coor_x_1 = grid_x + flow[:, :, :, 0]
    # coor_y_1 = grid_y + flow[:, :, :, 1]

    # coor_x_2 = grid_x + 0.8*flow[:, :, :, 0]
    # coor_y_2 = grid_y + 0.8*flow[:, :, :, 1]

    # coor_x_3 = grid_x + 0.6*flow[:, :, :, 0]
    # coor_y_3 = grid_y + 0.6*flow[:, :, :, 1]

    # coor_x_4 = grid_x + 0.4*flow[:, :, :, 0]
    # coor_y_4 = grid_y + 0.4*flow[:, :, :, 1]

    # coor_x_5 = grid_x + 0.2*flow[:, :, :, 0]
    # coor_y_5 = grid_y + 0.2*flow[:, :, :, 1]

    # coor_x_6 = grid_x - 0.2*flow[:, :, :, 0]
    # coor_y_6 = grid_y - 0.2*flow[:, :, :, 1]

    # coor_x_7 = grid_x - 0.4*flow[:, :, :, 0]
    # coor_y_7 = grid_y - 0.4*flow[:, :, :, 1]

    # coor_x_8 = grid_x - 0.6*flow[:, :, :, 0]
    # coor_y_8 = grid_y - 0.6*flow[:, :, :, 1]

    # coor_x_9 = grid_x - 0.8*flow[:, :, :, 0]
    # coor_y_9 = grid_y - 0.8*flow[:, :, :, 1]

    # coor_x_10 = grid_x - flow[:, :, :, 0]
    # coor_y_10 = grid_y - flow[:, :, :, 1]    
    

    # coor_x_1_2 = grid_x + flow2[:, :, :, 0]
    # coor_y_1_2 = grid_y + flow2[:, :, :, 1]

    # coor_x_2_2 = grid_x + 0.8*flow2[:, :, :, 0]
    # coor_y_2_2 = grid_y + 0.8*flow2[:, :, :, 1]

    # coor_x_3_2 = grid_x + 0.6*flow2[:, :, :, 0]
    # coor_y_3_2 = grid_y + 0.6*flow2[:, :, :, 1]

    # coor_x_4_2 = grid_x + 0.4*flow2[:, :, :, 0]
    # coor_y_4_2 = grid_y + 0.4*flow2[:, :, :, 1]

    # coor_x_5_2 = grid_x + 0.2*flow2[:, :, :, 0]
    # coor_y_5_2 = grid_y + 0.2*flow2[:, :, :, 1]

    # coor_x_6_2 = grid_x - 0.2*flow2[:, :, :, 0]
    # coor_y_6_2 = grid_y - 0.2*flow2[:, :, :, 1]

    # coor_x_7_2 = grid_x - 0.4*flow2[:, :, :, 0]
    # coor_y_7_2 = grid_y - 0.4*flow2[:, :, :, 1]

    # coor_x_8_2 = grid_x - 0.6*flow2[:, :, :, 0]
    # coor_y_8_2 = grid_y - 0.6*flow2[:, :, :, 1]

    # coor_x_9_2 = grid_x - 0.8*flow2[:, :, :, 0]
    # coor_y_9_2 = grid_y - 0.8*flow2[:, :, :, 1]

    # coor_x_10_2 = grid_x - flow2[:, :, :, 0]
    # coor_y_10_2 = grid_y - flow2[:, :, :, 1]    

    # coor_x_1_3 = grid_x + flow3[:, :, :, 0]
    # coor_y_1_3 = grid_y + flow3[:, :, :, 1]

    # coor_x_2_3 = grid_x + 0.8*flow3[:, :, :, 0]
    # coor_y_2_3 = grid_y + 0.8*flow3[:, :, :, 1]

    # coor_x_3_3 = grid_x + 0.6*flow3[:, :, :, 0]
    # coor_y_3_3 = grid_y + 0.6*flow3[:, :, :, 1]

    # coor_x_4_3 = grid_x + 0.4*flow3[:, :, :, 0]
    # coor_y_4_3 = grid_y + 0.4*flow3[:, :, :, 1]

    # coor_x_5_3 = grid_x + 0.2*flow3[:, :, :, 0]
    # coor_y_5_3 = grid_y + 0.2*flow3[:, :, :, 1]

    # coor_x_6_3 = grid_x - 0.2*flow3[:, :, :, 0]
    # coor_y_6_3 = grid_y - 0.2*flow3[:, :, :, 1]

    # coor_x_7_3 = grid_x - 0.4*flow3[:, :, :, 0]
    # coor_y_7_3 = grid_y - 0.4*flow3[:, :, :, 1]

    # coor_x_8_3 = grid_x - 0.6*flow3[:, :, :, 0]
    # coor_y_8_3 = grid_y - 0.6*flow3[:, :, :, 1]

    # coor_x_9_3 = grid_x - 0.8*flow3[:, :, :, 0]
    # coor_y_9_3 = grid_y - 0.8*flow3[:, :, :, 1]

    # coor_x_10_3 = grid_x - flow3[:, :, :, 0]
    # coor_y_10_3 = grid_y - flow3[:, :, :, 1]    

    # coor_x_1_4 = grid_x + flow4[:, :, :, 0]
    # coor_y_1_4 = grid_y + flow4[:, :, :, 1]

    # coor_x_2_4 = grid_x + 0.8*flow4[:, :, :, 0]
    # coor_y_2_4 = grid_y + 0.8*flow4[:, :, :, 1]

    # coor_x_3_4 = grid_x + 0.6*flow4[:, :, :, 0]
    # coor_y_3_4 = grid_y + 0.6*flow4[:, :, :, 1]

    # coor_x_4_4 = grid_x + 0.4*flow4[:, :, :, 0]
    # coor_y_4_4 = grid_y + 0.4*flow4[:, :, :, 1]

    # coor_x_5_4 = grid_x + 0.2*flow4[:, :, :, 0]
    # coor_y_5_4 = grid_y + 0.2*flow4[:, :, :, 1]

    # coor_x_6_4 = grid_x - 0.2*flow4[:, :, :, 0]
    # coor_y_6_4 = grid_y - 0.2*flow4[:, :, :, 1]

    # coor_x_7_4 = grid_x - 0.4*flow4[:, :, :, 0]
    # coor_y_7_4 = grid_y - 0.4*flow4[:, :, :, 1]

    # coor_x_8_4 = grid_x - 0.6*flow4[:, :, :, 0]
    # coor_y_8_4 = grid_y - 0.6*flow4[:, :, :, 1]

    # coor_x_9_4 = grid_x - 0.8*flow4[:, :, :, 0]
    # coor_y_9_4 = grid_y - 0.8*flow4[:, :, :, 1]

    # coor_x_10_4 = grid_x - flow4[:, :, :, 0]
    # coor_y_10_4 = grid_y - flow4[:, :, :, 1]   

    for j in range(num_in):
        exec("output_{}_1=bilinear_interp(input_images[:,:,:,{}:{}],coor_x_{}_1,coor_y_{}_1,'interpolate')".format(j+1,j*3,(j+1)*3,j+1,j+1),g,l)

    # output_1 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1, coor_y_1, 'interpolate')
    # output_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2, coor_y_2, 'interpolate')
    # output_3 = bilinear_interp(input_images[:, :, :, 6:9], coor_x_3, coor_y_3, 'interpolate')
    # output_4 = bilinear_interp(input_images[:, :, :, 9:12], coor_x_4, coor_y_4, 'interpolate')
    # output_5 = bilinear_interp(input_images[:, :, :, 12:15], coor_x_5, coor_y_5, 'interpolate')
    # output_6 = bilinear_interp(input_images[:, :, :, 15:18], coor_x_6, coor_y_6, 'interpolate')
    # output_7 = bilinear_interp(input_images[:, :, :, 18:21], coor_x_7, coor_y_7, 'interpolate')
    # output_8 = bilinear_interp(input_images[:, :, :, 21:24], coor_x_8, coor_y_8, 'interpolate')
    # output_9 = bilinear_interp(input_images[:, :, :, 24:27], coor_x_9, coor_y_9, 'interpolate')
    # output_10 = bilinear_interp(input_images[:, :, :, 27:30], coor_x_10, coor_y_10, 'interpolate')

    # output_1_2 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1_2, coor_y_1_2, 'interpolate')
    # output_2_2 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2_2, coor_y_2_2, 'interpolate')
    # output_3_2 = bilinear_interp(input_images[:, :, :, 6:9], coor_x_3_2, coor_y_3_2, 'interpolate')
    # output_4_2 = bilinear_interp(input_images[:, :, :, 9:12], coor_x_4_2, coor_y_4_2, 'interpolate')
    # output_5_2 = bilinear_interp(input_images[:, :, :, 12:15], coor_x_5_2, coor_y_5_2, 'interpolate')
    # output_6_2 = bilinear_interp(input_images[:, :, :, 15:18], coor_x_6_2, coor_y_6_2, 'interpolate')
    # output_7_2 = bilinear_interp(input_images[:, :, :, 18:21], coor_x_7_2, coor_y_7_2, 'interpolate')
    # output_8_2 = bilinear_interp(input_images[:, :, :, 21:24], coor_x_8_2, coor_y_8_2, 'interpolate')
    # output_9_2 = bilinear_interp(input_images[:, :, :, 24:27], coor_x_9_2, coor_y_9_2, 'interpolate')
    # output_10_2 = bilinear_interp(input_images[:, :, :, 27:30], coor_x_10_2, coor_y_10_2, 'interpolate')

    # output_1_3 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1_3, coor_y_1_3, 'interpolate')
    # output_2_3 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2_3, coor_y_2_3, 'interpolate')
    # output_3_3 = bilinear_interp(input_images[:, :, :, 6:9], coor_x_3_3, coor_y_3_3, 'interpolate')
    # output_4_3 = bilinear_interp(input_images[:, :, :, 9:12], coor_x_4_3, coor_y_4_3, 'interpolate')
    # output_5_3 = bilinear_interp(input_images[:, :, :, 12:15], coor_x_5_3, coor_y_5_3, 'interpolate')
    # output_6_3 = bilinear_interp(input_images[:, :, :, 15:18], coor_x_6_3, coor_y_6_3, 'interpolate')
    # output_7_3 = bilinear_interp(input_images[:, :, :, 18:21], coor_x_7_3, coor_y_7_3, 'interpolate')
    # output_8_3 = bilinear_interp(input_images[:, :, :, 21:24], coor_x_8_3, coor_y_8_3, 'interpolate')
    # output_9_3 = bilinear_interp(input_images[:, :, :, 24:27], coor_x_9_3, coor_y_9_3, 'interpolate')
    # output_10_3 = bilinear_interp(input_images[:, :, :, 27:30], coor_x_10_3, coor_y_10_3, 'interpolate')

    # output_1_4 = bilinear_interp(input_images[:, :, :, 0:3], coor_x_1_4, coor_y_1_4, 'interpolate')
    # output_2_4 = bilinear_interp(input_images[:, :, :, 3:6], coor_x_2_4, coor_y_2_4, 'interpolate')
    # output_3_4 = bilinear_interp(input_images[:, :, :, 6:9], coor_x_3_4, coor_y_3_4, 'interpolate')
    # output_4_4 = bilinear_interp(input_images[:, :, :, 9:12], coor_x_4_4, coor_y_4_4, 'interpolate')
    # output_5_4 = bilinear_interp(input_images[:, :, :, 12:15], coor_x_5_4, coor_y_5_4, 'interpolate')
    # output_6_4 = bilinear_interp(input_images[:, :, :, 15:18], coor_x_6_4, coor_y_6_4, 'interpolate')
    # output_7_4 = bilinear_interp(input_images[:, :, :, 18:21], coor_x_7_4, coor_y_7_4, 'interpolate')
    # output_8_4 = bilinear_interp(input_images[:, :, :, 21:24], coor_x_8_4, coor_y_8_4, 'interpolate')
    # output_9_4 = bilinear_interp(input_images[:, :, :, 24:27], coor_x_9_4, coor_y_9_4, 'interpolate')
    # output_10_4 = bilinear_interp(input_images[:, :, :, 27:30], coor_x_10_4, coor_y_10_4, 'interpolate')

    for i in range(num_out):
        for j in range(num_in-1):
            exec('mask{}_{}=0.5*(1.0+mask{}_{})'.format(j+1,i+1,j+1,i+1),g,l)
            exec('mask{}_{}=tf.tile(mask{}_{},[1,1,1,3])'.format(j+1,i+1,j+1,i+1),g,l)

    # mask1 = 0.5 * (1.0 + mask1)
    # mask2 = 0.5 * (1.0 + mask2)
    # mask3 = 0.5 * (1.0 + mask3)
    # mask4 = 0.5 * (1.0 + mask4)
    # mask5 = 0.5 * (1.0 + mask5)
    # mask6 = 0.5 * (1.0 + mask6)
    # mask7 = 0.5 * (1.0 + mask7)
    # mask8 = 0.5 * (1.0 + mask8)
    # mask9 = 0.5 * (1.0 + mask9)

    # mask1_2 = 0.5 * (1.0 + mask1_2)
    # mask2_2 = 0.5 * (1.0 + mask2_2)
    # mask3_2 = 0.5 * (1.0 + mask3_2)
    # mask4_2 = 0.5 * (1.0 + mask4_2)
    # mask5_2 = 0.5 * (1.0 + mask5_2)
    # mask6_2 = 0.5 * (1.0 + mask6_2)
    # mask7_2 = 0.5 * (1.0 + mask7_2)
    # mask8_2 = 0.5 * (1.0 + mask8_2)
    # mask9_2 = 0.5 * (1.0 + mask9_2)

    # mask1_3 = 0.5 * (1.0 + mask1_3)
    # mask2_3 = 0.5 * (1.0 + mask2_3)
    # mask3_3 = 0.5 * (1.0 + mask3_3)
    # mask4_3 = 0.5 * (1.0 + mask4_3)
    # mask5_3 = 0.5 * (1.0 + mask5_3)
    # mask6_3 = 0.5 * (1.0 + mask6_3)
    # mask7_3 = 0.5 * (1.0 + mask7_3)
    # mask8_3 = 0.5 * (1.0 + mask8_3)
    # mask9_3 = 0.5 * (1.0 + mask9_3)

    # mask1_4 = 0.5 * (1.0 + mask1_4)
    # mask2_4 = 0.5 * (1.0 + mask2_4)
    # mask3_4 = 0.5 * (1.0 + mask3_4)
    # mask4_4 = 0.5 * (1.0 + mask4_4)
    # mask5_4 = 0.5 * (1.0 + mask5_4)
    # mask6_4 = 0.5 * (1.0 + mask6_4)
    # mask7_4 = 0.5 * (1.0 + mask7_4)
    # mask8_4 = 0.5 * (1.0 + mask8_4)
    # mask9_4 = 0.5 * (1.0 + mask9_4)

    # mask1 = tf.tile(mask1, [1, 1, 1, 3])
    # mask2 = tf.tile(mask2, [1, 1, 1, 3])
    # mask3 = tf.tile(mask3, [1, 1, 1, 3])
    # mask4 = tf.tile(mask4, [1, 1, 1, 3])
    # mask5 = tf.tile(mask5, [1, 1, 1, 3])
    # mask6 = tf.tile(mask6, [1, 1, 1, 3])
    # mask7 = tf.tile(mask7, [1, 1, 1, 3])
    # mask8 = tf.tile(mask8, [1, 1, 1, 3])
    # mask9 = tf.tile(mask9, [1, 1, 1, 3])

    # mask1_2 = tf.tile(mask1_2, [1, 1, 1, 3])
    # mask2_2 = tf.tile(mask2_2, [1, 1, 1, 3])
    # mask3_2 = tf.tile(mask3_2, [1, 1, 1, 3])
    # mask4_2 = tf.tile(mask4_2, [1, 1, 1, 3])
    # mask5_2 = tf.tile(mask5_2, [1, 1, 1, 3])
    # mask6_2 = tf.tile(mask6_2, [1, 1, 1, 3])
    # mask7_2 = tf.tile(mask7_2, [1, 1, 1, 3])
    # mask8_2 = tf.tile(mask8_2, [1, 1, 1, 3])
    # mask9_2 = tf.tile(mask9_2, [1, 1, 1, 3])

    # mask1_3 = tf.tile(mask1_3, [1, 1, 1, 3])
    # mask2_3 = tf.tile(mask2_3, [1, 1, 1, 3])
    # mask3_3 = tf.tile(mask3_3, [1, 1, 1, 3])
    # mask4_3 = tf.tile(mask4_3, [1, 1, 1, 3])
    # mask5_3 = tf.tile(mask5_3, [1, 1, 1, 3])
    # mask6_3 = tf.tile(mask6_3, [1, 1, 1, 3])
    # mask7_3 = tf.tile(mask7_3, [1, 1, 1, 3])
    # mask8_3 = tf.tile(mask8_3, [1, 1, 1, 3])
    # mask9_3 = tf.tile(mask9_3, [1, 1, 1, 3])

    # mask1_4 = tf.tile(mask1_4, [1, 1, 1, 3])
    # mask2_4 = tf.tile(mask2_4, [1, 1, 1, 3])
    # mask3_4 = tf.tile(mask3_4, [1, 1, 1, 3])
    # mask4_4 = tf.tile(mask4_4, [1, 1, 1, 3])
    # mask5_4 = tf.tile(mask5_4, [1, 1, 1, 3])
    # mask6_4 = tf.tile(mask6_4, [1, 1, 1, 3])
    # mask7_4 = tf.tile(mask7_4, [1, 1, 1, 3])
    # mask8_4 = tf.tile(mask8_4, [1, 1, 1, 3])
    # mask9_4 = tf.tile(mask9_4, [1, 1, 1, 3])
    
    for i in range(num_out):
      exec("st{}_1=''".format(i+1),g,l)
    
    for i in range(num_out):
      for j in range(num_in-1):
        exec("temp='tf.multiply(mask{}_{},output_{}_1) + '".format(j+1,i+1,j+1),g,l)#
        exec("st{}_1+=temp".format(i+1),g,l)
    
    for i in range(num_out):
      exec("st{}_2='tf.multiply(1.0 - ('".format(i+1),g,l)
      
    for i in range(num_out):
      for j in range(num_in-1):
        exec("temp='mask{}_{}+'".format(j+1,i+1),g,l)
        exec("st{}_2+=temp".format(i+1),g,l)
      exec("st{}_2=st{}_2[:-1]+')'".format(i+1,i+1),g,l)
    
    
    for i in range(num_out):
      temp='net{} = '.format(i+1)+l['st{}_1'.format(i+1)]+l['st{}_2'.format(i+1)]+', output_{}_1)'.format(num_in)#
      l['temp']=temp
      exec(l['temp'],g,l)

    # exec("net1 = tf.multiply(mask1_1 , output_1_1) + tf.multiply(mask2_1 , output_2_1) + tf.multiply(mask3_1 , output_3_1) + tf.multiply(mask4_1 , output_4_1) + tf.multiply(mask5_1 , output_5_1) + tf.multiply(mask6_1 , output_6_1) + tf.multiply(mask7_1 , output_7_1) + tf.multiply(mask8_1 , output_8_1) + tf.multiply(mask9_1 , output_9_1) + tf.multiply(1.0 - (mask1_1+mask2_1+mask3_1+mask4_1+mask5_1+mask6_1+mask7_1+mask8_1+mask9_1) , output_10_1)",g,l)
    # exec("net2 = tf.multiply(mask1_2 , output_1_2) + tf.multiply(mask2_2 , output_2_2) + tf.multiply(mask3_2 , output_3_2) + tf.multiply(mask4_2 , output_4_2) + tf.multiply(mask5_2 , output_5_2) + tf.multiply(mask6_2 , output_6_2) + tf.multiply(mask7_2 , output_7_2) + tf.multiply(mask8_2 , output_8_2) + tf.multiply(mask9_2 , output_9_2) + tf.multiply(1.0 - (mask1_2+mask2_2+mask3_2+mask4_2+mask5_2+mask6_2+mask7_2+mask8_2+mask9_2) , output_10_2)",g,l)
    # exec("net3 = tf.multiply(mask1_3 , output_1_3) + tf.multiply(mask2_3 , output_2_3) + tf.multiply(mask3_3 , output_3_3) + tf.multiply(mask4_3 , output_4_3) + tf.multiply(mask5_3 , output_5_3) + tf.multiply(mask6_3 , output_6_3) + tf.multiply(mask7_3 , output_7_3) + tf.multiply(mask8_3 , output_8_3) + tf.multiply(mask9_3 , output_9_3) + tf.multiply(1.0 - (mask1_3+mask2_3+mask3_3+mask4_3+mask5_3+mask6_3+mask7_3+mask8_3+mask9_3) , output_10_3)",g,l)
    # exec("net4 = tf.multiply(mask1_4 , output_1_4) + tf.multiply(mask2_4 , output_2_4) + tf.multiply(mask3_4 , output_3_4) + tf.multiply(mask4_4 , output_4_4) + tf.multiply(mask5_4 , output_5_4) + tf.multiply(mask6_4 , output_6_4) + tf.multiply(mask7_4 , output_7_4) + tf.multiply(mask8_4 , output_8_4) + tf.multiply(mask9_4 , output_9_4) + tf.multiply(1.0 - (mask1_4+mask2_4+mask3_4+mask4_4+mask5_4+mask6_4+mask7_4+mask8_4+mask9_4) , output_10_4)",g,l)
    exec("net=tf.concat((net1,net2,net3,net4),axis=3)",g,l)
    # exec("net = tf.multiply(mask1_1 , output_1_1) + tf.multiply(mask2_1 , output_2_1) + tf.multiply(mask3_1 , output_3_1) + tf.multiply(mask4_1 , output_4_1) + tf.multiply(mask5_1 , output_5_1) + tf.multiply(mask6_1 , output_6_1) + tf.multiply(mask7_1 , output_7_1) + tf.multiply(mask8_1 , output_8_1) + tf.multiply(mask9_1 , output_9_1) + tf.multiply(mask10_1 , output_10_1) + tf.multiply(mask11_1 , output_11_1) + tf.multiply(mask12_1 , output_12_1) + tf.multiply(mask13_1 , output_13_1) + tf.multiply(mask14_1 , output_14_1) + tf.multiply(mask15_1 , output_15_1) + tf.multiply(mask16_1 , output_16_1) + tf.multiply(mask17_1 , output_17_1) + tf.multiply(mask18_1 , output_18_1) + tf.multiply(mask19_1 , output_19_1)  + tf.multiply(mask20_1 , output_20_1)  + tf.multiply(mask21_1 , output_21_1)  + tf.multiply(mask22_1 , output_22_1)  + tf.multiply(mask23_1 , output_23_1)  + tf.multiply(mask24_1 , output_24_1)  + tf.multiply(mask25_1 , output_25_1)  + tf.multiply(mask26_1 , output_26_1)  + tf.multiply(mask27_1 , output_27_1)  + tf.multiply(mask28_1 , output_28_1)  + tf.multiply(mask29_1 , output_29_1)  + tf.multiply(mask30_1 , output_30_1)  + tf.multiply(mask31_1 , output_31_1)  + tf.multiply(mask32_1 , output_32_1)  + tf.multiply(mask33_1 , output_33_1)  + tf.multiply(mask34_1 , output_34_1)  + tf.multiply(mask35_1 , output_35_1)  + tf.multiply(mask36_1 , output_36_1)  + tf.multiply(mask37_1 , output_37_1)  + tf.multiply(mask38_1 , output_38_1)  + tf.multiply(mask39_1 , output_39_1) + tf.multiply(1.0 - (mask1_1+mask2_1+mask3_1+mask4_1+mask5_1+mask6_1+mask7_1+mask8_1+mask9_1+mask10_1+mask11_1+mask12_1+mask13_1+mask14_1+mask15_1+mask16_1+mask17_1+mask18_1+mask19_1+mask20_1+mask21_1+mask22_1+mask23_1+mask24_1+mask25_1+mask26_1+mask27_1+mask28_1+mask29_1+mask30_1+mask31_1+mask32_1+mask33_1+mask34_1+mask35_1+mask36_1+mask37_1+mask38_1+mask39_1) , output_40_1)",g,l)

    return l['net']
