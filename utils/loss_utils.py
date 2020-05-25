"""Implements various tensorflow loss layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def l1_loss(predictions, targets):
  """Implements tensorflow l1 loss.
  Args:
  Returns:
  """
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.abs(predictions- targets))
  loss = tf.div(loss, total_elements)
  return loss

def l2_loss(predictions, targets):
  """Implements tensorflow l2 loss, normalized by number of elements.
  Args:
  Returns:
  """
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.square(predictions-targets))
  loss = tf.div(loss, total_elements)
  return loss

def hue_loss(predictions, targets):
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)
  pred_hue=tf.atan2(tf.sqrt(3.0)*(predictions[:,:,:,1]-predictions[:,:,:,2]),(2*(predictions[:,:,:,0]-predictions[:,:,:,1]-predictions[:,:,:,2])))
  target_hue=tf.atan2(tf.sqrt(3.0)*(targets[0,:,:]-targets[2,:,:]),(2*(targets[0,:,:]-targets[1,:,:]-targets[2,:,:])))
  loss = tf.reduce_sum(tf.square(pred_hue-target_hue))
  loss = tf.div(loss, total_elements)
  return loss
  
def lightness_loss(predictions, targets):
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)
  total_loss = tf.div(0.0,total_elements)
  for i in range(4): #num_out
    pred_lig = tf.div(tf.math.reduce_max(predictions[:,:,:,i*3:(i+1)*3],axis=3)+tf.math.reduce_min(predictions[:,:,:,i*3:(i+1)*3],axis=3),2)
    target_lig = tf.div(tf.math.reduce_max(targets[:,:,:,i*3:(i+1)*3],axis=3)+tf.math.reduce_min(targets[:,:,:,i*3:(i+1)*3],axis=3),2)
    loss = tf.reduce_sum(tf.square(pred_lig-target_lig))
    loss = tf.div(loss, total_elements)
    total_loss+=loss
  return total_loss

def tv_loss():
  #TODO
  pass
def vae_loss(z_mean, z_logvar, prior_weight=1.0):
  """Implements the VAE reguarlization loss.
  """
  total_elements = (tf.shape(z_mean)[0] * tf.shape(z_mean)[1] * tf.shape(z_mean)[2]
      * tf.shape(z_mean)[3])
  total_elements = tf.to_float(total_elements)

  vae_loss = -0.5 * tf.reduce_sum(1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
  vae_loss = tf.div(vae_loss, total_elements)
  return vae_loss

def bilateral_loss():
  #TODO
  pass
