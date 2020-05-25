"""Train a voxel flow model on ucf101 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataset
from utils.prefetch_queue_shuffle import PrefetchQueue
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
import random
from random import shuffle
from voxel_flow_model import Voxel_flow_model
from utils.image_utils import imwrite
from utils.image_utils import imsave
from functools import partial
import pdb
from copy import copy

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', './voxel_flow_checkpoints/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_image_dir', './voxel_flow_train_image/',
			   """Directory where to output images.""")
tf.app.flags.DEFINE_string('test_image_dir', './40_1_test/',
			   """Directory where to output images.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', './40_1_checkpoints/',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 4001,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer(
        'batch_size', 8, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_integer('num_in',4,"""number of input frames""")
tf.app.flags.DEFINE_integer('num_out',4,"""number of output frames""")


def train(dataset_frame1, dataset_frame2, dataset_frame3, dataset_frame4, dataset_frame5, dataset_frame6, dataset_frame7, dataset_frame8):# dataset_frame9, dataset_frame10, dataset_frame11, dataset_frame12, dataset_frame13, dataset_frame14, dataset_frame15, dataset_frame16, dataset_frame17, dataset_frame18, dataset_frame19, dataset_frame20,dataset_frame21):
# , dataset_frame22, dataset_frame23, dataset_frame24, dataset_frame25, dataset_frame26, dataset_frame27, dataset_frame28, dataset_frame29, dataset_frame30, dataset_frame31, dataset_frame32, dataset_frame33, dataset_frame34, dataset_frame35, dataset_frame36, dataset_frame37, dataset_frame38, dataset_frame39, dataset_frame40,
# dataset_frame41, dataset_frame42, dataset_frame43, dataset_frame44, dataset_frame45, dataset_frame46, dataset_frame47, dataset_frame48, dataset_frame49, dataset_frame50, dataset_frame51, dataset_frame52, dataset_frame53, dataset_frame54):
  """Trains a model."""
  with tf.Graph().as_default():
    # Create input and target placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, FLAGS.num_in*3))
    target_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, FLAGS.num_out*3))

    # input_resized = tf.image.resize_area(input_placeholder, [128, 128])
    # target_resized = tf.image.resize_area(target_placeholder,[128, 128])

    # Prepare model.
    model = Voxel_flow_model()
    prediction = model.inference(input_placeholder)
    # reproduction_loss, prior_loss = model.loss(prediction, target_placeholder)
    reproduction_loss = model.loss(prediction, target_placeholder)
    # total_loss = reproduction_loss + prior_loss
    total_loss = reproduction_loss
    
    # Perform learning rate scheduling.
    learning_rate = FLAGS.initial_learning_rate

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(learning_rate)
    grads = opt.compute_gradients(total_loss)
    update_op = opt.apply_gradients(grads)

    # Create summaries
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    summaries.append(tf.summary.scalar('reproduction_loss', reproduction_loss))
    # summaries.append(tf.summary.scalar('prior_loss', prior_loss))
    summaries.append(tf.summary.image('Input Image', input_placeholder, 3))
    summaries.append(tf.summary.image('Output Image', prediction, 3))
    summaries.append(tf.summary.image('Target Image', target_placeholder, 3))

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    # Summary Writter
    summary_writer = tf.summary.FileWriter(
      FLAGS.train_dir,
      graph=sess.graph)

    # Training loop using feed dict method.
    # data_list_frame1 = dataset_frame1.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame1)

    # data_list_frame2 = dataset_frame2.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame2)

    # data_list_frame3 = dataset_frame3.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame3)

    # data_list_frame4 = dataset_frame4.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame4)

    # data_list_frame5 = dataset_frame5.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame5)

    # data_list_frame6 = dataset_frame6.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame6)

    # data_list_frame7 = dataset_frame7.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame7)

    # data_list_frame8 = dataset_frame8.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame8)

    # data_list_frame9 = dataset_frame9.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame9)

    # data_list_frame10 = dataset_frame10.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame10)

    # data_list_frame11 = dataset_frame11.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame11)

    # data_list_frame12 = dataset_frame12.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame12)

    # data_list_frame13 = dataset_frame13.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame13)

    # data_list_frame14 = dataset_frame14.read_data_list_file()
    # random.seed(1)
    # shuffle(data_list_frame14)
    feed_str="{input_placeholder:np.concatenate(("
    for i in range(FLAGS.num_in):
      feed_str+="batch_data_frame{}, ".format(i+1)
    feed_str=feed_str[:-2]+'),3)'
    feed_str+=", target_placeholder:np.concatenate(("
    for i in range(FLAGS.num_in,FLAGS.num_in+FLAGS.num_out):
      feed_str+="batch_data_frame{}, ".format(i+1)
    feed_str=feed_str[:-2]+'),3)}'
    m=globals()
    n=locals()
    for i in range(FLAGS.num_out+FLAGS.num_in):
      exec("data_list_frame{}=dataset_frame{}.read_data_list_file()".format(i+1,i+1),m,n)
      exec("random.seed(1)",m,n)
      exec("shuffle(data_list_frame{})".format(i+1),m,n)

    exec("data_size = len(data_list_frame1)",m,n)
    exec("epoch_num = int(data_size / FLAGS.batch_size)",m,n)
    results=np.array([[0,0]])
    # num_workers = 1
      
    # load_fn_frame1 = partial(dataset_frame1.process_func)
    # p_queue_frame1 = PrefetchQueue(load_fn_frame1, data_list_frame1, FLAGS.batch_size, shuffle=False, num_workers=num_workers)

    # load_fn_frame2 = partial(dataset_frame2.process_func)
    # p_queue_frame2 = PrefetchQueue(load_fn_frame2, data_list_frame2, FLAGS.batch_size, shuffle=False, num_workers=num_workers)

    # load_fn_frame3 = partial(dataset_frame3.process_func)
    # p_queue_frame3 = PrefetchQueue(load_fn_frame3, data_list_frame3, FLAGS.batch_size, shuffle=False, num_workers=num_workers)

    for step in range(0, FLAGS.max_steps):
      n['step']=step
      exec("batch_idx = step % epoch_num",m,n)
      
      # batch_data_list_frame1 = data_list_frame1[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame2 = data_list_frame2[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame3 = data_list_frame3[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame4 = data_list_frame4[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame5 = data_list_frame5[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame6 = data_list_frame6[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame7 = data_list_frame7[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame8 = data_list_frame8[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame9 = data_list_frame9[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame10 = data_list_frame10[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame11 = data_list_frame11[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame12 = data_list_frame12[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame13 = data_list_frame13[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      # batch_data_list_frame14 = data_list_frame14[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]
      
      for i in range(FLAGS.num_out+FLAGS.num_in):
        exec("batch_data_list_frame{} = data_list_frame{}[int(batch_idx * FLAGS.batch_size) : int((batch_idx + 1) * FLAGS.batch_size)]".format(i+1,i+1),m,n)
      
      # Load batch data.
      # batch_data_frame1 = np.array([dataset_frame1.process_func(line) for line in batch_data_list_frame1])
      # batch_data_frame2 = np.array([dataset_frame2.process_func(line) for line in batch_data_list_frame2])
      # batch_data_frame3 = np.array([dataset_frame3.process_func(line) for line in batch_data_list_frame3])
      # batch_data_frame4 = np.array([dataset_frame4.process_func(line) for line in batch_data_list_frame4])
      # batch_data_frame5 = np.array([dataset_frame5.process_func(line) for line in batch_data_list_frame5])
      # batch_data_frame6 = np.array([dataset_frame6.process_func(line) for line in batch_data_list_frame6])
      # batch_data_frame7 = np.array([dataset_frame7.process_func(line) for line in batch_data_list_frame7])
      # batch_data_frame8 = np.array([dataset_frame8.process_func(line) for line in batch_data_list_frame8])
      # batch_data_frame9 = np.array([dataset_frame9.process_func(line) for line in batch_data_list_frame9])
      # batch_data_frame10 = np.array([dataset_frame10.process_func(line) for line in batch_data_list_frame10])
      # batch_data_frame11 = np.array([dataset_frame11.process_func(line) for line in batch_data_list_frame11])
      # batch_data_frame12 = np.array([dataset_frame12.process_func(line) for line in batch_data_list_frame12])
      # batch_data_frame13 = np.array([dataset_frame13.process_func(line) for line in batch_data_list_frame13])
      # batch_data_frame14 = np.array([dataset_frame14.process_func(line) for line in batch_data_list_frame14])
      
      # print(np.array([n['dataset_frame1'].process_func(line) for line in n['batch_data_list_frame1']]))
      for i in range(FLAGS.num_out+FLAGS.num_in):
        exec("tr = np.array([dataset_frame{}.process_func(batch_data_list_frame{}[0])])".format(i+1,i+1),m,n)
        exec("np.expand_dims(tr,axis=0)")
        exec("s='batch_data_list_frame{}'".format(i+1),m,n)
        for line in n[n['s']][1:]:
          n['line']=line
          exec("tr=np.append(tr,np.expand_dims(dataset_frame1.process_func(line),axis=0),axis=0)",m,n)
        exec('batch_data_frame{} = tr'.format(i+1),m,n)
      # batch_data_frame1 = p_queue_frame1.get_batch()
      # batch_data_frame2 = p_queue_frame2.get_batch()
      # batch_data_frame3 = p_queue_frame3.get_batch()
      
      exec("feed_dict = "+feed_str,m,n)
      # exec("feed_dict = "+feed_str+", target_placeholder: batch_data_frame21}",m,n)
    
      # exec("feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame2, batch_data_frame3, batch_data_frame4, batch_data_frame5, batch_data_frame6, batch_data_frame7, batch_data_frame8, batch_data_frame9, batch_data_frame10, batch_data_frame11, batch_data_frame12, batch_data_frame13, batch_data_frame14, batch_data_frame15, batch_data_frame16, batch_data_frame17, batch_data_frame18, batch_data_frame19, batch_data_frame20), 3), target_placeholder: np.concatenate((batch_data_frame11,batch_data_frame12,batch_data_frame13,batch_data_frame14),3)}",m,n)
      # exec("feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame2, batch_data_frame3, batch_data_frame4, batch_data_frame5, batch_data_frame6, batch_data_frame7, batch_data_frame8, batch_data_frame9, batch_data_frame10, batch_data_frame11, batch_data_frame12, batch_data_frame13, batch_data_frame14, batch_data_frame15, batch_data_frame16, batch_data_frame17, batch_data_frame18, batch_data_frame19, batch_data_frame20, batch_data_frame21, batch_data_frame22, batch_data_frame23, batch_data_frame24, batch_data_frame25, batch_data_frame26, batch_data_frame27, batch_data_frame28, batch_data_frame29, batch_data_frame30, batch_data_frame31, batch_data_frame32, batch_data_frame33, batch_data_frame34, batch_data_frame35, batch_data_frame36, batch_data_frame37, batch_data_frame38, batch_data_frame39, batch_data_frame40), 3), target_placeholder: batch_data_frame41}",m,n)
     
      # Run single step update.
      _, loss_value = sess.run([update_op, total_loss], feed_dict = n['feed_dict'])
      
      if n['batch_idx'] == 0:
        # Shuffle data at each epoch.
        # random.seed(1)
        # shuffle(data_list_frame1)
        # random.seed(1)
        # shuffle(data_list_frame2)
        # random.seed(1)
        # shuffle(data_list_frame3)
        # random.seed(1)
        # shuffle(data_list_frame4)
        # random.seed(1)
        # shuffle(data_list_frame5)
        # random.seed(1)
        # shuffle(data_list_frame6)
        # random.seed(1)
        # shuffle(data_list_frame7)
        # random.seed(1)
        # shuffle(data_list_frame8)
        # random.seed(1)
        # shuffle(data_list_frame9)
        # random.seed(1)
        # shuffle(data_list_frame10)
        # random.seed(1)
        # shuffle(data_list_frame11)
        # random.seed(1)
        # shuffle(data_list_frame12)
        # random.seed(1)
        # shuffle(data_list_frame13)
        # random.seed(1)
        # shuffle(data_list_frame14)
        for i in range(FLAGS.num_in+FLAGS.num_out):
          exec("random.seed(1)")
          exec("shuffle(data_list_frame{})".format(i+1),m,n)
        print('Epoch Number: %d' % int(step / n['epoch_num']))
        
      
      # Output Summary 
      if step % 10 == 0:
        # summary_str = sess.run(summary_op, feed_dict = feed_dict)
        # summary_writer.add_summary(summary_str, step)
	      print("Loss at step %d: %f" % (step, loss_value))
	      results=np.append(results,[[step,loss_value]],axis=0)

      if step % 100 == 0:
        # Run a batch of images
        try:	
          prediction_np, target_np = sess.run([prediction, target_placeholder], feed_dict = n['feed_dict']) 
          for i in range(0,prediction_np.shape[0]):
            for j in range(0,FLAGS.num_out):
              file_name = FLAGS.train_image_dir+str(i)+'_out_{}.png'.format(j)
              file_name_label = FLAGS.train_image_dir+str(i)+'_gt_{}.png'.format(j)
              imsave(prediction_np[i,:,:,j*3:(j+1)*3], file_name)
              imsave(target_np[i,:,:,j*3:(j+1)*3], file_name_label)
        except ValueError:
          print(prediction_np[0,:,:,0:3].shape)
          print(target_np[0,:,:,0:3].shape)
          break

      # Save checkpoint 
      if step % 200 == 0 or (step +1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        np.savetxt(FLAGS.train_image_dir+"results.csv",results[1:,:],delimiter=",",header="iter,loss",comments='')

def validate(dataset_frame1, dataset_frame2, dataset_frame3):
  """Performs validation on model.
  Args:  
  """
  pass

def test(dataset_frame1, dataset_frame2, dataset_frame3, dataset_frame4, dataset_frame5, dataset_frame6, dataset_frame7, dataset_frame8, dataset_frame9, dataset_frame10, dataset_frame11, dataset_frame12, dataset_frame13, dataset_frame14, dataset_frame15, dataset_frame16, dataset_frame17, dataset_frame18, dataset_frame19, dataset_frame20,dataset_frame21,
dataset_frame22, dataset_frame23, dataset_frame24, dataset_frame25, dataset_frame26, dataset_frame27, dataset_frame28, dataset_frame29, dataset_frame30, dataset_frame31, dataset_frame32, dataset_frame33, dataset_frame34, dataset_frame35, dataset_frame36, dataset_frame37, dataset_frame38, dataset_frame39, dataset_frame40,
dataset_frame41):#, dataset_frame42, dataset_frame43, dataset_frame44, dataset_frame45, dataset_frame46, dataset_frame47, dataset_frame48, dataset_frame49, dataset_frame50, dataset_frame51, dataset_frame52, dataset_frame53, dataset_frame54):
  """Perform test on a trained model."""
  with tf.Graph().as_default():
    # Create input and target placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, FLAGS.num_in*3))
    target_placeholder = tf.placeholder(tf.float32, shape=(None, 256, 256, FLAGS.num_out*3))
    
    # input_resized = tf.image.resize_area(input_placeholder, [128, 128])
    # target_resized = tf.image.resize_area(target_placeholder,[128, 128])

    # Prepare model.
    model = Voxel_flow_model(is_train=True)
    prediction = model.inference(input_placeholder)
    # reproduction_loss, prior_loss = model.loss(prediction, target_placeholder)
    reproduction_loss = model.loss(prediction, target_placeholder)
    # total_loss = reproduction_loss + prior_loss
    total_loss = reproduction_loss

    # Create a saver and load.
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()
    # Restore checkpoint from file.
    if FLAGS.pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      ckpt = tf.train.get_checkpoint_state(
               FLAGS.pretrained_model_checkpoint_path)
      restorer = tf.train.Saver()
      restorer.restore(sess, ckpt.model_checkpoint_path)
      print('%s: Pre-trained model restored from %s' %
        (datetime.now(), ckpt.model_checkpoint_path))
    
    # Process on test dataset.
    # data_list_frame1 = dataset_frame1.read_data_list_file()
    # data_list_frame2 = dataset_frame2.read_data_list_file()
    # data_list_frame3 = dataset_frame3.read_data_list_file()
    # data_list_frame4 = dataset_frame4.read_data_list_file()
    # data_list_frame5 = dataset_frame5.read_data_list_file()
    # data_list_frame6 = dataset_frame6.read_data_list_file()
    # data_list_frame7 = dataset_frame7.read_data_list_file()
    # data_list_frame8 = dataset_frame8.read_data_list_file()
    # data_list_frame9 = dataset_frame9.read_data_list_file()
    # data_list_frame10 = dataset_frame10.read_data_list_file()
    # data_list_frame11 = dataset_frame11.read_data_list_file()
    # data_list_frame12 = dataset_frame12.read_data_list_file()
    # data_list_frame13 = dataset_frame13.read_data_list_file()
    # data_list_frame14 = dataset_frame14.read_data_list_file()
    feed_str="{input_placeholder:np.concatenate(("
    for i in range(FLAGS.num_in):
      feed_str+="batch_data_frame{}, ".format(i+1)
    feed_str=feed_str[:-2]+'),3)'
    m=globals()
    n=locals()
    for i in range(FLAGS.num_out+FLAGS.num_in):
      exec("data_list_frame{}=dataset_frame{}.read_data_list_file()".format(i+1,i+1),m,n)
    exec("data_size = len(data_list_frame1)",m,n)
    exec("epoch_num = int(data_size / FLAGS.batch_size)",m,n)
    print("feed_dict = "+feed_str+", target_placeholder: batch_data_frame41}")

    j = 0 
    PSNR = 0
    # batch_data_frame1 = [dataset_frame1.process_func(ll) for ll in data_list_frame1[-8:]]
    # batch_data_frame2 = [dataset_frame2.process_func(ll) for ll in data_list_frame2[-8:]]
    # batch_data_frame3 = [dataset_frame3.process_func(ll) for ll in data_list_frame3[-8:]]
    # batch_data_frame4 = [dataset_frame4.process_func(ll) for ll in data_list_frame4[-8:]]
    # batch_data_frame5 = [dataset_frame5.process_func(ll) for ll in data_list_frame5[-8:]]
    # batch_data_frame6 = [dataset_frame6.process_func(ll) for ll in data_list_frame6[-8:]]
    # batch_data_frame7 = [dataset_frame7.process_func(ll) for ll in data_list_frame7[-8:]]
    # batch_data_frame8 = [dataset_frame8.process_func(ll) for ll in data_list_frame8[-8:]]
    # batch_data_frame9 = [dataset_frame9.process_func(ll) for ll in data_list_frame9[-8:]]
    # batch_data_frame10 = [dataset_frame10.process_func(ll) for ll in data_list_frame10[-8:]]
    # batch_data_frame11 = [dataset_frame11.process_func(ll) for ll in data_list_frame11[-8:]]
    # batch_data_frame12 = [dataset_frame12.process_func(ll) for ll in data_list_frame12[-8:]]
    # batch_data_frame13 = [dataset_frame13.process_func(ll) for ll in data_list_frame13[-8:]]
    # batch_data_frame14 = [dataset_frame14.process_func(ll) for ll in data_list_frame14[-8:]]

    # predicting using batch size 8 and input starting from 101
    for i in range(FLAGS.num_out+FLAGS.num_in):
      exec("tr = np.array([dataset_frame{}.process_func(data_list_frame{}[73])])".format(i+1,i+1),m,n)
      exec("np.expand_dims(tr,axis=0)",m,n)
      exec("s='data_list_frame{}'".format(i+1),m,n)
      for line in n[n['s']][74:81]:
        n['line']=line
        exec("tr=np.append(tr,np.expand_dims(dataset_frame1.process_func(line),axis=0),axis=0)",m,n)
      exec('batch_data_frame{} = tr'.format(i+1),m,n)
    
    # batch_data_frame1 = np.array(batch_data_frame1)
    # batch_data_frame2 = np.array(batch_data_frame2)
    # batch_data_frame3 = np.array(batch_data_frame3)
    # batch_data_frame4 = np.array(batch_data_frame4)
    # batch_data_frame5 = np.array(batch_data_frame5)
    # batch_data_frame6 = np.array(batch_data_frame6)
    # batch_data_frame7 = np.array(batch_data_frame7)
    # batch_data_frame8 = np.array(batch_data_frame8)
    # batch_data_frame9 = np.array(batch_data_frame9)
    # batch_data_frame10 = np.array(batch_data_frame10)
    # batch_data_frame11 = np.array(batch_data_frame11)
    # batch_data_frame12 = np.array(batch_data_frame12)
    # batch_data_frame13 = np.array(batch_data_frame13)
    # batch_data_frame14 = np.array(batch_data_frame14)
    
    for i in range(FLAGS.num_out+FLAGS.num_in):
      exec("batch_data_frame{} = np.array(batch_data_frame{})".format(i+1,i+1),m,n)

    for id_img in range(0, 10):  
      # Load single data.
      # line_image_frame1 = dataset_frame1.process_func(data_list_frame1[id_img])
      # line_image_frame2 = dataset_frame2.process_func(data_list_frame2[id_img])
      # line_image_frame3 = dataset_frame3.process_func(data_list_frame3[id_img])
      
      
      # batch_data_frame1.append(line_image_frame1)
      # batch_data_frame2.append(line_image_frame2)
      # batch_data_frame3.append(line_image_frame3)
      
      
      
      # feed_dict = {input_placeholder: np.concatenate((batch_data_frame1, batch_data_frame2, batch_data_frame3, batch_data_frame4,
      # batch_data_frame5, batch_data_frame6, batch_data_frame7, batch_data_frame8, batch_data_frame9, batch_data_frame10), 3), target_placeholder: np.concatenate((batch_data_frame11,batch_data_frame12,batch_data_frame13,batch_data_frame14),3)}
      # Run single step update.
      
      exec("feed_dict = "+feed_str+", target_placeholder: batch_data_frame41}",m,n)
      
    
      prediction_np, target_np, loss_value = sess.run([prediction,
                                                target_placeholder,
                                                total_loss],
                                                feed_dict = n['feed_dict'])
      # print("Loss for image %d: %f" % (i,loss_value))
      # for i in range(0,prediction_np.shape[0]):
      #       for j in range(0,4):
      #         file_name = FLAGS.test_image_dir+str(i)+'_out_{}.png'.format(j)
      #         file_name_label = FLAGS.test_image_dir+str(i)+'_gt_{}.png'.format(j)
      #         imsave(prediction_np[i,:,:,j*3:(j+1)*3], file_name)
              # imsave(target_np[i,:,:,j*3:(j+1)*3], file_name_label)
      file_name = FLAGS.test_image_dir+str(j)+'_out.png'
      # file_name_label = FLAGS.test_image_dir+str(j)+'_gt.png'
      imsave(prediction_np[-1,:,:,:], file_name)
      # imsave(target_np[-1,:,:,:], file_name_label)
      j += 1
      print(id_img)
      PSNR += 10*np.log10(255.0*255.0/np.sum(np.square(prediction_np-target_np)))
      # batch_data_frame1[-1]=batch_data_frame2[-1]
      # batch_data_frame2[-1]=batch_data_frame3[-1]
      # batch_data_frame3[-1]=batch_data_frame4[-1]
      # batch_data_frame4[-1]=batch_data_frame5[-1]
      # batch_data_frame5[-1]=batch_data_frame6[-1]
      # batch_data_frame6[-1]=batch_data_frame7[-1]
      # batch_data_frame7[-1]=batch_data_frame8[-1]
      # batch_data_frame8[-1]=batch_data_frame9[-1]
      # batch_data_frame9[-1]=batch_data_frame10[-1]
      # batch_data_frame10[-1]=batch_data_frame11[-1]
      # batch_data_frame11[-1]=prediction_np[-1,:,:,:]
      
      for i in range(FLAGS.num_out+FLAGS.num_in-1):
        exec("batch_data_frame{}[-1]=batch_data_frame{}[-1]".format(i+1,i+2),m,n)
      n['batch_data_frame41'][-1]=prediction_np[-1,:,:,:]
        
      
    print("Overall PSNR: %f db" % (PSNR/n['data_size']))
      
if __name__ == '__main__':
  
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  if FLAGS.subset == 'train':
    
    # data_list_path_frame1 = "../k8img2/k8frame1.txt"
    # data_list_path_frame2 = "../k8img2/k8frame2.txt"
    # data_list_path_frame3 = "../k8img2/k8frame3.txt"
    # data_list_path_frame4 = "../k8img2/k8frame4.txt"
    # data_list_path_frame5 = "../k8img2/k8frame5.txt"
    # data_list_path_frame6 = "../k8img2/k8frame6.txt"
    # data_list_path_frame7 = "../k8img2/k8frame7.txt"
    # data_list_path_frame8 = "../k8img2/k8frame8.txt"
    # data_list_path_frame9 = "../k8img2/k8frame9.txt"
    # data_list_path_frame10 = "../k8img2/k8frame10.txt"
    # data_list_path_frame11 = "../k8img2/k8frame11.txt"
    # data_list_path_frame12 = "../k8img2/k8frame12.txt"
    # data_list_path_frame13 = "../k8img2/k8frame13.txt"
    # data_list_path_frame14 = "../k8img2/k8frame14.txt"
    # data_list_path_frame15 = "../k8img2/k8frame15.txt"
    # data_list_path_frame16 = "../k8img2/k8frame16.txt"
    # data_list_path_frame17 = "../k8img2/k8frame17.txt"
    # data_list_path_frame18 = "../k8img2/k8frame18.txt"
    # data_list_path_frame19 = "../k8img2/k8frame19.txt"
    # data_list_path_frame20 = "../k8img2/k8frame20.txt"
    # data_list_path_frame21 = "../k8img2/k8frame21.txt"
    # data_list_path_frame22 = "../k8img2/k8frame22.txt"
    # data_list_path_frame23 = "../k8img2/k8frame23.txt"
    # data_list_path_frame24 = "../k8img2/k8frame24.txt"
    # data_list_path_frame25 = "../k8img2/k8frame25.txt"
    # data_list_path_frame26 = "../k8img2/k8frame26.txt"
    # data_list_path_frame27 = "../k8img2/k8frame27.txt"
    # data_list_path_frame28 = "../k8img2/k8frame28.txt"
    # data_list_path_frame29 = "../k8img2/k8frame29.txt"
    # data_list_path_frame30 = "../k8img2/k8frame30.txt"
    # data_list_path_frame31 = "../k8img2/k8frame31.txt"
    # data_list_path_frame32 = "../k8img2/k8frame32.txt"
    # data_list_path_frame33 = "../k8img2/k8frame33.txt"
    # data_list_path_frame34 = "../k8img2/k8frame34.txt"
    # data_list_path_frame35 = "../k8img2/k8frame35.txt"
    # data_list_path_frame36 = "../k8img2/k8frame36.txt"
    # data_list_path_frame37 = "../k8img2/k8frame37.txt"
    # data_list_path_frame38 = "../k8img2/k8frame38.txt"
    # data_list_path_frame39 = "../k8img2/k8frame39.txt"
    # data_list_path_frame40 = "../k8img2/k8frame40.txt"
    # data_list_path_frame41 = "../k8img2/k8frame41.txt"
    
    for i in range(FLAGS.num_in+FLAGS.num_out):
      exec("data_list_path_frame{} = '../k8img2/k8frame{}.txt'".format(i+1,i+1))
    
    # ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1) 
    # ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2) 
    # ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3)
    # ucf101_dataset_frame4 = dataset.Dataset(data_list_path_frame4) 
    # ucf101_dataset_frame5 = dataset.Dataset(data_list_path_frame5) 
    # ucf101_dataset_frame6 = dataset.Dataset(data_list_path_frame6)
    # ucf101_dataset_frame7 = dataset.Dataset(data_list_path_frame7) 
    # ucf101_dataset_frame8 = dataset.Dataset(data_list_path_frame8) 
    # ucf101_dataset_frame9 = dataset.Dataset(data_list_path_frame9)
    # ucf101_dataset_frame10 = dataset.Dataset(data_list_path_frame10) 
    # ucf101_dataset_frame11 = dataset.Dataset(data_list_path_frame11)
    # ucf101_dataset_frame12 = dataset.Dataset(data_list_path_frame12)
    # ucf101_dataset_frame13 = dataset.Dataset(data_list_path_frame13)
    # ucf101_dataset_frame14 = dataset.Dataset(data_list_path_frame14)
    # ucf101_dataset_frame15 = dataset.Dataset(data_list_path_frame15)
    # ucf101_dataset_frame16 = dataset.Dataset(data_list_path_frame16)
    # ucf101_dataset_frame17 = dataset.Dataset(data_list_path_frame17)
    # ucf101_dataset_frame18 = dataset.Dataset(data_list_path_frame18)
    # ucf101_dataset_frame19 = dataset.Dataset(data_list_path_frame19)
    # ucf101_dataset_frame20 = dataset.Dataset(data_list_path_frame20)
    # ucf101_dataset_frame21 = dataset.Dataset(data_list_path_frame21)
    # ucf101_dataset_frame22 = dataset.Dataset(data_list_path_frame22)
    # ucf101_dataset_frame23 = dataset.Dataset(data_list_path_frame23)
    # ucf101_dataset_frame24 = dataset.Dataset(data_list_path_frame24)
    # ucf101_dataset_frame25 = dataset.Dataset(data_list_path_frame25)
    # ucf101_dataset_frame26 = dataset.Dataset(data_list_path_frame26)
    # ucf101_dataset_frame27 = dataset.Dataset(data_list_path_frame27)
    # ucf101_dataset_frame28 = dataset.Dataset(data_list_path_frame28)
    # ucf101_dataset_frame29 = dataset.Dataset(data_list_path_frame29)
    # ucf101_dataset_frame30 = dataset.Dataset(data_list_path_frame30)
    # ucf101_dataset_frame31 = dataset.Dataset(data_list_path_frame31)
    # ucf101_dataset_frame32 = dataset.Dataset(data_list_path_frame32)
    # ucf101_dataset_frame33 = dataset.Dataset(data_list_path_frame33)
    # ucf101_dataset_frame34 = dataset.Dataset(data_list_path_frame34)
    # ucf101_dataset_frame35 = dataset.Dataset(data_list_path_frame35)
    # ucf101_dataset_frame36 = dataset.Dataset(data_list_path_frame36)
    # ucf101_dataset_frame37 = dataset.Dataset(data_list_path_frame37)
    # ucf101_dataset_frame38 = dataset.Dataset(data_list_path_frame38)
    # ucf101_dataset_frame39 = dataset.Dataset(data_list_path_frame39)
    # ucf101_dataset_frame40 = dataset.Dataset(data_list_path_frame40)
    # ucf101_dataset_frame41 = dataset.Dataset(data_list_path_frame41)
    
    for i in range(FLAGS.num_in+FLAGS.num_out):
      exec("ucf101_dataset_frame{} = dataset.Dataset(data_list_path_frame{})".format(i+1,i+1))
    
    exec("train_str=''")
    for i in range(FLAGS.num_in+FLAGS.num_out):
      exec("train_str+='ucf101_dataset_frame{}, '".format(i+1))
    exec("train_str=train_str[:-2]")
    exec("train("+train_str+")")
    # train(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3, ucf101_dataset_frame4,
    # ucf101_dataset_frame5, ucf101_dataset_frame6, ucf101_dataset_frame7, ucf101_dataset_frame8,
    #   ucf101_dataset_frame9, ucf101_dataset_frame10, ucf101_dataset_frame11, ucf101_dataset_frame12,
    #   ucf101_dataset_frame13, ucf101_dataset_frame14, ucf101_dataset_frame15, ucf101_dataset_frame16, ucf101_dataset_frame17,
    #   ucf101_dataset_frame18, ucf101_dataset_frame19, ucf101_dataset_frame20, ucf101_dataset_frame21, ucf101_dataset_frame22,
    #   ucf101_dataset_frame23, ucf101_dataset_frame24, ucf101_dataset_frame25, ucf101_dataset_frame26, ucf101_dataset_frame27,
    #   ucf101_dataset_frame28, ucf101_dataset_frame29, ucf101_dataset_frame30, ucf101_dataset_frame31, ucf101_dataset_frame32, ucf101_dataset_frame33, ucf101_dataset_frame34, ucf101_dataset_frame35, ucf101_dataset_frame36, ucf101_dataset_frame37, ucf101_dataset_frame38, ucf101_dataset_frame39, ucf101_dataset_frame40, ucf101_dataset_frame41)
  
  elif FLAGS.subset == 'test':
    
    # data_list_path_frame1 = "../k8img2/k8frame1.txt"
    # data_list_path_frame2 = "../k8img2/k8frame2.txt"
    # data_list_path_frame3 = "../k8img2/k8frame3.txt"
    # data_list_path_frame4 = "../k8img2/k8frame4.txt"
    # data_list_path_frame5 = "../k8img2/k8frame5.txt"
    # data_list_path_frame6 = "../k8img2/k8frame6.txt"
    # data_list_path_frame7 = "../k8img2/k8frame7.txt"
    # data_list_path_frame8 = "../k8img2/k8frame8.txt"
    # data_list_path_frame9 = "../k8img2/k8frame9.txt"
    # data_list_path_frame10 = "../k8img2/k8frame10.txt"
    # data_list_path_frame11 = "../k8img2/k8frame11.txt"
    # data_list_path_frame12 = "../k8img2/k8frame12.txt"
    # data_list_path_frame13 = "../k8img2/k8frame13.txt"
    # data_list_path_frame14 = "../k8img2/k8frame14.txt"
    
    for i in range(FLAGS.num_in+FLAGS.num_out):
      exec("data_list_path_frame{} = '../k8img2/k8frame{}.txt'".format(i+1,i+1))
    
    # ucf101_dataset_frame1 = dataset.Dataset(data_list_path_frame1) 
    # ucf101_dataset_frame2 = dataset.Dataset(data_list_path_frame2) 
    # ucf101_dataset_frame3 = dataset.Dataset(data_list_path_frame3)
    # ucf101_dataset_frame4 = dataset.Dataset(data_list_path_frame4) 
    # ucf101_dataset_frame5 = dataset.Dataset(data_list_path_frame5) 
    # ucf101_dataset_frame6 = dataset.Dataset(data_list_path_frame6)
    # ucf101_dataset_frame7 = dataset.Dataset(data_list_path_frame7) 
    # ucf101_dataset_frame8 = dataset.Dataset(data_list_path_frame8) 
    # ucf101_dataset_frame9 = dataset.Dataset(data_list_path_frame9)
    # ucf101_dataset_frame10 = dataset.Dataset(data_list_path_frame10) 
    # ucf101_dataset_frame11 = dataset.Dataset(data_list_path_frame11)  
    # ucf101_dataset_frame12 = dataset.Dataset(data_list_path_frame12)  
    # ucf101_dataset_frame13 = dataset.Dataset(data_list_path_frame13)  
    # ucf101_dataset_frame14 = dataset.Dataset(data_list_path_frame14)
    
    for i in range(FLAGS.num_in+FLAGS.num_out):
      exec("ucf101_dataset_frame{} = dataset.Dataset(data_list_path_frame{})".format(i+1,i+1))
      
    exec("test_str=''")
    for i in range(FLAGS.num_in+FLAGS.num_out):
      exec("test_str+='ucf101_dataset_frame{}, '".format(i+1))
    exec("test_str=test_str[:-2]")
    exec("test("+test_str+")")
    
    # test(ucf101_dataset_frame1, ucf101_dataset_frame2, ucf101_dataset_frame3, ucf101_dataset_frame4,
    # ucf101_dataset_frame5, ucf101_dataset_frame6, ucf101_dataset_frame7, ucf101_dataset_frame8,
    #   ucf101_dataset_frame9, ucf101_dataset_frame10, ucf101_dataset_frame11, ucf101_dataset_frame12, ucf101_dataset_frame13, ucf101_dataset_frame14)
