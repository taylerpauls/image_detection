#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:09:10 2018

@author: athar
"""
import tensorflow as tf

def rgbd_dataset_generator(dataset_name, batch_size):
    pass
#    assert dataset_name in ['train', 'test']
#    assert batch_size > 0 or batch_size == -1  # -1 for entire dataset
#    
#    path = './svhn_mat/' # path to the SVHN dataset you will download in Q1.1
#    file_name = '%s_32x32.mat' % dataset_name
#    file_dict = scipy.io.loadmat(os.path.join(path, file_name))
#    X_all = file_dict['X'].transpose((3, 0, 1, 2))
#    y_all = file_dict['y']
#    data_len = X_all.shape[0]
#    batch_size = batch_size if batch_size > 0 else data_len
#    
#    X_all_padded = np.concatenate([X_all, X_all[:batch_size]], axis=0)
#    y_all_padded = np.concatenate([y_all, y_all[:batch_size]], axis=0)
#    y_all_padded[y_all_padded == 10] = 0
#    
#    for slice_i in range(int(math.ceil(data_len / batch_size))):
#        idx = slice_i * batch_size
#        X_batch = X_all_padded[idx:idx + batch_size]
#        y_batch = np.ravel(y_all_padded[idx:idx + batch_size])
#        yield X_batch, y_batch
    
def mdl_rgb_d(x_rbg,x_depth):
    
    """
    First we define the stram for the rgb images
    """
    
    convR1 = tf.layers.conv2d(
            inputs=x_rbg,
            filters= 96,  # number of filters, Integer, the dimensionality of the output space 
            strides= 4, # convolution stride
            kernel_size=[11, 11],
            padding="valid",
            activation=tf.nn.relu)
    
   poolR1 = tf.layers.max_pooling2d(inputs=convR1, 
                                    pool_size=[3, 3], 
                                    strides=2)  # strides of the pooling operation 
    
   convR2 = tf.layers.conv2d(
            inputs = poolR1,
            filters = 256, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [5,5],
            padding="valid",
            activation=tf.nn.relu)
    
   poolR2 = tf.layers.max_pooling2d(inputs=convR2, 
                                    pool_size=[3, 3], 
                                    strides = 2)   # strides of the pooling operation 
    
   convR3 = tf.layers.conv2d(
            inputs = poolR2,
            filters = 384, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
    
   convR4 = tf.layers.conv2d(
            inputs = convR3,
            filters = 384, # number of filters
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
   
    
   convR5 = tf.layers.conv2d(
            inputs = convR4,
            filters = 256, # number of filters
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
   poolR5 = tf.layers.max_pooling2d(inputs=convR5, 
                                    pool_size=[3, 3], 
                                    strides=2)   # strides of the pooling operation 
   
   
   fcR6 = tf.layers.conv2d(
            inputs = poolR5,
            num_outputs = 4096,
            activation=tf.nn.relu)
   
   fcR7 = tf.layers.conv2d(
            inputs = fcR6,
            num_outputs = 4096,
            activation=tf.nn.relu)
    
   """
     define the stram for the depth images
     
    """    
    
    convD1 = tf.layers.conv2d(
            inputs=x_depth,
            filters= 96,  # number of filters, Integer, the dimensionality of the output space 
            strides= 4, # convolution stride
            kernel_size=[11, 11],
            padding="valid",
            activation=tf.nn.relu)
    
   poolD1 = tf.layers.max_pooling2d(inputs=convD1, 
                                    pool_size=[3, 3], 
                                    strides=2)  # strides of the pooling operation 
    
   convD2 = tf.layers.conv2d(
            inputs = poolD1,
            filters = 256, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [5,5],
            padding="valid",
            activation=tf.nn.relu)
    
   poolD2 = tf.layers.max_pooling2d(inputs=convD2, 
                                    pool_size=[3, 3], 
                                    strides = 2)   # strides of the pooling operation 
    
   convD3 = tf.layers.conv2d(
            inputs = poolD2,
            filters = 384, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
    
   convD4 = tf.layers.conv2d(
            inputs = convD3,
            filters = 384, # number of filters
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
   
    
   convD5 = tf.layers.conv2d(
            inputs = convD4,
            filters = 256, # number of filters
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
   poolD5 = tf.layers.max_pooling2d(inputs=convD5, 
                                    pool_size=[3, 3], 
                                    strides=2)   # strides of the pooling operation 
   
   
   fcD6 =  tf.contrib.layers.fully_connected (
            inputs = poolD5,
            num_outputs = 4096,
            activation=tf.nn.relu)
   
   fcD7 = tf.contrib.layers.fully_connected (
            inputs = fcD6,
            num_outputs = 4096,
            activation=tf.nn.relu)
   
   fc8 = tf.contrib.layers.fully_connected (
            inputs = tf.concat((fcR7, fcD7), axis=1),
            num_outputs = 4096,
            activation=tf.nn.relu)
   
   fc9 = tf.contrib.layers.fully_connected (
            inputs = fc8,
            num_outputs = 51,
            activation=tf.nn.relu)
   """
   pool_flat = tf.contrib.layers.flatten(pool2, scope='pool2flat')
   dense = tf.layers.dense(inputs=pool_flat, units=500, activation=tf.nn.relu)
   logits = tf.layers.dense(inputs=dense, units=10)
   """
   
   return fc9


"""
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss
"""
def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_rgb = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x_rgb')
            x_depth = tf.placeholder(tf.float32, [None, 227, 227, 1], name='x_depth')
            y_ = tf.placeholder(tf.int32, [None], name='y_')
            y_logits = model_function(x_rgb, x_depth)
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy_loss)
           
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), axis=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model_dict = {'graph': g, 'inputs': [x_rgb,x_depth, y_], 'train_op': train_op,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    
    return model_dict



def train_model(model_dict, dataset_generators, epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess.run(model_dict['train_op'], feed_dict=train_feed_dict)
                
                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict))
                    averages = np.mean(collect_arr, axis=0)
                    fmt = (epoch_i, iter_i, ) + tuple(averages)
                    print('epoch {:d} iter {:d}, loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))
                    
                    
dataset_generators = {
        'train': rgbd_dataset_generator('train', 256),
        'test': rgbd_dataset_generator('test', 256)
}
    
model_dict = apply_classification_loss(mdl_rgb_d)
train_model(model_dict, dataset_generators, epoch_n=50, print_every=20)                    