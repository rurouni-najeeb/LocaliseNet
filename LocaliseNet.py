## Importing required libraries
import tensorflow as tf
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler

## Class Definition for LocaliseNet
class LocaliseNet(object):
    
    def __init__(self, params):
        
        self.N_EPOCHS = params['N_EPOCHS']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.LEARNING_RATE = params['LEARNING_RATE']
        self.NUM_CLASSES = params['NUM_CLASSES']
        self.NUM_COORDINATES = params['NUM_COORDINATES']
        self.var_list = []
        self.placeholder = None
        self.category_loss = None
        self.regressor_loss = None
        self.global_step_category = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step_category")
        self.global_step_regressor = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step_regressor")
        self.c_optimizer = None
        self.r_optimizer = None

    def _create_placeholder(self):
        
        if not self.placeholder:
            
            with tf.variable_scope('placeholder') as scope:
                input_tensor = tf.placeholder(dtype=tf.float32,shape=[None,28,28,3],name="Input")
                class_tensor = tf.placeholder(dtype=tf.float32,shape=[None,self.NUM_CLASSES],name="Label")
                box_tensor = tf.placeholder(dtype=tf.float32,shape=[None,self.NUM_COORDINATES],name="Box")
                self.placeholder = (input_tensor, class_tensor, box_tensor)
                return self.placeholder
        
    def _create_conv_relu_layer(self, prev_layer, layer_name, kernel_size=[5,5,3], n_kernels=32):
            
        with tf.variable_scope(layer_name) as scope:
            kernel_size.append(n_kernels)
            w = tf.get_variable(name='weights',shape=kernel_size,initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable(name="biases",shape=[n_kernels],initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(prev_layer,w,strides=[1,1,1,1],padding="SAME")
            relu = tf.nn.relu(conv + b)
            return relu

    def _create_max_pool(self, prev_layer, layer_name, kernel_size=[1,2,2,1]):
        
        with tf.variable_scope(layer_name) as scope:
            maxpool = tf.nn.max_pool(prev_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            return maxpool

    def _create_fully_connected(self, prev_layer, num_neurones, layer_name, save_vars=False):
        
        with tf.variable_scope(layer_name) as scope:
            try:
                b, x, y, z = prev_layer.get_shape().as_list()
                flat_size = x*y*z
            except:
                flat_size = prev_layer.get_shape().as_list()[1]

            flat = tf.reshape(prev_layer,shape=[-1,flat_size])
            w = tf.get_variable(name="weights",shape=[flat_size,num_neurones],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="biases",shape=[num_neurones],initializer=tf.random_normal_initializer())
            out = tf.matmul(flat,w) + b
            full = tf.nn.relu(out)

            if save_vars:
                self.var_list += [w, b]

            return full

    def _create_softmax(self, prev_layer, num_output, layer_name):
        
        with tf.variable_scope(layer_name) as scope:
            fan_in = prev_layer.get_shape().as_list()[1]
            w = tf.get_variable(name="weights",shape=[fan_in,num_output],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="biases",shape=[num_output],initializer=tf.random_normal_initializer())
            softmax = tf.nn.softmax(tf.matmul(prev_layer,w) + b)
            return softmax

    def _create_xentropy_loss(self, prev_layer, layer_name):
        
         with tf.variable_scope(layer_name) as scope:
            input_tensor, class_tensor, box_tensor = self.placeholder
            loss = tf.reduce_mean(-tf.reduce_sum(class_tensor*tf.log(prev_layer),reduction_indices=[1]))
            return loss

    def _create_squared_loss(self, prev_layer, layer_name):
        
        with tf.variable_scope(layer_name) as scope:
            input_tensor, class_tensor, box_tensor = self.placeholder
            loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(box_tensor,prev_layer),reduction_indices=[1]))
            return loss

    def _create_optimizer(self, loss, layer_name, regression=False):
        
        if not regression and not self.c_optimizer:
            with tf.variable_scope(layer_name) as scope:
                self.c_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss, global_step=self.global_step_category)

        if regression and not self.r_optimizer:
            with tf.variable_scope(layer_name) as scope:
                self.r_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(loss, var_list=self.var_list, global_step=self.global_step_regressor)
            
    def _create_summaries(self, loss, layer_name, loss_name="loss"):
        
        with tf.variable_scope(layer_name) as scope:
            tf.summary.scalar(loss_name, loss)
            tf.summary.histogram(loss_name, loss)
            summary_op = tf.summary.merge_all()
            return summary_op

    def train(self, X, y, box, loss, summary_op, regression_head=False, directory_name='two_layer', checkpoint_dir='two_layer'):
        
        print 'Training the graph...'

        with tf.Session() as sess:
            ## Initialising variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ## Computation graph
            writer = tf.summary.FileWriter('graphs/{}/'.format(directory_name), sess.graph)
            
            ## Checkpoint
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/{}/checkpoint'.format(checkpoint_dir)))
            if ckpt and ckpt.model_checkpoint_path :
                print 'Model checkpoint found. Restoring session'
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            for i in range(1, self.N_EPOCHS+1):
                epoch_loss = 0
                start_time = time.time()
                N_BATCHES = X.shape[0]/self.BATCH_SIZE

                for j in tqdm(range(N_BATCHES)):
                    
                    ## Generating batches for data
                    input_tensor, class_tensor, box_tensor = self.placeholder
                    cur_batch, next_batch = j*self.BATCH_SIZE, (j+1)*self.BATCH_SIZE
                    x_batch = X[cur_batch:next_batch,:,:,:]
                    y_batch = y[cur_batch:next_batch,:]
                    box_batch = box[cur_batch:next_batch,:]
                    
                    ## Running the optimizer
                    if not regression_head:
                        
                        _, l, summary = sess.run([self.c_optimizer, loss, summary_op],
                        feed_dict = {input_tensor:x_batch, class_tensor:y_batch})
                    
                    else:
                        
                        _, l, summary = sess.run([self.r_optimizer, loss, summary_op],
                        feed_dict = {input_tensor:x_batch, box_tensor:box_batch, class_tensor:y_batch})

                    epoch_loss += l
                        
                ## Writing summary
                writer.add_summary(summary, global_step = i)

                end_time = time.time()
                print 'Epoch: {}\t Loss:{}\tTime: {}'.format(i, epoch_loss, end_time-start_time)

                ## Saving Checkpoint
                if i % 5 == 0:
                    print 'Storing session as checkpoint'
                    saver.save(sess, 'checkpoints/{}/{}'.format(directory_name, directory_name), i)


    def predict(self, X, class_output_node, box_output_node, checkpoint_dir='two_layer'):
    
        with tf.Session() as sess:

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/{}/checkpoint'.format(checkpoint_dir)))
            if ckpt and ckpt.model_checkpoint_path:
                print 'Restoring session'
                saver.restore(sess, ckpt.model_checkpoint_path)

            input_tensor, class_tensor, box_tensor = self.placeholder
            
            class_out, box_out = sess.run([class_output_node, box_output_node],
                                feed_dict = {input_tensor:X})

            return (class_out, box_out)
