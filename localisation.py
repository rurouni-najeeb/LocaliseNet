## Importing required libraries
import tensorflow as tf
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler

## Defining Constants
path = '../../../Dataset/VOC2007'
data_file = 'data.pkl'
box_file = 'labels.pkl'
class_file = 'name.pkl'
N_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
trainable = True
var_list = []

print 'Acquiring data.....'

with open(os.path.join(path, data_file), 'rb') as fp:
    data = pkl.load(fp)
    data = np.asarray(data, dtype=np.float32)

with open(os.path.join(path, box_file), 'rb') as fp:
    labels = pkl.load(fp)
    labels = np.asarray(labels,dtype=np.float32)

with open(os.path.join(path, class_file), 'rb') as fp:
    name = pkl.load(fp)
    class_labels = pd.get_dummies(name).values

## Characteritics of Data
print 'VOC Data Characteristics'
print 'Shape of Data: ', data.shape
print 'Shape of Bounding Box Labels: ', labels.shape
print 'Number of Categories: ', len(np.unique(name))


## Preprocessing the data
print 'Normalising the data'
b, x, y, z = data.shape
scaler = StandardScaler()
data = scaler.fit_transform(data.reshape([b,x*y*z]))
print 'Standard Deviation: ', np.std(data)
print 'Mean: ', np.mean(data)
data = data.reshape([b,x,y,z])

## Graph Definition
print'Constructing the graph...'
tf.reset_default_graph()

with tf.variable_scope('placeholder') as scope:
    input_tensor = tf.placeholder(dtype=tf.float32,shape=[None,28,28,3],name="Input")
    class_tensor = tf.placeholder(dtype=tf.float32,shape=[None,20],name="Label")
    box_tensor = tf.placeholder(dtype=tf.float32,shape=[None,4],name="Box")

with tf.variable_scope('conv1') as scope:
    w = tf.get_variable(name='weights',shape=[5,5,3,32],initializer=tf.contrib.layers.xavier_initializer_conv2d(),trainable=trainable)
    b = tf.get_variable(name="biases",shape=[32],initializer=tf.random_normal_initializer(),trainable=trainable)
    conv = tf.nn.conv2d(input_tensor,w,strides=[1,1,1,1],padding="SAME")
    conv1 = tf.nn.relu(conv + b)

with tf.variable_scope('max_pool1') as scope:
    max_pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

with tf.variable_scope('conv2') as scope:
    w = tf.get_variable(name='weights',shape=[5,5,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d(),trainable=trainable)
    b = tf.get_variable(name='biases',shape=[64],initializer=tf.random_normal_initializer(),trainable=trainable)
    conv = tf.nn.conv2d(max_pool1,w,strides=[1,1,1,1],padding="SAME")
    conv2 = tf.nn.relu(conv + b)

with tf.variable_scope('max_pool2') as scope:
    max_pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

with tf.variable_scope('fully_connected') as scope:
    flat = tf.reshape(max_pool2,shape=[-1,7*7*64])
    w = tf.get_variable(name="weights",shape=[7*7*64,128],initializer=tf.contrib.layers.xavier_initializer(),trainable=trainable)
    b = tf.get_variable(name="biases",shape=[128],initializer=tf.random_normal_initializer(),trainable=trainable)
    out = tf.matmul(flat,w) + b
    full = tf.nn.relu(out)

with tf.variable_scope('softmax') as scope:
    w = tf.get_variable(name="weights",shape=[128,20],initializer=tf.contrib.layers.xavier_initializer(),trainable=trainable)
    b = tf.get_variable(name="biases",shape=[20],initializer=tf.random_normal_initializer(),trainable=trainable)
    softmax = tf.nn.softmax(tf.matmul(full,w) + b,name=scope.name)



with tf.variable_scope('fully_connected_regressor') as scope:
    flat = tf.reshape(max_pool2,shape=[-1,7*7*64])
    w = tf.get_variable(name="weights",shape=[7*7*64,128],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name="biases",shape=[128],initializer=tf.random_normal_initializer())
    out = tf.matmul(flat,w) + b
    regressor = tf.nn.relu(out)
    var_list += [w, b]


with tf.variable_scope('box_output') as scope:
    w = tf.get_variable(name="weights",shape=[128,4],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name="biases",shape=[4],initializer=tf.random_normal_initializer())
    out = tf.matmul(regressor,w) + b
    box_output = tf.nn.relu(out)
    var_list += [w, b]

with tf.variable_scope('xentropy_loss') as scope:
    category_loss = tf.reduce_mean(-tf.reduce_sum(class_tensor*tf.log(softmax),reduction_indices=[1]))

with tf.variable_scope('regression_loss') as scope:
    regressor_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(box_tensor,box_output),reduction_indices=[1]))

global_step_category = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step_category")
global_step_regressor = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step_regressor")

with tf.variable_scope("category_optimiser") as scope:
    c_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(category_loss,global_step=global_step_category)

with tf.variable_scope("regression_optimizer") as scope:
    r_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(regressor_loss, var_list = var_list, global_step=global_step_regressor)

with tf.variable_scope("classification_summaries") as scope:
    tf.summary.scalar("category_loss", category_loss)
    tf.summary.histogram("category_loss", category_loss)
    summary_op_class = tf.summary.merge_all()

with tf.variable_scope("regressor_summaries") as scope:
    tf.summary.scalar("regression_loss", regressor_loss)
    tf.summary.histogram("regression_loss", regressor_loss)
    summary_op_regression = tf.summary.merge_all()
## Training the graph
ch = str(raw_input('Want to train the classification head: '))
if ch == 'y' or ch == 'Y':
    
    print 'Training the graph...'

    with tf.Session() as sess:
        ## Initialising variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ## Computation graph
        writer = tf.summary.FileWriter('graphs/two_layer/', sess.graph)
        
        ## Checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/two_layer/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path :
            print 'Model checkpoint found. Restoring session'
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        ## Training the classification head
        print 'Training for Classification Head'
        for i in range(1, N_EPOCHS+1):
            epoch_loss = 0
            start_time = time.time()
            N_BATCHES = data.shape[0]/BATCH_SIZE

            for j in tqdm(range(N_BATCHES)):
                
                ## Generating batches for data
                cur_batch, next_batch = j*BATCH_SIZE, (j+1)*BATCH_SIZE
                x_batch = data[cur_batch:next_batch,:,:,:]
                y_batch = class_labels[cur_batch:next_batch,:]
                
                ## Running the optimizer
                _, l, summary = sess.run([c_optimizer, category_loss, summary_op_class],
                            feed_dict = {input_tensor:x_batch, class_tensor:y_batch})
                epoch_loss += l
                    
            ## Writing summary
            writer.add_summary(summary, global_step = i)

            end_time = time.time()
            print 'Epoch: {}\t Loss:{}\tTime: {}'.format(i, epoch_loss, end_time-start_time)

            ## Saving Checkpoint
            if i % 5 == 0:
                print 'Storing session as checkpoint'
                saver.save(sess, 'checkpoints/two_layer/two_layer', i)

        
        
## Training the Regression Head
ch = str(raw_input('Want to train regression head?'))
if ch == 'y' or ch == 'Y':

    with tf.Session() as sess:

        ## Initialising the model
        init = tf.global_variables_initializer()
        sess.run(init)

        ## Computation graph
        writer = tf.summary.FileWriter('graphs/two_layer', sess.graph)

        ## Checkpoint
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/two_layer/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print 'Model checkpoint found. Restoring session'
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        ## Training the regression head  
        for i in range(1, N_EPOCHS+1):
            epoch_loss = 0
            start_time = time.time()
            N_BATCHES = data.shape[0]/BATCH_SIZE

            for j in tqdm(range(N_BATCHES)):
                
                ## Generating batches for data
                cur_batch, next_batch = j*BATCH_SIZE, (j+1)*BATCH_SIZE
                x_batch = data[cur_batch:next_batch,:,:,:]
                y_batch = labels[cur_batch:next_batch,:]
                false_batch = class_labels[cur_batch:next_batch,:]

                ## Running the optimizer
                _, l, summary = sess.run([r_optimizer, regressor_loss, summary_op_regression],
                                        feed_dict = {input_tensor:x_batch, box_tensor:y_batch, class_tensor:false_batch})
                
                epoch_loss += l
            
            ## Writing the summary
            writer.add_summary(summary, global_step=i)

            end_time = time.time()

            print 'Epoch: {}\t Loss:{}\t Time:{}'.format(i, epoch_loss, end_time - start_time)

            ## Saving Checkpoint
            if i % 10 == 0:
                print 'Saving the checkpoint'
                saver.save(sess, 'checkpoints/two_layer_reg/two_layer', i)




