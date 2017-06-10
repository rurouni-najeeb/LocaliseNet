## Importing required libraries
import tensorflow as tf
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import os
from sklearn.preprocessing import StandardScaler
from LocaliseNet import LocaliseNet

def main():
    
    ## Defining constants
    path = '../../Dataset/VOC2007'
    data_file = 'data.pkl'
    box_file = 'labels.pkl'
    class_file = 'name.pkl'
    params = {}
    params['LEARNING_RATE'] = 1e-3
    params['N_EPOCHS'] = 10
    params['BATCH_SIZE'] = 128
    params['NUM_CLASSES'] = 20
    params['NUM_COORDINATES'] = 4

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
    model = LocaliseNet(params)
    image_tensor, class_tensor, box_tensor = model._create_placeholder()
    conv1 = model._create_conv_relu_layer(image_tensor, "conv1")
    maxpool1 = model._create_max_pool(conv1, "maxpool1")
    conv2 = model._create_conv_relu_layer(maxpool1, "conv2", kernel_size=[5,5,32], n_kernels=64)
    maxpool2 = model._create_max_pool(conv2, "maxpool2")
    fc_class = model._create_fully_connected(maxpool2, num_neurones=128, layer_name="fc_class")
    softmax = model._create_softmax(fc_class, num_output=20, layer_name="softmax")
    fc_regress = model._create_fully_connected(maxpool2, num_neurones=128, layer_name="fc_regress", save_vars=True)
    box_cood = model._create_fully_connected(fc_regress, num_neurones=4, layer_name="box_cood", save_vars=True)
    cat_loss = model._create_xentropy_loss(softmax, layer_name="xentropy")
    reg_loss = model._create_squared_loss(box_cood, layer_name="squared_loss")
    model._create_optimizer(cat_loss, "category_optimizer")
    model._create_optimizer(reg_loss, "box_optimizer", regression=True)
    summary_op_class = model._create_summaries(cat_loss, layer_name="category_summary", loss_name="cat_loss")
    summary_op_regression = model._create_summaries(reg_loss, layer_name="regression_summary", loss_name="reg_loss")

    ## Predicting the output
    class_scores, box_coordinates = model.predict(data, softmax, box_cood, checkpoint_dir="two_layer")

    ## Accuracy of predictions
    print 'Accuracy on training set: {}'.format(np.mean(np.equal(np.argmax(class_scores,1), np.argmax(class_labels,1))))


if __name__ == "__main__":
    main()