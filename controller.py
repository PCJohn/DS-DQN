import cv2
import tensorflow as tf
from tensorflow import nn
import tfac
import numpy as np

#Build the net in the session
def build_net(sess):
    in_len = 32
    in_dep = 1

    x_hold = tf.placeholder(tf.float32,shape=[None,in_dep*in_len*in_len])
    y_hold = tf.placeholder(tf.float32,shape=[None,2])
    keep_prob = tf.placeholder(tf.float32)

    xt = tf.reshape(x_hold,[-1,in_len,in_len,in_dep])

    #Layer 1 - 5x5 convolution
    w1 = tfac.weight([5,5,in_dep,4])
    b1 = tfac.bias([4])
    c1 = nn.relu(nn.conv2d(xt,w1,strides=[1,2,2,1],padding='VALID')+b1)
    o1 = c1

    #Layer 2 - 3x3 convolution
    w2 = tfac.weight([3,3,4,16])
    b2 = tfac.bias([16])
    c2 = nn.relu(nn.conv2d(o1,w2,strides=[1,2,2,1],padding='VALID')+b2)
    o2 = c2

    #Layer 3 - 3x3 convolution
    w3 = tfac.weight([3,3,16,32])
    b3 = tfac.bias([32])
    c3 = nn.relu(nn.conv2d(o2,w3,strides=[1,1,1,1],padding='VALID')+b3)
    o3 = c3

    dim = 32 * 4*4
        
    #Fully connected layer - 600 units
    of = tf.reshape(o3,[-1,dim])
    w4 = tfac.weight([dim,600])
    b4 = tfac.bias([600])
    o4 = nn.relu(tf.matmul(of,w4)+b4)

    o4 = nn.dropout(o4, keep_prob)

    #Output softmax layer - 2 units
    w5 = tfac.weight([600,2])
    b5 = tfac.bias([2])
    y = nn.softmax(tf.matmul(o4,w5)+b5)

    sess.run(tf.initialize_all_variables())

    return y,x_hold,y_hold,keep_prob

#Method to run the training
def train_net():
    train,val,test = face_ds.load_find_ds()
    sess = tfac.start_sess()
    y,x_hold,y_hold,keep_prob = build_net(sess)
    acc = tfac.train(sess,
                    y,
                    x_hold,
                    y_hold,
                    keep_prob,
                    train[0],train[1],
                    test[0],test[1],
                    lrate=1e-4,
                    epsilon=1e-16,
                    n_epoch=8,
                    batch_size=100,
                    print_epoch=1,
                    save_path=model_path)
    print "Accuracy:",acc
    sess.close()
