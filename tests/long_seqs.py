import argparse
import os
import tensorflow as tf
import numpy as np
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from linear_recurrent_net.layers import linear_surrogate_lstm
from tensorflow import nn
from tensorflow.contrib import rnn


INPUT_DIM = 128

def gen_sample(seq_length):

    # Choose element to be predicted
    desired_class = np.random.choice([0,1], size=1)
    desired_vec = np.zeros((1,INPUT_DIM))
    desired_vec[0, desired_class] = 1

    # Make random one hot vectors for the data 
    all_vectors = np.eye(INPUT_DIM)
    X = all_vectors[np.random.choice(all_vectors.shape[0], size=seq_length)]

    X = np.concatenate((desired_vec, X), axis=0)

    desired_vec = desired_vec.squeeze(0)
    return X, desired_vec

def gen_data(batch_size, seq_length):

    data_X = []
    data_y = []

    for _ in range(batch_size):
        sample_X, sample_y = gen_sample(seq_length)
        data_X.append(sample_X)
        data_y.append(sample_y)

    batch_X = np.array(data_X)
    batch_y = np.array(data_y)

    batch_X = np.swapaxes(batch_X, 0, 1)

    return batch_X, batch_y

def ls_lstm(X):
    n_hidden = 512

    W = tf.get_variable('W', initializer=
                         tf.random_normal([n_hidden, INPUT_DIM]), dtype='float')
    b = tf.get_variable('b', initializer=tf.zeros([INPUT_DIM]), dtype='float')

    layer1 = linear_surrogate_lstm(X, n_hidden, name='ls-lstm')
    outputs = linear_surrogate_lstm(layer1, n_hidden, name='ls-lstm2')    
    print("*"*40, outputs.shape)
    pred = tf.matmul(outputs[-1], W) + b

    return pred

def lstm(X):
    n_hidden = 512
    n_layers = 2
    cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for _ in range(n_layers)])
    outputs, _ = nn.static_rnn(cell, [X[i] for i in range(X.shape[0])], dtype="float")
    W = tf.get_variable('W', initializer=
                         tf.random_normal([n_hidden, INPUT_DIM]), dtype='float')
    b = tf.get_variable('b', initializer=tf.zeros([INPUT_DIM]), dtype='float')
    pred = tf.matmul(outputs[-1], W) + b

    return pred

def run(args):
    if args.ls:
        print("Using LS-LSTM")
    else:
        print("Using traditional LSTM")
    seq_len = args.seq_len
    batch_size = 2**16 // seq_len
    print(f"Using batch size: {batch_size}")
    learning_rate = args.lr
    training_iters = args.num_epochs

    X = tf.placeholder("float", [seq_len+1, batch_size, INPUT_DIM])
    y = tf.placeholder("float", [batch_size, INPUT_DIM])

    pred = ls_lstm(X) if args.ls else lstm(X)
    pred_max = tf.argmax(pred, axis=1)
    y_max = tf.argmax(y, axis=1)
    num_err = tf.count_nonzero((pred_max - y_max))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    streak = 0
    target_streak = 5
    tolerance = 0.01

    with tf.Session() as sess:
        sess.run(init)
        start_time = time.time()

        for train_iter in range(training_iters):
            if train_iter % args.print_epoch == 0:
                print("Epoch", train_iter)
            X_data, y_data = gen_data(batch_size, seq_len)
            # print(X_data.shape)
            # print(y_data.shape)
            _, loss_val, err, p_max, ys_max = sess.run([train_op, loss, num_err, pred_max, y_max], feed_dict={X: X_data, y: y_data})
            error = err/batch_size
            print("Loss:", loss_val, "Accuracy:", 1 - error)

            if error < tolerance:
                streak += 1
            else:
                streak = 0
            if streak == target_streak:
                break

    duration = time.time() - start_time
    print("Train time:", duration)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--ls", action="store_true", help="Use LS-LSTM")
    args.add_argument("--seq-len", required=True, type=int)

    # Traditional: 0.1  LS-LSTM: 0.005
    args.add_argument("--lr", type=float, default=0.005)
    args.add_argument("--num-epochs", type=int, default=500)

    args.add_argument("--print-epoch", default=1, help="How often to print epoch number")
    args.add_argument("--gpuid", default=-1, type=int)
    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    run(args)

