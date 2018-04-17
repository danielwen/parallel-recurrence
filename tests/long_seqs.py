import argparse
import os
import tensorflow as tf
import numpy as np

INPUT_DIM = 10 #128
NUM_CLASSES = 2 # +1 or -1

def gen_sample(seq_length):

    # Choose element to be predicted
    desired_class = np.random.choice([-1,1], size=1)
    desired_vec = np.zeros((1,INPUT_DIM))
    desired_vec[0,0] = desired_class

    # Make random one hot vectors for the data 
    all_vectors = np.eye(INPUT_DIM)
    X = all_vectors[np.random.choice(all_vectors.shape[0], size=seq_length)]

    X = np.concatenate((desired_vec, X), axis=0)

    return X, desired_class

def gen_data(batch_size, seq_length):

    data_X = []
    data_y = []

    for _ in range(batch_size):
        sample_X, sample_y = gen_sample(seq_length)
        data_X.append(sample_X)
        data_y.append(sample_y)

    batch_X = np.array(data_X)
    batch_y = np.array(data_y)

    return batch_X, batch_y

def run(args):
    seq_len = args.seq_len
    batch_size = args.batch_size

    X_data, y_data = gen_data(batch_size, seq_len)

    print(X_data)
    print(X_data.shape)
    print(y_data)
    print(y_data.shape)

    X = tf.placeholder("float", [seq_len, batch_size, INPUT_DIM])
    y = tf.placeholder("float", [batch_size, NUM_CLASSES])

    # Do training loop

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--seq-len", required=True, type=int)

    args.add_argument("--lr", default=1e-4)
    args.add_argument("--batch-size", type=int, default=256)
    args.add_argument("--num-epochs", default=10)

    args.add_argument("--print-epoch", default=1, help="How often to print epoch number")
    args.add_argument("--gpuid", default=-1, type=int)
    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    run(args)

