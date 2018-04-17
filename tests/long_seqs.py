import tensorflow as tf

def ls_lstm(seq_len)
    n_hidden = 256
    n_classes = 2
    n_steps = seq_len
    batch_size = 65536 // seq_len
    n_input = 24
    n_layers = 2

    x = tf.placeholder("float", [n_steps, batch_size, n_input])
    y = tf.placeholder("float", [batch_size, n_classes])
    W = tf.get_variable('W', initializer=
                         tf.random_normal([n_hidden, n_classes]), dtype='float')
    b = tf.get_variable('b', initializer=tf.zeros([n_classes]), dtype='float')

    layer1 = linear_surrogate_lstm(x, n_hidden, name='ls-lstm')
    outputs = linear_surrogate_lstm(layer1, n_hidden, name='ls-lstm2')    
    pred = tf.matmul(outputs[-1], W) + b

    return x, y, pred

def run():
    learning_rate = 0.001
    training_iters = 5000000
    seq_len = 1024

    x, y, pred = ls_lstm(seq_len)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    init = tf.global_variables_initializer()
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(init)
        for train_iter in range(training_iters):
            sess.run(train_op, feed_dict={x: x_in, y: y_in})

if __name__=="__main__":
    run()

