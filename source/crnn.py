import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



class Model(object):

    def __init__(self, num_classes, seq_len, word_dict_size, num_filters1=150, num_filters2=100, filter1_size=2,
                 filter2_size=5, emb_size=100, dist_emb_size=10, l2_reg_lambda=0.01):
        tf.reset_default_graph()
        # model_path = './ckpt/lstm-cnn-att/model.ckpt'

        self.w = tf.placeholder(tf.int32, [None, seq_len], name="x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Initialization
        # W_emb = tf.Variable(wv, name='W_emb', trainable=True)
        W_emb = tf.Variable(tf.truncated_normal([100000,emb_size], stddev=0.1),trainable=True,name='W_emb')

        # Embedding layer
        X = tf.nn.embedding_lookup(W_emb, self.w)

        # LSTM layer
        with tf.variable_scope('lstm'):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_filters1, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_filters1, state_is_tuple=True)

            _X = tf.unstack(X, num=seq_len, axis=1)
            outputs, state_fw, state_bw = tf.nn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                                                                                  cell_bw=lstm_bw_cell, inputs=_X,
                                                                                  dtype=tf.float32)
            outputs = tf.stack(outputs, axis=1)

            h1_rnn = tf.expand_dims(outputs, -1)

            ## Max pooling
            h1_pool = tf.nn.max_pool(h1_rnn, ksize=[1, filter1_size, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

        # CNN+Maxpooling Layer

        with tf.variable_scope('cnn'):
            filter_shape = [filter2_size, 2 * num_filters1, 1, num_filters2]
            W_cnn = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")  # Convolution parameter
            b_cnn = tf.Variable(tf.constant(0.1, shape=[num_filters2]), name="b")  # Convolution bias parameter
            conv = tf.nn.conv2d(h1_pool,
                                W_cnn,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
            h1_cnn = tf.nn.relu(tf.nn.bias_add(conv, b_cnn))

            ##Maxpooling
            h2_pool = tf.nn.max_pool(h1_cnn,
                                     ksize=[1, seq_len - (filter1_size - 1) - (filter2_size - 1), 1, 1],
                                     strides=[1, 1, 1, 1],
                                     padding="VALID")
            h2_cnn = tf.squeeze(h2_pool, axis=[1, 2])

        ##Dropout
        h_flat = tf.reshape(h2_pool, [-1, num_filters2])
        # h_flat = tf.reshape(h2_cnn,[-1,(seq_len-3*(filter_size-1))*2*num_filters])
        h_drop = tf.nn.dropout(h_flat, self.dropout_keep_prob)

        # Fully connetected layer
        W = tf.Variable(tf.truncated_normal([num_filters2, num_classes], stddev=0.1), name="W")
        # W = tf.Variable(tf.truncated_normal([(seq_len-3*(filter_size-1))*2*num_filters, num_classes], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")

        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(W)

        # prediction and loss function
        self.predictions = tf.argmax(scores, 1)
        self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
        self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        # Accuracy
        self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss, global_step=self.global_step)

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=session_conf)

        self.sess.run(tf.global_variables_initializer())

    def train_step(self, W_batch, y_batch):
        feed_dict = {
            self.w: W_batch,
            self.dropout_keep_prob: 0.7,
            self.input_y: y_batch
        }
        _, step, loss, accuracy, predictions = self.sess.run(
            [self.optimizer, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
        print("step " + str(step) + " loss " + str(loss) + " accuracy " + str(accuracy))
        return step, accuracy

    def test_step(self, W_batch, y_batch):
        feed_dict = {
            self.w: W_batch,
            self.dropout_keep_prob: 1.0,
            self.input_y: y_batch
        }
        loss, accuracy, predictions = self.sess.run([self.loss, self.accuracy, self.predictions], feed_dict)
        return predictions
