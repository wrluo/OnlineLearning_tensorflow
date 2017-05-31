import tensorflow as tf
import numpy as np

import time
import inspect

from CommonObject import PathConfig

def data_type():
  # return tf.float16 if FLAGS.use_fp16 else tf.float32
  return tf.float64


class stockPricePredictionModel(object):
    """docstring for stockPredictionModel"""

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size

        def lstm_cell():
            if 'reuse' in inspect.getfullargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
            	return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            else:
            	return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell

        if is_training and config.keep_prob < 1:
        	def attn_cell():
        		return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        if is_training and config.keep_prob < 1:
        	inputs = tf.nn.dropout(input_.input_data, config.keep_prob)
        else:
        	inputs = input_.input_data


        # below code can be replaced by theses two lines
        #
        #
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self._initial_state)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
        	for time_step in range(num_steps):
        		if time_step > 0: tf.get_variable_scope().reuse_variables()
        		(cell_output, state) = cell(inputs[:, time_step, :], state)
        		outputs.append(cell_output)

        ###############################################################################################################
        ###############################################################################################################
        ##############should be further developed######################################################################
        ###############################################################################################################
        ###############################################################################################################
        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        regression_w = tf.get_variable("regression_w", [size, 1], dtype=data_type())
        regression_b = tf.get_variable("regression_b", [1], dtype=data_type())
        predict = tf.matmul(output, regression_w) + regression_b
        loss = tf.reduce_sum(tf.square(predict - input_.targets))
        ###############################################################################################################
        ###############################################################################################################
        ##############should be further developed######################################################################
        ###############################################################################################################
        ###############################################################################################################

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
        	return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
        	zip(grads, tvars),
        	global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
        	tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
    	session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
    	return self._input

    @property
    def initial_state(self):
    	return self._initial_state

    @property
    def cost(self):
    	return self._cost

    @property
    def final_state(self):
    	return self._final_state

    @property
    def lr(self):
    	return self._lr

    @property
    def train_op(self):
    	return self._train_op