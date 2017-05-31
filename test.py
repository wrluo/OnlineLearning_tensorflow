import tensorflow as tf
import numpy as np
import time
import NetworkStructure as network
import DataProcessor
import CommonObject
import datetime

from CommonObject import PathConfig

class StockInput(object):
	"""The input data."""

	def __init__(self, config, dataConfig, features, targets, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		# self.epoch_size = ((len(features)*dataConfig.observedWindowLength // batch_size) - 1) // num_steps
		self.epoch_size = 10
		#######################To be developed
		self.input_data = tf.convert_to_tensor(features)
		self.targets =tf.reshape(tf.convert_to_tensor(targets), [len(features)*dataConfig.observedWindowLength, 1])


class networkConfig(object):
	"""docstring for networkConfig"""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 230
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 44
	vocab_size = 10000

class dataConfig(object):
	"""docstring for dataConfig"""
	train_start = datetime.datetime(2015, 5, 1, 9, 0, 0)
	train_end = datetime.datetime(2015, 5, 6, 8, 0, 0)

	validate_start = datetime.datetime(2015, 5, 20, 9, 0, 0)
	validate_end = datetime.datetime(2015, 5, 21, 8, 0, 0)

	test_start = datetime.datetime(2015, 5, 21, 9, 0, 0)
	test_end = datetime.datetime(2015, 5, 22, 8, 0, 0)

	symbol = "600000-SSE"
	level = 1
	interval = 1
	periodicity = CommonObject.Periodicity.Minutely
	observedWindowLength = 230
	forwardWindowLength = 10
	leastSeqLength = 10
		
def run_epoch(session, model, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state)

	fetches = {
		"cost": model.cost,
		"final_state": model.final_state,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
				(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
					iters * model.input.batch_size / (time.time() - start_time)))

	return np.exp(costs / iters)

def get_config(config_name):
	if config_name == "network":
		return networkConfig()
	elif config_name == "data":
		return dataConfig()

def main(_):
	config_data = get_config("data")
	dp = DataProcessor.DataProcessor(config_data.train_start, config_data.train_end, 
		config_data.validate_start, config_data.validate_end, 
		config_data.test_start, config_data.test_end, 
		config_data.symbol, config_data.level, config_data.interval, config_data.periodicity, 
		config_data.observedWindowLength, config_data.forwardWindowLength, config_data.leastSeqLength)
	dp.generateTensor_movingWindow(CommonObject.dataType.Train)
	dp.generateTensor_movingWindow(CommonObject.dataType.Validate)
	dp.generateTensor_movingWindow(CommonObject.dataType.Test)

	config = get_config("network")
	eval_config = get_config("network")
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

	with tf.name_scope("Train"):
		train_input = StockInput(config=config, dataConfig=config_data, features=dp.train_features, targets=dp.train_labels, name="TrainInput")
		with tf.variable_scope("Model", reuse=None, initializer=initializer):
			m = network.stockPricePredictionModel(is_training=True, config=config, input_=train_input)
		tf.summary.scalar("Training Loss", m.cost)
		tf.summary.scalar("Learning Rate", m.lr)

	# with tf.name_scope("Valid"):
	# 	valid_input = StockInput(config=config, dataConfig=config_data, features=dp.validate_features, targets=dp.validate_labels, name="ValidInput")
	# 	with tf.variable_scope("Model", reuse=True, initializer=initializer):
	# 		mvalid = network.stockPricePredictionModel(is_training=False, config=config, input_=valid_input)
	# 	tf.summary.scalar("Validation Loss", mvalid.cost)

	# with tf.name_scope("Test"):
	# 	test_input = StockInput(config=eval_config, dataConfig=config_data, features=dp.test_features, targets=dp.test_labels, name="TestInput")
	# 	with tf.variable_scope("Model", reuse=True, initializer=initializer):
	# 		mtest = network.stockPricePredictionModel(is_training=False, config=eval_config, input_=test_input)

	#######################To be changed directory
	sv = tf.train.Supervisor(logdir="~/Desktop/Online-Learning/models")
	with sv.managed_session() as session:
		for i in range(config.max_max_epoch):
			lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
			m.assign_lr(session, config.learning_rate * lr_decay)

			print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
			train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)

			print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
			# valid_perplexity = run_epoch(session, mvalid)
			# print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

		# test_perplexity = run_epoch(session, mtest)
		# print("Test Perplexity: %.3f" % test_perplexity)

		##################To be changed directory
		# if FLAGS.save_path:
		if "~/Desktop/Online-Learning/models":
			print("Saving model to %s." % "~/Desktop/Online-Learning/models")
			sv.saver.save(session, "~/Desktop/Online-Learning/models", global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()