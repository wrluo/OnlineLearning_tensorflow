# -*- coding: utf-8 -*-
# @Author: larry
# @Date:   2017-02-20 18:32:05
# @Last Modified by:   larry
# @Last Modified time: 2017-03-01 22:30:49

import DataReader
import CommonObject
import datetime
#import ipdb
import time
import numpy as np
import pywt
import matplotlib.pyplot as plt
import Wavelet as myWavelet
import DataProcessor
import tensorflow as tf
import NetworkStructure

import tensorflow as tf

train_start = datetime.datetime(2015, 5, 1, 9, 0, 0)
train_end = datetime.datetime(2015, 5, 6, 8, 0, 0)

validate_start = datetime.datetime(2015, 5, 20, 9, 0, 0)
validate_end = datetime.datetime(2015, 5, 25, 8, 0, 0)

test_start = datetime.datetime(2015, 5, 25, 9, 0, 0)
test_end = datetime.datetime(2015, 5, 30, 8, 0, 0)
symbol = "600000-SSE"
level = 1
interval = 1
periodicity = CommonObject.Periodicity.Minutely
observedWindowLength = 230
forwardWindowLength = 10
leastSeqLength = 10

dp = DataProcessor.DataProcessor(train_start, train_end, 
	validate_start, validate_end, 
	test_start, test_end, 
	symbol, level, interval, periodicity, 
	observedWindowLength, forwardWindowLength, leastSeqLength)


dp.generateTensor_movingWindow(CommonObject.dataType.Train)
label = tf.convert_to_tensor(dp.train_labels)
label = tf.reshape(label, [len(dp.train_labels)*230, 1])

print(label)

# print(dp.train_features.shape)
# dp.generateTensor_incremental(CommonObject.dataType.validate)
# dp.generateTensor_incremental(CommonObject.dataType.test)

# train_features = dp.train_features
# train_labels = dp.train_labels
# # train_features = dp.train_features
# # train_labels = dp.train_labels
# # validate_features = np.array(dp.validate_features)
# # validate_labels = np.array(dp.validate_labels)
# # test_features = np.array(dp.test_features)
# # test_labels = np.array(dp.test_labels)

# print(train_features)
# print(train_labels)
# # print(train_features[0].shape)
# # print(train_labels[0].shape)

# estimator = tf.contrib.learn.Estimator(model_fn=NetworkStructure.basic)

# def input_fn():
# 	x = tf.constant(train_features[0])
# 	y = tf.constant(train_labels[0])

# 	return x, y
# # input_fn = tf.contrib.learn.io.numpy_input_fn({'x', train_features}, train_labels, 10, num_epochs=1000)

# # train
# estimator.fit(input_fn=input_fn, steps=1000)
# # evaluate our model
# print(estimator.evaluate(input_fn=input_fn, steps=10))


###################################################Test Code
# while True:
#     try:
#     	if (dr.pointer < dr.end):
#     		print(dr.pointer.strftime("%Y-%m-%d") + " " + str(len(dr.getMinuteData(CommonObject.Periodicity.Daily).index)))
#     		time.sleep(1)
#     	else:
#     		print("Pointer out of range")
#     		break
#     except KeyboardInterrupt:
#         print('Manual break by user')
#         break

# ipdb.set_trace()


################################################################Larry Test
# df = dr.formatSequenceLength(CommonObject.Periodicity.Daily, 1)
# denoised_open, denoised_high, denoised_low, denoised_close, denoised_volume = dp.waveletProcess(df, 'db4', 4, 1, 4)
# plt.plot(denoised_open, label='open')
# plt.plot(denoised_high, label='high')
# plt.plot(denoised_low, label='low')
# plt.plot(denoised_close, label='close')
# plt.legend()
# plt.show()
