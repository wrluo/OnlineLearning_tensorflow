# -*- coding: utf-8 -*-
# @Author: larry
# @Date:   2017-02-27 00:29:48
# @Last Modified by:   larry
# @Last Modified time: 2017-02-27 00:39:26

from aenum import Enum
import tensorflow as tf


class Periodicity(Enum):
    Secondly = 1
    Minutely = 2
    Hourly = 3
    Daily = 4
    Weekly = 5
    Monthly = 6

class dataType(Enum):
	Train = 1
	Validate = 2
	Test = 3

def PathConfig():
	flags = tf.flags
	logging = tf.loggin

	flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, medium, large.")
	flags.DEFINE_string("data_path", None, "Where the training/test data is stored.")
	flags.DEFINE_string("save_path", "~/Desktop/Online-Learning/models", "Model output directory.")
	flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")

	return flags.FLAGS