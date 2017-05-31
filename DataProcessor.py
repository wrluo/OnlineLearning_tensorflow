import datetime
import DataReader
import numpy as np
import Wavelet as myWavelet
import CommonObject
import tensorflow as tf


class DataProcessor:
    """docstring for DataProcessor"""

    def __init__(self, train_Start, train_End, validate_Start, validate_End, test_Start, test_End, symbol, level, interval, periodicity, observedWindowLength, forwardWindowLength, leastSeqLength):
        self.train_dataReader = DataReader.DataReader(
            train_Start, train_End, symbol, level, interval, periodicity)
        self.validate_dataReader = DataReader.DataReader(
            validate_Start, validate_End, symbol, level, interval, periodicity)
        self.test_dataReader = DataReader.DataReader(
            test_Start, test_End, symbol, level, interval, periodicity)

        self.train_features = []
        self.train_labels = []
        self.validate_features = []
        self.validate_labels = []
        self.test_features = []
        self.test_labels = []

        self.forwardWindowLength = forwardWindowLength
        self.observedWindowLength = observedWindowLength
        self.leastSeqLength = leastSeqLength

    def waveletProcess(self, dataframe, waveletFunc, levelNum, denoiseStartLayer, denoiseEndLayer):
        raw_open = (dataframe['STARTPRC'].values).astype(np.float)
        raw_high = (dataframe['HIGHPRC'].values).astype(np.float)
        raw_low = (dataframe['LOWPRC'].values).astype(np.float)
        raw_close = (dataframe['ENDPRC'].values).astype(np.float)
        raw_volume = (dataframe['MINTQ'].values).astype(np.float)

        return(myWavelet.wt(raw_open, waveletFunc, levelNum, denoiseStartLayer, denoiseEndLayer),
               myWavelet.wt(raw_high, waveletFunc, levelNum,
                            denoiseStartLayer, denoiseEndLayer),
               myWavelet.wt(raw_low, waveletFunc, levelNum,
                            denoiseStartLayer, denoiseEndLayer),
               myWavelet.wt(raw_close, waveletFunc, levelNum,
                            denoiseStartLayer, denoiseEndLayer),
               myWavelet.wt(raw_volume, waveletFunc, levelNum, denoiseStartLayer, denoiseEndLayer))

    def generateTensor_movingWindow(self, dataReader):
    	if (dataReader == CommonObject.dataType.Train):
    		myDataReader = self.train_dataReader
    	elif (dataReader == CommonObject.dataType.Validate):
    		myDataReader = self.validate_dataReader
    	else:
    		myDataReader = self.test_dataReader

    	while myDataReader.pointer < myDataReader.end:
    		df = myDataReader.formatSequenceLength(CommonObject.Periodicity.Daily, 1)
    		if len(df) >= self.observedWindowLength:
    			for i in range(self.observedWindowLength, len(df)):
    				features_df = df[i-self.observedWindowLength:i]

    				raw_open = (features_df['STARTPRC'].values).astype(np.float)
    				raw_high = (features_df['HIGHPRC'].values).astype(np.float)
    				raw_low = (features_df['LOWPRC'].values).astype(np.float)
    				raw_close = (features_df['ENDPRC'].values).astype(np.float)
    				feature1 = self.myNormalize((raw_high - raw_low) / raw_close)
    				feature2 = self.myNormalize((raw_high - raw_open) / raw_close)
    				# self.train_features.append(np.array([feature1, feature2]).T)
    				self.train_features.append(tf.convert_to_tensor(np.array([feature1, feature2]).T))

    				self.train_labels.append(tf.convert_to_tensor(raw_close))

    		print(myDataReader.pointer)

    def generateTensor_test(self, dataReader):
    	if (dataReader == CommonObject.dataType.Train):
    		myDataReader = self.train_dataReader
    	elif (dataReader == CommonObject.dataType.Validate):
    		myDataReader = self.validate_dataReader
    	else:
    		myDataReader = self.test_dataReader

    	df = myDataReader.formatSequenceLength(CommonObject.Periodicity.Daily, 1)

    	features = []
    	labels = []

    	for i in range(1, len(df)):
    		features.append(df['ENDPRC'].iloc[i-1: i+1])
    		labels.append(float(df['ENDPRC'].iloc[i: i+1]))

    	return features, labels

    def myNormalize(self, feature):
    	feature = np.array(feature)
    	feature = [(float(i) - np.mean(feature)) / np.std(feature) for i in feature]
    	return feature
