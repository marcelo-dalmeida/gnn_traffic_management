import os

import numpy as np
import pandas as pd

import config


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape


def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y


def loadData(dataset_file, attribute):
    # Traffic
    df = pd.read_hdf(os.path.join(config.ROOT_DIR, dataset_file), key='data').loc[:, attribute]
    Traffic = df.values
    # train/val/test 
    num_step = df.shape[0]
    train_steps = round(config.AGENT.TRAIN_RATIO * num_step)
    test_steps = round(config.AGENT.TEST_RATIO * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps: train_steps + val_steps]
    test = Traffic[-test_steps:]
    # X, Y 
    trainX, trainY = seq2instance(train, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS)
    valX, valY = seq2instance(val, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS)
    testX, testY = seq2instance(test, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    se_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, 'SE.txt')

    # spatial embedding 
    f = open(se_file, mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1:]

    # temporal embedding 
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps: train_steps + val_steps]
    test = Time[-test_steps:]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
    valTE = seq2instance(val, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)
    testTE = seq2instance(test, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)
