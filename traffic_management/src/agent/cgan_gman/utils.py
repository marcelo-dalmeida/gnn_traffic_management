import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import config
from const import CONST


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


def calculate_detection_metrics(pred, label, detection_times):
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()

    # Detection Rate (DR)
    dr = tp / (tp + fn)

    # False Positive Rate (FPR)
    fpr = fp / (fp + tn)

    # F-measurement (F-score)
    f_score = f1_score(label, pred)

    # Mean Time to Detection (MTTD)
    mttd = np.mean(detection_times)

    return dr, fpr, f_score, mttd


def count_continuous_freq(arr):
    count = 0
    items = []
    counts = []

    for i in range(0, len(arr)-1):
        if arr[i] == arr[i+1]:
            count += 1
        else:
            items.append(arr[i])
            counts.append(count)
            count = 0

    items.append(arr[len(arr)-1])
    counts.append(count)

    return np.array(items), np.array(counts)


def seq2instance(data, P, Q, trafpatY_label):

    items, counts = count_continuous_freq(trafpatY_label)

    num_step, dims = data.shape
    num_samples = counts - P - Q + 1
    x = np.zeros(shape=(sum(num_samples), P, dims))
    y = np.zeros(shape=(sum(num_samples), Q, dims))
    trafpatY = np.zeros(shape=(sum(num_samples)))

    total = 0
    for num_sample in num_samples:
        for i in range(num_sample):
            k = total + i
            x[k] = data[k: k + P]
            y[k] = data[k + P: k + P + Q]
            trafpatY[k] = trafpatY_label[k + P + Q]

        total += num_sample

    return x, y, trafpatY


def loadData(dataset_file, attribute):

    traffic_patterns = {
        CONST.REGULAR_TRAFFIC: 0,
        CONST.ANOMALOUS_TRAFFIC: 1,
        CONST.ACCIDENT_BASED_ANOMALOUS_TRAFFIC: 2
    }

    # Traffic
    df = pd.read_hdf(os.path.join(config.ROOT_DIR, dataset_file), key='data')
    df = df.reset_index(level=0)
    df['traffic_pattern'] = [CONST.REGULAR_TRAFFIC] * (len(df) // 3) + [CONST.ANOMALOUS_TRAFFIC] * (len(df) // 3) + [
        CONST.ACCIDENT_BASED_ANOMALOUS_TRAFFIC] * (len(df) // 3)
    df = df.sort_index()

    df.loc[:, 'traffic_pattern'] = df.loc[:, 'traffic_pattern'].replace(traffic_patterns)

    # train/val/test
    num_step = df.shape[0] // len(config.EXPERIMENT.TRAFFIC_PATTERNS)
    train_steps = round(config.AGENT.TRAIN_RATIO * num_step) * len(config.EXPERIMENT.TRAFFIC_PATTERNS)
    test_steps = round(config.AGENT.TEST_RATIO * num_step) * len(config.EXPERIMENT.TRAFFIC_PATTERNS)
    val_steps = num_step * len(config.EXPERIMENT.TRAFFIC_PATTERNS) - train_steps - test_steps

    train_data = df[: train_steps].sort_values(by=['traffic_pattern', 'time'])
    val_data = df[train_steps: train_steps + val_steps].sort_values(by=['traffic_pattern', 'time'])
    test_data = df[-test_steps:].sort_values(by=['traffic_pattern', 'time'])

    train = train_data.loc[:, attribute].values
    val = val_data.loc[:, attribute].values
    test = test_data.loc[:, attribute].values

    traintrafpatY_label = train_data.loc[:, 'traffic_pattern'].values
    valtrafpatY_label = val_data.loc[:, 'traffic_pattern'].values
    testtrafpatY_label = test_data.loc[:, 'traffic_pattern'].values

    # X, Y
    trainX, trainY, traintrafpatY = seq2instance(
        train, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, traintrafpatY_label)
    valX, valY, valtrafpatY = seq2instance(
        val, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, valtrafpatY_label)
    testX, testY, testtrafpatY = seq2instance(
        test, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, testtrafpatY_label)
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
    no_duplicates_Time = Time.drop_duplicates()
    no_duplicates_Time.freq = pd.infer_freq(no_duplicates_Time)
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) // no_duplicates_Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)

    Time = list(zip(Time, df['traffic_pattern'].values))

    train = np.array(list(zip(*sorted(Time[: train_steps], key=lambda x: (x[1], 0))))[0])
    val = np.array(list(zip(*sorted(Time[train_steps: train_steps + val_steps], key=lambda x: (x[1], 0))))[0])
    test = np.array(list(zip(*sorted(Time[-test_steps:], key=lambda x: (x[1], 0))))[0])

    # shape = (num_sample, P + Q, 2)
    trainTE_1, trainTE_2, _ = seq2instance(train, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, traintrafpatY_label)
    trainTE = np.concatenate((trainTE_1, trainTE_2), axis=1).astype(np.int32)
    valTE_1, valTE_2, _ = seq2instance(val, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, valtrafpatY_label)
    valTE = np.concatenate((valTE_1, valTE_2), axis=1).astype(np.int32)
    testTE_1, testTE_2, _ = seq2instance(test, config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, testtrafpatY_label)
    testTE = np.concatenate((testTE_1, testTE_2), axis=1).astype(np.int32)

    return (trainX, trainTE, trainY, traintrafpatY,
            valX, valTE, valY, valtrafpatY,
            testX, testTE, testY, testtrafpatY,
            SE, mean, std)
