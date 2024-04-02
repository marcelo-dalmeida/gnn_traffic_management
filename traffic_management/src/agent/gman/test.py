import os
import math
import time

import numpy as np
import tensorflow as tf

import utils


def test():

    import config

    dataset_file = os.path.join(config.PATH_TO_RECORDS, 'data', 'dataset.h5')

    start = time.time()

    log_file = os.path.join(config.PATH_TO_RECORDS, config.EXPERIMENT.SCENARIO_NAME)
    log = open(os.path.join(config.ROOT_DIR, log_file), 'w')

    model_file = os.path.join(config.PATH_TO_RECORDS, f"{config.EXPERIMENT.SCENARIO_NAME}_gman_model")

    # load data
    utils.log_string(log, 'loading data...')
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
     SE, mean, std) = utils.loadData(dataset_file, config.AGENT.PREDICTED_ATTRIBUTE)
    num_train, num_val, num_test = trainX.shape[0], valX.shape[0], testX.shape[0]
    utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
    utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
    utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
    utils.log_string(log, 'data loaded!')

    # test model
    utils.log_string(log, '**** testing model ****')
    utils.log_string(log, 'loading model from %s' % os.path.join(config.ROOT_DIR, model_file))
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(config.ROOT_DIR, model_file) + '.meta')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        saver.restore(sess, os.path.join(config.ROOT_DIR, model_file))
        parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            parameters += np.product([x.value for x in variable.get_shape()])
        utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
        pred = graph.get_collection(name='pred')[0]
        utils.log_string(log, 'model restored!')
        utils.log_string(log, 'evaluating...')
        trainPred = []
        num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)
            feed_dict = {
                'X:0': trainX[start_idx: end_idx],
                'TE:0': trainTE[start_idx: end_idx],
                'is_training:0': False}
            pred_batch = sess.run(pred, feed_dict=feed_dict)
            trainPred.append(pred_batch)
        trainPred = np.concatenate(trainPred, axis=0)
        valPred = []
        num_batch = math.ceil(num_val / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_val, (batch_idx + 1) * config.AGENT.BATCH_SIZE)
            feed_dict = {
                'X:0': valX[start_idx: end_idx],
                'TE:0': valTE[start_idx: end_idx],
                'is_training:0': False}
            pred_batch = sess.run(pred, feed_dict=feed_dict)
            valPred.append(pred_batch)
        valPred = np.concatenate(valPred, axis=0)
        testPred = []
        num_batch = math.ceil(num_test / config.AGENT.BATCH_SIZE)
        start_test = time.time()
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_test, (batch_idx + 1) * config.AGENT.BATCH_SIZE)
            feed_dict = {
                'X:0': testX[start_idx: end_idx],
                'TE:0': testTE[start_idx: end_idx],
                'is_training:0': False}
            pred_batch = sess.run(pred, feed_dict=feed_dict)
            testPred.append(pred_batch)
        end_test = time.time()
        testPred = np.concatenate(testPred, axis=0)
    train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
    val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
    test_mae, test_rmse, test_mape = utils.metric(testPred, testY)
    utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
                     (train_mae, train_rmse, train_mape * 100))
    utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                     (val_mae, val_rmse, val_mape * 100))
    utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
                     (test_mae, test_rmse, test_mape * 100))
    utils.log_string(log, 'performance in each prediction step')
    MAE, RMSE, MAPE = [], [], []
    for q in range(config.AGENT.PREDICTION_STEPS):
        mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        utils.log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                         (q + 1, mae, rmse, mape * 100))
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    utils.log_string(
        log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
             (average_mae, average_rmse, average_mape * 100))
    end = time.time()
    utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()


if __name__ == "__main__":
    test()
