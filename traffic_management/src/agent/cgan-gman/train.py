import math
import os
import time
import datetime

import numpy as np
import tensorflow as tf

import agent.gman.utils as utils
import agent.gman.model as model


def train():

    start = time.time()

    import config

    dataset_file = os.path.join(config.PATH_TO_RECORDS, 'data', 'dataset.h5')

    log_file = os.path.join(config.PATH_TO_RECORDS,
                            f"{config.EXPERIMENT.SCENARIO_NAME}_gman_log")
    log = open(os.path.join(config.ROOT_DIR, log_file), 'w')

    model_file = os.path.join(config.PATH_TO_RECORDS,
                              f"{config.EXPERIMENT.SCENARIO_NAME}_gman_model")

    # load data
    utils.log_string(log, 'loading data...')
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE,
     mean, std) = utils.loadData(dataset_file, config.AGENT.PREDICTED_ATTRIBUTE)
    utils.log_string(log, 'trainX: %s\ttrainY: %s' %
                     (trainX.shape, trainY.shape))
    utils.log_string(log, 'valX:   %s\t\tvalY:   %s' %
                     (valX.shape, valY.shape))
    utils.log_string(log, 'testX:  %s\t\ttestY:  %s' %
                     (testX.shape, testY.shape))
    utils.log_string(log, 'data loaded!')

    # train model
    utils.log_string(log, 'compiling model...')
    T = 24 * 60 // config.AGENT.TIME_SLOT
    num_train, _, N = trainX.shape
    X, TE, label, is_training = model.placeholder(
        config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, N)
    global_step = tf.Variable(0, trainable=False)
    bn_momentum = tf.compat.v1.train.exponential_decay(
        0.5, global_step,
        decay_steps=config.AGENT.DECAY_EPOCH * num_train // config.AGENT.BATCH_SIZE,
        decay_rate=0.5, staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    pred = model.GMAN(
        X,
        TE,
        SE,
        T,
        bn=True,
        bn_decay=bn_decay,
        is_training=is_training
    )
    pred = pred * std + mean
    loss = model.mae_loss(pred, label)
    tf.compat.v1.add_to_collection('pred', pred)
    tf.compat.v1.add_to_collection('loss', loss)
    learning_rate = tf.compat.v1.train.exponential_decay(
        config.AGENT.LEARNING_RATE, global_step,
        decay_steps=config.AGENT.DECAY_EPOCH * num_train // config.AGENT.BATCH_SIZE,
        decay_rate=0.7, staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-5)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
    utils.log_string(log, 'model compiled!')
    saver = tf.compat.v1.train.Saver()
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf_config)
    sess.run(tf.compat.v1.global_variables_initializer())
    utils.log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    wait = 0
    val_loss_min = np.inf
    for epoch in range(config.AGENT.MAX_EPOCH):
        if wait >= config.AGENT.PATIENCE:
            utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
            break
        # shuffle
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        # train loss
        start_train = time.time()
        train_loss = 0
        num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)
            feed_dict = {
                X: trainX[start_idx:end_idx],
                TE: trainTE[start_idx:end_idx],
                label: trainY[start_idx:end_idx],
                is_training: True}
            _, loss_batch = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += loss_batch * (end_idx - start_idx)
        train_loss /= num_train
        end_train = time.time()
        # val loss
        start_val = time.time()
        val_loss = 0
        num_batch = math.ceil(num_val / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_val, (batch_idx + 1) * config.AGENT.BATCH_SIZE)
            feed_dict = {
                X: valX[start_idx:end_idx],
                TE: valTE[start_idx:end_idx],
                label: valY[start_idx:end_idx],
                is_training: False}
            loss_batch = sess.run(loss, feed_dict=feed_dict)
            val_loss += loss_batch * (end_idx - start_idx)
        val_loss /= num_val
        end_val = time.time()
        utils.log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             config.AGENT.MAX_EPOCH, end_train - start_train, end_val - start_val))
        utils.log_string(
            log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
        if val_loss <= val_loss_min:
            utils.log_string(
                log,
                'val loss decrease from %.4f to %.4f, saving model to %s' %
                (val_loss_min, val_loss, os.path.join(config.ROOT_DIR, model_file)))
            wait = 0
            val_loss_min = val_loss
            saver.save(sess, os.path.join(config.ROOT_DIR, model_file))
        else:
            wait += 1

    # test model
    utils.log_string(log, '**** testing model ****')
    utils.log_string(log, 'loading model from %s' %
                     os.path.join(config.ROOT_DIR, model_file))
    saver = tf.compat.v1.train.import_meta_graph(
        os.path.join(config.ROOT_DIR, model_file) + '.meta')
    saver.restore(sess, os.path.join(config.ROOT_DIR, model_file))
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    num_test = testX.shape[0]
    trainPred = []
    num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * config.AGENT.BATCH_SIZE
        end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)
        feed_dict = {
            X: trainX[start_idx: end_idx],
            TE: trainTE[start_idx: end_idx],
            is_training: False}
        pred_batch = sess.run(pred, feed_dict=feed_dict)
        trainPred.append(pred_batch)
    trainPred = np.concatenate(trainPred, axis=0)
    valPred = []
    num_batch = math.ceil(num_val / config.AGENT.BATCH_SIZE)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * config.AGENT.BATCH_SIZE
        end_idx = min(num_val, (batch_idx + 1) * config.AGENT.BATCH_SIZE)
        feed_dict = {
            X: valX[start_idx: end_idx],
            TE: valTE[start_idx: end_idx],
            is_training: False}
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
            X: testX[start_idx:end_idx],
            TE: testTE[start_idx:end_idx],
            is_training: False}
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
    sess.close()
    log.close()


if __name__ == "__main__":
    train()
