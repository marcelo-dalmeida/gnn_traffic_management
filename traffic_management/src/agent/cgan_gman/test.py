import os
import math
import time

import numpy as np
import tensorflow as tf

import utils
from agent.cgan_gman import model


def test():

    import config

    dataset_file = os.path.join(config.PATH_TO_DATA, 'dataset.h5')

    start = time.time()

    log_file = os.path.join(config.PATH_TO_MODEL, config.EXPERIMENT.SCENARIO_NAME)
    log = open(os.path.join(config.ROOT_DIR, log_file), 'w')

    model_file = os.path.join(config.PATH_TO_MODEL, f"{config.EXPERIMENT.SCENARIO_NAME}_gman_model")

    # load data
    utils.log_string(log, 'loading data...')
    (trainX, trainTE, trainY, traintrafpatY, valX, valTE, valY, valtrafpatY, testX, testTE, testY, testtrafpatY, SE,
     mean, std) = utils.loadData(dataset_file, config.AGENT.PREDICTED_ATTRIBUTE)
    num_train, num_val, num_test = trainX.shape[0], valX.shape[0], testX.shape[0]
    utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
    utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
    utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
    utils.log_string(log, 'data loaded!')

    N = trainX.shape[2]
    X, TE, trafpatY, genTE, label, gen_out, is_training = model.placeholder(
        config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, N)

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
        gen_pred = graph.get_collection(name='gen_pred')[0]
        disc_pred = graph.get_collection(name='disc_pred')[0]
        utils.log_string(log, 'model restored!')

        utils.log_string(log, 'evaluating...')
        gen_trainPred = []
        disc_trainPred = []
        num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

            gen_feed_dict = {
                X: trainX[start_idx:end_idx],
                TE: trainTE[start_idx:end_idx],
                trafpatY: traintrafpatY[start_idx:end_idx],
                is_training: False
            }

            gen_output = sess.run(gen_pred, feed_dict=gen_feed_dict)

            disc_generated_feed_dict = {
                X: trainX[start_idx:end_idx],
                gen_out: gen_output,
                TE: trainTE[start_idx:end_idx],
                genTE: trainTE[start_idx:end_idx],
                is_training: False
            }

            disc_generated_output = sess.run(disc_pred, feed_dict=disc_generated_feed_dict)

            gen_trainPred.append(gen_output)
            disc_trainPred.append(disc_generated_output)

        gen_trainPred = np.concatenate(gen_trainPred, axis=0)
        disc_trainPred = np.concatenate(disc_trainPred, axis=0)

        gen_valPred = []
        disc_valPred = []
        num_batch = math.ceil(num_val / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_val, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

            gen_feed_dict = {
                X: valX[start_idx:end_idx],
                TE: valTE[start_idx:end_idx],
                trafpatY: valtrafpatY[start_idx:end_idx],
                is_training: False
            }

            gen_output = sess.run(gen_pred, feed_dict=gen_feed_dict)

            disc_generated_feed_dict = {
                X: valX[start_idx:end_idx],
                gen_out: gen_output,
                TE: valTE[start_idx:end_idx],
                genTE: valTE[start_idx:end_idx],
                is_training: False
            }

            disc_generated_output = sess.run(disc_pred, feed_dict=disc_generated_feed_dict)

            gen_valPred.append(gen_output)
            disc_valPred.append(disc_generated_output)
        gen_valPred = np.concatenate(gen_valPred, axis=0)
        disc_valPred = np.concatenate(disc_valPred, axis=0)

        gen_testPred = []
        disc_testPred = []
        num_batch = math.ceil(num_test / config.AGENT.BATCH_SIZE)
        start_test = time.time()
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE

            gen_feed_dict = {
                X: testX[start_idx:end_idx],
                TE: testTE[start_idx:end_idx],
                trafpatY: testtrafpatY[start_idx:end_idx],
                is_training: False
            }

            gen_output = sess.run(gen_pred, feed_dict=gen_feed_dict)

            disc_generated_feed_dict = {
                X: testX[start_idx:end_idx],
                gen_out: gen_output,
                TE: testTE[start_idx:end_idx],
                genTE: testTE[start_idx:end_idx],
                is_training: False
            }

            disc_generated_output = sess.run(disc_pred, feed_dict=disc_generated_feed_dict)

            gen_testPred.append(gen_output)
            disc_testPred.append(disc_generated_output)
        end_test = time.time()
        gen_testPred = np.concatenate(gen_testPred, axis=0)
        disc_testPred = np.concatenate(disc_testPred, axis=0)

    # summary
    gen_train_mae, gen_train_rmse, gen_train_mape = utils.metric(gen_trainPred, trainY)
    gen_val_mae, gen_val_rmse, gen_val_mape = utils.metric(gen_valPred, valY)
    gen_test_mae, gen_test_rmse, gen_test_mape = utils.metric(gen_testPred, testY)
    utils.log_string(log, 'gen testing time: %.1fs' % (end_test - start_test))
    utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    utils.log_string(log, 'gen train            %.2f\t\t%.2f\t\t%.2f%%' %
                     (gen_train_mae, gen_train_rmse, gen_train_mape * 100))
    utils.log_string(log, 'gen val              %.2f\t\t%.2f\t\t%.2f%%' %
                     (gen_val_mae, gen_val_rmse, gen_val_mape * 100))
    utils.log_string(log, 'gen test             %.2f\t\t%.2f\t\t%.2f%%' %
                     (gen_test_mae, gen_test_rmse, gen_test_mape * 100))
    utils.log_string(log, 'performance in each prediction step')
    gen_MAE, gen_RMSE, gen_MAPE = [], [], []
    for q in range(config.AGENT.PREDICTION_STEPS):
        gen_mae, gen_rmse, gen_mape = utils.metric(gen_testPred[:, q], testY[:, q])
        gen_MAE.append(gen_mae)
        gen_RMSE.append(gen_rmse)
        gen_MAPE.append(gen_mape)
        utils.log_string(log, 'gen step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                         (q + 1, gen_mae, gen_rmse, gen_mape * 100))
    gen_average_mae = np.mean(gen_MAE)
    gen_average_rmse = np.mean(gen_RMSE)
    gen_average_mape = np.mean(gen_MAPE)
    utils.log_string(
        log, 'gen average:         %.2f\t\t%.2f\t\t%.2f%%' %
             (gen_average_mae, gen_average_rmse, gen_average_mape * 100))

    disc_train_mae, disc_train_rmse, disc_train_mape = utils.metric(disc_trainPred, traintrafpatY)
    disc_val_mae, disc_val_rmse, disc_val_mape = utils.metric(disc_valPred, valtrafpatY)
    disc_test_mae, disc_test_rmse, disc_test_mape = utils.metric(disc_testPred, testtrafpatY)
    utils.log_string(log, 'disc testing time: %.1fs' % (end_test - start_test))
    utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    utils.log_string(log, 'disc train            %.2f\t\t%.2f\t\t%.2f%%' %
                     (disc_train_mae, disc_train_rmse, disc_train_mape * 100))
    utils.log_string(log, 'disc val              %.2f\t\t%.2f\t\t%.2f%%' %
                     (disc_val_mae, disc_val_rmse, disc_val_mape * 100))
    utils.log_string(log, 'disc test             %.2f\t\t%.2f\t\t%.2f%%' %
                     (disc_test_mae, disc_test_rmse, disc_test_mape * 100))
    utils.log_string(log, 'disc performance in each prediction step')
    disc_MAE, disc_RMSE, disc_MAPE = [], [], []
    for q in range(config.AGENT.PREDICTION_STEPS):
        disc_mae, disc_rmse, disc_mape = utils.metric(disc_testPred[:, q], testtrafpatY[:, q])
        disc_MAE.append(disc_mae)
        disc_RMSE.append(disc_rmse)
        disc_MAPE.append(disc_mape)
        utils.log_string(log, 'disc step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                         (q + 1, disc_mae, disc_rmse, disc_mape * 100))
    disc_average_mae = np.mean(disc_MAE)
    disc_average_rmse = np.mean(disc_RMSE)
    disc_average_mape = np.mean(disc_MAPE)
    utils.log_string(
        log, 'disc average:         %.2f\t\t%.2f\t\t%.2f%%' %
             (disc_average_mae, disc_average_rmse, disc_average_mape * 100))

    end = time.time()
    utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()


if __name__ == "__main__":
    test()
