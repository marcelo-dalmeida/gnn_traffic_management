import math
import os
import time
import datetime

import numpy as np
import tensorflow as tf

import agent.gman_classifier.utils as utils
import agent.gman_classifier.model as model


def train():
    start = time.time()

    import config

    dataset_file = os.path.join(config.PATH_TO_DATA, 'dataset.h5')

    log_file = os.path.join(config.PATH_TO_MODEL, f"{config.EXPERIMENT.SCENARIO_NAME}_gman_log")
    log = open(os.path.join(config.ROOT_DIR, log_file), 'w')

    model_file = os.path.join(config.PATH_TO_MODEL, f"{config.EXPERIMENT.SCENARIO_NAME}_gman_model")

    # load data
    utils.log_string(log, 'loading data...')
    (trainX, trainTE, trainY, traintrafpatY, valX, valTE, valY, valtrafpatY, testX, testTE, testY, testtrafpatY, SE,
     mean, std) = utils.loadData(dataset_file, config.AGENT.PREDICTED_ATTRIBUTE)
    utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
    utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
    utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
    utils.log_string(log, 'data loaded!')

    # train model
    utils.log_string(log, 'compiling model...')

    T = 24 * 60 // config.AGENT.TIME_SLOT

    num_train, _, N = trainX.shape

    X, TE, trafpatY, is_training = model.placeholder(
        config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, N)

    global_step = tf.Variable(0, trainable=False)

    bn_momentum = tf.compat.v1.train.exponential_decay(
        0.5, global_step,
        decay_steps=config.AGENT.DECAY_EPOCH * num_train // config.AGENT.BATCH_SIZE,
        decay_rate=0.5, staircase=True)

    bn_decay = tf.minimum(0.99, 1 - bn_momentum)

    clas_pred = model.GMAN_class(
        X,
        TE,
        SE,
        T,
        bn=True,
        bn_decay=bn_decay,
        is_training=is_training
    )

    clas_pred = clas_pred * std + mean

    clas_loss = model.clas_loss(clas_pred, trafpatY)

    tf.compat.v1.add_to_collection('clas_pred', clas_pred)
    tf.compat.v1.add_to_collection('clas_loss', clas_loss)

    learning_rate = tf.compat.v1.train.exponential_decay(
        config.AGENT.LEARNING_RATE, global_step,
        decay_steps=config.AGENT.DECAY_EPOCH * num_train // config.AGENT.BATCH_SIZE,
        decay_rate=0.7, staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-5)

    clas_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    clas_train_op = clas_optimizer.minimize(clas_loss, global_step=global_step)

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

    special_epoch_status = None
    epoch_start = 0
    if os.path.isfile(os.path.join(config.ROOT_DIR, model_file) + '.meta'):
        try:
            with open(os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'epochs.txt'), "r") as file:
                rounds = file.readlines()
                try:
                    epoch_start = int(rounds[-1]) + 1
                except ValueError:
                    special_epoch_status = rounds[-1]

        except FileNotFoundError:
            pass

        loader = tf.compat.v1.train.import_meta_graph(
            os.path.join(config.ROOT_DIR, model_file) + '.meta')
        loader.restore(sess, os.path.join(config.ROOT_DIR, model_file))
        utils.log_string(log, 'model restored!')

    utils.log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    wait = 0
    clas_val_loss_min = np.inf

    for epoch in range(epoch_start, config.AGENT.MAX_EPOCH):

        if special_epoch_status == 'early_stop':
            utils.log_string(log, 'Stopping training of an early stopped model')
            break

        if wait >= config.AGENT.PATIENCE:
            utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
            with open(os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'epochs.txt'), "a+") as file:
                file.write(f"early_stop\n")
            break
        # shuffle
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]

        # train loss
        start_train = time.time()
        clas_train_loss = 0
        num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

            clas_feed_dict = {
                X: trainX[start_idx:end_idx],
                TE: trainTE[start_idx:end_idx],
                trafpatY: traintrafpatY[start_idx:start_idx+1],
                is_training: True
            }
            _, clas_total_loss = sess.run([clas_train_op, clas_loss], feed_dict=clas_feed_dict)

            clas_loss_batch = clas_total_loss

            clas_train_loss += clas_loss_batch * (end_idx - start_idx)
        clas_train_loss /= num_train
        end_train = time.time()

        # val loss
        start_val = time.time()
        clas_val_loss = 0
        num_batch = math.ceil(num_val / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_val, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

            clas_feed_dict = {
                X: valX[start_idx:end_idx],
                TE: valTE[start_idx:end_idx],
                trafpatY: valtrafpatY[start_idx:end_idx],
                is_training: False
            }

            clas_total_loss = sess.run(clas_loss, feed_dict=clas_feed_dict)

            clas_val_loss += clas_total_loss * (end_idx - start_idx)
        clas_val_loss /= num_val
        end_val = time.time()

        with open(os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'epochs.txt'), "a+") as file:
            file.write(f"{epoch}\n")

        utils.log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             config.AGENT.MAX_EPOCH, end_train - start_train, end_val - start_val))
        utils.log_string(
            log, 'clas train loss: %.4f, clas val_loss: %.4f' % (clas_train_loss, clas_val_loss))
        if clas_val_loss <= clas_val_loss_min:
            utils.log_string(
                log,
                'clas val loss decrease from %.4f to %.4f, saving model to %s' %
                (clas_val_loss_min, clas_val_loss, os.path.join(config.ROOT_DIR, model_file)))
            wait = 0
            clas_val_loss_min = clas_val_loss
            saver.save(sess, os.path.join(config.ROOT_DIR, model_file))
        else:
            wait += 1

    # test model
    utils.log_string(log, '**** testing model ****')
    utils.log_string(log, 'loading model from %s' % os.path.join(config.ROOT_DIR, model_file))
    loader = tf.compat.v1.train.import_meta_graph(os.path.join(config.ROOT_DIR, model_file) + '.meta')
    loader.restore(sess, os.path.join(config.ROOT_DIR, model_file))
    utils.log_string(log, 'model restored!')

    utils.log_string(log, 'evaluating...')
    num_test = testX.shape[0]
    clas_trainPred = []
    num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * config.AGENT.BATCH_SIZE
        end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

        clas_feed_dict = {
            X: trainX[start_idx:end_idx],
            TE: trainTE[start_idx:end_idx],
            is_training: False
        }

        clas_output = sess.run(clas_pred, feed_dict=clas_feed_dict)

        clas_trainPred.append(clas_output)
    clas_trainPred = np.concatenate(clas_trainPred, axis=0)

    clas_valPred = []
    num_batch = math.ceil(num_val / config.AGENT.BATCH_SIZE)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * config.AGENT.BATCH_SIZE
        end_idx = min(num_val, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

        clas_feed_dict = {
            X: valX[start_idx:end_idx],
            TE: valTE[start_idx:end_idx],
            is_training: False
        }

        clas_output = sess.run(clas_pred, feed_dict=clas_feed_dict)

        clas_valPred.append(clas_output)
    clas_valPred = np.concatenate(clas_valPred, axis=0)

    clas_testPred = []
    num_batch = math.ceil(num_test / config.AGENT.BATCH_SIZE)
    start_test = time.time()
    for batch_idx in range(num_batch):
        start_idx = batch_idx * config.AGENT.BATCH_SIZE
        end_idx = min(num_test, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

        clas_feed_dict = {
            X: testX[start_idx:end_idx],
            TE: testTE[start_idx:end_idx],
            is_training: False
        }

        clas_output = sess.run(clas_pred, feed_dict=clas_feed_dict)

        clas_testPred.append(clas_output)
    end_test = time.time()
    clas_testPred = np.concatenate(clas_testPred, axis=0)

    # summary
    clas_train_mae, clas_train_rmse, clas_train_mape = utils.metric(clas_trainPred, traintrafpatY)
    clas_val_mae, clas_val_rmse, clas_val_mape = utils.metric(clas_valPred, valtrafpatY)
    clas_test_mae, clas_test_rmse, clas_test_mape = utils.metric(clas_testPred, testtrafpatY)
    utils.log_string(log, 'clas testing time: %.1fs' % (end_test - start_test))
    utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    utils.log_string(log, 'clas train            %.2f\t\t%.2f\t\t%.2f%%' %
                     (clas_train_mae, clas_train_rmse, clas_train_mape * 100))
    utils.log_string(log, 'clas val              %.2f\t\t%.2f\t\t%.2f%%' %
                     (clas_val_mae, clas_val_rmse, clas_val_mape * 100))
    utils.log_string(log, 'clas test             %.2f\t\t%.2f\t\t%.2f%%' %
                     (clas_test_mae, clas_test_rmse, clas_test_mape * 100))
    utils.log_string(log, 'clas performance in each prediction step')
    clas_MAE, clas_RMSE, clas_MAPE = [], [], []
    for q in range(config.AGENT.PREDICTION_STEPS):
        clas_mae, clas_rmse, clas_mape = utils.metric(clas_testPred[:, q], testtrafpatY[:, q])
        clas_MAE.append(clas_mae)
        clas_RMSE.append(clas_rmse)
        clas_MAPE.append(clas_mape)
        utils.log_string(log, 'clas step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                         (q + 1, clas_mae, clas_rmse, clas_mape * 100))
    clas_average_mae = np.mean(clas_MAE)
    clas_average_rmse = np.mean(clas_RMSE)
    clas_average_mape = np.mean(clas_MAPE)
    utils.log_string(
        log, 'clas average:         %.2f\t\t%.2f\t\t%.2f%%' %
             (clas_average_mae, clas_average_rmse, clas_average_mape * 100))

    # TODO FIX detection times
    detection_times = [5, 7, 4, 8, 6, 3]

    dr, fpr, f_score, mttd = utils.calculate_detection_metrics(clas_testPred, testtrafpatY, detection_times)

    utils.log_string(log, f"Detection Rate (DR): {dr}")
    utils.log_string(log, f"False Positive Rate (FPR): {fpr}")
    utils.log_string(log, f"F-measurement: {f_score}")
    # utils.log_string(log, f"Mean Time to Detection (MTTD): {mttd} minutes")

    end = time.time()
    utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    sess.close()
    log.close()


if __name__ == "__main__":
    train()
