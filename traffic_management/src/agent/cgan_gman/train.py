import math
import os
import time
import datetime

import numpy as np
import tensorflow as tf

import agent.cgan_gman.utils as utils
import agent.cgan_gman.model as model


def train():
    start = time.time()

    import config

    dataset_file = os.path.join(config.PATH_TO_DATA, 'data', 'dataset.h5')

    log_file = os.path.join(config.PATH_TO_MODEL, f"{config.EXPERIMENT.SCENARIO_NAME}_gman_log")
    log = open(os.path.join(config.ROOT_DIR, log_file), 'w')

    model_file = os.path.join(config.PATH_TO_MODEL, f"{config.EXPERIMENT.SCENARIO_NAME}_gman_model")

    # load data
    utils.log_string(log, 'loading data...')
    (trainX, trainTE, trainY, traintrafpatY, valX, valTE, valY, valtrafpatY, testX, testTE, testY, testtrafpatY, SE,
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

    X, TE, trafpatY, genTE, label, gen_out, is_training = model.placeholder(
        config.AGENT.HISTORY_STEPS, config.AGENT.PREDICTION_STEPS, N)

    global_step = tf.Variable(0, trainable=False)

    bn_momentum = tf.compat.v1.train.exponential_decay(
        0.5, global_step,
        decay_steps=config.AGENT.DECAY_EPOCH * num_train // config.AGENT.BATCH_SIZE,
        decay_rate=0.5, staircase=True)

    bn_decay = tf.minimum(0.99, 1 - bn_momentum)

    gen_pred = model.GMAN_gen(
        X,
        TE,
        SE,
        trafpatY,
        T,
        bn=True,
        bn_decay=bn_decay,
        is_training=is_training
    )
    disc_pred = model.GMAN_disc(
        X,
        gen_out,
        TE,
        genTE,
        SE,
        T,
        bn=True,
        bn_decay=bn_decay,
        is_training=is_training
    )

    gen_pred = gen_pred * std + mean
    disc_pred = disc_pred * std + mean

    gen_loss = model.gen_loss(disc_pred, gen_pred, label)
    disc_loss = model.disc_loss(disc_pred, disc_pred)

    learning_rate = tf.compat.v1.train.exponential_decay(
        config.AGENT.LEARNING_RATE, global_step,
        decay_steps=config.AGENT.DECAY_EPOCH * num_train // config.AGENT.BATCH_SIZE,
        decay_rate=0.7, staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-5)

    gen_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    disc_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    gen_train_op = gen_optimizer.minimize(gen_loss, global_step=global_step)
    disc_train_op = disc_optimizer.minimize(disc_loss, global_step=global_step)

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

    epoch_start = 0
    if os.path.isfile(os.path.join(config.ROOT_DIR, model_file) + '.meta'):
        try:
            with open(os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'epochs.txt'), "r") as file:
                rounds = file.readlines()
                epoch_start = int(rounds[-1])
        except FileNotFoundError:
            pass

        loader = tf.compat.v1.train.import_meta_graph(
            os.path.join(config.ROOT_DIR, model_file) + '.meta')
        loader.restore(sess, os.path.join(config.ROOT_DIR, model_file))

    utils.log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    wait = 0
    gen_val_loss_min = np.inf
    disc_val_loss_min = np.inf

    for epoch in range(epoch_start, config.AGENT.MAX_EPOCH):
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
        gen_train_loss = 0
        disc_train_loss = 0
        num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

            gen_feed_dict = {
                X: trainX[start_idx:end_idx],
                TE: trainTE[start_idx:end_idx],
                trafpatY: traintrafpatY[start_idx:start_idx+1],
                label: trainY[start_idx:end_idx],
                is_training: True
            }

            disc_real_feed_dict = {
                X: trainX[start_idx:end_idx],
                label: trainY[start_idx:end_idx],
                gen_out: trainY[start_idx:end_idx],
                TE: trainTE[start_idx:end_idx],
                genTE: trainTE[start_idx:end_idx],
                trafpatY: traintrafpatY[start_idx:start_idx+1],
                is_training: True
            }

            gen_output = sess.run(gen_pred, feed_dict=gen_feed_dict)

            disc_generated_feed_dict = {
                X: trainX[start_idx:end_idx],
                gen_out: gen_output,
                TE: trainTE[start_idx:end_idx],
                genTE: trainTE[start_idx:end_idx],
                trafpatY: traintrafpatY[start_idx:start_idx+1],
                is_training: True
            }

            _, gen_total_loss = sess.run([gen_train_op, gen_loss], feed_dict={**gen_feed_dict, **disc_generated_feed_dict})

            _, disc_real_loss = sess.run([disc_train_op, disc_loss], feed_dict=disc_real_feed_dict)
            _, disc_generated_loss = sess.run([disc_train_op, disc_loss], feed_dict=disc_generated_feed_dict)
            disc_total_loss = disc_real_loss + disc_generated_loss

            gen_loss_batch = gen_total_loss
            disc_loss_batch = disc_total_loss

            gen_train_loss += gen_loss_batch * (end_idx - start_idx)
            disc_train_loss += disc_loss_batch * (end_idx - start_idx)
        gen_train_loss /= num_train
        disc_train_loss /= num_train
        end_train = time.time()

        # val loss
        start_val = time.time()
        gen_val_loss = 0
        disc_val_loss = 0
        num_batch = math.ceil(num_val / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_val, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

            gen_feed_dict = {
                X: valX[start_idx:end_idx],
                TE: valTE[start_idx:end_idx],
                trafpatY: valtrafpatY[start_idx:end_idx],
                label: valY[start_idx:end_idx],
                is_training: False}

            gen_output = sess.run(gen_pred, feed_dict=gen_feed_dict)

            disc_generated_feed_dict = {
                X: valX[start_idx:end_idx],
                gen_out: gen_output,
                TE: valTE[start_idx:end_idx],
                genTE: valTE[start_idx:end_idx],
                trafpatY: valtrafpatY[start_idx:end_idx],
                is_training: False}

            gen_total_loss = sess.run(gen_loss, feed_dict={**gen_feed_dict, **disc_generated_feed_dict})
            disc_total_loss = sess.run(disc_loss, feed_dict=disc_generated_feed_dict)

            gen_val_loss += gen_total_loss * (end_idx - start_idx)
            disc_val_loss += disc_total_loss * (end_idx - start_idx)
        gen_val_loss /= num_val
        disc_val_loss /= num_val
        end_val = time.time()

        with open(os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'epochs.txt'), "a+") as file:
            file.write(f"{epoch}\n")

        utils.log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             config.AGENT.MAX_EPOCH, end_train - start_train, end_val - start_val))
        utils.log_string(
            log, 'gen train loss: %.4f, gen val_loss: %.4f' % (gen_train_loss, gen_val_loss))
        utils.log_string(
            log, 'disc train loss: %.4f, disc val_loss: %.4f' % (disc_train_loss, disc_val_loss))
        if gen_val_loss <= gen_val_loss_min:
            utils.log_string(
                log,
                'gen val loss decrease from %.4f to %.4f, saving model to %s' %
                (gen_val_loss_min, gen_val_loss, os.path.join(config.ROOT_DIR, model_file)))
            wait = 0
            gen_val_loss_min = gen_val_loss
            saver.save(sess, os.path.join(config.ROOT_DIR, model_file))
        if disc_val_loss <= disc_val_loss_min:
            utils.log_string(
                log,
                'disc val loss decrease from %.4f to %.4f, saving model to %s' %
                (disc_val_loss_min, disc_val_loss, os.path.join(config.ROOT_DIR, model_file)))
            wait = 0
            disc_val_loss_min = disc_val_loss
            saver.save(sess, os.path.join(config.ROOT_DIR, model_file))
        else:
            wait += 1

    # test model
    utils.log_string(log, '**** testing model ****')
    utils.log_string(log, 'loading model from %s' %
                     os.path.join(config.ROOT_DIR, model_file))
    loader = tf.compat.v1.train.import_meta_graph(
        os.path.join(config.ROOT_DIR, model_file) + '.meta')
    loader.restore(sess, os.path.join(config.ROOT_DIR, model_file))
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    num_test = testX.shape[0]
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
            is_training: False}

        gen_output = sess.run(gen_pred, feed_dict=gen_feed_dict)

        disc_generated_feed_dict = {
            X: trainX[start_idx:end_idx],
            gen_out: gen_output,
            TE: trainTE[start_idx:end_idx],
            genTE: trainTE[start_idx:end_idx],
            is_training: False}

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
            is_training: False}

        gen_output = sess.run(gen_pred, feed_dict=gen_feed_dict)

        disc_generated_feed_dict = {
            X: valX[start_idx:end_idx],
            gen_out: gen_output,
            TE: valTE[start_idx:end_idx],
            genTE: valTE[start_idx:end_idx],
            is_training: False}

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
        end_idx = min(num_test, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

        gen_feed_dict = {
            X: testX[start_idx:end_idx],
            TE: testTE[start_idx:end_idx],
            trafpatY: testtrafpatY[start_idx:end_idx],
            is_training: False}

        gen_output = sess.run(gen_pred, feed_dict=gen_feed_dict)

        disc_generated_feed_dict = {
            X: testX[start_idx:end_idx],
            gen_out: gen_output,
            TE: testTE[start_idx:end_idx],
            genTE: testTE[start_idx:end_idx],
            is_training: False}

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
    sess.close()
    log.close()


if __name__ == "__main__":
    train()
