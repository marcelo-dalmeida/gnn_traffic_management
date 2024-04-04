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

    dataset_file = os.path.join(config.PATH_TO_RECORDS, 'data', 'dataset.h5')

    log_file = os.path.join(config.PATH_TO_RECORDS,
                            f"{config.EXPERIMENT.SCENARIO_NAME}_gman_log")
    log = open(os.path.join(config.ROOT_DIR, log_file), 'w')

    model_file = os.path.join(config.PATH_TO_RECORDS,
                              f"{config.EXPERIMENT.SCENARIO_NAME}_gman_model")

    # load data
    utils.log_string(log, 'loading data...')
    # (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE,
    #  mean, std) = utils.loadData(dataset_file, config.AGENT.PREDICTED_ATTRIBUTE)
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

    X, TE, trafpatY, genTE, label, is_training = model.placeholder(
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
        gen_pred,
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

    gen_loss = model.generator_loss(disc_pred, gen_pred, label)
    disc_loss = model.discriminator_loss(X, disc_pred)

    tf.compat.v1.add_to_collection('gen_pred', gen_pred)
    tf.compat.v1.add_to_collection('gen_loss', gen_loss)
    tf.compat.v1.add_to_collection('disc_pred', disc_pred)
    tf.compat.v1.add_to_collection('disc_loss', disc_loss)

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
    gen_sess = tf.compat.v1.Session(config=tf_config)
    gen_sess.run(tf.compat.v1.global_variables_initializer())

    disc_sess = tf.compat.v1.Session(config=tf_config)
    disc_sess.run(tf.compat.v1.global_variables_initializer())

    utils.log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    wait = 0
    gen_val_loss_min = np.inf
    disc_val_loss_min = np.inf

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
        gen_train_loss = 0
        disc_train_loss = 0
        num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
        for batch_idx in range(num_batch):
            start_idx = batch_idx * config.AGENT.BATCH_SIZE
            end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)
            gen_feed_dict = {
                X: trainX[start_idx:end_idx],
                TE: trainTE[start_idx:end_idx],
                trafpatY: traintrafpatY[start_idx:end_idx],
                label: trainY[start_idx:end_idx],
                is_training: True}

            disc_real_feed_dict = {
                X: trainX[start_idx:end_idx],
                label: trainY[start_idx:end_idx],
                TE: trainTE[start_idx:end_idx],
                trafpatY: traintrafpatY[start_idx:end_idx],
                is_training: True}

            gen_output, gen_loss_batch = gen_sess.run(
                [gen_train_op, gen_loss], feed_dict=gen_feed_dict)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = gen_pred(**gen_feed_dict)

            disc_generated_feed_dict = {
                X: trainX[start_idx:end_idx],
                label: gen_output,
                TE: trainTE[start_idx:end_idx],
                trafpatY: traintrafpatY[start_idx:end_idx],
                is_training: True}

            disc_real_output = disc_pred(**disc_real_feed_dict)
            disc_generated_output = disc_pred(**disc_generated_feed_dict)

            gen_total_loss, gen_gan_loss, gen_l1_loss = gen_loss(
                disc_generated_output, gen_output, trainY[start_idx:end_idx])
            disc_total_loss = disc_loss(
                disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(
                gen_total_loss, gen_pred.trainable_variables)
            discriminator_gradients = disc_tape.gradient(
                disc_total_loss, disc_pred.trainable_variables)

            gen_optimizer.apply_gradients(
                zip(generator_gradients, gen_pred.trainable_variables))
            disc_optimizer.apply_gradients(
                zip(discriminator_gradients, disc_pred.trainable_variables))

            _, disc_loss_batch = disc_sess.run(
                [disc_train_op, disc_loss], feed_dict=disc_real_feed_dict)

            _, disc_loss_batch = disc_sess.run(
                [disc_train_op, disc_loss], feed_dict=disc_generated_feed_dict)

            # gen_loss_batch = gen_total_loss
            # disc_loss_batch = disc_total_loss

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

            gen_output = gen_pred(**gen_feed_dict)

            disc_generated_feed_dict = {
                X: valX[start_idx:end_idx],
                label: gen_output,
                TE: valTE[start_idx:end_idx],
                trafpatY: valtrafpatY[start_idx:end_idx],
                is_training: False}

            disc_generated_output = disc_pred(**disc_generated_feed_dict)

            gen_total_loss, _, _ = gen_loss(
                disc_generated_output, gen_output, valY[start_idx:end_idx])
            disc_total_loss = disc_loss(
                disc_real_output, disc_generated_output)

            # loss_batch = sess.run(loss, feed_dict=feed_dict)

            gen_val_loss += gen_total_loss * (end_idx - start_idx)
            disc_val_loss += disc_total_loss * (end_idx - start_idx)
        gen_val_loss /= num_val
        disc_val_loss /= num_val
        end_val = time.time()

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
            saver.save(gen_sess, os.path.join(config.ROOT_DIR, model_file))
        if disc_val_loss <= disc_val_loss_min:
            utils.log_string(
                log,
                'disc val loss decrease from %.4f to %.4f, saving model to %s' %
                (disc_val_loss_min, disc_val_loss, os.path.join(config.ROOT_DIR, model_file)))
            wait = 0
            disc_val_loss_min = disc_val_loss
            saver.save(disc_sess, os.path.join(config.ROOT_DIR, model_file))
        else:
            wait += 1

    # test model
    utils.log_string(log, '**** testing model ****')
    utils.log_string(log, 'loading model from %s' %
                     os.path.join(config.ROOT_DIR, model_file))
    saver = tf.compat.v1.train.import_meta_graph(
        os.path.join(config.ROOT_DIR, model_file) + '.meta')
    saver.restore(gen_sess, os.path.join(config.ROOT_DIR, model_file))
    saver.restore(disc_sess, os.path.join(config.ROOT_DIR, model_file))
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    num_test = testX.shape[0]
    trainPred = []
    num_batch = math.ceil(num_train / config.AGENT.BATCH_SIZE)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * config.AGENT.BATCH_SIZE
        end_idx = min(num_train, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

        gen_feed_dict = {
            X: trainX[start_idx:end_idx],
            TE: trainTE[start_idx:end_idx],
            trafpatY: traintrafpatY[start_idx:end_idx],
            is_training: False}

        gen_output = gen_pred(**gen_feed_dict)

        disc_generated_feed_dict = {
            X: trainX[start_idx:end_idx],
            label: gen_output,
            TE: trainTE[start_idx:end_idx],
            is_training: False}

        disc_generated_output = disc_pred(**disc_generated_feed_dict)

        # pred_batch = sess.run(pred, feed_dict=feed_dict)

        trainPred.append(disc_generated_output)
    trainPred = np.concatenate(trainPred, axis=0)
    valPred = []
    num_batch = math.ceil(num_val / config.AGENT.BATCH_SIZE)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * config.AGENT.BATCH_SIZE
        end_idx = min(num_val, (batch_idx + 1) * config.AGENT.BATCH_SIZE)

        gen_feed_dict = {
            X: valX[start_idx:end_idx],
            TE: valTE[start_idx:end_idx],
            trafpatY: valtrafpatY[start_idx:end_idx],
            is_training: False}

        gen_output = gen_pred(**gen_feed_dict)

        disc_generated_feed_dict = {
            X: valX[start_idx:end_idx],
            label: gen_output,
            TE: valTE[start_idx:end_idx],
            is_training: False}

        disc_generated_output = disc_pred(**disc_generated_feed_dict)

        # pred_batch = sess.run(pred, feed_dict=feed_dict)

        valPred.append(disc_generated_output)
    valPred = np.concatenate(valPred, axis=0)
    testPred = []
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

        gen_output = gen_pred(**gen_feed_dict)

        disc_generated_feed_dict = {
            X: testX[start_idx:end_idx],
            label: gen_output,
            TE: testTE[start_idx:end_idx],
            is_training: False}

        disc_generated_output = disc_pred(**disc_generated_feed_dict)

        # pred_batch = sess.run(pred, feed_dict=feed_dict)

        testPred.append(disc_generated_output)
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
    gen_sess.close()
    disc_sess.close()

    log.close()


if __name__ == "__main__":
    train()
