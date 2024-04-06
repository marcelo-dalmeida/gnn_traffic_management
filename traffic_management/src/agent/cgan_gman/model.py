import numpy as np
import tensorflow as tf

import agent.cgan_gman.tf_utils as tf_utils
import config


LAMBDA = 100


def placeholder(P, Q, N):
    X = tf.compat.v1.placeholder(
        shape=(None, P, N), dtype=tf.float32, name='X')
    TE = tf.compat.v1.placeholder(
        shape=(None, P + Q, 2), dtype=tf.int32, name='TE')
    trafpatY = tf.compat.v1.placeholder(
        shape=(1), dtype=tf.int32, name='trafpatY')
    genTE = tf.compat.v1.placeholder(
        shape=(None, P + Q, 2), dtype=tf.int32, name='genTE')
    label = tf.compat.v1.placeholder(
        shape=(None, Q, N), dtype=tf.float32, name='label')
    gen_out = tf.compat.v1.placeholder(
        shape=(None, Q, N), dtype=tf.float32, name='gen_out')
    is_training = tf.compat.v1.placeholder(
        shape=(), dtype=tf.bool, name='is_training')
    return X, TE, trafpatY, genTE, label, gen_out, is_training


def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True, drop=None):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = tf_utils.dropout(x, drop=drop, is_training=is_training)
        x = tf_utils.conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x


def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    '''
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, P + Q, N, D]
    '''
    # spatial embedding
    SE = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
    SE = FC(
        SE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # temporal embedding
    dayofweek = tf.one_hot(TE[..., 0], depth=7)
    timeofday = tf.one_hot(TE[..., 1], depth=T)
    TE = tf.concat((dayofweek, timeofday), axis=-1)
    TE = tf.expand_dims(TE, axis=2)
    TE = FC(
        TE, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return tf.add(SE, TE)


def spatialAttention(X, STE, K, d, bn, bn_decay, is_training):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    X = tf.concat((X, STE), axis=-1)
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # [K * batch_size, num_step, N, N]
    attention = tf.matmul(query, key, transpose_b=True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=True):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    X = tf.concat((X, STE), axis=-1)
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, num_step, num_step]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    # mask attention score
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape=(num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)
        mask = tf.tile(mask, multiples=(K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype=tf.bool)
        attention = tf.compat.v2.where(
            condition=mask, x=attention, y=-2 ** 15 + 1)
    # softmax
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=False)
    XT = FC(
        HT, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return H


def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask=True):
    HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
    HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=mask)
    H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
    return tf.add(X, H)


def transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''
    D = K * d
    # query: [batch_size, Q, N, K * d]
    # key:   [batch_size, P, N, K * d]
    # value: [batch_size, P, N, K * d]
    query = FC(
        STE_Q, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        STE_P, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # query: [K * batch_size, Q, N, d]
    # key:   [K * batch_size, P, N, d]
    # value: [K * batch_size, P, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, Q, d]
    # key:   [K * batch_size, N, d, P]
    # value: [K * batch_size, N, P, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, Q, P]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, Q, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def trafpat_transformAttention(X, STE_P, STE_Q, trafpat, K, d, bn, bn_decay, is_training):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''

    dense_layer = tf.keras.layers.Dense(d)
    transformed_trafpat = dense_layer(trafpat)

    D = K * d

    # query: [batch_size, Q, N, K * d]
    # key:   [batch_size, P, N, K * d]
    # value: [batch_size, P, N, K * d]
    query = FC(
        STE_Q, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        STE_P, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # query: [K * batch_size, Q, N, d]
    # key:   [K * batch_size, P, N, d]
    # value: [K * batch_size, P, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, Q, d]
    # key:   [K * batch_size, N, d, P]
    # value: [K * batch_size, N, P, d]
    query += transformed_trafpat
    key += transformed_trafpat

    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))

    # [K * batch_size, N, Q, P]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, Q, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def GMAN(X, TE, SE, T, bn, bn_decay, is_training):
    '''
    GMAN
    X:       [batch_size, P, N]
    TE:      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE:      [N, K * d]
    P:       number of history steps
    Q:       number of prediction steps
    T:       one day is divided into T steps
    L:       number of STAtt blocks in the encoder/decoder
    K:       number of attention heads
    d:       dimension of each attention head outputs
    return:  [batch_size, Q, N]
    '''

    P = config.AGENT.HISTORY_STEPS
    Q = config.AGENT.PREDICTION_STEPS
    L = config.AGENT.NUMBER_OF_STATT_BLOCKS
    K = config.AGENT.NUMBER_OF_ATTENTION_HEADS
    d = config.AGENT.HEAD_ATTENTION_OUTPUT_DIM

    D = K * d
    # input
    X = tf.expand_dims(X, axis=-1)

    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # STE
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P:]
    # encoder
    for _ in range(L):
        X = STAttBlock(X, STE_P, K, d, bn, bn_decay, is_training)
    # transAtt
    X = transformAttention(
        X, STE_P, STE_Q, K, d, bn, bn_decay, is_training)
    # decoder
    for _ in range(L):
        X = STAttBlock(X, STE_Q, K, d, bn, bn_decay, is_training)
    # output
    X = FC(
        X, units=[D, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training,
        use_bias=True, drop=0.1)
    return tf.squeeze(X, axis=3)


def GMAN_gen(X, TE, SE, trafpatY, T, bn, bn_decay, is_training):
    '''
    GMAN
    X:       [batch_size, P, N]
    TE:      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE:      [N, K * d]
    trafpatY:labels/conditions [C] c = # of classes (3)
    P:       number of history steps
    Q:       number of prediction steps
    T:       one day is divided into T steps
    L:       number of STAtt blocks in the encoder/decoder
    K:       number of attention heads
    d:       dimension of each attention head outputs
    return:  [batch_size, Q, N]
    '''

    P = config.AGENT.HISTORY_STEPS
    Q = config.AGENT.PREDICTION_STEPS
    L = config.AGENT.NUMBER_OF_STATT_BLOCKS
    K = config.AGENT.NUMBER_OF_ATTENTION_HEADS
    d = config.AGENT.HEAD_ATTENTION_OUTPUT_DIM

    D = K * d
    # input
    X = tf.expand_dims(X, axis=-1)

    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # STE
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P:]

    # trafpat
    trafpatY = tf.one_hot(trafpatY, depth=3)

    # encoder
    for _ in range(L):
        X = STAttBlock(X, STE_P, K, d, bn, bn_decay, is_training)
    # transAtt
    X = trafpat_transformAttention(
        X, STE_P, STE_Q, trafpatY, K, d, bn, bn_decay, is_training)
    # decoder
    for _ in range(L):
        X = STAttBlock(X, STE_Q, K, d, bn, bn_decay, is_training)
    # output
    X = FC(
        X, units=[D, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training,
        use_bias=True, drop=0.1)

    X = tf.squeeze(X, axis=3)

    return X


def GMAN_disc(X, gen_out, TE, genTE, SE, T, bn, bn_decay, is_training):
    '''
    GMAN
    X:       [batch_size, P, N]
    gen_out: [batch_size, Q, N]
    TE:      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    genTE:   [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE:      [N, K * d]
    P:       number of history steps
    Q:       number of prediction steps
    T:       one day is divided into T steps
    L:       number of STAtt blocks in the encoder/decoder
    K:       number of attention heads
    d:       dimension of each attention head outputs
    return:  [batch_size, Q, N]
    '''

    P = config.AGENT.HISTORY_STEPS
    Q = config.AGENT.PREDICTION_STEPS
    L = config.AGENT.NUMBER_OF_STATT_BLOCKS
    K = config.AGENT.NUMBER_OF_ATTENTION_HEADS
    d = config.AGENT.HEAD_ATTENTION_OUTPUT_DIM

    D = K * d
    # input
    X = tf.expand_dims(X, axis=-1)

    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)

    gen_out = tf.expand_dims(gen_out, axis=-1)

    gen_out = FC(
        gen_out, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)

    X = tf.concat((X, gen_out), axis=2)  # concat pred

    # STE
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    genSTE = STEmbedding(SE, genTE, T, D, bn, bn_decay, is_training)
    STE = tf.concat((STE, genSTE), axis=2)  # concat STE
    STE_Q = STE[:, P:]

    # encoder
    for _ in range(L):
        X = STAttBlock(X, STE_Q, K, d, bn, bn_decay, is_training)
    # transAtt
    X = transformAttention(
        X, STE_Q, STE_Q, K, d, bn, bn_decay, is_training)

    # Reduces to the original input size
    shape = X.get_shape().as_list()
    X = tf.reshape(X, [-1, shape[2]])
    X = tf.keras.layers.Dense(shape[2] // 2)(X)
    X = tf.reshape(X, [-1, shape[1], shape[2] // 2, shape[3]])

    shape = STE_Q.get_shape().as_list()
    STE_Q = tf.reshape(STE_Q, [-1, shape[2]])
    STE_Q = tf.keras.layers.Dense(shape[2] // 2)(STE_Q)
    STE_Q = tf.reshape(STE_Q, [-1, shape[1], shape[2] // 2, shape[3]])

    # decoder
    for _ in range(L):
        X = STAttBlock(X, STE_Q, K, d, bn, bn_decay, is_training)
    # output
    X = FC(
        X, units=[D, 1], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training,
        use_bias=True, drop=0.1)
    X = tf.squeeze(X, axis=3)
    N = X.shape[2]

    X = tf.reshape(X, (-1, Q * N))

    trafpatY = tf.keras.layers.Dense(3, activation='softmax')(X)

    return trafpatY


def mae_loss(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition=tf.math.is_nan(mask), x=0., y=mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition=tf.math.is_nan(loss), x=0., y=loss)
    loss = tf.reduce_mean(loss)
    return loss


def gen_loss(disc_generated_output, gen_output, target):
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = mae_loss(gen_output, target)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss
    #return total_gen_loss, gan_loss, l1_loss


def disc_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
