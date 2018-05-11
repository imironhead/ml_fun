"""
"""
import tensorflow as tf


def gaussian(tensors):
    """
    """
    return tf.exp(- tf.square(tensors))


def build_gaussian_model(world_height, world_width):
    """
    """
    print('building model: gaussian')

    # NOTE: world at T, should be padded circularly.
    source_tensors = tf.placeholder(
        shape=[None, world_height + 2, world_width + 2, 1],
        dtype=tf.float32,
        name='source_tensors')

    # NOTE: world at T+1.
    target_tensors = tf.placeholder(
        shape=[None, world_height, world_width, 1],
        dtype=tf.float32,
        name='target_tensors')

    # NOTE: kernel initializer
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # NOTE: valid padding to down-size to shape of  target_tensors
    tensors = tf.layers.conv2d(
        source_tensors,
        filters=1,
        kernel_size=3,
        padding='valid',
        activation=gaussian,
        use_bias=True,
        kernel_initializer=initializer)

    # NOTE: to probabilities
    predictions = tensors

    # NOTE: corss entropy loss
    loss = tf.losses.mean_squared_error(target_tensors, tensors)

    step = tf.get_variable(
        'training_step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    trainer = tf.train \
       .AdamOptimizer(learning_rate=0.01) \
       .minimize(loss, global_step=step)

    model = {
        'source_tensors': source_tensors,
        'target_tensors': target_tensors,
        'predictions': predictions,
        'loss': loss,
        'step': step,
        'trainer': trainer,
    }

    for var in tf.global_variables():
        if var.name.endswith('kernel:0') or var.name.endswith('bias:0'):
            model['var_' + var.name[:-2]] = var

    return model


def build_cnn_model(world_height, world_width):
    """
    """
    print('building model: cnn')

    # NOTE: world at T, should be padded circularly.
    source_tensors = tf.placeholder(
        shape=[None, world_height + 2, world_width + 2, 1],
        dtype=tf.float32,
        name='source_tensors')

    # NOTE: world at T+1.
    target_tensors = tf.placeholder(
        shape=[None, world_height, world_width, 1],
        dtype=tf.float32,
        name='target_tensors')

    # NOTE: kernel initializer
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # NOTE: valid padding to down-size to shape of  target_tensors
    tensors = tf.layers.conv2d(
        source_tensors,
        filters=4,
        kernel_size=3,
        padding='valid',
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=initializer,
        name='conv2d_0')

    # NOTE: conv net chain
    for i in range(2):
        tensors = tf.layers.conv2d(
            tensors,
            filters=4,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='conv2d_{}'.format(i + 1))

    # NOTE: map to target layer
    tensors = tf.layers.conv2d(
        tensors,
        filters=1,
        kernel_size=3,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        name='conv2d_3')

    # NOTE: to probabilities
    predictions = tf.nn.sigmoid(tensors, name='predictions')

    # NOTE: corss entropy loss
    loss = tf.losses.sigmoid_cross_entropy(target_tensors, tensors)

    step = tf.get_variable(
        'training_step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    trainer = tf.train \
       .AdamOptimizer(learning_rate=0.001) \
       .minimize(loss, global_step=step)

    model = {
        'source_tensors': source_tensors,
        'target_tensors': target_tensors,
        'predictions': predictions,
        'loss': loss,
        'step': step,
        'trainer': trainer,
    }

    for var in tf.global_variables():
        if var.name.endswith('kernel:0') or var.name.endswith('bias:0'):
            model['var_' + var.name[:-2]] = var

    return model


def build_model(world_height, world_width, name='cnn'):
    """
    """
    if name == 'gaussian':
        return build_gaussian_model(world_height, world_width)
    else:
        return build_cnn_model(world_height, world_width)


