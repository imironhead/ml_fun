"""
"""
import tensorflow as tf


def build_model(world_size):
    """
    """
    # NOTE: world at T, should be padded circularly.
    source_tensors = tf.placeholder(
        shape=[None, world_size + 2, world_size + 2, 1],
        dtype=tf.float32,
        name='source_tensors')

    # NOTE: world at T+1.
    target_tensors = tf.placeholder(
        shape=[None, world_size, world_size, 1],
        dtype=tf.float32,
        name='target_tensors')

    # NOTE: kernel initializer
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # NOTE: valid padding to down-size toshape of  target_tensors
    tensors = tf.layers.conv2d(
        source_tensors,
        filters=4,
        kernel_size=3,
        padding='valid',
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=initializer)

    # NOTE: conv net chain
    for _ in range(2):
        tensors = tf.layers.conv2d(
            tensors,
            filters=4,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer)

    # NOTE: map to target layer
    tensors = tf.layers.conv2d(
        tensors,
        filters=1,
        kernel_size=3,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer)

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

    return {
        'source_tensors': source_tensors,
        'target_tensors': target_tensors,
        'predictions': predictions,
        'loss': loss,
        'step': step,
        'trainer': trainer,
    }
