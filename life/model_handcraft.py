"""
"""
import numpy as np
import tensorflow as tf


def build_model(world_height, world_width):
    """
    """
    # NOTE: world at T, should be padded circularly.
    tensors = source_tensors = tf.placeholder(
        shape=[None, world_height + 2, world_width + 2, 1],
        dtype=tf.float32,
        name='source_tensors')

    # NOTE:
    weights = np.array([
        [[[1.0]], [[1.0]], [[1.0]]],
        [[[1.0]], [[0.5]], [[1.0]]],
        [[[1.0]], [[1.0]], [[1.0]]]], dtype=np.float32)

    const_conv = tf.constant(weights, name='w')

    tensors = tf.nn.conv2d(tensors, const_conv, [1] * 4, padding='VALID')

    # NOTE:
    bias = np.array([-3.0], dtype=np.float32);

    const_bias = tf.constant(bias, name='b')

    tensors = tf.nn.bias_add(tensors, bias)

    # NOTE:
    predictions = tf.exp(- tf.square(tensors))

    return {
        'source_tensors': source_tensors,
        'predictions': predictions,
    }

