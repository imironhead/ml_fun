"""
"""
import numpy as np
import tensorflow as tf


def build_gaussian_model(world_height, world_width):
    """
    """
    print('build gaussian model')

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


def build_relu_model(world_height, world_width):
    """
    """
    print('building relu model')

    # NOTE: world at T, should be padded circularly.
    tensors = source_tensors = tf.placeholder(
        shape=[None, world_height + 2, world_width + 2, 1],
        dtype=tf.float32,
        name='source_tensors')

    # NOTE: x => 0.0/0.5/1.0/...2.5/3/3/5...8.5
    weights = np.array([
        [[[1.0]], [[1.0]], [[1.0]]],
        [[[1.0]], [[0.5]], [[1.0]]],
        [[[1.0]], [[1.0]], [[1.0]]]], dtype=np.float32)

    const_conv = tf.constant(weights, name='k0')

    tensors = tf.nn.conv2d(tensors, const_conv, [1] * 4, padding='VALID')

    # NOTE: ReLU(5x-10.25) / ReLU(5x-12.25) / ReLU(5x-17.75) / ReLU(5x-19.75)
    weights = np.array([[[[5.0, 5.0, 5.0, 5.0]]]], dtype=np.float32)

    const_conv = tf.constant(weights, name='k1')

    tensors = tf.nn.conv2d(tensors, const_conv, [1] * 4, padding='VALID')

    # NOTE:
    bias = np.array([-10.25, -12.25, -17.75, -19.75], dtype=np.float32);

    const_bias = tf.constant(bias, name='b1')

    tensors = tf.nn.bias_add(tensors, bias)

    # NOTE:
    tensors = tf.nn.relu(tensors)

    # NOTE: ReLU(5x-10.25) - ReLU(5x-12.25) - ReLU(5x-17.75) + ReLU(5x-19.75)
    weights = np.array([[[[1.0], [-1.0], [-1.0], [1.0]]]], dtype=np.float32)

    const_conv = tf.constant(weights, name='k2')

    tensors = tf.nn.conv2d(tensors, const_conv, [1] * 4, padding='VALID')

    # NOTE:
    bias = np.array([-1.0], dtype=np.float32);

    const_bias = tf.constant(bias, name='b2')

    tensors = tf.nn.bias_add(tensors, bias)

    # NOTE: z = sigmoid(y)
    predictions = tensors

    return {
        'source_tensors': source_tensors,
        'predictions': predictions,
    }


def build_model(world_height, world_width, name='gaussian'):
    """
    """
    if name == 'gaussian':
        return build_gaussian_model(world_height, world_width)
    else:
        return build_relu_model(world_height, world_width)

