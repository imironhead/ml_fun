"""
"""
import numpy as np
import os
import skimage.io
import skimage.transform
import tensorflow as tf

import life
import model

FLAGS = tf.app.flags.FLAGS


def circular_pad(tensors):
    """
    """
    tensors = np.pad(
        tensors,
        pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]],
        mode='constant')

    # NOTE: vertical cycle
    tensors[:, 0, :, 0] = tensors[:, -2, :, 0]
    tensors[:, -1, :, 0] = tensors[:, 1, :, 0]

    # NOTE: horizontal cycle
    tensors[:, :, 0, 0] = tensors[:, :, -2, 0]
    tensors[:, :, -1, 0] = tensors[:, :, 1, 0]

    # NOTE: 4 corners
    tensors[:, 0, 0, 0] = tensors[:, -2, -2, 0]
    tensors[:, -1, 0, 0] = tensors[:, 1, -2, 0]

    tensors[:, 0, -1, 0] = tensors[:, -2, 1, 0]
    tensors[:, -1, -1, 0] = tensors[:, 1, 1, 0]

    return tensors

def generate_mini_batch(size, height, width, threshold=0.5):
    """
    """
    # NOTE: pad source for vertical and horizontal cycles
    source_tensors = np.random.rand(size, height + 2, width + 2)
    target_tensors = np.zeros((size, height, width), dtype=np.float)

    # NOTE: transfer source_tensors into binary matrics
    source_tensors[source_tensors >= threshold] = 1.0
    source_tensors[source_tensors < threshold] = 0.0

    # NOTE: vertical cycle
    source_tensors[:, 0, :] = source_tensors[:, -2, :]
    source_tensors[:, -1, :] = source_tensors[:, 1, :]

    # NOTE: horizontal cycle
    source_tensors[:, :, 0] = source_tensors[:, :, -2]
    source_tensors[:, :, -1] = source_tensors[:, :, 1]

    # NOTE: 4 corners
    source_tensors[:, 0, 0] = source_tensors[:, -2, -2]
    source_tensors[:, -1, 0] = source_tensors[:, 1, -2]

    source_tensors[:, 0, -1] = source_tensors[:, -2, 1]
    source_tensors[:, -1, -1] = source_tensors[:, 1, 1]

    # NOTE: live goes on
    for i in range(size):
        target_tensors[i] = life.life_goes_on(source_tensors[i, 1:-1, 1:-1])

    # NOTE: reshape for cnn
    source_tensors = np.expand_dims(source_tensors, axis=3)
    target_tensors = np.expand_dims(target_tensors, axis=3)

    return source_tensors, target_tensors


def train_one_step(session, life_model):
    """
    """
    source_tensors, target_tensors = \
        generate_mini_batch(128, 32, 32)

    feeds = {
        life_model['source_tensors']: source_tensors,
        life_model['target_tensors']: target_tensors,
    }

    fetches = {
        'step': life_model['step'],
        'loss': life_model['loss'],
        'trainer': life_model['trainer'],
    }

    fetched = session.run(fetches, feed_dict=feeds)

    return fetched['step'], fetched['loss']


def validate(session, life_model):
    """
    """
    num_samples = 0
    num_correct = 0

    for _ in range(10):
        source_tensors, target_tensors = generate_mini_batch(128, 32, 32)

        feeds = {
            life_model['source_tensors']: source_tensors,
        }

        fetches = {
            'predictions': life_model['predictions'],
         }

        fetched = session.run(fetches, feed_dict=feeds)

        num_samples += 128

        guess = fetched['predictions'].reshape(128, -1)
        truth = target_tensors.reshape(128, -1)

        guess[guess > 0.5] = 1.0
        guess[guess < 1.0] = 0.0

        num_correct += np.sum(np.sum(truth == guess, axis=1).astype(np.int) == 1024)


    return float(num_correct) / float(num_samples)


def train():
    """
    """
    # NOTE: a simple model, skip loading checkpoint
    # NOTE: a simple model, skip summary

    life_model = model.build_model(32)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for _ in range(20000):
            step, loss = train_one_step(session, life_model)

            if step % 1000 == 0:
                print('loss[{:>8}]: {}'.format(step, loss))

            if step % 1000 == 0:
                accuracy = validate(session, life_model)

                print('accu[{:>8}]: {}'.format(step, accuracy))

        tf.train.Saver().save(
            session, FLAGS.ckpt_path, global_step=life_model['step'])


def read_world_tensors(path):
    """
    """
    if path is None or not os.path.isfile(path):
        raise Exception('invalid path: {}'.format(path))

    with open(path, encoding='utf-8') as f:
        lines = [line.strip() for line in f]

        world = [[1 if x == 'o' else 0 for x in line] for line in lines]

    lengths = [len(line) for line in world]

    if len(lengths) == 0 or any([l != lengths[0] for l in lengths]):
        raise Exception('invalid file')

    h, w = len(lengths), lengths[0]

    world_tensors = np.array(world).astype(np.float32)
    world_tensors = np.reshape(world_tensors, [1, h, w, 1])

    return world_tensors


def predict():
    """
    """
    world_tensors = read_world_tensors(FLAGS.world_path)

    h, w = world_tensors.shape[1:3]

    life_model = model.build_model(h)

    with tf.Session() as session:
        tf.train.Saver().restore(session, FLAGS.ckpt_path)

        for step in range(FLAGS.predict_length):
            world_tensors = circular_pad(world_tensors)

            feeds = {life_model['source_tensors']: world_tensors}

            world_tensors = session.run(
                life_model['predictions'], feed_dict=feeds)

            # NOTE: save
            output_image_path = os.path.join(
                FLAGS.output_path, '{:0>8}.png'.format(step))

            scaled_image = skimage.transform.resize(
                world_tensors[0, :, :, 0], [32 * h, 32 * w], order=0)

            skimage.io.imsave(output_image_path, scaled_image)

            # NOTE: save 2nd
            world_tensors[world_tensors > 0.5] = 1.0
            world_tensors[world_tensors < 1.0] = 0.0


def main(_):
    """
    """
    if FLAGS.predict:
        predict()
    else:
        train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('meta_path', None, '')
    tf.app.flags.DEFINE_string('world_path', None, '')
    tf.app.flags.DEFINE_string('output_path', None, '')

    tf.app.flags.DEFINE_boolean('predict', False, '')
    tf.app.flags.DEFINE_integer('predict_length', 128, '')

    tf.app.run()

