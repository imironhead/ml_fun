"""
"""
import numpy as np
import tensorflow as tf

import life
import model_handcraft


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
    generate a batch of data. the source tensors are padded (we build a model
    with padded source tensors input)
    """
    # NOTE: pad source for vertical and horizontal cycles
    source_tensors = np.random.rand(size, height, width, 1)
    target_tensors = np.zeros((size, height, width, 1), dtype=np.float)

    # NOTE: transfer source_tensors into binary matrics
    source_tensors[source_tensors >= threshold] = 1.0
    source_tensors[source_tensors < threshold] = 0.0

    # NOTE: pad for the model
    source_tensors = circular_pad(source_tensors)

    # NOTE: live goes on
    for i in range(size):
        target_tensors[i, :, :, 0] = \
            life.life_goes_on(source_tensors[i, 1:-1, 1:-1, 0])

    return source_tensors, target_tensors


def trial(num_trials):
    """
    """
    num_samples = 0
    num_correct = 0

    # NOTE: build the handcrafted model
    model = model_handcraft.build_model(32, 32)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for step in range(num_trials):
            source_tensors, target_tensors = \
                generate_mini_batch(128, 32, 32)

            feeds = {
                model['source_tensors']: source_tensors,
            }

            predictions = session.run(model['predictions'], feed_dict=feeds)

            num_samples += 128

            guess = predictions.reshape(128, -1)
            truth = target_tensors.reshape(128, -1)

            guess[guess > 0.5] = 1.0
            guess[guess < 1.0] = 0.0

            num_correct_cells = np.sum(truth == guess, axis=1).astype(np.int)
            num_correct += np.sum(num_correct_cells == 1024)

        print('accuracy: {}'.format(float(num_correct) / float(num_samples)))


def main(_):
    """
    """
    trial(tf.app.flags.FLAGS.num_trials)


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer(
        'num_trials', 128, 'number of prediction trials')

    tf.app.run()

