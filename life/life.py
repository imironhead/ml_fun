"""
"""
import numpy as np


def life_goes_on(world):
    """
    """
    old_world = np.zeros_like(world, dtype=np.int)

    # NOTE: sanity check
    old_world[world >= 0.5] = 1

    # NOTE: counting horizontal neighbors
    tmp_world = old_world.copy()

    tmp_world[:, :-1] += old_world[:, 1:]
    tmp_world[:, 1:] += old_world[:, :-1]
    tmp_world[:, -1] += old_world[:, 0]
    tmp_world[:, 0] += old_world[:, -1]

    # NOTE: counting vertical neighbors
    new_world = tmp_world.copy()

    new_world[:-1, :] += tmp_world[1:, :]
    new_world[1:, :] += tmp_world[:-1, :]
    new_world[-1, :] += tmp_world[0, :]
    new_world[0, :] += tmp_world[-1, :]

    # NOTE: remove duplicated self
    new_world -= old_world

    # NOTE: positions with 2 live neighbors
    position = (new_world == 2)

    # NOTE: if old_world[x] is live, then new_world[x] becomes 3
    #       if old_world[x] is dead, then new_world[x] remains 2
    new_world[position] += old_world[position]

    # NOTE: now only if new_world[x] is 3 will be live
    new_world[new_world != 3] = 0
    new_world[new_world == 3] = 1

    return new_world.astype(np.float)


def print_life(world):
    """
    """
    temp_world = np.zeros_like(world, dtype=np.int)

    temp_world[world >= 0.5] = 1

    for row in temp_world:
        print(''.join(['.' if x == 0 else 'X' for x in row]))


if __name__ == '__main__':
    print_life(life_goes_on(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.int)))

    print_life(life_goes_on(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]], dtype=np.int)))

    print_life(life_goes_on(np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]], dtype=np.int)))

    print_life(life_goes_on(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]], dtype=np.int)))

    print_life(life_goes_on(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]], dtype=np.int)))
