import numpy as np
import jax.scipy.signal as signal
import jax.numpy as jnp


def setup_neighbor_filter(with_center=False):
    m = jnp.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
        ], dtype=int)
    if with_center:
        return m.at[1, 1].set(1)
    return m


def test_filter():
    f = setup_neighbor_filter()
    print(f)
    m = (f == 1).astype(int)
    print(m)


NEIGHBOR_FILTER = setup_neighbor_filter(with_center=True)


def find_reach(board, neighbor_filter, color: int, num_steps: int):
    """ find where colored stones can reach in empty spaces. Implemented as convolution
    which could be faster on GPUs by exploiting parallelism.

    num_steps: how many iterations to apply convolution. go.N * 2 should be safe.
    We could try detect work_board not expanding, but it will cost a little

    :return: an indicator array (only in empty spaces)
    """
    empty_spaces = board == 0

    work_board = (board == color).astype(int)
    for i in range(num_steps):
        m = signal.convolve(work_board, neighbor_filter, mode='same')
        m = jnp.logical_and(m > 0, empty_spaces)
        work_board = m.astype(int)
    return work_board


def tromp_score(board: np.ndarray, komi=0.5, max_steps=5):
    """
    max_steps: the higher it is, the more accurate. go.N * 2 is safe, but it's expensive.
    5 should be enough for a typical end-game (even on 19x19 board).
    For a game in earlier stages, Tromp score is not useful anyways.
    """
    black_reach = find_reach(board, NEIGHBOR_FILTER, 1, num_steps=max_steps)
    white_reach = find_reach(board, NEIGHBOR_FILTER, -1, num_steps=max_steps)

    # seki area can be reached by both black and white, therefore cancels out
    score_board = black_reach - white_reach + board
    return score_board.sum() - komi, score_board
