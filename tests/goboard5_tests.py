import numpy as np
import scipy.signal as signal
import jax.numpy as jnp
import pax

from games.dsu import DSU
from games.go_game import GoBoard5x5
from tests import go, coords

assert go.N == 5

BEST_C2_GAME = "B[cd];W[cc];B[dc];W[dd];B[de];W[bd];B[ed];W[cb];B[be];W[ad];B[db];W[ca];B[ab];W[bb];B[ce];W[ac];B[da];W[aa];B[ae]"


def board_from_sgf(s):
    env = GoBoard5x5()
    for i, move_str in enumerate(s.split(';')):
        move = coords.to_flat(coords.from_sgf(move_str[2:4]))
        env, reward = env.step(jnp.array(move, dtype=int))
        assert reward == 0
        assert not env.is_terminated()
        # invalid_actions = env.invalid_actions()
        # assert jnp.sum(invalid_actions) <= i + 1

    return env


def board_from_gtp(s):
    env = GoBoard5x5()
    for i, move_str in enumerate(s.split(';')):
        move = coords.to_flat(coords.from_gtp(move_str))
        env, reward = env.step(jnp.array(move, dtype=int))
        assert reward == 0
        assert not env.is_terminated()
    return env


def apply_move(env, move_gtp: str):
    move = coords.to_flat(coords.from_gtp(move_gtp))
    return env.step(jnp.array(move, dtype=int))


def test_simple():
    env = GoBoard5x5()
    assert env.num_actions() == 26 and env.turn == 1
    env, reward = apply_move(env, 'C2')
    assert reward == 0
    assert env.num_actions() == 26
    assert not env.is_terminated()
    assert env.turn == -1
    env.render()
    score_black = env.final_score(env.board, 1)
    score_white = env.final_score(env.board, -1)
    assert score_black + score_white == 0
    print(score_black)
    invalid_actions = env.invalid_actions()
    assert jnp.sum(invalid_actions) == 1
    assert invalid_actions.shape == (26,)


def test_BEST_C2_GAME():
    env = board_from_sgf(BEST_C2_GAME)
    score = env.final_score(env.board, 1)
    env.render()
    # black only counted 2 eyes, missing 3 pts
    assert score + 3 == 2.5

    # black E4 will convert that space into two eyes
    env, _ = apply_move(env, 'pass')
    env, _ = apply_move(env, 'E4')
    score = env.final_score(env.board, 1)
    env.render()
    assert score == 2.5


def test_ko():
    KO_SETUP = 'C2;C3;D3;B2;E2;C1;D1;' + 'D2'
    env = board_from_gtp(KO_SETUP)
    env.render()
    # C2 is not allowed
    env, reward = apply_move(env, 'C2')
    assert reward == -1
    # and game over
    assert env.is_terminated()

    # env, reward = apply_move(env, 'pass')
    # assert reward == 0
    # env, reward = apply_move(env, 'pass')
    # assert reward != 0

    env.render()
    score = env.final_score(env.board, 1)
    print(reward, score)


@pax.pure
def get_all_roots_pure(dsu: DSU):
    res = dsu.get_all_roots()
    return res


def test_dsu():
    env = GoBoard5x5()
    A5_flat = coords.to_flat(coords.from_gtp('A5'))
    B5_flat = coords.to_flat(coords.from_gtp('B5'))
    env, reward = apply_move(env, 'A5')
    env, reward = apply_move(env, 'C2')
    env, reward = apply_move(env, 'B5')
    # env, reward = apply_move(env, 'C3')
    env.render()
    env.dsu.pp()
    result = env.dsu.find_set_pure(A5_flat)
    result2 = env.dsu.find_set_pure(B5_flat)
    print(result)
    assert result == result2
    roots = get_all_roots_pure(env.dsu)
    print(roots, len(roots))


def test_chain_border():
    """ given a chain, find its border
    1. given a chain (indicator bitmap), find its 1-neighbor expansion: just convolution with
    the cross-shaped kernel?
    2. expansion - original chain = border

    chain_border() quite doable; but this is only one chain. We may need to loop over all chains (of empty spaces).
    Also need to identify those chains in the first place. DSU?
    """


def setup_neighbor_filter(with_center=False):
    m = np.zeros((3, 3), dtype=int)
    m[0, 1] = m[2, 1] = m[1, 0] = m[1, 2] = 1
    if with_center:
        m[1, 1] = 1
    return m


def test_filter():
    f = setup_neighbor_filter()
    m = (f == 1).astype(int)
    print(m)


def find_reach(board, neighbor_filter, color: int, max_steps=go.N*2):
    """ find where colored stones can reach in empty spaces, in parallel

    max_steps: how many steps to reach every empty space from the closet stone.
      For a typical game, it's less than 5 for 19x19 game

    :return: an indicator array (only in empty spaces)
    """
    empty_spaces = board == 0

    work_board = (board == color).astype(int)
    for i in range(max_steps):
        m = signal.convolve(work_board, neighbor_filter, mode='same')
        m = np.logical_and(m > 0, empty_spaces)
        work_board = m.astype(int)
    return work_board


def tromp_score(board: np.ndarray, komi=0.5, max_steps=5):
    filter = setup_neighbor_filter(with_center=True)
    black_reach = find_reach(board, filter, 1, max_steps=max_steps)
    white_reach = find_reach(board, filter, -1, max_steps=max_steps)

    score_board = black_reach - white_reach + board
    return score_board.sum() - komi, score_board


def test_black_floodfill():
    """ from all black stones, flood-fill on neighboring empty spaces """
    # setup test board
    board = np.zeros((go.N, go.N), dtype=int)
    board[1, :] = 1
    board[3, :] = -1
    board[0, 1] = 1
    board[0, 2] = -1
    print('original board')
    print(board)

    score, score_board = tromp_score(board, max_steps=2)
    print('score:', score)
    print(score_board)


def test_tromp_score():
    """ for every empty spot, find the chain & its border (colors);
    determine ownership of those chains (black/white/both); sum up scores

    Imagine we propagate binary (white/black) messages across empty spaces in parallel: each empty space remembers
    what messages (a set) it has seen. After 2*go.N cycles, we know the ownership of each space.
      The key is that messages can only pass thru empty spaces, not stones.
      Now I only know convolution works this way; and we have 3 unique states. Even/odd number may not work, but
    real/im/0 for black/white/space should work. Or we do it in two passes: one for black, one for white.
    """
    env = board_from_sgf(BEST_C2_GAME)
    env.render()
    board = env.board.to_py()
    score, score_board = tromp_score(board, max_steps=2)
    assert score == 2.5
    print(score_board)
