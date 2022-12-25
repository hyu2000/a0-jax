import pytest
import numpy as np
import jax.numpy as jnp
from go_game import GoBoard5x5
import coords


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


def test_simple():
    env = GoBoard5x5()
    assert env.num_actions() == 26 and env.turn == 1
    c2 = coords.to_flat(coords.from_gtp('C2'))
    env, reward = env.step(jnp.array(c2, dtype=int))
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


def test_ko():
    KO_SETUP = 'C2;C3;D3;B2;E2;C1;D1;' + 'D2'
    env = board_from_gtp(KO_SETUP)
    env.render()
    # C2 is not allowed
    move_c2 = coords.to_flat(coords.from_gtp('C2'))
    _, reward = env.step(jnp.array(move_c2, dtype=int))
    assert reward == -1

    env, reward = env.step(jnp.array(coords.to_flat(None)))
    assert reward == 0
    env, reward = env.step(jnp.array(coords.to_flat(None)))
    assert reward != 0

    env.render()
    score = env.final_score(env.board, 1)
    print(reward, score)
