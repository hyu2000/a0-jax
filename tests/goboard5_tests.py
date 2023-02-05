import pickle

import numpy as np
import jax.scipy.signal as signal
import jax.numpy as jnp
import pax
import tensorflow as tf

from games.dsu import DSU
from games.go_game import GoBoard5x5, GoBoard5C2
import go
import coords
from games.go_logic import tromp_score
from policies.resnet_policy import ResnetPolicyValueNet128

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
    score_black = env.rough_score(env.board, 1)
    score_white = env.rough_score(env.board, -1)
    assert score_black + score_white == 0
    print(score_black)
    invalid_actions = env.invalid_actions()
    assert jnp.sum(invalid_actions) == 1
    assert invalid_actions.shape == (26,)
    obs = env.observation()
    print(obs.shape, obs.dtype)


def test_BEST_C2_GAME():
    env = board_from_sgf(BEST_C2_GAME)
    score = env.rough_score(env.board, 1)
    env.render()
    # black only counted 2 eyes, missing 3 pts
    assert score + 3 == 2.5

    # black E4 will convert that space into two eyes
    env, _ = apply_move(env, 'pass')
    env, _ = apply_move(env, 'E4')
    score = env.rough_score(env.board, 1)
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
    score = env.rough_score(env.board, 1)
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


def test_tromp_score_simple():
    """ from all black stones, flood-fill on neighboring empty spaces """

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
    assert score == -4.5


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


def test_np_indexing():
    arr = np.ones((2, 2))
    arrnone = arr[None]  # newaxis
    assert arrnone.ndim == arr.ndim + 1
    print(arrnone.shape)
    arrns = arr[None, None]
    assert arrns.shape == (1, 1, 2, 2)


def test_check_suicide():
    """ check whether a move is suicide. The current game logic detects it after the fact
    """


def test_saved_model():
    tf_model_path: str = "../exp-go5C2/tfmodel/myconv"

    env0 = GoBoard5x5()
    env01, _ = apply_move(env0, 'E3')
    env02, _ = apply_move(env0, 'D2')
    env1 = GoBoard5C2()
    env1, _ = apply_move(env1, 'C3')
    env11, reward = apply_move(env1, 'D3')
    env12, reward = apply_move(env1, 'D1')
    input1 = env1.canonical_observation()
    m1 = tf.saved_model.load(tf_model_path)
    o1 = m1.f(input1)
    print('step', env1.count, o1)
    obss = []
    for env in [env0, env01, env02, env11, env12]:
        # env.render()
        obss.append(env.canonical_observation())
    xs = jnp.stack(obss)
    probs, values = m1.f_batched(xs)
    print(probs.shape, values)


def setup5x5(ckpt_filename='../exp-go5C2/colab/go_agent_5-25.ckpt'):
    env0 = GoBoard5x5()
    agent = ResnetPolicyValueNet128(
        input_dims=env0.observation().shape,
        num_actions=env0.num_actions(),
    )
    with open(ckpt_filename, "rb") as f:
        dic = pickle.load(f)
        agent = agent.load_state_dict(dic["agent"])
    agent = agent.eval()
    return env0, agent


def test_batch_eval():
    """ dnn(, batched=True) """
    env0, agent = setup5x5()

    obs = []
    obs.append(env0.canonical_observation())
    env, _ = apply_move(env0, 'C2')
    obs.append(env.canonical_observation())
    env, _ = apply_move(env0, 'D1')
    obs.append(env.canonical_observation())

    xs = jnp.stack(obs)
    probs, values = agent(xs, batched=True)
    print([coords.flat_to_gtp(jnp.argmax(x)) for x in probs])
    print(values)
